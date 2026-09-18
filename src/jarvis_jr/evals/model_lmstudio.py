"""The single adapter for a local OpenAI-compatible chat server (LM Studio, Ollama).

This is the only file in evals that knows the wire format. The rest sees
ModelReply and ToolCall. Swapping models is a constructor argument; swapping
servers is one new class with the same `complete` method.

Unlike the assistant's per-provider clients in jarvis_jr.llm, this adapter
owns NO tool loop — the loop lives once, in evals.agent — so the same
trajectory recording and grading apply regardless of backend.
"""

from __future__ import annotations

import json
import os
import re
import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from jarvis_jr.evals.domain import ModelReply, ToolCall

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "google/gemma-4-26b-a4b-qat"

# Thinking blocks that some local models leak into content when the server
# doesn't split reasoning out (Qwen <think>, Gemma 4 thought channel).
# Deliberately duplicated from jarvis_jr.llm.ollama_client rather than
# imported: evals should not depend on the assistant's client stack.
_THINKING = re.compile(
    r"<think>.*?</think>|</?think>|<\|channel>thought\n.*?<channel\|>|<\|channel>(thought\n)?|<channel\|>",
    re.DOTALL | re.IGNORECASE,
)


def parse_reply(response_json: dict) -> ModelReply:
    """Turn one /chat/completions response (as a dict) into a ModelReply. Pure; testable."""
    message = response_json["choices"][0]["message"]
    calls = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_raw": fn.get("arguments", "")}
        calls.append(ToolCall(tool=fn.get("name", ""), args=args, call_id=tc.get("id", "")))
    content = _THINKING.sub("", message.get("content") or "").strip()
    return ModelReply(content=content, tool_calls=tuple(calls), raw_message=message)


class LMStudioModel:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        timeout: int = 300,
        api_key: str = "",
    ):
        self.model = model
        self.temperature = temperature
        # LM Studio may require a token (same env var the assistant uses).
        key = api_key or os.environ.get("LOCAL_LLM_API_KEY") or "lm-studio"
        self._client = OpenAI(base_url=base_url, api_key=key, timeout=timeout)

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> ModelReply:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        # LM Studio can unload a JIT model mid-run ("Model unloaded") or drop a
        # connection; the next request reloads it. Retry a few times with backoff
        # rather than losing a 20-minute episode to a transient.
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self._client.chat.completions.create(**kwargs)
                return parse_reply(response.model_dump())
            except (APIConnectionError, APITimeoutError, APIStatusError) as e:
                last_error = e
                msg = str(e).lower()
                transient = isinstance(e, (APIConnectionError, APITimeoutError)) or (
                    any(k in msg for k in ("unloaded", "failed to load", "loading", "canceled"))
                    or getattr(e, "status_code", 0) >= 500
                )
                if not transient or attempt == 3:
                    raise
                time.sleep(15 * (attempt + 1))  # a 15 GB model takes a while to (re)load
        raise last_error  # pragma: no cover
