"""Ollama backend via the OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

import base64
import json
import os
import re
from collections.abc import Iterable
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from openai import OpenAI

from jarvis_jr.confirm import Confirmer
from jarvis_jr.tools.registry import ToolRegistry


# Qwen/DeepSeek-style chain-of-thought blocks that leak into the spoken reply.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Sometimes the closing tag arrives without a matching opener — strip a stray one too.
_TRAILING_THINK = re.compile(r"</?think>", re.IGNORECASE)
# Gemma 4-style thought channel: <|channel>thought\n ... <channel|>
_GEMMA_THOUGHT = re.compile(r"<\|channel>thought\n.*?<channel\|>", re.DOTALL)
# Strip stray/unclosed channel markers too (e.g. truncated thought blocks).
_GEMMA_CHANNEL_TAG = re.compile(r"<\|channel>(thought\n)?|<channel\|>")


def _strip_thinking(text: str) -> str:
    text = _THINK_BLOCK.sub("", text)
    text = _TRAILING_THINK.sub("", text)
    text = _GEMMA_THOUGHT.sub("", text)
    text = _GEMMA_CHANNEL_TAG.sub("", text)
    return text.strip()


def _to_openai_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in schemas
    ]


class OllamaClient:
    """Multi-turn local-LLM client using Ollama's OpenAI-compatible endpoint.

    Defaults to a vision-capable Qwen2.5 VL model. If the chosen model has no
    vision support, images are silently dropped.
    """

    def __init__(
        self,
        model: str,
        tools: ToolRegistry,
        confirmer: Confirmer,
        system_prompt: str,
        require_confirmation_for: Iterable[str] = (),
        max_tokens: int = 1024,
        max_iterations: int | None = None,
        base_url: str = "http://localhost:11434",
        api_key: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        # Ollama ignores the key; LM Studio may require one (set LOCAL_LLM_API_KEY
        # in .env or `api_key` in the config). "ollama" is a harmless placeholder.
        self.api_key = api_key or os.getenv("LOCAL_LLM_API_KEY") or "ollama"
        self._client = OpenAI(base_url=f"{self.base_url}/v1", api_key=self.api_key)
        self.model = model
        self.tools = tools
        self.confirmer = confirmer
        self.system_prompt = system_prompt
        self.require_confirmation_for = set(require_confirmation_for)
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self._openai_tools = _to_openai_tools(tools.schemas)
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def health_check(self) -> bool:
        """Return True if the server is reachable (Ollama or any OpenAI-compatible server)."""
        for path in ("/api/tags", "/v1/models"):  # Ollama-native, then OpenAI-compatible (LM Studio etc.)
            try:
                req = Request(
                    f"{self.base_url}{path}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                with urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except (URLError, TimeoutError, OSError):
                continue
        return False

    def submit(self, user_text: str, image_jpeg_bytes: bytes | None = None) -> str:
        if not self.health_check():
            raise RuntimeError(
                f"No local LLM server reachable at {self.base_url}. "
                "Start Ollama (`ollama serve`) or LM Studio's server "
                "(Developer tab -> Start Server, then set base_url to http://localhost:1234)."
            )
        if image_jpeg_bytes is not None:
            b64 = base64.b64encode(image_jpeg_bytes).decode("ascii")
            # Image before text — Gemma 4 best practice; harmless for other models.
            content: Any = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
                {"type": "text", "text": user_text},
            ]
        else:
            content = user_text
        self.messages.append({"role": "user", "content": content})

        iteration = 0
        while True:
            iteration += 1
            if self.max_iterations is not None and iteration > self.max_iterations:
                return (
                    f"[autocoder] max_iterations ({self.max_iterations}) reached "
                    "without the model finishing. Stopping."
                )
            response = self._client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self._openai_tools,
                tool_choice="auto",
                max_tokens=self.max_tokens,
                # Gemma 4 recommended sampling; sensible for most local models.
                temperature=1.0,
                top_p=0.95,
                extra_body={"top_k": 64},
            )
            msg = response.choices[0].message
            assistant_text = _strip_thinking(msg.content or "")
            tool_calls = msg.tool_calls or []

            # Echo assistant message back into history (must include tool_calls if any)
            assistant_entry: dict[str, Any] = {"role": "assistant", "content": assistant_text or ""}
            if tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            self.messages.append(assistant_entry)

            if not tool_calls:
                return assistant_text

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if name in self.require_confirmation_for:
                    prompt = assistant_text or self.tools.describe(name, args)
                    if not self.confirmer(prompt):
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": "User declined.",
                            }
                        )
                        continue
                result = self.tools.dispatch(name, args)
                self.messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
