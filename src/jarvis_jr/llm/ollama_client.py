"""Ollama backend via the OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Iterable
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from openai import OpenAI

from jarvis_jr.confirm import Confirmer
from jarvis_jr.tools.registry import ToolRegistry


# Qwen/DeepSeek-style chain-of-thought blocks that leak into the spoken reply.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Sometimes the closing tag arrives without a matching opener — strip a stray one too.
_TRAILING_THINK = re.compile(r"</?think>", re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    text = _THINK_BLOCK.sub("", text)
    text = _TRAILING_THINK.sub("", text)
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
    ):
        self.base_url = base_url.rstrip("/")
        self._client = OpenAI(base_url=f"{self.base_url}/v1", api_key="ollama")
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
        """Return True if the Ollama server is reachable."""
        try:
            with urlopen(f"{self.base_url}/api/tags", timeout=2) as resp:
                return resp.status == 200
        except (URLError, TimeoutError, OSError):
            return False

    def submit(self, user_text: str, image_jpeg_bytes: bytes | None = None) -> str:
        if not self.health_check():
            raise RuntimeError(
                f"Ollama is not reachable at {self.base_url}. "
                "Start it in another terminal with: `ollama serve`"
            )
        if image_jpeg_bytes is not None:
            b64 = base64.b64encode(image_jpeg_bytes).decode("ascii")
            content: Any = [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
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
