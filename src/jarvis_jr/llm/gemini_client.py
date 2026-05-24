"""Google Gemini backend via the google-genai SDK."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable
from typing import Any

from google import genai
from google.genai import types

from jarvis_jr.confirm import Confirmer
from jarvis_jr.tools.registry import ToolRegistry


# Retry tuning for 429 RESOURCE_EXHAUSTED. The free tier is 5 RPM; bursts will
# trip it. Gemini's error includes a `retryDelay: "Ns"` hint we honor.
_MAX_429_RETRIES = 4
_DEFAULT_429_BACKOFF_S = 8.0
_MAX_429_BACKOFF_S = 65.0
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", re.IGNORECASE)


def _parse_retry_delay_seconds(err_str: str) -> float | None:
    m = _RETRY_DELAY_RE.search(err_str)
    return float(m.group(1)) + 0.5 if m else None  # small buffer past the hinted wait


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc)
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg


def _to_gemini_tools(schemas: list[dict[str, Any]]) -> list[types.Tool]:
    """Convert the registry's Anthropic-style tool schemas into a Gemini Tool list."""
    decls = [
        types.FunctionDeclaration(
            name=s["name"],
            description=s["description"],
            parameters_json_schema=s["input_schema"],
        )
        for s in schemas
    ]
    return [types.Tool(function_declarations=decls)]


class GeminiClient:
    """Multi-turn Gemini client with vision + tool use + manual confirmation."""

    def __init__(
        self,
        model: str,
        tools: ToolRegistry,
        confirmer: Confirmer,
        system_prompt: str,
        require_confirmation_for: Iterable[str] = (),
        max_tokens: int = 1024,
        max_iterations: int | None = None,
        api_key: str | None = None,
    ):
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Get a free key from "
                "https://aistudio.google.com/apikey and add it to your .env file."
            )
        self._client = genai.Client(api_key=key)
        self.model = model
        self.tools = tools
        self.confirmer = confirmer
        self.system_prompt = system_prompt
        self.require_confirmation_for = set(require_confirmation_for)
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self._gemini_tools = _to_gemini_tools(tools.schemas)
        self.history: list[types.Content] = []

    def reset(self) -> None:
        self.history = []

    def submit(self, user_text: str, image_jpeg_bytes: bytes | None = None) -> str:
        parts: list[types.Part] = []
        if image_jpeg_bytes is not None:
            parts.append(
                types.Part(inline_data=types.Blob(data=image_jpeg_bytes, mime_type="image/jpeg"))
            )
        parts.append(types.Part(text=user_text))
        self.history.append(types.Content(role="user", parts=parts))

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            tools=self._gemini_tools,
            max_output_tokens=self.max_tokens,
            # Disable SDK-side auto-calling so we can run confirmations between calls.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        iteration = 0
        while True:
            iteration += 1
            if self.max_iterations is not None and iteration > self.max_iterations:
                return (
                    f"[autocoder] max_iterations ({self.max_iterations}) reached "
                    "without the model finishing. Stopping."
                )
            response = self._generate_with_retry(config)
            candidate = response.candidates[0]
            self.history.append(candidate.content)

            response_parts = candidate.content.parts or []
            text_parts = [p.text for p in response_parts if p.text]
            function_calls = [p.function_call for p in response_parts if p.function_call]
            assistant_text = " ".join(t.strip() for t in text_parts).strip()

            if not function_calls:
                return assistant_text

            tool_response_parts: list[types.Part] = []
            for fc in function_calls:
                args = dict(fc.args) if fc.args else {}
                name = fc.name
                if name in self.require_confirmation_for:
                    prompt = assistant_text or self.tools.describe(name, args)
                    if not self.confirmer(prompt):
                        tool_response_parts.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=name,
                                    response={"declined": True, "message": "User declined"},
                                )
                            )
                        )
                        continue
                raw = self.tools.dispatch(name, args)
                tool_response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=name, response=_as_response_dict(raw)
                        )
                    )
                )

            self.history.append(types.Content(role="user", parts=tool_response_parts))

    def _generate_with_retry(self, config: types.GenerateContentConfig):
        """Call models.generate_content with 429-aware exponential backoff."""
        backoff = _DEFAULT_429_BACKOFF_S
        last_exc: BaseException | None = None
        for attempt in range(_MAX_429_RETRIES + 1):
            try:
                return self._client.models.generate_content(
                    model=self.model, contents=self.history, config=config
                )
            except Exception as e:  # noqa: BLE001
                if not _is_rate_limit_error(e) or attempt == _MAX_429_RETRIES:
                    raise
                last_exc = e
                hinted = _parse_retry_delay_seconds(str(e))
                wait = min(_MAX_429_BACKOFF_S, hinted if hinted is not None else backoff)
                print(
                    f"[gemini] rate-limited (attempt {attempt + 1}/{_MAX_429_RETRIES}); "
                    f"sleeping {wait:.0f}s before retry."
                )
                time.sleep(wait)
                backoff *= 2
        # Unreachable due to the raise above, but satisfies the type checker.
        raise RuntimeError("Gemini retry loop exited without returning") from last_exc


def _as_response_dict(raw: str) -> dict[str, Any]:
    """Gemini wants function_response.response to be a dict; the registry returns a string."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
        return {"result": parsed}
    except (json.JSONDecodeError, TypeError):
        return {"result": raw}
