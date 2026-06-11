"""LLM backend protocol + factory.

All LLM client implementations follow the `LLMClient` Protocol. Each backend
manages its own conversation state and tool-use loop; the caller just gives
it a user turn (text + optional JPEG bytes) and gets back a final assistant
string after any tool calls have been dispatched and confirmed.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

import cv2
import numpy as np

from jarvis_jr.confirm import Confirmer
from jarvis_jr.tools.registry import ToolRegistry


class LLMClient(Protocol):
    """A multi-turn LLM client with vision + tool use."""

    def submit(self, user_text: str, image_jpeg_bytes: bytes | None = None) -> str:
        """Send one user turn. Returns the final assistant text after any tool calls."""
        ...

    def reset(self) -> None:
        """Clear the conversation history."""
        ...


class _AnthropicAdapter:
    """Bridges the existing AnthropicClient (which takes a BGR ndarray) to the
    Protocol's `image_jpeg_bytes: bytes` shape. Keeps AnthropicClient untouched.
    """

    def __init__(self, wrapped: Any):
        self._wrapped = wrapped

    def submit(self, user_text: str, image_jpeg_bytes: bytes | None = None) -> str:
        image = None
        if image_jpeg_bytes is not None:
            arr = np.frombuffer(image_jpeg_bytes, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("AnthropicAdapter: could not decode JPEG bytes")
        return self._wrapped.submit(user_text, image=image)

    def reset(self) -> None:
        self._wrapped.reset()


def build_llm_client(
    provider: str,
    model: str,
    tools: ToolRegistry,
    confirmer: Confirmer,
    system_prompt: str,
    require_confirmation_for: Iterable[str] = (),
    max_tokens: int = 1024,
    max_iterations: int | None = None,
    **provider_kwargs: Any,
) -> LLMClient:
    """Construct an LLMClient by provider name.

    Args:
        provider: "anthropic" | "google" | "ollama"
        model: provider-specific model ID
        tools: ToolRegistry instance
        confirmer: Confirmer callable for tool-use approval
        system_prompt: system prompt string
        require_confirmation_for: tool names that require user approval before running
        max_tokens: cap on response length
        **provider_kwargs: provider-specific options (e.g. `base_url` for Ollama)
    """
    provider = provider.lower()
    if provider == "anthropic":
        from jarvis_jr.llm.anthropic_client import AnthropicClient

        wrapped = AnthropicClient(
            registry=tools,
            model=model,
            max_tokens=max_tokens,
            require_confirmation_for=require_confirmation_for,
            confirmer=confirmer,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
        )
        return _AnthropicAdapter(wrapped)

    if provider == "google":
        from jarvis_jr.llm.gemini_client import GeminiClient

        return GeminiClient(
            model=model,
            tools=tools,
            confirmer=confirmer,
            system_prompt=system_prompt,
            require_confirmation_for=require_confirmation_for,
            max_tokens=max_tokens,
            max_iterations=max_iterations,
        )

    if provider == "ollama":
        from jarvis_jr.llm.ollama_client import OllamaClient

        return OllamaClient(
            model=model,
            tools=tools,
            confirmer=confirmer,
            system_prompt=system_prompt,
            require_confirmation_for=require_confirmation_for,
            max_tokens=max_tokens,
            max_iterations=max_iterations,
            base_url=provider_kwargs.get("base_url", "http://localhost:11434"),
            api_key=provider_kwargs.get("api_key"),
        )

    raise ValueError(f"Unknown LLM provider: {provider!r}")
