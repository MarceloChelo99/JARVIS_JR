"""Anthropic Messages API client with multi-turn memory + tool loop."""

from __future__ import annotations

import base64
from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np
from anthropic import Anthropic

from jarvis_jr.confirm import Confirmer, stdin_confirmer
from jarvis_jr.llm.prompts import build_system_prompt
from jarvis_jr.tools.registry import ToolRegistry


class AnthropicClient:
    """Holds a running conversation and runs the tool-use loop for each turn."""

    def __init__(
        self,
        registry: ToolRegistry,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        require_confirmation_for: Iterable[str] = (),
        confirmer: Confirmer | None = None,
        system_prompt: str | None = None,
        jpeg_quality: int = 85,
    ):
        self._client = Anthropic()
        self.registry = registry
        self.model = model
        self.max_tokens = max_tokens
        self.require_confirmation_for = set(require_confirmation_for)
        self.confirmer = confirmer or stdin_confirmer
        self.system_prompt = system_prompt or build_system_prompt()
        self.jpeg_quality = jpeg_quality
        self.messages: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.messages = []

    def submit(self, user_text: str, image: np.ndarray | None = None) -> str:
        """Send one user turn and return the final assistant text after any tool calls."""
        self.messages.append({"role": "user", "content": self._user_blocks(user_text, image)})

        while True:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                tools=self.registry.schemas,
                messages=self.messages,
            )
            assistant_content = [self._serialize_block(b) for b in response.content]
            self.messages.append({"role": "assistant", "content": assistant_content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            text_parts = [b.text for b in response.content if b.type == "text"]
            assistant_text = " ".join(t.strip() for t in text_parts).strip()

            if response.stop_reason != "tool_use" or not tool_uses:
                return assistant_text

            tool_results = []
            for tu in tool_uses:
                approved = True
                if tu.name in self.require_confirmation_for:
                    prompt = assistant_text or self.registry.describe(tu.name, tu.input)
                    approved = self.confirmer(prompt)
                if not approved:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": "User declined.",
                        }
                    )
                    continue
                content = self.registry.dispatch(tu.name, tu.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": content,
                    }
                )

            self.messages.append({"role": "user", "content": tool_results})

    # ---- helpers -----------------------------------------------------------

    def _user_blocks(self, text: str, image: np.ndarray | None) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        if image is not None:
            blocks.append(self._image_block(image))
        blocks.append({"type": "text", "text": text})
        return blocks

    def _image_block(self, image_bgr: np.ndarray) -> dict[str, Any]:
        ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("Failed to JPEG-encode camera frame")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        }

    @staticmethod
    def _serialize_block(block: Any) -> dict[str, Any]:
        """Convert a response content block into the dict form the API expects on re-send."""
        if block.type == "text":
            return {"type": "text", "text": block.text}
        if block.type == "tool_use":
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            }
        # Future-proof: thinking / other blocks. Round-trip via model_dump if available.
        if hasattr(block, "model_dump"):
            return block.model_dump()
        raise RuntimeError(f"Cannot serialize content block of type {block.type!r}")


def encode_image_file(path: str, jpeg_quality: int = 85) -> np.ndarray:
    """Load any image file from disk and return a BGR ndarray suitable for AnthropicClient.submit()."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        # Try via PIL for formats cv2 chokes on (e.g. some HEIC paths)
        from PIL import Image

        with Image.open(path) as pil:
            arr = np.asarray(pil.convert("RGB"))
        img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    _ = jpeg_quality  # currently unused; reserved for downscaling
    return img
