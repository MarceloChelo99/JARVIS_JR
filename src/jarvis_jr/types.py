"""Shared dataclasses used across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class CapturedFrame:
    """One frame grabbed from the camera."""

    image: np.ndarray  # BGR, HxWx3, uint8
    timestamp: datetime
    width: int
    height: int


@dataclass
class Transcription:
    """Output of a single STT pass."""

    text: str
    language: str
    duration_seconds: float


@dataclass
class ToolCall:
    """A tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result of a tool invocation, returned to the LLM."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Turn:
    """One user-assistant exchange."""

    user_text: str
    user_image: np.ndarray | None
    assistant_text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
