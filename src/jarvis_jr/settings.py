"""Load configs/default.yaml + .env into a typed Settings object."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


@dataclass
class AudioSettings:
    sample_rate: int = 16000
    channels: int = 1
    input_device: int | str | None = None
    output_device: int | str | None = None


@dataclass
class CameraSettings:
    name_contains: str = "iPhone"
    width: int = 640
    height: int = 360
    jpeg_quality: int = 80
    attach_policy: str = "auto"  # "auto" | "always" | "never"


@dataclass
class STTSettings:
    model: str = "base.en"
    compute_type: str = "int8"
    vad_filter: bool = True


@dataclass
class TTSSettings:
    voice: str = "en_US-amy-medium"
    speed: float = 1.0


@dataclass
class LLMSettings:
    provider: str = "google"
    model: str = "gemini-2.5-flash"
    max_tokens: int = 1024
    system_prompt_path: str | None = None
    anthropic: dict[str, Any] = field(default_factory=lambda: {"model": "claude-sonnet-4-6"})
    google: dict[str, Any] = field(default_factory=lambda: {"model": "gemini-2.5-flash"})
    ollama: dict[str, Any] = field(
        default_factory=lambda: {"model": "qwen3.5:9b", "base_url": "http://localhost:11434"}
    )


@dataclass
class ToolsSettings:
    # Active profile + named profiles. Each profile is a list of tool-name
    # patterns (fnmatch: "edgar_*" matches all EDGAR tools, "*" matches all).
    profile: str = "voice"
    profiles: dict[str, list[str]] = field(
        default_factory=lambda: {"voice": ["*"], "desk": ["*"]}
    )

    def active_patterns(self) -> list[str]:
        return self.profiles.get(self.profile, ["*"])


@dataclass
class MCPSettings:
    # Each entry: {name, command, args?, env?}. Tools are exposed to the LLM
    # as "<name>_<tool>". Empty list = MCP disabled.
    servers: list[dict[str, Any]] = field(default_factory=list)
    call_timeout_s: float = 30.0


@dataclass
class ConfirmationSettings:
    require_for: list[str] = field(default_factory=list)
    yes_words: list[str] = field(default_factory=list)
    no_words: list[str] = field(default_factory=list)


@dataclass
class CalendarSettings:
    default_calendar_id: str = "primary"
    default_duration_minutes: int = 30
    default_timezone: str | None = None


@dataclass
class Settings:
    audio: AudioSettings = field(default_factory=AudioSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    stt: STTSettings = field(default_factory=STTSettings)
    tts: TTSSettings = field(default_factory=TTSSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    tools: ToolsSettings = field(default_factory=ToolsSettings)
    mcp: MCPSettings = field(default_factory=MCPSettings)
    confirmation: ConfirmationSettings = field(default_factory=ConfirmationSettings)
    calendar: CalendarSettings = field(default_factory=CalendarSettings)


def _build_section(cls: type, data: dict[str, Any] | None):
    return cls(**(data or {}))


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load settings from yaml + .env. Falls back to dataclass defaults."""
    load_dotenv(REPO_ROOT / ".env", override=False)
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        with path.open() as f:
            raw = yaml.safe_load(f) or {}
    return Settings(
        audio=_build_section(AudioSettings, raw.get("audio")),
        camera=_build_section(CameraSettings, raw.get("camera")),
        stt=_build_section(STTSettings, raw.get("stt")),
        tts=_build_section(TTSSettings, raw.get("tts")),
        llm=_build_section(LLMSettings, raw.get("llm")),
        tools=_build_section(ToolsSettings, raw.get("tools")),
        mcp=_build_section(MCPSettings, raw.get("mcp")),
        confirmation=_build_section(ConfirmationSettings, raw.get("confirmation")),
        calendar=_build_section(CalendarSettings, raw.get("calendar")),
    )
