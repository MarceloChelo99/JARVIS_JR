"""M4 verification: text-only LLM round trip with tools, any backend.

Usage:
    uv run python scripts/test_llm.py "what's on my calendar today?"
    uv run python scripts/test_llm.py "make an event tomorrow at 3pm called dentist"
    uv run python scripts/test_llm.py "what is this?" --image path/to/photo.jpg
    uv run python scripts/test_llm.py --provider ollama "set a 5 minute timer for laundry"

Sensitive tools (create_event, update_event, delete_event) prompt for stdin y/n
confirmation before executing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from jarvis_jr.confirm import stdin_confirmer
from jarvis_jr.llm import build_llm_client
from jarvis_jr.llm.prompts import build_system_prompt
from jarvis_jr.settings import load_settings
from jarvis_jr.tools.calendar import CalendarError, GoogleCalendar
from jarvis_jr.tools.registry import ToolRegistry
from jarvis_jr.tools.timer import TimerManager


def load_image_as_jpeg_bytes(path: str, quality: int = 85) -> bytes:
    """Read an image file from disk and return its JPEG-encoded bytes."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        from PIL import Image

        with Image.open(path) as pil:
            arr = np.asarray(pil.convert("RGB"))
        img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Failed to JPEG-encode {path}")
    return buf.tobytes()


def _resolve_model(settings, provider: str) -> tuple[str, dict]:
    """Pick the model + extra kwargs for the chosen provider."""
    per_provider = getattr(settings.llm, provider, {}) or {}
    if provider == settings.llm.provider:
        model = settings.llm.model
    else:
        model = per_provider.get("model")
        if not model:
            raise SystemExit(
                f"No model configured for provider {provider!r}. "
                f"Add llm.{provider}.model to configs/default.yaml."
            )
    extra = {k: v for k, v in per_provider.items() if k != "model"}
    return model, extra


def main() -> int:
    parser = argparse.ArgumentParser(description="Text-only LLM round trip with tools.")
    parser.add_argument("prompt", help="What to ask the assistant.")
    parser.add_argument("--image", help="Optional image file to attach.")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "google", "ollama"],
        help="Override llm.provider for this run.",
    )
    parser.add_argument(
        "--no-calendar",
        action="store_true",
        help="Skip Google Calendar setup (useful for image-only smoke tests).",
    )
    args = parser.parse_args()

    settings = load_settings()
    provider = args.provider or settings.llm.provider
    model, extra_kwargs = _resolve_model(settings, provider)

    calendar: GoogleCalendar | None = None
    if not args.no_calendar:
        try:
            calendar = GoogleCalendar(
                calendar_id=settings.calendar.default_calendar_id,
                default_duration_minutes=settings.calendar.default_duration_minutes,
                default_timezone=settings.calendar.default_timezone,
            )
        except CalendarError as e:
            print(f"[calendar] {e}")
            print("Continuing without calendar tools — pass --no-calendar to silence this.")

    timer_manager = TimerManager()
    registry = ToolRegistry(calendar=calendar, timer_manager=timer_manager)

    # Trace tool dispatches for visibility.
    original_dispatch = registry.dispatch

    def traced_dispatch(name, tool_args):
        print(f"\n🔧 tool_use: {name}({tool_args})")
        result = original_dispatch(name, tool_args)
        print(f"   → {result}")
        return result

    registry.dispatch = traced_dispatch  # type: ignore[method-assign]

    client = build_llm_client(
        provider=provider,
        model=model,
        tools=registry,
        confirmer=stdin_confirmer,
        system_prompt=build_system_prompt(),
        require_confirmation_for=settings.confirmation.require_for,
        max_tokens=settings.llm.max_tokens,
        **extra_kwargs,
    )

    image_bytes = (
        load_image_as_jpeg_bytes(args.image, quality=settings.camera.jpeg_quality)
        if args.image
        else None
    )

    print(f"[provider={provider} model={model}]")
    print(f"USER: {args.prompt}")
    if args.image:
        print(f"      [+image {Path(args.image).name}]")

    reply = client.submit(args.prompt, image_jpeg_bytes=image_bytes)
    print(f"\nASSISTANT: {reply}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
