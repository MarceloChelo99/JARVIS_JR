"""Main entry point: full voice assistant, push-to-talk.

Hold SPACE to speak. Release when done. The assistant sees the camera, calls
tools when needed (with voice confirmation for sensitive actions), and speaks
its reply.

Override the LLM backend per run without editing the config:

    uv run python scripts/run_jarvis.py                       # uses configs/default.yaml
    uv run python scripts/run_jarvis.py --provider ollama     # local Qwen via Ollama
    uv run python scripts/run_jarvis.py --provider anthropic  # Claude
    uv run python scripts/run_jarvis.py --model gemini-2.5-pro

Ctrl+C to quit.
"""

from __future__ import annotations

import argparse
import sys

from jarvis_jr.session import Session
from jarvis_jr.settings import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the JARVIS Jr. voice assistant.")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "google", "ollama"],
        help="Override llm.provider for this run.",
    )
    parser.add_argument("--model", help="Override the LLM model ID for this run.")
    parser.add_argument(
        "--profile",
        help="Tool profile for this run (e.g. 'voice', 'desk'). See configs/default.yaml.",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help=(
            "Don't open the camera and never attach frames to the LLM. "
            "Skips vision prefill entirely — much faster turns for text-style chat."
        ),
    )
    args = parser.parse_args()

    settings = load_settings()

    if args.provider:
        settings.llm.provider = args.provider
        per_provider = getattr(settings.llm, args.provider, {}) or {}
        if not args.model and per_provider.get("model"):
            settings.llm.model = per_provider["model"]
    if args.model:
        settings.llm.model = args.model
    if args.no_vision:
        settings.camera.attach_policy = "never"
    if args.profile:
        settings.tools.profile = args.profile

    session = Session(settings)
    print("\nJARVIS Jr. is ready. Hold SPACE to talk. Ctrl+C to quit.\n")
    session.speak("Ready.")
    try:
        session.run_forever()
    except KeyboardInterrupt:
        print("\nGoodbye.")
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
