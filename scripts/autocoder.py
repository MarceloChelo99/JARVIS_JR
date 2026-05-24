"""Run the autocoder on a single spec.

Examples:

    uv run python scripts/autocoder.py "Add a --verbose flag to scripts/run_jarvis.py"

    uv run python scripts/autocoder.py \
        --provider anthropic --model claude-sonnet-4-6 \
        "Add a sentence-streaming TTS implementation behind a --stream-tts flag"

    uv run python scripts/autocoder.py --spec-file specs/refactor-camera.md

Defaults to Gemini 2.5 Flash (free tier). Switches to Ollama (`--provider
ollama`) for offline / fully-free use, or Anthropic Claude for the hardest
specs (paid).

Each run:
- Requires a clean working tree.
- Creates a fresh branch `autocoder/<run-id>` off the current HEAD.
- Streams tool calls to the console and to `notes/autocoder/<run-id>/turns.jsonl`.
- Writes a markdown summary to `notes/autocoder/<run-id>/summary.md`.
- Leaves the branch checked out for you to `gh pr create` or `git branch -D`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jarvis_jr.autocoder import RunConfig, run_autocoder
from jarvis_jr.settings import REPO_ROOT, load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the autocoder on one spec.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("spec", nargs="?", help="The spec text (in quotes).")
    src.add_argument("--spec-file", type=Path, help="Path to a file containing the spec.")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "google", "ollama"],
        help="Override llm.provider for this run (default: from configs/default.yaml).",
    )
    parser.add_argument("--model", help="Override the model ID for the chosen provider.")
    parser.add_argument(
        "--max-iterations", type=int, default=30,
        help="Hard cap on tool-use iterations (default: 30).",
    )
    parser.add_argument(
        "--bash-timeout", type=int, default=60,
        help="Per-command bash timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--allow-network", action="store_true",
        help="Allow the agent to run network commands (uv add / git push / curl etc.).",
    )
    parser.add_argument(
        "--base-branch", default="main",
        help="Branch name to assume as base for the PR (default: main).",
    )
    args = parser.parse_args()

    spec = args.spec if args.spec else args.spec_file.read_text(encoding="utf-8")

    settings = load_settings()
    provider = args.provider or settings.llm.provider
    per_provider = getattr(settings.llm, provider, {}) or {}
    if args.model:
        model = args.model
    elif provider == settings.llm.provider:
        model = settings.llm.model
    else:
        model = per_provider.get("model")
        if not model:
            print(f"ERROR: no model configured for provider {provider!r}.")
            return 2
    extras = {k: v for k, v in per_provider.items() if k != "model"}

    cfg = RunConfig(
        spec=spec,
        repo_root=REPO_ROOT,
        provider=provider,
        model=model,
        provider_kwargs=extras,
        max_iterations=args.max_iterations,
        bash_timeout_sec=args.bash_timeout,
        allow_network=args.allow_network,
        base_branch=args.base_branch,
    )

    result = run_autocoder(cfg)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
