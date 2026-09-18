"""Run the agent eval suite against the local model.

Usage:
    uv run python scripts/run_evals.py                          # evals/tasks, native tools
    uv run python scripts/run_evals.py --profile evals          # + assistant tools (read-only set)
    uv run python scripts/run_evals.py evals/tasks_gmail        # needs GMAIL_* in .env
    uv run python scripts/run_evals.py --graders checks         # fast, model-free grading only
    uv run python scripts/run_evals.py --model qwen/qwen3.8-27b # A/B another model

Model and server default to llm.ollama in configs/default.yaml. Results land
in notes/evals/<timestamp>/ (per-task workspaces + report.json).
"""

from __future__ import annotations

from jarvis_jr.evals.run import main

if __name__ == "__main__":
    main()
