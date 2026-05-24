"""Autonomous coding agent that takes a spec and produces a finished change.

M6: single-agent (coder) only. Uses the existing `LLMClient` abstraction so the
backend is pluggable — defaults to Gemini Flash for $0 variable cost.
"""

from jarvis_jr.autocoder.base import RunConfig, RunResult
from jarvis_jr.autocoder.runner import run_autocoder

__all__ = ["RunConfig", "RunResult", "run_autocoder"]
