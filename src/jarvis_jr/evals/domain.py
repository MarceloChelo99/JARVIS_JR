"""The harness vocabulary. Typed shapes that cross every boundary.

This module imports nothing but the standard library. Everything else
imports from here; nothing here imports from anywhere else in the package.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolCall:
    """The model asked for a tool to be run."""

    tool: str
    args: dict
    call_id: str = ""


@dataclass(frozen=True)
class ToolResult:
    """The one result/error shape every tool returns."""

    ok: bool
    output: str = ""
    error: str = ""

    @staticmethod
    def success(output: str) -> "ToolResult":
        return ToolResult(ok=True, output=output)

    @staticmethod
    def failure(error: str) -> "ToolResult":
        return ToolResult(ok=False, error=error)

    def as_text(self) -> str:
        return self.output if self.ok else f"ERROR: {self.error}"


@dataclass(frozen=True)
class ModelReply:
    """One turn of model output, normalized away from any API's wire format."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    raw_message: dict | None = None  # verbatim assistant message for the chat log


@dataclass
class Step:
    """One agent-loop iteration: what the model said, did, and got back."""

    thought: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class Trajectory:
    """Everything the agent did for one task, in order."""

    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    hit_turn_limit: bool = False

    def tool_sequence(self) -> list[str]:
        return [c.tool for s in self.steps for c in s.tool_calls]


@dataclass(frozen=True)
class Check:
    """One programmatic assertion about the outcome.

    kind: file_exists | file_contains | file_matches | answer_contains
    """

    kind: str
    path: str = ""
    expect: str = ""


@dataclass(frozen=True)
class TaskSpec:
    """One task the agent is asked to do, plus how to grade it."""

    id: str
    prompt: str
    checks: tuple[Check, ...] = ()
    expected_tools: tuple[str, ...] = ()  # ordered subsequence that must appear
    forbidden_tools: tuple[str, ...] = ()
    rubric: str = ""  # empty -> LLM judge is skipped for this task
    fixture_dir: str = ""  # absolute path; copied into the workspace before the run
    max_turns: int = 12


@dataclass(frozen=True)
class Verdict:
    """One grader's opinion of one task run."""

    grader: str
    score: float  # 0.0 .. 1.0
    passed: bool
    detail: str = ""


@dataclass
class TaskReport:
    """All verdicts for one task run."""

    task_id: str
    verdicts: list[Verdict] = field(default_factory=list)
    trajectory: Trajectory = field(default_factory=Trajectory)

    @property
    def passed(self) -> bool:
        return all(v.passed for v in self.verdicts) if self.verdicts else False

    @property
    def score(self) -> float:
        if not self.verdicts:
            return 0.0
        return sum(v.score for v in self.verdicts) / len(self.verdicts)
