"""Three graders, one contract: (task, trajectory, workspace) -> Verdict.

ChecksGrader     — deterministic outcome assertions (did the right state result?)
TrajectoryGrader — did the agent use the right tools, in order, and none forbidden?
JudgeGrader      — a second model scores the answer against the task's rubric.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

from jarvis_jr.evals.domain import Check, TaskSpec, Trajectory, Verdict


class Grader(Protocol):
    name: str

    def grade(self, task: TaskSpec, trajectory: Trajectory, workspace: Path) -> Verdict: ...


class ChecksGrader:
    name = "checks"

    def grade(self, task: TaskSpec, trajectory: Trajectory, workspace: Path) -> Verdict:
        if not task.checks:
            return Verdict(self.name, 1.0, True, "no checks defined")
        results = [(c, self._run_check(c, trajectory, workspace)) for c in task.checks]
        passed = sum(1 for _, ok in results if ok)
        detail = "; ".join(
            f"{'PASS' if ok else 'FAIL'} {c.kind} {c.path or ''} {c.expect or ''}".strip()
            for c, ok in results
        )
        return Verdict(self.name, passed / len(results), passed == len(results), detail)

    def _run_check(self, check: Check, trajectory: Trajectory, workspace: Path) -> bool:
        if check.kind == "answer_contains":
            return check.expect.lower() in trajectory.final_answer.lower()
        target = workspace / check.path
        if check.kind == "file_exists":
            return target.is_file()
        if not target.is_file():
            return False
        text = target.read_text()
        if check.kind == "file_contains":
            return check.expect in text
        if check.kind == "file_matches":
            return re.search(check.expect, text) is not None
        raise ValueError(f"unknown check kind: {check.kind}")


class TrajectoryGrader:
    name = "trajectory"

    def grade(self, task: TaskSpec, trajectory: Trajectory, workspace: Path) -> Verdict:
        if not task.expected_tools and not task.forbidden_tools:
            return Verdict(self.name, 1.0, True, "no trajectory expectations")

        actual = trajectory.tool_sequence()
        matched = _ordered_matches(list(task.expected_tools), actual)
        expected_score = matched / len(task.expected_tools) if task.expected_tools else 1.0

        forbidden_used = sorted(set(actual) & set(task.forbidden_tools))
        passed = expected_score == 1.0 and not forbidden_used

        parts = []
        if task.expected_tools:
            parts.append(f"expected {list(task.expected_tools)}, matched {matched}, actual {actual}")
        if forbidden_used:
            parts.append(f"used forbidden tools: {forbidden_used}")
        score = expected_score if not forbidden_used else 0.0
        return Verdict(self.name, score, passed, "; ".join(parts))


def _ordered_matches(expected: list[str], actual: list[str]) -> int:
    """How many of `expected` appear in `actual` as an ordered subsequence."""
    i = 0
    for tool in actual:
        if i < len(expected) and tool == expected[i]:
            i += 1
    return i


JUDGE_PROMPT = """You are grading the work of an AI agent. Be strict but fair.

## Task given to the agent
{prompt}

## Grading rubric
{rubric}

## Tools the agent called, in order
{tools}

## Agent's final answer
{answer}

Reply with ONLY a JSON object, no other text:
{{"score": <number between 0.0 and 1.0>, "reasoning": "<one or two sentences>"}}"""


class JudgeGrader:
    """LLM-as-judge. Give it a small model (e.g. Gemma 4 12B QAT) so it can
    run alongside the agent model.

    # GROWTH: sample the judge 3x and take the median when score variance
    # becomes a problem — not before.
    """

    name = "judge"

    def __init__(self, model, threshold: float = 0.7):
        self.model = model
        self.threshold = threshold

    def grade(self, task: TaskSpec, trajectory: Trajectory, workspace: Path) -> Verdict:
        if not task.rubric:
            return Verdict(self.name, 1.0, True, "no rubric; judge skipped")
        prompt = JUDGE_PROMPT.format(
            prompt=task.prompt,
            rubric=task.rubric,
            tools=", ".join(trajectory.tool_sequence()) or "(none)",
            answer=trajectory.final_answer or "(empty)",
        )
        reply = self.model.complete([{"role": "user", "content": prompt}])
        score, reasoning = _parse_judge(reply.content)
        return Verdict(self.name, score, score >= self.threshold, reasoning)


def _parse_judge(text: str) -> tuple[float, str]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return 0.0, f"judge reply was not JSON: {text[:200]}"
    try:
        data = json.loads(match.group(0))
        score = max(0.0, min(1.0, float(data.get("score", 0.0))))
        return score, str(data.get("reasoning", ""))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0, f"could not parse judge JSON: {text[:200]}"
