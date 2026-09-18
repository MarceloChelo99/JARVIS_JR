"""Composition root: wire model + tools + graders, run a task suite, report.

Each task gets a fresh workspace under notes/evals/<timestamp>/<task_id>/
(fixture files copied in first). Results go to stdout and report.json.

Defaults come from configs/default.yaml (llm.ollama.base_url / model) so
evals hit the same server the assistant does; every default is overridable
on the command line.

# GROWTH: run tasks in parallel when the suite passes ~20 tasks and wall time
# hurts — not before.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

from jarvis_jr.evals.agent import Agent
from jarvis_jr.evals.domain import TaskReport, TaskSpec
from jarvis_jr.evals.grading import ChecksGrader, JudgeGrader, TrajectoryGrader
from jarvis_jr.evals.model_lmstudio import LMStudioModel
from jarvis_jr.evals.tasks import load_tasks
from jarvis_jr.evals.tools import available_tools, registry_tools
from jarvis_jr.settings import REPO_ROOT, Settings, load_settings

DEFAULT_OUT = REPO_ROOT / "notes" / "evals"


def run_task(task: TaskSpec, agent: Agent, graders: list, workspace: Path) -> TaskReport:
    workspace.mkdir(parents=True, exist_ok=True)
    if task.fixture_dir:
        shutil.copytree(task.fixture_dir, workspace, dirs_exist_ok=True)

    trajectory = agent.run(task.prompt, workspace, max_turns=task.max_turns)
    verdicts = [g.grade(task, trajectory, workspace) for g in graders]
    return TaskReport(task_id=task.id, verdicts=verdicts, trajectory=trajectory)


def build_graders(names: str, base_url: str, judge_model: str) -> list:
    """names: comma-separated subset of checks,trajectory,judge."""
    available = {
        "checks": lambda: ChecksGrader(),
        "trajectory": lambda: TrajectoryGrader(),
        "judge": lambda: JudgeGrader(LMStudioModel(base_url, judge_model, temperature=0.0)),
    }
    graders = []
    for name in [n.strip() for n in names.split(",") if n.strip()]:
        if name not in available:
            raise ValueError(f"unknown grader {name!r} (valid: {sorted(available)})")
        graders.append(available[name]())
    if not graders:
        raise ValueError("at least one grader is required")
    return graders


def build_tools(settings: Settings, profile: str | None) -> list:
    """Native eval tools, plus the assistant's own tools when a profile is named.

    The registry is built without calendar/MCP/voice — evals exercise tools,
    not the assistant's session — and the profile's fnmatch patterns decide
    which of its tools the agent can see.
    """
    tools = available_tools()
    if profile is None:
        return tools
    if profile not in settings.tools.profiles:
        raise ValueError(f"unknown tool profile {profile!r} (valid: {sorted(settings.tools.profiles)})")
    # Imported here so evals stay importable without the assistant's tool deps.
    from jarvis_jr.tools.registry import ToolRegistry
    from jarvis_jr.tools.timer import TimerManager

    registry = ToolRegistry(
        calendar=None,
        timer_manager=TimerManager(),
        enabled_patterns=settings.tools.profiles[profile],
    )
    native = {t.name for t in tools}
    return tools + [t for t in registry_tools(registry) if t.name not in native]


def run_suite(
    tasks_dir: Path,
    out_dir: Path = DEFAULT_OUT,
    base_url: str = "",
    model: str = "",
    judge_model: str = "",
    temperature: float = 0.2,
    graders: str = "checks,trajectory,judge",
    profile: str | None = None,
) -> list[TaskReport]:
    settings = load_settings()
    base_url = base_url or settings.llm.ollama["base_url"].rstrip("/") + "/v1"
    model = model or settings.llm.ollama["model"]

    tasks = load_tasks(tasks_dir)
    agent = Agent(LMStudioModel(base_url, model, temperature), build_tools(settings, profile))
    grader_list = build_graders(graders, base_url, judge_model or model)

    run_dir = out_dir / time.strftime("%Y%m%d-%H%M%S")
    reports = []
    print(f"[evals] model={model}  server={base_url}  profile={profile or '-'}")
    for task in tasks:
        print(f"[{task.id}] running ...", flush=True)
        report = run_task(task, agent, grader_list, run_dir / task.id / "workspace")
        reports.append(report)
        status = "PASS" if report.passed else "FAIL"
        print(f"[{task.id}] {status}  score={report.score:.2f}")
        for v in report.verdicts:
            print(f"    {v.grader:<10} {'ok ' if v.passed else 'BAD'} {v.score:.2f}  {v.detail}")

    _write_report(reports, run_dir / "report.json")
    total = sum(1 for r in reports if r.passed)
    print(f"\n{total}/{len(reports)} tasks passed. Full report: {run_dir / 'report.json'}")
    return reports


def _write_report(reports: list[TaskReport], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "task_id": r.task_id,
            "passed": r.passed,
            "score": r.score,
            "verdicts": [dataclasses.asdict(v) for v in r.verdicts],
            "tool_sequence": r.trajectory.tool_sequence(),
            "final_answer": r.trajectory.final_answer,
            "steps": [dataclasses.asdict(s) for s in r.trajectory.steps],
        }
        for r in reports
    ]
    path.write_text(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run an agent eval suite against a local model.")
    parser.add_argument(
        "tasks_dir", type=Path, nargs="?", default=REPO_ROOT / "evals" / "tasks",
        help="Directory of .yaml task files (default: evals/tasks)",
    )
    parser.add_argument("--base-url", default="", help="Default: configs/default.yaml llm.ollama")
    parser.add_argument("--model", default="", help="Default: configs/default.yaml llm.ollama")
    parser.add_argument("--judge-model", default="", help="Defaults to --model if unset")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--graders",
        default="checks,trajectory,judge",
        help="Comma-separated subset of: checks,trajectory,judge (default: all three)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Also expose the assistant's tools from this configs/default.yaml profile",
    )
    args = parser.parse_args(argv)

    reports = run_suite(
        args.tasks_dir,
        args.out,
        args.base_url,
        args.model,
        args.judge_model,
        args.temperature,
        args.graders,
        args.profile,
    )
    sys.exit(0 if all(r.passed for r in reports) else 1)


if __name__ == "__main__":
    main()
