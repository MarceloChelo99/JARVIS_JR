"""Load TaskSpecs from YAML files. One loader, no task framework.

A task file is one YAML document; see evals/tasks/ for examples. fixture_dir is
resolved relative to the task file and must exist if given.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from jarvis_jr.evals.domain import Check, TaskSpec

VALID_CHECK_KINDS = {"file_exists", "file_contains", "file_matches", "answer_contains"}


def load_task(path: Path) -> TaskSpec:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "id" not in data or "prompt" not in data:
        raise ValueError(f"{path}: a task needs at least `id` and `prompt`")

    checks = []
    for raw in data.get("checks", []):
        kind = raw.get("kind", "")
        if kind not in VALID_CHECK_KINDS:
            raise ValueError(f"{path}: unknown check kind {kind!r} (valid: {sorted(VALID_CHECK_KINDS)})")
        checks.append(Check(kind=kind, path=raw.get("path", ""), expect=str(raw.get("expect", ""))))

    fixture_dir = ""
    if data.get("fixture_dir"):
        fixture = (path.parent / data["fixture_dir"]).resolve()
        if not fixture.is_dir():
            raise ValueError(f"{path}: fixture_dir does not exist: {fixture}")
        fixture_dir = str(fixture)

    return TaskSpec(
        id=str(data["id"]),
        prompt=str(data["prompt"]),
        checks=tuple(checks),
        expected_tools=tuple(data.get("expected_tools", [])),
        forbidden_tools=tuple(data.get("forbidden_tools", [])),
        rubric=str(data.get("rubric", "")),
        fixture_dir=fixture_dir,
        max_turns=int(data.get("max_turns", 12)),
    )


def load_tasks(tasks_dir: Path) -> list[TaskSpec]:
    files = sorted(p for p in tasks_dir.glob("*.yaml")) + sorted(tasks_dir.glob("*.yml"))
    if not files:
        raise FileNotFoundError(f"no .yaml task files in {tasks_dir}")
    return [load_task(p) for p in files]
