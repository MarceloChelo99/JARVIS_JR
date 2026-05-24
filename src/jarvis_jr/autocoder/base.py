"""Shared dataclasses for the autocoder."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class RunConfig:
    """Inputs that govern one autocoder run."""

    spec: str
    repo_root: Path
    provider: str = "google"
    model: str = "gemini-2.5-flash"
    provider_kwargs: dict = field(default_factory=dict)
    max_iterations: int = 30
    bash_timeout_sec: int = 60
    allow_network: bool = False
    base_branch: str = "main"
    branch_prefix: str = "autocoder"
    logs_root: Path | None = None  # defaults to repo_root/notes/autocoder


@dataclass
class RunResult:
    """Outputs of one autocoder run."""

    run_id: str
    branch: str
    started_at: datetime
    finished_at: datetime
    success: bool
    final_message: str
    log_dir: Path
    iterations: int
    commits: list[str] = field(default_factory=list)
