"""Top-level autocoder runner: git branch + log management + LLM dispatch."""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from jarvis_jr.autocoder.base import RunConfig, RunResult
from jarvis_jr.autocoder.prompts import CODER_SYSTEM_PROMPT
from jarvis_jr.autocoder.tools import CoderTools
from jarvis_jr.confirm import always_yes
from jarvis_jr.llm import build_llm_client


def run_autocoder(cfg: RunConfig) -> RunResult:
    """Execute one autocoder run end-to-end."""
    started_at = datetime.now()
    run_id = f"{started_at.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    branch = f"{cfg.branch_prefix}/{run_id}"

    # Verify clean working tree FIRST — before creating any files that would
    # dirty it themselves (notes/autocoder/<run_id>/...).
    _require_clean_worktree(cfg.repo_root)
    head_before = _git(cfg.repo_root, "rev-parse", "HEAD").strip()
    _git(cfg.repo_root, "checkout", "-b", branch)

    logs_root = cfg.logs_root or (cfg.repo_root / "notes" / "autocoder")
    log_dir = logs_root / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    turns_path = log_dir / "turns.jsonl"
    spec_path = log_dir / "spec.txt"
    summary_path = log_dir / "summary.md"
    spec_path.write_text(cfg.spec, encoding="utf-8")

    print(f"[autocoder] run_id={run_id}")
    print(f"[autocoder] branch={branch}")
    print(f"[autocoder] logs={log_dir}")

    def log_call(name: str, args: dict, result: str) -> None:
        with turns_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "tool": name,
                        "args": _shrink_args(args),
                        "result_preview": result[:600],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        # Streamed console echo for live visibility.
        print(f"  🔧 {name}({_shrink_args(args)}) → {_one_line(result)[:160]}")

    tools = CoderTools(
        repo_root=cfg.repo_root,
        on_call=log_call,
        allow_network=cfg.allow_network,
        default_bash_timeout=cfg.bash_timeout_sec,
    )

    llm = build_llm_client(
        provider=cfg.provider,
        model=cfg.model,
        tools=tools,
        confirmer=always_yes,  # autocoder is unattended; no human in the loop
        system_prompt=CODER_SYSTEM_PROMPT,
        require_confirmation_for=(),
        max_tokens=4096,
        max_iterations=cfg.max_iterations,
        **cfg.provider_kwargs,
    )

    print(f"[autocoder] provider={cfg.provider} model={cfg.model}")
    print(f"[autocoder] spec:\n  {cfg.spec}\n")

    final_message = ""
    crashed = False
    hit_cap = False
    try:
        final_message = llm.submit(cfg.spec)
        hit_cap = final_message.startswith("[autocoder] max_iterations")
    except Exception as e:  # noqa: BLE001
        final_message = f"FAILED with exception: {type(e).__name__}: {e}"
        crashed = True

    finished_at = datetime.now()
    commits = _commits_since(cfg.repo_root, head_before)
    iterations = _count_lines(turns_path)

    # Outcome semantics:
    # - Any new commits on the run branch = the agent produced something.
    # - "success" means it shipped at least one commit AND the loop ended cleanly.
    # - "partial" means it shipped commits but then crashed or hit the cap.
    # - "fail" means no commits.
    success = bool(commits) and not (crashed or hit_cap)
    if commits and (crashed or hit_cap):
        outcome_label = "partial (commits shipped, loop ended on " + (
            "crash" if crashed else "max_iterations") + ")"
    elif commits:
        outcome_label = "✓ success"
    elif crashed:
        outcome_label = "✗ crashed before any commit"
    elif hit_cap:
        outcome_label = "✗ hit max_iterations without committing"
    else:
        outcome_label = "✗ stopped without committing"

    summary_path.write_text(
        _render_summary(
            run_id=run_id,
            branch=branch,
            cfg=cfg,
            started_at=started_at,
            finished_at=finished_at,
            success=success,
            iterations=iterations,
            commits=commits,
            final_message=final_message,
        ),
        encoding="utf-8",
    )

    print(f"\n[autocoder] {outcome_label}; {iterations} tool calls; "
          f"{len(commits)} commits on {branch}")
    print(f"[autocoder] summary written to {summary_path}")
    print(f"[autocoder] inspect & ship: git log {branch} ; gh pr create --head {branch}")

    return RunResult(
        run_id=run_id,
        branch=branch,
        started_at=started_at,
        finished_at=finished_at,
        success=success,
        final_message=final_message,
        log_dir=log_dir,
        iterations=iterations,
        commits=commits,
    )


# ---- helpers ---------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return proc.stdout


def _require_clean_worktree(repo_root: Path) -> None:
    out = _git(repo_root, "status", "--porcelain")
    if out.strip():
        raise RuntimeError(
            "Working tree is not clean. Commit or stash your changes before running "
            f"the autocoder. Dirty files:\n{out}"
        )


def _commits_since(repo_root: Path, ref: str) -> list[str]:
    out = _git(repo_root, "log", "--oneline", f"{ref}..HEAD")
    return [line.strip() for line in out.splitlines() if line.strip()]


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


def _shrink_args(args: dict) -> dict:
    """Shrink long strings in tool args for log readability."""
    shrunk = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 200:
            shrunk[k] = v[:200] + f"... [{len(v)-200} more chars]"
        else:
            shrunk[k] = v
    return shrunk


def _one_line(s: str) -> str:
    return " ".join(s.split())


def _render_summary(
    *,
    run_id: str,
    branch: str,
    cfg: RunConfig,
    started_at: datetime,
    finished_at: datetime,
    success: bool,
    iterations: int,
    commits: list[str],
    final_message: str,
) -> str:
    duration = (finished_at - started_at).total_seconds()
    commits_md = "\n".join(f"- `{c}`" for c in commits) or "_no commits_"
    return f"""# Autocoder run {run_id}

- **Branch**: `{branch}`
- **Provider/model**: `{cfg.provider}` / `{cfg.model}`
- **Started**: {started_at.isoformat(timespec="seconds")}
- **Finished**: {finished_at.isoformat(timespec="seconds")}  (duration: {duration:.0f}s)
- **Outcome**: {"✓ success" if success else "✗ stopped"}
- **Tool calls**: {iterations}
- **Max iterations**: {cfg.max_iterations}

## Spec

```
{cfg.spec}
```

## Commits

{commits_md}

## Final agent message

{final_message}
"""
