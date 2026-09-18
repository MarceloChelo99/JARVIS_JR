"""Run one trading scenario — or a whole family — one agent turn per simulated day.

Usage:
    uv run python scripts/run_trading_episode.py evals/scenarios/dip_and_recover.yaml
    uv run python scripts/run_trading_episode.py evals/scenarios/cpi_shock.yaml --days 3
    uv run python scripts/run_trading_episode.py evals/scenarios/generated/cpi_release   # a family (directory)
    uv run python scripts/run_trading_episode.py <scenario> --model qwen/qwen3.8-27b

Model/server default to llm.ollama in configs/default.yaml. Output goes to
notes/evals/trading/<timestamp>/ (report.json per scenario; family.json for a directory).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from jarvis_jr.evals.agent import Agent
from jarvis_jr.evals.model_lmstudio import LMStudioModel
from jarvis_jr.evals.tools import registry_tools
from jarvis_jr.settings import REPO_ROOT, load_settings
from jarvis_jr.trading.data_tools import DataTools
from jarvis_jr.trading.episode import run_episode
from jarvis_jr.trading.scenario import build_feeds, grade, load_scenario
from jarvis_jr.trading.tools import TradingTools


def run_one(scenario_path: Path, base_url: str, model: str, temperature: float,
            days: int | None, run_dir: Path, max_turns: int | None = None) -> dict:
    scenario = load_scenario(scenario_path)
    turns = max_turns or scenario.max_turns
    broker = scenario.broker()
    tools = TradingTools(broker, scenario.limits)
    feeds = build_feeds(scenario)
    research = DataTools(clock=lambda: broker.portfolio().day, broker=broker, **feeds)
    agent = Agent(
        LMStudioModel(base_url, model, temperature),
        registry_tools(tools) + registry_tools(research),
    )
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    print(f"[trading] scenario={scenario.id} model={model} days={len(scenario.days)} max_turns={turns} "
          f"research={[s['name'] for s in research.schemas]}", flush=True)

    def on_day(rec):
        print(
            f"  {rec.day}  equity {rec.equity_before:>10,.0f} -> {rec.equity_after:>10,.0f}  "
            f"orders={rec.orders_placed} rejected={rec.rejections}"
            f"{'  TURN-LIMIT' if rec.hit_turn_limit else ''}  "
            f"tools={rec.trajectory.tool_sequence()}",
            flush=True,
        )

    report = run_episode(
        agent, broker, tools, scenario.symbols, workspace,
        max_days=days, max_turns=turns, on_day=on_day,
    )
    verdicts = grade(report, scenario.expect)
    passed = all(v.passed for v in verdicts)
    print(f"[trading] {json.dumps(report.summary())}")
    for v in verdicts:
        print(f"  {'PASS' if v.passed else 'FAIL'} {v.name}: {v.detail}")

    payload = {
        "scenario": scenario.id,
        "scenario_path": str(scenario_path),
        "model": model,
        "max_turns": turns,
        "passed": passed,
        "summary": report.summary(),
        "verdicts": [dataclasses.asdict(v) for v in verdicts],
        "days": [
            {
                "day": d.day.isoformat(),
                "equity_before": d.equity_before,
                "equity_after": d.equity_after,
                "orders_placed": d.orders_placed,
                "rejections": d.rejections,
                "hit_turn_limit": d.hit_turn_limit,
                "tool_sequence": d.trajectory.tool_sequence(),
                "final_answer": d.trajectory.final_answer,
                "steps": [dataclasses.asdict(s) for s in d.trajectory.steps],
            }
            for d in report.days
        ],
    }
    (run_dir / "report.json").write_text(json.dumps(payload, indent=2))
    print(f"[trading] report: {run_dir / 'report.json'}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenario", type=Path, help="a scenario .yaml, or a directory of them (a family)")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--days", type=int, default=None, help="Stop after N days (default: all)")
    parser.add_argument("--max-turns", type=int, default=None, help="Override the scenario's per-day turn budget")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "notes" / "evals" / "trading")
    args = parser.parse_args()

    settings = load_settings()
    base_url = args.base_url or settings.llm.ollama["base_url"].rstrip("/") + "/v1"
    model = args.model or settings.llm.ollama["model"]
    stamp = time.strftime("%Y%m%d-%H%M%S")

    if args.scenario.is_file():
        result = run_one(args.scenario, base_url, model, args.temperature, args.days, args.out / stamp, args.max_turns)
        sys.exit(0 if result["passed"] else 1)

    paths = sorted(args.scenario.rglob("scenario.yaml")) + sorted(args.scenario.glob("*.yaml"))
    if not paths:
        sys.exit(f"no scenarios under {args.scenario}")
    family_dir = args.out / f"{stamp}-family-{args.scenario.name}"
    results = []
    for i, path in enumerate(paths, 1):
        print(f"\n===== [{i}/{len(paths)}] {path} =====", flush=True)
        try:
            results.append(run_one(path, base_url, model, args.temperature, args.days,
                                   family_dir / path.parent.name, args.max_turns))
        except Exception as e:  # noqa: BLE001 — one bad scenario must not kill the family
            print(f"[trading] {path} CRASHED: {type(e).__name__}: {e}", flush=True)
            results.append({"scenario": path.parent.name, "scenario_path": str(path), "passed": False, "crashed": str(e)})

    n_pass = sum(1 for r in results if r.get("passed"))
    rows = []
    for r in results:
        s = r.get("summary", {})
        rows.append(
            f"  {'PASS' if r.get('passed') else 'FAIL':<4} {r['scenario']:<32} "
            f"ret={s.get('return_pct', 0) * 100:+.2f}%  dd={s.get('max_drawdown', 0) * 100:.2f}%  "
            f"exp={s.get('end_exposure', 0) * 100:.0f}%  turnlimit={s.get('turn_limit_days', '-')}"
            + (f"  CRASH {r['crashed'][:60]}" if r.get("crashed") else "")
        )
    print(f"\n[family] {n_pass}/{len(results)} passed  ({args.scenario})")
    print("\n".join(rows))
    (family_dir / "family.json").write_text(json.dumps(results, indent=2))
    print(f"[family] {family_dir / 'family.json'}")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
