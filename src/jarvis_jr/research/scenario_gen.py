"""Stamp real-data scenarios out of labeled events.

For each event: the real adjusted closes for the configured symbols over
[day - before, day + after] trading days, the real first prints (with release
dates) for the event's series, and ONE templated headline on the event day
built from the instance's own numbers — never a hallucinated story. The
output is a directory the existing runner consumes unchanged.

A generated family (evals/scenarios/generated/<type>/<day>/) turns pass/fail
into a rate over real history instead of a verdict on one hand-written path.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from jarvis_jr.research.bars import BarStore
from jarvis_jr.research.extractors import ResearchData
from jarvis_jr.research.taxonomy import EventType, LabeledEvent, Taxonomy
from jarvis_jr.settings import REPO_ROOT

DEFAULT_OUT = REPO_ROOT / "evals" / "scenarios" / "generated"


@dataclass(frozen=True)
class GenerateResult:
    written: list[Path]
    skipped: list[tuple[date, str]]  # (event day, reason)


def _headline(et: EventType, ev: LabeledEvent) -> str:
    ctx = {**ev.instance.context, "value": ev.instance.value, "direction": ev.direction,
           "magnitude": ev.magnitude, "expected": ev.instance.expected}
    try:
        return et.headline.format(**ctx) if et.headline else f"{et.id} event: {ev.label}"
    except (KeyError, ValueError):
        return f"{et.id}: {ev.label} (value {ev.instance.value:g})"


def _macro_rows(et: EventType, data: ResearchData, start: date, end: date) -> list[dict]:
    """First prints of the event's series released inside the window, plus the 3 before it."""
    series = et.source.get("series")
    if not series or et.source["kind"] not in ("fred_vintage", "fred_daily_step", "threshold"):
        return []
    prints = data.prints(series)
    inside = [p for p in prints if start <= p.release_day <= end]
    before = [p for p in prints if p.release_day < start][-3:]
    return [
        {"series": series, "date": p.obs_day.isoformat(), "value": p.value, "release_date": p.release_day.isoformat()}
        for p in before + inside
    ]


def generate(
    et: EventType,
    events: list[LabeledEvent],
    taxonomy: Taxonomy,
    data: ResearchData,
    out_root: Path = DEFAULT_OUT,
    cash: float = 100_000.0,
    max_turns: int = 12,
) -> GenerateResult:
    bars: BarStore = data.bars
    before, after = int(et.scenario.get("before_days", 2)), int(et.scenario.get("after_days", 5))
    written, skipped = [], []
    for ev in events:
        symbols = [
            (ev.instance.context.get("ticker", "") if s == "TICKER" else s).upper()
            for s in et.scenario.get("symbols", ["SPY"])
        ]
        if any(not s for s in symbols):
            skipped.append((ev.instance.day, "no ticker in context"))
            continue
        series = {s: bars.get(s) for s in symbols}
        anchor = series[symbols[0]]
        i = anchor.index_on_or_after(ev.instance.day)
        if i is None or i - before < 0 or i + after >= len(anchor):
            skipped.append((ev.instance.day, "bars do not cover the window"))
            continue
        days = [b.day for b in anchor.bars[i - before : i + after + 1]]
        prices: dict[str, list[float]] = {}
        ok = True
        for s, bs in series.items():
            closes = {b.day: b.adj_close for b in bs.bars}
            if any(d not in closes for d in days):
                ok = False
                break
            prices[s] = [round(closes[d], 4) for d in days]
        if not ok:
            skipped.append((ev.instance.day, "a symbol is missing bars in the window"))
            continue

        scen_dir = out_root / et.id / ev.instance.day.isoformat()
        scen_dir.mkdir(parents=True, exist_ok=True)
        macro_rows = _macro_rows(et, data, days[0], days[-1])
        if macro_rows:
            with (scen_dir / "macro.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["series", "date", "value", "release_date"])
                w.writeheader()
                w.writerows(macro_rows)
        (scen_dir / "news.jsonl").write_text(
            json.dumps(
                {"date": ev.instance.day.isoformat(), "title": _headline(et, ev), "source": "generated",
                 "symbols": [], "summary": f"event type {et.id}; label {ev.label}"}
            ) + "\n"
        )
        spec = {
            "id": f"{et.id}_{ev.instance.day.isoformat()}",
            "start_day": days[0].isoformat(),
            "cash": cash,
            "prices": prices,
            "limits": {"max_position_pct": 0.30, "max_order_notional": 25_000, "long_only": True},
            "slippage_bps": 5,
            "commission": 1.0,
            "max_turns": max_turns,
            "data": {
                "news": "news.jsonl",
                **({"macro": "macro.csv"} if macro_rows else {}),
                "filings": "secfiler",
                "playbook": "macro",
                "reactions": "taxonomy",
                "sizing": "bandit",
            },
            "expect": taxonomy.expectations_for(ev.label),
            "event": {
                "type": et.id, "category": et.category, "day": ev.instance.day.isoformat(),
                "label": ev.label, "value": ev.instance.value,
                "expected": ev.instance.expected, "event_index_in_window": before,
            },
            "notes": f"Generated from {et.id} on {ev.instance.day} ({ev.label}); real bars, real prints.",
        }
        path = scen_dir / "scenario.yaml"
        path.write_text(yaml.safe_dump(spec, sort_keys=False))
        written.append(path)
    return GenerateResult(written, skipped)
