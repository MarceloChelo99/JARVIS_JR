"""Extract, label, and stamp real-data scenarios for an event type.

Usage:
    uv run python scripts/research_scenarios.py cpi_release --list                 # labeled instances
    uv run python scripts/research_scenarios.py cpi_release --direction above --magnitude large --since 2021-01-01
    uv run python scripts/research_scenarios.py fomc_decision --direction hike --since 2022-01-01
    uv run python scripts/research_scenarios.py vix_spike --since 2015-01-01 --limit 5

Writes evals/scenarios/generated/<type>/<day>/{scenario.yaml,macro.csv,news.jsonl}.
Run a whole family with: uv run python scripts/run_trading_episode.py evals/scenarios/generated/cpi_release
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date

from jarvis_jr.research.extractors import ResearchData, extract
from jarvis_jr.research.labels import label_events
from jarvis_jr.research.scenario_gen import DEFAULT_OUT, generate
from jarvis_jr.research.taxonomy import load_taxonomy
from jarvis_jr.settings import load_settings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("event_type")
    p.add_argument("--direction")
    p.add_argument("--magnitude")
    p.add_argument("--since", type=date.fromisoformat, default=date(2015, 1, 1))
    p.add_argument("--until", type=date.fromisoformat, default=None)
    p.add_argument("--limit", type=int, default=None, help="most recent N")
    p.add_argument("--list", action="store_true", help="print instances, write nothing")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()
    load_settings()

    taxonomy = load_taxonomy()
    et = taxonomy[args.event_type]
    data = ResearchData()
    labeled = label_events(extract(et, data), et)
    print(f"[{et.id}] {len(labeled)} instances; labels: {Counter(e.label for e in labeled).most_common(8)}")

    picked = [
        e for e in labeled
        if e.instance.day >= args.since
        and (args.until is None or e.instance.day <= args.until)
        and (args.direction is None or e.direction == args.direction)
        and (args.magnitude is None or e.magnitude == args.magnitude)
    ]
    if args.limit:
        picked = picked[-args.limit:]
    print(f"[{et.id}] {len(picked)} selected")
    if args.list:
        for e in picked:
            extra = f" surprise={e.instance.surprise:+.4g}" if e.instance.surprise is not None else ""
            print(f"  {e.instance.day}  {e.label:<16} value={e.instance.value:g}{extra} {e.instance.context.get('ticker','')}")
        return

    from pathlib import Path

    result = generate(et, picked, taxonomy, data, Path(args.out))
    print(f"[{et.id}] wrote {len(result.written)} scenarios under {args.out}/{et.id}/")
    for day, why in result.skipped[:10]:
        print(f"  skipped {day}: {why}")
    if len(result.skipped) > 10:
        print(f"  ... and {len(result.skipped) - 10} more skipped")


if __name__ == "__main__":
    main()
