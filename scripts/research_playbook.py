"""Print the macro playbook for a series, exactly as the agent would see it.

Usage:
    uv run python scripts/research_playbook.py CPIAUCSL
    uv run python scripts/research_playbook.py UNRATE --direction above --as-of 2024-06-01
    uv run python scripts/research_playbook.py PAYEMS --symbols SPY QQQ TLT
"""

from __future__ import annotations

import argparse
from datetime import date

from jarvis_jr.research.bars import BarStore
from jarvis_jr.research.playbook import DEFAULT_SYMBOLS, MacroPlaybook


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("series")
    p.add_argument("--direction", default="all", choices=["above", "below", "all"])
    p.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    p.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    p.add_argument("--lookback-years", type=int, default=3)
    p.add_argument("--all-surprises", action="store_true", help="not just the largest third")
    args = p.parse_args()

    pb = MacroPlaybook(BarStore(), symbols=tuple(args.symbols))
    directions = [args.direction] if args.direction != "all" else ["above", "below", "all"]
    for d in directions:
        e = pb.lookup(args.series, d, args.as_of, args.lookback_years, not args.all_surprises)
        print(
            f"\n{e.series} {d.upper()} expectation, as of {e.as_of}, last {e.lookback_years}y"
            f"{' (largest-third surprises)' if e.big_only else ''}: {e.n_events} events"
        )
        print(f"  {'symbol':<6} {'h':>2} {'n':>4} {'mean%':>7} {'med%':>7} {'up':>5} {'worst%':>7} {'best%':>7}")
        for r in e.reactions:
            print(
                f"  {r.symbol:<6} {r.horizon:>2} {r.n:>4} {r.mean*100:>7.2f} {r.median*100:>7.2f} "
                f"{r.hit_rate:>5.0%} {r.worst*100:>7.2f} {r.best*100:>7.2f}"
            )


if __name__ == "__main__":
    main()
