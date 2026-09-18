"""Walk-forward test of the exposure-sizing policy against constant baselines.

For every event of a type after --since, the policy is asked for an exposure
as of the day BEFORE the event (trailing window, no lookahead), and the
realized 5-day SPY return is recorded. Reported: cumulative and mean
contribution (exposure x return), worst case, hit rate — for the policy and
for constant 0 / 0.6 / 1.0 exposure.

Usage:
    uv run python scripts/research_sizing.py cpi_release --since 2021-01-01
    uv run python scripts/research_sizing.py fomc_decision --since 2019-01-01 --risk-aversion 2
    uv run python scripts/research_sizing.py cpi_release --since 2021-01-01 --regime --full-label   # finer state (ablation)
"""

from __future__ import annotations

import argparse
import json
from datetime import date

from jarvis_jr.research.reactions import ReactionGraph
from jarvis_jr.research.sizing import ExposurePolicy, summarize, walk_forward
from jarvis_jr.settings import load_settings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("event_type")
    p.add_argument("--since", type=date.fromisoformat, default=date(2021, 1, 1))
    p.add_argument("--until", type=date.fromisoformat, default=None)
    p.add_argument("--risk-aversion", type=float, default=4.0)
    p.add_argument("--prior", type=float, default=0.6)
    p.add_argument("--min-n", type=int, default=6)
    p.add_argument("--regime", action="store_true", help="add the VIX regime to the state (default off)")
    p.add_argument("--full-label", action="store_true", help="finer state: match direction/magnitude (default direction only)")
    p.add_argument("--lookback-years", type=int, default=6)
    p.add_argument("--rows", action="store_true", help="print every event")
    args = p.parse_args()
    load_settings()

    policy = ExposurePolicy(
        ReactionGraph(), risk_aversion=args.risk_aversion, prior_exposure=args.prior,
        min_n=args.min_n, use_regime=args.regime, direction_only=not args.full_label,
        lookback_years=args.lookback_years,
    )
    rows = walk_forward(policy, args.event_type, args.since, args.until, args.lookback_years)
    if args.rows:
        for r in rows:
            print(f"  {r.day}  {r.label:<14} {r.regime:<9} r5={r.r5*100:+6.2f}%  exposure={r.policy_exposure:.1f}"
                  f"{'  (prior)' if r.fallback else ''}")
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
