"""Print the event → reaction graph: how each event type/label mapped to reaction
archetypes over a trailing window ending at --as-of.

Usage:
    uv run python scripts/research_reactions.py                       # every type, direction-level labels
    uv run python scripts/research_reactions.py cpi_release --as-of 2023-06-01
    uv run python scripts/research_reactions.py fomc_decision --labels hike/75bp cut/25bp --horizon 5
    uv run python scripts/research_reactions.py cpi_release --as-of 2019-12-31   # compare regimes
"""

from __future__ import annotations

import argparse
from datetime import date

from jarvis_jr.research.reactions import ARCHETYPES, ReactionGraph
from jarvis_jr.settings import load_settings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("event_type", nargs="?", default=None)
    p.add_argument("--labels", nargs="*", default=None, help="default: above/below (or all labels for non-surprise types)")
    p.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    p.add_argument("--lookback-years", type=int, default=None)
    p.add_argument("--horizon", type=int, default=1)
    args = p.parse_args()
    load_settings()

    g = ReactionGraph()
    types = [args.event_type] if args.event_type else g.available_types()
    head = f"{'event / label':<38}{'n':>4}  " + "".join(f"{a[:9]:>10}" for a in ARCHETYPES) + f"{'SPY%':>7}{'TLT%':>7}{'def-gro%':>9}"
    print(f"as of {args.as_of}, {args.horizon}-day reaction\n{head}")
    for t in types:
        try:
            all_labels = g.labels_for(t)
        except ValueError as e:
            print(f"{t:<38} (skipped: {e})")
            continue
        if args.labels:
            labels = args.labels
        elif any(lbl.startswith(("above", "below")) for lbl in all_labels):
            labels = ["above", "below"]
        else:
            labels = all_labels or ["all"]
        for lbl in labels:
            pr = g.profile(t, lbl, args.as_of, args.lookback_years, args.horizon)
            if pr.n == 0:
                continue
            shares = "".join(f"{pr.shares[a]:>10.0%}" for a in ARCHETYPES)
            spy = pr.mean_returns.get("SPY", 0) * 100
            tlt = pr.mean_returns.get("TLT", 0) * 100
            spread = (pr.mean_defensive_spread or 0) * 100
            print(f"{t + ' / ' + lbl:<38}{pr.n:>4}  {shares}{spy:>7.2f}{tlt:>7.2f}{spread:>9.2f}")


if __name__ == "__main__":
    main()
