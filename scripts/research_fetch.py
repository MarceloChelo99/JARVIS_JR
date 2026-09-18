"""Fill the local research stores: daily bars from Tiingo, macro vintages from FRED.

Usage:
    uv run python scripts/research_fetch.py                       # defaults below
    uv run python scripts/research_fetch.py --symbols SPY QQQ XLP TLT AAPL MSFT --start 2010-01-01
    uv run python scripts/research_fetch.py --series CPIAUCSL UNRATE PAYEMS FEDFUNDS --refresh

Needs TIINGO_API_KEY and FRED_API_KEY in .env. Re-running is cheap: bars are
merged into data/bars/<T>.csv, vintages are cached unless --refresh.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date

from jarvis_jr.research.bars import BarStore, TiingoBars
from jarvis_jr.research.macro_events import FredVintages, macro_events
from jarvis_jr.settings import load_settings

DEFAULT_SYMBOLS = ["SPY", "QQQ", "XLP", "TLT", "AAPL", "MSFT", "KO"]
DEFAULT_SERIES = [
    "CPIAUCSL", "CPILFESL", "PCEPI", "PPIFIS", "RSAFS", "GDPC1", "ICSA", "UMCSENT", "HOUST",
    "INDPRO", "UNRATE", "PAYEMS", "FEDFUNDS", "DFEDTARU", "VIXCLS",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    p.add_argument("--series", nargs="*", default=DEFAULT_SERIES)
    p.add_argument("--start", type=date.fromisoformat, default=date(2010, 1, 1))
    p.add_argument("--refresh", action="store_true", help="re-download FRED vintages")
    args = p.parse_args()
    load_settings()  # loads .env

    failures = 0
    if args.symbols:
        try:
            tiingo = TiingoBars()
        except ValueError as e:
            print(f"[bars] skipped: {e}")
            tiingo = None
        if tiingo:
            store = BarStore()
            for sym in args.symbols:
                try:
                    bars = tiingo.bars(sym, args.start)
                    path = store.put(sym, bars)
                    print(f"[bars] {sym}: {len(bars)} rows -> {path}")
                    time.sleep(0.3)  # be polite to the free tier
                except Exception as e:  # noqa: BLE001
                    failures += 1
                    print(f"[bars] {sym} FAILED: {e}")

    if args.series:
        try:
            fred = FredVintages()
        except ValueError as e:
            print(f"[macro] skipped: {e}")
            fred = None
        if fred:
            for s in args.series:
                try:
                    fred.raw(s, refresh=args.refresh)
                    ev = macro_events(s, fred)
                    above = sum(1 for e in ev if e.direction == "above")
                    print(f"[macro] {s}: {len(ev)} events ({above} above / {len(ev) - above} below), "
                          f"{ev[0].release_day if ev else '-'} .. {ev[-1].release_day if ev else '-'}")
                except Exception as e:  # noqa: BLE001
                    failures += 1
                    print(f"[macro] {s} FAILED: {e}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
