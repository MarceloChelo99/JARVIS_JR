"""FRED macro feed with true point-in-time data (ALFRED vintages).

FRED's `realtime_start`/`realtime_end` parameters return a series *as it was
known on that date* — observations that had been released by then, with the
values as first published (before later revisions). Setting both to `as_of`
is exactly the no-lookahead guarantee the feed protocol demands.

Responses are cached on disk keyed by (series, as_of, n) so a scenario run
is reproducible and offline after its first pass. Env: FRED_API_KEY
(free at https://fred.stlouisfed.org/docs/api/api_key.html).

The only file that knows FRED's URL or JSON shape.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import date
from pathlib import Path

import requests

from jarvis_jr.trading.feed import MacroObservation

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_CACHE = Path(__file__).resolve().parents[3] / "data" / "fred_cache"

# Common series ids, for the tool description. Any FRED id works.
COMMON_SERIES = {
    "CPIAUCSL": "CPI (all items, SA, monthly)",
    "UNRATE": "unemployment rate (monthly)",
    "FEDFUNDS": "effective fed funds rate (monthly)",
    "DGS10": "10-year Treasury yield (daily)",
    "DGS2": "2-year Treasury yield (daily)",
    "T10Y2Y": "10y-2y spread (daily)",
    "VIXCLS": "VIX close (daily)",
    "PAYEMS": "nonfarm payrolls (monthly)",
}


def _http_fetch(params: dict) -> dict:
    resp = requests.get(FRED_URL, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


class FredMacro:
    def __init__(
        self,
        api_key: str = "",
        cache_dir: Path = DEFAULT_CACHE,
        fetch: Callable[[dict], dict] = _http_fetch,
    ):
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        if not self.api_key:
            raise ValueError("FredMacro needs FRED_API_KEY")
        self.cache_dir = Path(cache_dir)
        self._fetch = fetch

    def observations(self, series: str, as_of: date, n: int) -> list[MacroObservation]:
        series = series.upper()
        cache = self.cache_dir / f"{series}_{as_of.isoformat()}_{n}.json"
        if cache.is_file():
            data = json.loads(cache.read_text())
        else:
            data = self._fetch(
                {
                    "series_id": series,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "realtime_start": as_of.isoformat(),
                    "realtime_end": as_of.isoformat(),
                    "observation_end": as_of.isoformat(),
                    "sort_order": "desc",
                    "limit": n,
                }
            )
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data))
        out = []
        for o in data.get("observations", []):
            if o.get("value") in (".", "", None):
                continue  # FRED's missing-value marker
            out.append(
                MacroObservation(
                    series=series,
                    day=date.fromisoformat(o["date"]),
                    value=float(o["value"]),
                    # With a single-day realtime window FRED clips realtime_start to as_of,
                    # so this is "known by as_of", not the first-release date. Good enough
                    # for no-lookahead; widen the window if true release dates are needed.
                    released=date.fromisoformat(o.get("realtime_start", as_of.isoformat())),
                )
            )
        # Belt and braces: the API already filtered by vintage, but the protocol
        # promises it, so enforce it here too.
        return [o for o in out if o.released <= as_of and o.day <= as_of][:n]
