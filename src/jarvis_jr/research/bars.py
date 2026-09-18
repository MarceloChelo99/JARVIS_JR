"""Historical daily bars: a CSV-per-ticker store plus the Tiingo fetcher.

Store layout: data/bars/<TICKER>.csv with columns date,close,adj_close.
Adjusted close (splits + dividends, CRSP method) is what event studies use.
Tiingo is the only foreign API this file knows; `fetch` is injectable so
tests never touch the network. Env: TIINGO_API_KEY.
"""

from __future__ import annotations

import bisect
import csv
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import requests

from jarvis_jr.settings import REPO_ROOT

DEFAULT_BARS_DIR = REPO_ROOT / "data" / "bars"
TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"


@dataclass(frozen=True)
class Bar:
    day: date
    close: float
    adj_close: float


class BarSeries:
    """One ticker's bars, sorted by day, with trading-day arithmetic."""

    def __init__(self, ticker: str, bars: list[Bar]):
        self.ticker = ticker
        self.bars = sorted(bars, key=lambda b: b.day)
        self._days = [b.day for b in self.bars]

    def __len__(self) -> int:
        return len(self.bars)

    @property
    def first_day(self) -> date | None:
        return self._days[0] if self._days else None

    @property
    def last_day(self) -> date | None:
        return self._days[-1] if self._days else None

    def index_on_or_after(self, day: date) -> int | None:
        i = bisect.bisect_left(self._days, day)
        return i if i < len(self._days) else None

    def forward_return(self, day: date, horizon: int) -> float | None:
        """Return from the close BEFORE `day` to the close `horizon` trading days on/after it.
        horizon=1 is the reaction on the first trading day on/after `day` itself."""
        i = self.index_on_or_after(day)
        if i is None or i == 0 or i + horizon - 1 >= len(self.bars):
            return None
        base = self.bars[i - 1].adj_close
        end = self.bars[i + horizon - 1].adj_close
        return end / base - 1 if base else None

    def close_on_or_before(self, day: date) -> Bar | None:
        i = bisect.bisect_right(self._days, day)
        return self.bars[i - 1] if i else None


class BarStore:
    def __init__(self, root: Path = DEFAULT_BARS_DIR):
        self.root = Path(root)

    def path(self, ticker: str) -> Path:
        return self.root / f"{ticker.upper()}.csv"

    def tickers(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.csv"))

    def get(self, ticker: str) -> BarSeries:
        path = self.path(ticker)
        bars: list[Bar] = []
        if path.is_file():
            with path.open() as f:
                for r in csv.DictReader(f):
                    bars.append(Bar(date.fromisoformat(r["date"]), float(r["close"]), float(r["adj_close"])))
        return BarSeries(ticker.upper(), bars)

    def put(self, ticker: str, bars: list[Bar]) -> Path:
        path = self.path(ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = {b.day: b for b in self.get(ticker).bars}
        merged.update({b.day: b for b in bars})
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "close", "adj_close"])
            for day in sorted(merged):
                b = merged[day]
                w.writerow([day.isoformat(), b.close, b.adj_close])
        return path


def _http_fetch(url: str, params: dict, headers: dict) -> list[dict]:
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


class TiingoBars:
    def __init__(self, api_key: str = "", fetch: Callable[[str, dict, dict], list[dict]] = _http_fetch):
        self.api_key = api_key or os.environ.get("TIINGO_API_KEY", "")
        if not self.api_key:
            raise ValueError("TiingoBars needs TIINGO_API_KEY")
        self._fetch = fetch

    def bars(self, ticker: str, start: date, end: date | None = None) -> list[Bar]:
        params = {"startDate": start.isoformat(), "format": "json"}
        if end:
            params["endDate"] = end.isoformat()
        rows = self._fetch(
            TIINGO_URL.format(ticker=ticker.lower()),
            params,
            {"Authorization": f"Token {self.api_key}", "Content-Type": "application/json"},
        )
        return [
            Bar(date.fromisoformat(r["date"][:10]), float(r["close"]), float(r["adjClose"]))
            for r in rows
        ]
