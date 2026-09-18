"""The macro playbook: "what usually happens after this kind of release".

Wraps event studies as a feed the agent can query. Point-in-time by
construction: every lookup takes `as_of` and only uses events whose reaction
window closed before it, over a TRAILING window (regimes change: see
lookup's docstring). Results are cached per exact as_of day — a coarser key
would let a later cutoff's answer leak into an earlier day.

Data comes from the local stores filled by scripts/research_fetch.py:
data/bars/*.csv (Tiingo) and data/macro_vintages/*.json (FRED).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from jarvis_jr.research.bars import BarSeries, BarStore
from jarvis_jr.research.event_study import HORIZONS, EventStudy, study
from jarvis_jr.research.macro_events import (
    DEFAULT_VINTAGES_DIR,
    SERIES_KIND,
    MacroEvent,
    events_from_prints,
    first_prints,
)

DEFAULT_SYMBOLS = ("SPY", "QQQ", "XLP", "TLT")
DEFAULT_LOOKBACK_YEARS = 3


@dataclass(frozen=True)
class PlaybookEntry:
    series: str
    direction: str
    as_of: date
    n_events: int
    reactions: tuple  # ReactionStats...
    lookback_years: int = DEFAULT_LOOKBACK_YEARS
    big_only: bool = True

    def to_json(self) -> str:
        return json.dumps(
            {
                "series": self.series,
                "direction": self.direction,
                "as_of": self.as_of.isoformat(),
                "window": f"last {self.lookback_years}y"
                + (", largest-third surprises only" if self.big_only else ""),
                "n_events": self.n_events,
                "caution": "small sample; treat as anecdote" if self.n_events < 8 else "",
                "reactions": [
                    {
                        "symbol": r.symbol,
                        "horizon_days": r.horizon,
                        "n": r.n,
                        "mean_pct": round(r.mean * 100, 2),
                        "median_pct": round(r.median * 100, 2),
                        "up_rate": round(r.hit_rate, 2),
                        "worst_pct": round(r.worst * 100, 2),
                        "best_pct": round(r.best * 100, 2),
                    }
                    for r in self.reactions
                ],
            }
        )


class MacroPlaybook:
    def __init__(
        self,
        bars: BarStore,
        vintages_dir: Path = DEFAULT_VINTAGES_DIR,
        symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
        horizons: tuple[int, ...] = HORIZONS,
    ):
        self.bars = bars
        self.vintages_dir = Path(vintages_dir)
        self.symbols = tuple(s.upper() for s in symbols)
        self.horizons = horizons
        self._events: dict[str, list[MacroEvent]] = {}
        self._bars: dict[str, BarSeries] = {}
        self._cache: dict[tuple[str, str, str], PlaybookEntry] = {}

    # ---- data access (lazy, cached) -----------------------------------------

    def available_series(self) -> list[str]:
        return sorted(p.stem for p in self.vintages_dir.glob("*.json"))

    def events(self, series: str) -> list[MacroEvent]:
        series = series.upper()
        if series not in self._events:
            path = self.vintages_dir / f"{series}.json"
            if not path.is_file():
                raise ValueError(
                    f"no vintage data for {series}; run scripts/research_fetch.py --series {series}"
                )
            raw = json.loads(path.read_text())
            kind = SERIES_KIND.get(series, "level")
            self._events[series] = events_from_prints(first_prints(series, raw), kind)
        return self._events[series]

    def bar_series(self, symbol: str) -> BarSeries:
        symbol = symbol.upper()
        if symbol not in self._bars:
            self._bars[symbol] = self.bars.get(symbol)
        return self._bars[symbol]

    # ---- the feed -------------------------------------------------------------

    def lookup(
        self,
        series: str,
        direction: str,
        as_of: date,
        lookback_years: int = DEFAULT_LOOKBACK_YEARS,
        big_only: bool = True,
    ) -> PlaybookEntry:
        """Reaction stats over a trailing window ending at as_of.

        Regime matters more than sample size here: a hot CPI print moved nothing in
        2010-2020 and moved everything in 2021-2023. So the window is trailing, not
        expanding, and `big_only` keeps the top third of |surprise| within it — the
        prints markets actually noticed. `n` is reported; treat n < 8 as anecdote.
        """
        series, direction = series.upper(), direction.lower()
        if direction not in ("above", "below", "all"):
            raise ValueError("direction must be 'above', 'below' or 'all'")
        key = (series, direction, as_of.isoformat(), lookback_years, big_only)
        if key in self._cache:
            return self._cache[key]
        start = date(as_of.year - lookback_years, as_of.month, min(as_of.day, 28))
        events = [e for e in self.events(series) if start <= e.release_day <= as_of]
        if big_only and len(events) >= 6:
            cut = sorted(abs(e.surprise) for e in events)[len(events) * 2 // 3]
            events = [e for e in events if abs(e.surprise) >= cut]
        bars_by_symbol = {s: self.bar_series(s) for s in self.symbols if len(self.bar_series(s))}
        result: EventStudy = study(events, bars_by_symbol, direction, self.horizons, as_of=as_of)
        entry = PlaybookEntry(
            series, direction, as_of, result.n_events, result.stats, lookback_years, big_only
        )
        self._cache[key] = entry
        return entry
