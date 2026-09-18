"""Event study: conditional reaction statistics for a set of dated events.

Identity, not model: given events and bars, the numbers below are computed
exactly. No fitting. `as_of` makes the study point-in-time — only events
whose full forward window closed on or before `as_of` count, so a playbook
consulted on a simulated day never contains that day's future.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from statistics import mean, median

from jarvis_jr.research.bars import BarSeries
from jarvis_jr.research.macro_events import MacroEvent

HORIZONS = (1, 5)


@dataclass(frozen=True)
class ReactionStats:
    symbol: str
    horizon: int
    n: int
    mean: float
    median: float
    hit_rate: float  # fraction of events with a positive return
    worst: float
    best: float


@dataclass(frozen=True)
class EventStudy:
    series: str
    direction: str  # "above" | "below" | "all"
    as_of: date | None
    n_events: int
    stats: tuple[ReactionStats, ...]

    def for_symbol(self, symbol: str, horizon: int) -> ReactionStats | None:
        return next((s for s in self.stats if s.symbol == symbol and s.horizon == horizon), None)


def study(
    events: list[MacroEvent],
    bars_by_symbol: dict[str, BarSeries],
    direction: str = "all",
    horizons: tuple[int, ...] = HORIZONS,
    as_of: date | None = None,
) -> EventStudy:
    selected = [e for e in events if direction == "all" or e.direction == direction]
    stats: list[ReactionStats] = []
    counted: set[date] = set()
    for symbol, bars in bars_by_symbol.items():
        for h in horizons:
            rets: list[float] = []
            for e in selected:
                i = bars.index_on_or_after(e.release_day)
                if i is None or i + h - 1 >= len(bars):
                    continue
                if as_of is not None and bars.bars[i + h - 1].day > as_of:
                    continue  # forward window not closed yet on as_of
                r = bars.forward_return(e.release_day, h)
                if r is not None:
                    rets.append(r)
                    counted.add(e.release_day)
            if rets:
                stats.append(
                    ReactionStats(
                        symbol=symbol,
                        horizon=h,
                        n=len(rets),
                        mean=mean(rets),
                        median=median(rets),
                        hit_rate=sum(1 for r in rets if r > 0) / len(rets),
                        worst=min(rets),
                        best=max(rets),
                    )
                )
    return EventStudy(
        series=events[0].series if events else "",
        direction=direction,
        as_of=as_of,
        n_events=len(counted),
        stats=tuple(stats),
    )
