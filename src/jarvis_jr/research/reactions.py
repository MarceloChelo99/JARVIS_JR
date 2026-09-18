"""Reaction archetypes and the event → reaction graph.

The interesting object is not the event but the cross-asset SIGNATURE the
market printed in response. Four archetypes are defined by identity — the
signs of stocks and bonds — plus "muted" when nothing moved:

    inflation_shock  stocks down, bonds down   (hot CPI, hawkish Fed)
    growth_scare     stocks down, bonds up     (weak jobs, VIX spell)
    goldilocks       stocks up,   bonds up     (cool CPI, solid growth)
    reflation        stocks up,   bonds down   (strong growth, yields up)
    muted            neither moved past the threshold

No fitting. Clustering, if ever, is a *check* on these, not a replacement.

The graph: node = (event_type, label); edge weights = the empirical
distribution over archetypes, computed over a TRAILING window ending at
`as_of` (the same regime logic as the playbook — hot CPI was "goldilocks"
in 2019 and "inflation_shock" in 2022, and the graph must show the shift).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import date
from statistics import mean

from jarvis_jr.research.bars import BarSeries, BarStore
from jarvis_jr.research.extractors import ResearchData, extract, window_start
from jarvis_jr.research.labels import label_events
from jarvis_jr.research.taxonomy import DEFAULT_TAXONOMY, LabeledEvent, Taxonomy, load_taxonomy

ARCHETYPES = ("inflation_shock", "growth_scare", "goldilocks", "reflation", "muted")
STOCKS, BONDS, DEFENSIVE, GROWTH = "SPY", "TLT", "XLP", "QQQ"
MUTED_THRESHOLD = 0.0015  # 15 bp on both legs = no signal


def archetype(stocks: float, bonds: float, threshold: float = MUTED_THRESHOLD) -> str:
    if abs(stocks) < threshold and abs(bonds) < threshold:
        return "muted"
    if stocks < 0:
        return "inflation_shock" if bonds < 0 else "growth_scare"
    return "goldilocks" if bonds >= 0 else "reflation"


@dataclass(frozen=True)
class Reaction:
    day: date
    horizon: int
    returns: dict  # symbol -> return
    archetype: str

    @property
    def defensive_spread(self) -> float | None:
        if DEFENSIVE in self.returns and GROWTH in self.returns:
            return self.returns[DEFENSIVE] - self.returns[GROWTH]
        return None


def reaction_for(bars: dict[str, BarSeries], day: date, horizon: int, as_of: date | None = None) -> Reaction | None:
    """Cross-asset returns from the close before `day` to `horizon` days on/after it."""
    rets: dict[str, float] = {}
    for sym, bs in bars.items():
        i = bs.index_on_or_after(day)
        if i is None or i + horizon - 1 >= len(bs):
            return None
        if as_of is not None and bs.bars[i + horizon - 1].day > as_of:
            return None  # window not closed on as_of
        r = bs.forward_return(day, horizon)
        if r is None:
            return None
        rets[sym] = r
    if STOCKS not in rets or BONDS not in rets:
        return None
    return Reaction(day, horizon, rets, archetype(rets[STOCKS], rets[BONDS]))


@dataclass(frozen=True)
class ReactionProfile:
    event_type: str
    label: str
    as_of: date
    lookback_years: int
    horizon: int
    n: int
    counts: dict  # archetype -> count
    mean_returns: dict  # symbol -> mean return
    mean_defensive_spread: float | None

    @property
    def shares(self) -> dict:
        return {a: (self.counts.get(a, 0) / self.n if self.n else 0.0) for a in ARCHETYPES}

    @property
    def dominant(self) -> str:
        return max(self.shares, key=self.shares.get) if self.n else "none"

    def to_json(self) -> str:
        return json.dumps(
            {
                "event_type": self.event_type,
                "label": self.label,
                "as_of": self.as_of.isoformat(),
                "window": f"last {self.lookback_years}y, {self.horizon}-day reaction",
                "n": self.n,
                "caution": "small sample; treat as anecdote" if self.n < 8 else "",
                "dominant_reaction": self.dominant,
                "reaction_shares": {a: round(s, 2) for a, s in self.shares.items() if s > 0},
                "mean_returns_pct": {s: round(r * 100, 2) for s, r in self.mean_returns.items()},
                "defensives_minus_growth_pct": (
                    round(self.mean_defensive_spread * 100, 2) if self.mean_defensive_spread is not None else None
                ),
            }
        )


class ReactionGraph:
    """Point-in-time event → reaction profiles over the whole taxonomy."""

    def __init__(
        self,
        data: ResearchData | None = None,
        taxonomy: Taxonomy | None = None,
        symbols: tuple[str, ...] = (STOCKS, BONDS, DEFENSIVE, GROWTH),
    ):
        self.data = data or ResearchData()
        self.taxonomy = taxonomy or load_taxonomy(DEFAULT_TAXONOMY)
        self.symbols = symbols
        self._events: dict[str, list[LabeledEvent]] = {}
        self._bars: dict[str, BarSeries] | None = None

    def available_types(self) -> list[str]:
        return sorted(self.taxonomy.types)

    def labeled(self, type_id: str) -> list[LabeledEvent]:
        if type_id not in self._events:
            et = self.taxonomy[type_id]
            self._events[type_id] = label_events(extract(et, self.data), et)
        return self._events[type_id]

    def labels_for(self, type_id: str) -> list[str]:
        return sorted({e.label for e in self.labeled(type_id)})

    def bars(self) -> dict[str, BarSeries]:
        if self._bars is None:
            store: BarStore = self.data.bars
            self._bars = {s: store.get(s) for s in self.symbols}
        return self._bars

    def profile(
        self, type_id: str, label: str, as_of: date, lookback_years: int | None = None, horizon: int = 1
    ) -> ReactionProfile:
        et = self.taxonomy[type_id]
        years = lookback_years or int(et.playbook.get("lookback_years", 3))
        start = window_start(as_of, years)
        wanted = [
            e for e in self.labeled(type_id)
            if start <= e.instance.day <= as_of and (label == "all" or e.label == label or e.direction == label)
        ]
        reactions = [r for r in (reaction_for(self.bars(), e.instance.day, horizon, as_of) for e in wanted) if r]
        counts = Counter(r.archetype for r in reactions)
        symbols_seen = {s for r in reactions for s in r.returns}
        means = {s: mean(r.returns[s] for r in reactions if s in r.returns) for s in symbols_seen} if reactions else {}
        spreads = [r.defensive_spread for r in reactions if r.defensive_spread is not None]
        return ReactionProfile(
            type_id, label, as_of, years, horizon, len(reactions), dict(counts), means,
            mean(spreads) if spreads else None,
        )
