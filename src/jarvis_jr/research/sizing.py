"""Exposure sizing after an event: a contextual bandit with a closed-form arm.

The learned component sizes RISK; it does not pick names. State = (event
type, label, VIX regime on the event day). Arm = an exposure level. The
"learning" is an identity, not a fit: for the events in a trailing window
that match the state, the mean-variance optimal exposure is

    e* = clip( mean(r) / (lambda * var(r)) , 0, 1 )

with r the 5-day SPY return after each event (Markowitz / fractional Kelly
with risk aversion lambda). It is then snapped to the exposure grid. Too few
matching events -> fall back to the prior exposure. Point-in-time: only
events whose reaction window closed before `as_of` count.

Why this and not RL on prices: ~600 macro events since 2010 is enough to
estimate a handful of state means; it is nowhere near enough to learn a
policy from raw prices, and regimes shift (see the reaction graph). Evaluate
walk-forward on generated families — never train on them.

What the first walk-forward taught (2026-09-18): a fine state (type x
direction/magnitude x VIX regime) fell back to the prior 90% of the time;
direction-only with no regime feature and a 6-year window is what the data
supports. It does not beat constant exposure on CPI prints in an up-market;
it clearly does around FOMC decisions (see DECISIONS.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from statistics import mean, pvariance

from jarvis_jr.research.extractors import window_start
from jarvis_jr.research.reactions import STOCKS, ReactionGraph, reaction_for

EXPOSURE_GRID = (0.0, 0.3, 0.6, 1.0)
VIX_BUCKETS = ((0, 20, "calm"), (20, 30, "elevated"), (30, 10_000, "stressed"))


def vix_regime(vix: float | None) -> str:
    if vix is None:
        return "unknown"
    for lo, hi, name in VIX_BUCKETS:
        if lo <= vix < hi:
            return name
    return "unknown"


@dataclass(frozen=True)
class SizingAdvice:
    event_type: str
    label: str
    regime: str
    as_of: date
    n: int
    mean_5d: float | None
    vol_5d: float | None
    raw_exposure: float | None  # mean/(lambda*var) before clipping
    exposure: float  # snapped to grid
    fallback: bool  # True when n was too small and the prior was used

    def to_json(self) -> str:
        return json.dumps(
            {
                "event_type": self.event_type,
                "label": self.label,
                "vix_regime": self.regime,
                "as_of": self.as_of.isoformat(),
                "n": self.n,
                "mean_5d_pct": round(self.mean_5d * 100, 2) if self.mean_5d is not None else None,
                "vol_5d_pct": round(self.vol_5d * 100, 2) if self.vol_5d is not None else None,
                "recommended_exposure": self.exposure,
                "basis": "prior (too few matching events)" if self.fallback else "mean-variance over matching events",
                "caution": "small sample; treat as anecdote" if self.n < 8 else "",
            }
        )


class ExposurePolicy:
    def __init__(
        self,
        graph: ReactionGraph,
        risk_aversion: float = 4.0,
        prior_exposure: float = 0.6,
        min_n: int = 6,
        horizon: int = 5,
        grid: tuple[float, ...] = EXPOSURE_GRID,
        use_regime: bool = False,  # VIX regime as a state feature: too fine for the data (see DECISIONS)
        direction_only: bool = True,  # match on above/below, ignore magnitude (coarser state)
        lookback_years: int = 6,
    ):
        self.graph = graph
        self.risk_aversion = risk_aversion
        self.prior = prior_exposure
        self.min_n = min_n
        self.horizon = horizon
        self.grid = grid
        self.use_regime = use_regime
        self.direction_only = direction_only
        self.lookback_years = lookback_years
        self._vix: dict[date, float] | None = None

    # ---- state features -----------------------------------------------------------

    def vix_on(self, day: date) -> float | None:
        if self._vix is None:
            try:
                self._vix = {p.obs_day: p.value for p in self.graph.data.prints("VIXCLS")}
            except ValueError:
                self._vix = {}
        # last close on or before the day
        candidates = [d for d in self._vix if d <= day]
        return self._vix[max(candidates)] if candidates else None

    def regime_on(self, day: date) -> str:
        return vix_regime(self.vix_on(day)) if self.use_regime else "any"

    # ---- the arm ------------------------------------------------------------------

    def advise(self, event_type: str, label: str, as_of: date, lookback_years: int | None = None) -> SizingAdvice:
        self.graph.taxonomy[event_type]  # raises KeyError for unknown types
        years = lookback_years or self.lookback_years
        start = window_start(as_of, years)
        regime = self.regime_on(as_of)
        bars = self.graph.bars()
        want = label.split("/")[0] if self.direction_only else label
        rets: list[float] = []
        for ev in self.graph.labeled(event_type):
            d = ev.instance.day
            if not (start <= d <= as_of):
                continue
            if want != "all" and ev.label != want and ev.direction != want:
                continue
            if self.use_regime and self.regime_on(d) != regime:
                continue
            r = reaction_for(bars, d, self.horizon, as_of)
            if r is not None and STOCKS in r.returns:
                rets.append(r.returns[STOCKS])
        if len(rets) < self.min_n:
            return SizingAdvice(event_type, label, regime, as_of, len(rets), None, None, None, self.snap(self.prior), True)
        m, v = mean(rets), pvariance(rets)
        raw = m / (self.risk_aversion * v) if v > 0 else (1.0 if m > 0 else 0.0)
        return SizingAdvice(event_type, label, regime, as_of, len(rets), m, v ** 0.5, raw, self.snap(raw), False)

    def snap(self, exposure: float) -> float:
        e = min(1.0, max(0.0, exposure))
        return min(self.grid, key=lambda g: abs(g - e))


# ---- walk-forward evaluation ---------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardRow:
    day: date
    label: str
    regime: str
    r5: float
    policy_exposure: float
    fallback: bool


def walk_forward(
    policy: ExposurePolicy, event_type: str, since: date, until: date | None = None,
    lookback_years: int | None = None,
) -> list[WalkForwardRow]:
    """For each event after `since`: advise as of the day BEFORE it (no lookahead),
    then record the realized 5-day SPY return. Test events never inform their own advice."""
    rows = []
    bars = policy.graph.bars()
    for ev in policy.graph.labeled(event_type):
        d = ev.instance.day
        if d < since or (until and d > until):
            continue
        realized = reaction_for(bars, d, policy.horizon)
        if realized is None:
            continue
        advice = policy.advise(event_type, ev.label, d - _one_day(), lookback_years)
        rows.append(WalkForwardRow(d, ev.label, advice.regime, realized.returns[STOCKS], advice.exposure, advice.fallback))
    return rows


def _one_day():
    from datetime import timedelta

    return timedelta(days=1)


def summarize(rows: list[WalkForwardRow], baselines: tuple[float, ...] = (0.0, 0.6, 1.0)) -> dict:
    """Cumulative contribution (exposure x 5-day return) of the policy vs constant baselines."""
    if not rows:
        return {"n": 0}
    out = {"n": len(rows), "fallback_share": round(sum(r.fallback for r in rows) / len(rows), 2)}
    policy_ret = [r.policy_exposure * r.r5 for r in rows]
    out["policy"] = _stats(policy_ret)
    for b in baselines:
        out[f"const_{b:g}"] = _stats([b * r.r5 for r in rows])
    out["mean_policy_exposure"] = round(mean(r.policy_exposure for r in rows), 2)
    return out


def _stats(x: list[float]) -> dict:
    return {
        "sum_pct": round(sum(x) * 100, 2),
        "mean_pct": round(mean(x) * 100, 3),
        "worst_pct": round(min(x) * 100, 2),
        "hit_rate": round(sum(1 for v in x if v > 0) / len(x), 2),
    }
