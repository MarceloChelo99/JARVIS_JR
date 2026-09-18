"""Exposure sizing: closed-form arm, point-in-time, regime feature, walk-forward."""

import json
from datetime import date, timedelta

from jarvis_jr.research.bars import Bar, BarSeries, BarStore
from jarvis_jr.research.extractors import ResearchData
from jarvis_jr.research.macro_events import FirstPrint
from jarvis_jr.research.reactions import ReactionGraph
from jarvis_jr.research.sizing import ExposurePolicy, summarize, vix_regime, walk_forward
from jarvis_jr.research.taxonomy import EventInstance, EventType, LabeledEvent, Taxonomy
from jarvis_jr.trading.data_tools import DataTools

D = date


def _weekdays(start, n):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


class _Graph(ReactionGraph):
    """Injected events + bars; VIX prints injected via ResearchData.prints."""

    def __init__(self, events, bars, vix_by_day):
        et = EventType("cpi_release", "scheduled_macro", {"kind": "fred_vintage"},
                       {"direction": {"rule": "surprise_sign"}}, {}, {"lookback_years": 3})
        data = ResearchData(bars=BarStore("/nonexistent"))
        data._prints["VIXCLS"] = [FirstPrint("VIXCLS", d, d, v) for d, v in sorted(vix_by_day.items())]
        super().__init__(data, Taxonomy({"cpi_release": et}, {}))
        self._events["cpi_release"] = events
        self._bars = bars


def _world(event_move_by_index, n_days=1200, start=D(2019, 1, 2)):
    """Bars where SPY moves by a given amount over the 5 days after each event day."""
    days = _weekdays(start, n_days)
    spy, tlt = [], []
    p = 100.0
    moves = {}
    for idx, mv in event_move_by_index.items():
        for k in range(5):
            moves[idx + k] = mv / 5
    for i, d in enumerate(days):
        p *= 1 + moves.get(i, 0.0)
        spy.append(Bar(d, p, p))
        tlt.append(Bar(d, 100.0, 100.0))
    return days, {"SPY": BarSeries("SPY", spy), "TLT": BarSeries("TLT", tlt)}


def test_vix_regime_buckets():
    assert vix_regime(12.0) == "calm" and vix_regime(25.0) == "elevated" and vix_regime(45.0) == "stressed"
    assert vix_regime(None) == "unknown"


def test_advise_is_mean_variance_and_point_in_time():
    # 8 "above" events, each followed by +2% over 5 days (low variance) -> full exposure
    idx = [50, 100, 150, 200, 250, 300, 350, 400]
    days, bars = _world({i: 0.02 for i in idx})
    events = [LabeledEvent(EventInstance("cpi_release", days[i], 1.0, 0.0), "above", "large") for i in idx]
    vix = {d: 15.0 for d in days}
    pol = ExposurePolicy(_Graph(events, bars, vix), risk_aversion=4.0, lookback_years=3)
    late = pol.advise("cpi_release", "above", days[450])
    assert late.n == 8 and not late.fallback and late.exposure == 1.0
    # as of day 103 the day-100 event's 5-day window (ends day 104) is still open -> only 1 event
    early = pol.advise("cpi_release", "above", days[103])
    assert early.fallback and early.exposure == 0.6 and early.n == 1
    # events after as_of never count
    assert pol.advise("cpi_release", "above", days[40]).n == 0


def test_negative_regime_sizes_to_zero_and_regime_is_a_feature():
    idx_calm = [50, 100, 150, 200, 250, 300]
    idx_stress = [450, 500, 550, 600, 650, 700]
    days, bars = _world({**{i: 0.02 for i in idx_calm}, **{i: -0.03 for i in idx_stress}})
    events = [LabeledEvent(EventInstance("cpi_release", days[i], 1.0, 0.0), "above", "large")
              for i in idx_calm + idx_stress]
    vix = {d: (35.0 if i >= 425 else 15.0) for i, d in enumerate(days)}
    pol = ExposurePolicy(_Graph(events, bars, vix), risk_aversion=4.0, use_regime=True)
    stressed = pol.advise("cpi_release", "above", days[720])
    assert stressed.regime == "stressed" and stressed.n == 6 and stressed.exposure == 0.0
    # ignoring the regime pools both eras and lands in between
    pooled = ExposurePolicy(_Graph(events, bars, vix), use_regime=False).advise("cpi_release", "above", days[720])
    assert pooled.n == 12 and 0.0 <= pooled.exposure < 1.0


def test_walk_forward_never_uses_the_test_event_and_beats_baselines_here():
    idx = list(range(50, 900, 50))
    days, bars = _world({i: (0.02 if k % 2 == 0 else -0.02) for k, i in enumerate(idx)})
    events = []
    for k, i in enumerate(idx):  # alternate labels so each label's history is consistently signed
        events.append(LabeledEvent(EventInstance("cpi_release", days[i], 1.0, 0.0), "above" if k % 2 == 0 else "below", "large"))
    vix = {d: 15.0 for d in days}
    pol = ExposurePolicy(_Graph(events, bars, vix), risk_aversion=4.0, min_n=3, lookback_years=3)
    rows = walk_forward(pol, "cpi_release", days[400])
    assert rows and all(r.r5 != 0 for r in rows)
    s = summarize(rows)
    assert s["n"] == len(rows)
    assert s["policy"]["sum_pct"] > s["const_0.6"]["sum_pct"] > s["const_0"]["sum_pct"]
    assert s["policy"]["worst_pct"] >= s["const_1"]["worst_pct"]


def test_sizing_tool_through_data_tools():
    idx = [50, 100, 150, 200, 250, 300, 350]
    days, bars = _world({i: 0.02 for i in idx})
    events = [LabeledEvent(EventInstance("cpi_release", days[i], 1.0, 0.0), "above", "large") for i in idx]
    pol = ExposurePolicy(_Graph(events, bars, {d: 15.0 for d in days}), use_regime=True)
    tools = DataTools(clock=lambda: days[400], sizing=pol)
    out = json.loads(tools.dispatch("get_exposure_recommendation", {"event_type": "cpi_release", "label": "above"}))
    assert out["recommended_exposure"] == 1.0 and out["n"] == 7 and out["vix_regime"] == "calm"
    assert "unknown event_type" in tools.dispatch("get_exposure_recommendation", {"event_type": "nope"})
