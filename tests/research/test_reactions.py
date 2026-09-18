"""Reaction archetypes (identity rules) and the point-in-time reaction graph."""

import json
from datetime import date, timedelta

import pytest

from jarvis_jr.research.bars import Bar, BarSeries, BarStore
from jarvis_jr.research.extractors import ResearchData
from jarvis_jr.research.reactions import ReactionGraph, archetype, reaction_for
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


def test_archetype_identity_rules():
    assert archetype(-0.01, -0.01) == "inflation_shock"
    assert archetype(-0.01, +0.01) == "growth_scare"
    assert archetype(+0.01, +0.01) == "goldilocks"
    assert archetype(+0.01, -0.01) == "reflation"
    assert archetype(0.0005, -0.0005) == "muted"
    assert archetype(0.0005, -0.01) == "reflation"  # bonds moved, stocks barely: not muted


def _bars_with_shock(days, shock_index, stocks_move, bonds_move):
    spy, tlt, xlp, qqq = [], [], [], []
    ps = pt = px = pq = 100.0
    for i, d in enumerate(days):
        if i == shock_index:
            ps *= 1 + stocks_move
            pt *= 1 + bonds_move
            px *= 1 + stocks_move / 2
            pq *= 1 + stocks_move * 1.5
        spy.append(Bar(d, ps, ps))
        tlt.append(Bar(d, pt, pt))
        xlp.append(Bar(d, px, px))
        qqq.append(Bar(d, pq, pq))
    return {"SPY": BarSeries("SPY", spy), "TLT": BarSeries("TLT", tlt),
            "XLP": BarSeries("XLP", xlp), "QQQ": BarSeries("QQQ", qqq)}


def test_reaction_for_measures_from_prior_close_and_respects_as_of():
    days = _weekdays(D(2024, 1, 2), 10)
    bars = _bars_with_shock(days, 4, -0.02, -0.01)
    r = reaction_for(bars, days[4], 1)
    assert r.archetype == "inflation_shock"
    assert r.returns["SPY"] == pytest.approx(-0.02)
    assert r.defensive_spread == pytest.approx(-0.01 - (-0.03))  # defensives beat growth
    assert reaction_for(bars, days[4], 5, as_of=days[6]) is None  # 5-day window not closed
    assert reaction_for(bars, days[4], 5, as_of=days[8]) is not None


class _StubGraph(ReactionGraph):
    """ReactionGraph with injected labeled events and bars (no stores)."""

    def __init__(self, events, bars):
        et = EventType("cpi_release", "scheduled_macro", {"kind": "fred_vintage"},
                       {"direction": {"rule": "surprise_sign"}}, {}, {"lookback_years": 3})
        super().__init__(ResearchData(bars=BarStore("/nonexistent")), Taxonomy({"cpi_release": et}, {}))
        self._events["cpi_release"] = events
        self._bars = bars


def test_profile_is_trailing_and_point_in_time():
    days = _weekdays(D(2018, 1, 2), 2000)
    # two regimes: 2018-2019 events -> goldilocks; 2022+ events -> inflation shock
    bars = _bars_with_shock(days, 0, 0, 0)
    spy, tlt = [], []
    ps = pt = 100.0
    event_days = [days[i] for i in (50, 150, 250, 350, 1100, 1200, 1300, 1400)]
    for i, d in enumerate(days):
        if d in event_days[:4]:
            ps *= 1.01
            pt *= 1.01
        elif d in event_days[4:]:
            ps *= 0.99
            pt *= 0.99
        spy.append(Bar(d, ps, ps))
        tlt.append(Bar(d, pt, pt))
    bars["SPY"], bars["TLT"] = BarSeries("SPY", spy), BarSeries("TLT", tlt)
    events = [LabeledEvent(EventInstance("cpi_release", d, 1.0, 0.0), "above", "large") for d in event_days]
    g = _StubGraph(events, bars)

    early = g.profile("cpi_release", "above", days[400])
    assert early.n == 4 and early.dominant == "goldilocks"
    late = g.profile("cpi_release", "above", days[1450])
    assert late.n == 4 and late.dominant == "inflation_shock"  # trailing 3y sees only the new regime
    mid = g.profile("cpi_release", "above", days[1150])
    assert mid.n == 1  # window closed only for the first late event; early ones aged out
    assert g.profile("cpi_release", "above/large", days[1450]).n == 4  # full label works too
    assert g.profile("cpi_release", "below", days[1450]).n == 0


def test_reaction_tool_through_data_tools():
    days = _weekdays(D(2024, 1, 2), 30)
    bars = _bars_with_shock(days, 10, -0.02, 0.01)
    events = [LabeledEvent(EventInstance("cpi_release", days[10], 1.0, 0.0), "above", "large")]
    g = _StubGraph(events, bars)
    tools = DataTools(clock=lambda: days[20], reactions=g)
    out = json.loads(tools.dispatch("get_reaction_profile", {"event_type": "CPI_release", "label": "above"}))
    assert out["n"] == 1 and out["dominant_reaction"] == "growth_scare" and out["caution"]
    assert "unknown event_type" in tools.dispatch("get_reaction_profile", {"event_type": "nope"})
    before = tools.dispatch("get_reaction_profile", {"event_type": "cpi_release", "label": "below"})
    assert before.startswith("no completed")
