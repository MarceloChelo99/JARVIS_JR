"""Research package: bar store, macro events from vintages, event study math,
and the point-in-time playbook (including the no-lookahead contract)."""

import json
from datetime import date, timedelta

import pytest

from jarvis_jr.research.bars import Bar, BarSeries, BarStore, TiingoBars
from jarvis_jr.research.event_study import study
from jarvis_jr.research.macro_events import (
    FredVintages,
    MacroEvent,
    events_from_prints,
    first_prints,
)
from jarvis_jr.research.playbook import MacroPlaybook
from jarvis_jr.trading.data_tools import DataTools

D = date


def _weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# ---- bars ----------------------------------------------------------------------


def test_bar_store_roundtrip_and_merge(tmp_path):
    store = BarStore(tmp_path)
    days = _weekdays(D(2026, 1, 5), 3)
    store.put("spy", [Bar(days[0], 100, 100), Bar(days[1], 101, 101)])
    store.put("SPY", [Bar(days[1], 101, 101.5), Bar(days[2], 102, 102)])  # merge + overwrite
    s = store.get("SPY")
    assert [b.adj_close for b in s.bars] == [100, 101.5, 102]
    assert store.tickers() == ["SPY"]


def test_forward_return_semantics():
    days = _weekdays(D(2026, 1, 5), 7)  # Jan 5-9, then Jan 12-13
    prices = [100, 110, 121, 121, 100, 105, 105]
    s = BarSeries("X", [Bar(d, p, p) for d, p in zip(days, prices)])
    # event on day[1]: base is day[0] close (100); h=1 -> day[1] close 110
    assert s.forward_return(days[1], 1) == pytest.approx(0.10)
    assert s.forward_return(days[1], 2) == pytest.approx(0.21)
    assert s.forward_return(days[0], 1) is None  # no prior close
    assert s.forward_return(days[6], 2) is None  # window runs past data
    # event on Saturday Jan 10 maps to Monday Jan 12; base is Friday's close (100)
    assert s.forward_return(D(2026, 1, 10), 1) == pytest.approx(0.05)


def test_tiingo_parses_and_authenticates(monkeypatch):
    seen = {}

    def fake(url, params, headers):
        seen.update(url=url, params=params, headers=headers)
        return [{"date": "2026-01-05T00:00:00.000Z", "close": 100.0, "adjClose": 99.5}]

    t = TiingoBars(api_key="abc", fetch=fake)
    bars = t.bars("SPY", D(2026, 1, 1))
    assert bars == [Bar(D(2026, 1, 5), 100.0, 99.5)]
    assert "spy/prices" in seen["url"] and seen["headers"]["Authorization"] == "Token abc"
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    with pytest.raises(ValueError):
        TiingoBars()


# ---- macro events -------------------------------------------------------------


RAW = {
    "observations": [
        # two vintages of the Jan obs: first print 100.0 on Feb 12, revised Mar 10
        {"date": "2026-01-01", "realtime_start": "2026-02-12", "value": "100.0"},
        {"date": "2026-01-01", "realtime_start": "2026-03-10", "value": "100.3"},
        {"date": "2025-12-01", "realtime_start": "2026-01-14", "value": "99.0"},
        {"date": "2025-11-01", "realtime_start": "2025-12-10", "value": "98.5"},
        {"date": "2025-10-01", "realtime_start": "2025-11-13", "value": "."},
        # back-history: 1980 observation whose earliest vintage is the 1994 ALFRED start
        {"date": "1980-01-01", "realtime_start": "1994-02-17", "value": "50.0"},
    ]
}


def test_back_history_without_real_release_is_dropped():
    assert all(p.obs_day.year >= 2025 for p in first_prints("CPIAUCSL", RAW))


def test_first_prints_take_earliest_vintage_and_drop_missing():
    fp = first_prints("CPIAUCSL", RAW)
    assert [(p.obs_day, p.release_day, p.value) for p in fp] == [
        (D(2025, 11, 1), D(2025, 12, 10), 98.5),
        (D(2025, 12, 1), D(2026, 1, 14), 99.0),
        (D(2026, 1, 1), D(2026, 2, 12), 100.0),  # not the 100.3 revision
    ]


def test_events_surprise_by_kind():
    fp = first_prints("CPIAUCSL", RAW)
    (e,) = events_from_prints(fp, "pct")
    # prior pct = 99/98.5-1 = 0.5076%; expected = 99*(1+0.005076)=99.5025; actual 100 -> above
    assert e.release_day == D(2026, 2, 12) and e.direction == "above"
    assert e.surprise == pytest.approx(100 - 99 * (99 / 98.5))
    (lvl,) = events_from_prints(fp, "level")
    assert lvl.expected == 99.0 and lvl.surprise == pytest.approx(1.0)


def test_fred_vintages_cache(tmp_path):
    calls = []
    v = FredVintages(api_key="k", cache_dir=tmp_path, fetch=lambda p: (calls.append(p), RAW)[1])
    v.raw("cpiaucsl")
    v.raw("CPIAUCSL")
    assert len(calls) == 1 and calls[0]["realtime_start"] == "1776-07-04"
    assert (tmp_path / "CPIAUCSL.json").is_file()


# ---- event study + playbook ---------------------------------------------------


def _synthetic():
    """Events on days 2, 6, 10 of a 14-day series; SPY drops 2% on 'above' days,
    rises 1% on 'below' days, flat otherwise."""
    days = _weekdays(D(2026, 1, 5), 14)
    events = [
        MacroEvent("CPIAUCSL", days[2], days[2], 1, 0, +1.0),   # above
        MacroEvent("CPIAUCSL", days[6], days[6], 1, 0, -1.0),   # below
        MacroEvent("CPIAUCSL", days[10], days[10], 1, 0, +1.0),  # above
    ]
    price, bars = 100.0, []
    for i, d in enumerate(days):
        if i in (2, 10):
            price *= 0.98
        elif i == 6:
            price *= 1.01
        bars.append(Bar(d, price, price))
    return days, events, BarSeries("SPY", bars)


def test_study_conditional_stats():
    days, events, spy = _synthetic()
    above = study(events, {"SPY": spy}, "above", horizons=(1,))
    r = above.for_symbol("SPY", 1)
    assert above.n_events == 2 and r.n == 2
    assert r.mean == pytest.approx(-0.02) and r.hit_rate == 0.0
    below = study(events, {"SPY": spy}, "below", horizons=(1,))
    assert below.for_symbol("SPY", 1).mean == pytest.approx(0.01)


def test_study_is_point_in_time():
    days, events, spy = _synthetic()
    # as_of day 8: the day-10 event hasn't happened; only the day-2 'above' event counts
    early = study(events, {"SPY": spy}, "above", horizons=(1, 5), as_of=days[8])
    assert early.n_events == 1
    # as_of day 3: day-2 event's 1-day window closed (day 2), 5-day window (ends day 6) has not
    edge = study(events, {"SPY": spy}, "above", horizons=(1, 5), as_of=days[3])
    assert edge.for_symbol("SPY", 1).n == 1 and edge.for_symbol("SPY", 5) is None


def test_playbook_tool_end_to_end(tmp_path):
    days, events, spy = _synthetic()
    store = BarStore(tmp_path / "bars")
    store.put("SPY", spy.bars)
    vint = tmp_path / "vint"
    vint.mkdir()
    # vintages that reproduce the synthetic events: 4 first prints -> 2 'pct' events... simpler:
    # write raw observations so that first_prints/events_from_prints('level') yield our events.
    obs = [
        {"date": days[0].isoformat(), "realtime_start": days[0].isoformat(), "value": "1"},
        {"date": days[1].isoformat(), "realtime_start": days[1].isoformat(), "value": "1"},
        {"date": days[2].isoformat(), "realtime_start": days[2].isoformat(), "value": "2"},  # above
        {"date": days[6].isoformat(), "realtime_start": days[6].isoformat(), "value": "1"},  # below
        {"date": days[10].isoformat(), "realtime_start": days[10].isoformat(), "value": "2"},  # above
    ]
    (vint / "UNRATE.json").write_text(json.dumps({"observations": obs}))

    pb = MacroPlaybook(store, vintages_dir=vint, symbols=("SPY",), horizons=(1,))
    assert pb.available_series() == ["UNRATE"]
    entry = pb.lookup("unrate", "above", days[12])  # 3 events < 6: big_only filter is a no-op
    assert entry.n_events == 2
    payload = json.loads(entry.to_json())
    assert payload["reactions"][0]["mean_pct"] == pytest.approx(-2.0)
    assert payload["caution"]  # small sample is flagged

    tools = DataTools(clock=lambda: days[8], playbook=pb)
    out = json.loads(tools.dispatch("get_event_playbook", {"series": "UNRATE", "direction": "above"}))
    assert out["n_events"] == 1  # point-in-time through the tool too
    assert out["window"].startswith("last 3y")
    assert "no playbook" in tools.dispatch("get_event_playbook", {"series": "NOPE", "direction": "all"})


def test_playbook_trailing_window_and_big_only():
    days = _weekdays(D(2020, 1, 6), 3000)
    bars = BarSeries("SPY", [Bar(d, 100.0, 100.0) for d in days])
    # 12 events: 6 old (2020) with tiny surprises, 6 recent (2025) with large surprises
    # (start at day 1: an event on the very first bar has no prior close to measure from)
    events = [MacroEvent("X", d, d, 1, 0, 0.1) for d in days[1:7]]
    events += [MacroEvent("X", d, d, 1, 0, 5.0) for d in days[1300:1306]]

    class PB(MacroPlaybook):
        def events(self, series):
            return events

    pb = PB(BarStore("/nonexistent"), symbols=("SPY",), horizons=(1,))
    pb._bars["SPY"] = bars
    as_of = days[1400]
    assert pb.lookup("X", "all", as_of, lookback_years=1).n_events == 6  # old ones excluded
    assert pb.lookup("X", "all", as_of, lookback_years=15, big_only=False).n_events == 12
    assert pb.lookup("X", "all", as_of, lookback_years=15, big_only=True).n_events == 6
