"""Taxonomy → extractors → labels → generated scenarios, on synthetic stores."""

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from jarvis_jr.research.bars import Bar, BarStore
from jarvis_jr.research.extractors import ResearchData, extract
from jarvis_jr.research.labels import label_events
from jarvis_jr.research.scenario_gen import generate
from jarvis_jr.research.taxonomy import DEFAULT_TAXONOMY, EventInstance, EventType, load_taxonomy
from jarvis_jr.trading.scenario import load_scenario

D = date


def _weekdays(start, n):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _monthly_vintages(series, values, start=D(2020, 1, 1)):
    """Monthly first prints released on the 12th of the following month."""
    obs = []
    y, m = start.year, start.month
    for v in values:
        rel_y, rel_m = (y, m + 1) if m < 12 else (y + 1, 1)
        obs.append({"date": f"{y}-{m:02d}-01", "realtime_start": f"{rel_y}-{rel_m:02d}-12", "value": str(v)})
        y, m = rel_y, rel_m
    return {"observations": obs}


@pytest.fixture
def stores(tmp_path):
    vint = tmp_path / "vint"
    vint.mkdir()
    # CPI index rising 0.3%/mo, with one +1.0% jump at index 9 (a hot print)
    vals, v = [], 100.0
    for i in range(14):
        v *= 1.01 if i == 9 else 1.003
        vals.append(round(v, 3))
    (vint / "CPIAUCSL.json").write_text(json.dumps(_monthly_vintages("CPIAUCSL", vals)))
    # policy rate: daily-ish steps
    steps = [
        {"date": "2022-03-15", "realtime_start": "2022-03-15", "value": "0.25"},
        {"date": "2022-03-17", "realtime_start": "2022-03-17", "value": "0.50"},  # +25 (decision 3/16)
        {"date": "2022-06-16", "realtime_start": "2022-06-16", "value": "1.75"},  # +125? no: 0.50->1.75
    ]
    (vint / "DFEDTARU.json").write_text(json.dumps({"observations": steps}))
    # VIX: crosses 30 once
    vix = [{"date": d.isoformat(), "realtime_start": d.isoformat(), "value": str(v)}
           for d, v in zip(_weekdays(D(2022, 1, 3), 6), [20, 25, 31, 35, 29, 33])]
    (vint / "VIXCLS.json").write_text(json.dumps({"observations": vix}))
    # bars: 2019-2022, gently rising
    bars = BarStore(tmp_path / "bars")
    days = _weekdays(D(2019, 1, 2), 1000)
    for sym in ("SPY", "QQQ", "XLP", "TLT"):
        bars.put(sym, [Bar(d, 100 + i * 0.1, 100 + i * 0.1) for i, d in enumerate(days)])
    return ResearchData(vintages_dir=vint, bars=bars)


def test_shipped_taxonomy_loads_and_validates():
    tax = load_taxonomy(DEFAULT_TAXONOMY)
    assert {"cpi_release", "fomc_decision", "vix_spike", "annual_filing"} <= set(tax.types)
    assert tax["cpi_release"].source["kind"] == "fred_vintage"
    assert tax.expectations_for("above/large")["max_end_exposure"] == 0.6
    assert "max_end_exposure" not in tax.expectations_for("below/small")
    with pytest.raises(KeyError):
        tax["nope"]


def test_fred_vintage_extraction_and_trailing3_expectation(stores):
    tax = load_taxonomy(DEFAULT_TAXONOMY)
    inst = extract(tax["cpi_release"], stores)
    assert len(inst) == 14 - 4  # trailing3 needs 4 prior prints
    hot = [e for e in inst if e.context["obs_day"] == "2020-10-01"][0]
    assert hot.surprise > 0 and hot.context["pct_change"] == pytest.approx(0.01, abs=1e-4)
    assert hot.day == D(2020, 11, 12)  # release day, not observation day


def test_fomc_step_and_vix_threshold(stores):
    tax = load_taxonomy(DEFAULT_TAXONOMY)
    fomc = extract(tax["fomc_decision"], stores)
    assert [(e.day, e.context["bps"]) for e in fomc] == [(D(2022, 3, 16), 25.0), (D(2022, 6, 15), 125.0)]
    vix = extract(tax["vix_spike"], stores)
    # two crossings 5 days apart are one spell under the 15-day cooldown
    assert [(e.day, e.value) for e in vix] == [(D(2022, 1, 5), 31.0)]


def test_labels_are_point_in_time_and_rules_apply(stores):
    tax = load_taxonomy(DEFAULT_TAXONOMY)
    cpi = label_events(extract(tax["cpi_release"], stores), tax["cpi_release"])
    hot = [e for e in cpi if e.instance.context["obs_day"] == "2020-10-01"][0]
    assert hot.direction == "above" and hot.magnitude == "unrated"  # < 6 prior instances
    assert cpi[-1].magnitude in ("small", "medium", "large")  # enough history by the end

    fomc = label_events(extract(tax["fomc_decision"], stores), tax["fomc_decision"])
    assert [e.label for e in fomc] == ["hike/25bp", "hike/75bp+"]
    vix = label_events(extract(tax["vix_spike"], stores), tax["vix_spike"])
    assert vix[0].label == "risk_off/30-40"


def test_unknown_rule_or_kind_rejected(stores):
    bad = EventType("x", "c", {"kind": "telepathy"}, {"direction": {"rule": "surprise_sign"}}, {}, {})
    with pytest.raises(ValueError, match="unknown source kind"):
        extract(bad, stores)
    bad2 = EventType("x", "c", {"kind": "threshold"}, {"direction": {"rule": "vibes"}}, {}, {})
    with pytest.raises(ValueError, match="unknown label rule"):
        label_events([EventInstance("x", D(2022, 1, 1), 1.0)], bad2)


def test_generated_scenario_roundtrips_through_the_runner_loader(stores, tmp_path):
    tax = load_taxonomy(DEFAULT_TAXONOMY)
    et = tax["cpi_release"]
    events = label_events(extract(et, stores), et)
    hot = [e for e in events if e.direction == "above"][:1]
    res = generate(et, hot, tax, stores, tmp_path / "gen")
    assert len(res.written) == 1 and not res.skipped
    scen = load_scenario(res.written[0])
    assert scen.symbols == ["SPY", "QQQ", "XLP", "TLT"]
    assert len(scen.days) == 2 + 5 + 1
    # the event sits at index `before_days` and its print is visible from that day
    assert scen.days[2] == hot[0].instance.day
    assert (Path(res.written[0]).parent / "macro.csv").is_file()
    news = json.loads((Path(res.written[0]).parent / "news.jsonl").read_text())
    assert news["date"] == hot[0].instance.day.isoformat() and "CPI" in news["title"]
    assert scen.expect.max_turn_limit_days == 1
