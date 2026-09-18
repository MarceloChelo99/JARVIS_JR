"""Feeds: fixture implementations, the FRED and SecFiler adapters, and the
point-in-time contract — nothing dated after as_of, ever."""

import json
from datetime import date
from pathlib import Path

import pytest

from jarvis_jr.trading.feed import FixtureMacro, FixtureNews
from jarvis_jr.trading.feed_fred import FredMacro
from jarvis_jr.trading.feed_secfiler import SecFilerFilings

D = date


# ---- fixtures ---------------------------------------------------------------


@pytest.fixture
def macro_csv(tmp_path):
    p = tmp_path / "macro.csv"
    p.write_text(
        "series,date,value,release_date\n"
        "CPIAUCSL,2026-01-01,310.1,2026-02-12\n"  # Jan CPI released Feb 12
        "CPIAUCSL,2025-12-01,309.4,2026-01-14\n"
        "UNRATE,2026-01-01,4.1,2026-02-06\n"
    )
    return p


@pytest.fixture
def news_jsonl(tmp_path):
    p = tmp_path / "news.jsonl"
    p.write_text(
        json.dumps({"date": "2026-01-05", "title": "AAPL cut", "symbols": ["AAPL"]}) + "\n"
        + json.dumps({"date": "2026-01-09", "title": "Apple rebounds", "symbols": ["AAPL"]}) + "\n"
        + json.dumps({"date": "2026-01-07", "title": "Fed patient", "symbols": []}) + "\n"
    )
    return p


@pytest.fixture
def secfiler_dir(tmp_path):
    """A two-filing SecFiler data dir: FY2024 filed 2024-11-01, FY2025 filed 2025-10-31."""
    root = tmp_path / "data"
    for acc, fy, filed, report in (
        ("0000320193-24-000123", 2024, "2024-11-01", "2024-09-28"),
        ("0000320193-25-000079", 2025, "2025-10-31", "2025-09-27"),
    ):
        raw = root / "raw" / "AAPL" / acc
        raw.mkdir(parents=True)
        (raw / "filing.json").write_text(
            json.dumps({"ticker": "AAPL", "form": "10-K", "filing_date": filed, "report_date": report})
        )
        summ = root / "summaries" / "AAPL" / f"FY{fy}"
        summ.mkdir(parents=True)
        (summ / "item_7.json").write_text(
            json.dumps({"form": "10-K", "title": "MD&A", "summary": f"MD&A for FY{fy}"})
        )
    analysis = root / "analysis"
    analysis.mkdir()
    (analysis / "features.csv").write_text(
        "ticker,company,period_label,filing_date,feature,value,note\n"
        "AAPL,Apple,FY2024,2024-11-01,drift_1A,0.02,vs FY2023\n"
        "AAPL,Apple,FY2025,2025-10-31,drift_1A,0.05,vs FY2024\n"
        "AAPL,Apple,FY2025,2025-10-31,risk_headings,30,\n"
    )
    (analysis / "risk_headings.csv").write_text(
        "ticker,company,fiscal_year,item,heading,period,cluster,label,theme\n"
        "AAPL,Apple,2024,1A,Old risk,0,1,supply,Supply chain\n"
        "AAPL,Apple,2025,1A,New risk,0,2,ai,AI regulation\n"
    )
    return root


# ---- FixtureMacro / FixtureNews ---------------------------------------------


def test_macro_respects_release_date_not_observation_date(macro_csv):
    feed = FixtureMacro(macro_csv)
    # On Feb 1 the January CPI (released Feb 12) is NOT known yet
    assert [o.value for o in feed.observations("CPIAUCSL", D(2026, 2, 1), 5)] == [309.4]
    assert [o.value for o in feed.observations("CPIAUCSL", D(2026, 2, 12), 5)] == [310.1, 309.4]
    assert feed.observations("UNRATE", D(2026, 1, 1), 5) == []


def test_news_filters_by_date_and_query(news_jsonl):
    feed = FixtureNews(news_jsonl)
    assert [h.title for h in feed.headlines("AAPL", D(2026, 1, 6), 5)] == ["AAPL cut"]
    assert [h.title for h in feed.headlines("AAPL", D(2026, 1, 31), 5)] == ["Apple rebounds", "AAPL cut"]
    assert [h.title for h in feed.headlines("", D(2026, 1, 8), 5)] == ["Fed patient", "AAPL cut"]
    assert [h.title for h in feed.headlines("fed", D(2026, 1, 31), 5)] == ["Fed patient"]


# ---- FredMacro (no network: injected fetch) ---------------------------------


def test_fred_sends_vintage_params_caches_and_drops_missing(tmp_path):
    calls = []

    def fake_fetch(params):
        calls.append(params)
        return {
            "observations": [
                {"date": "2026-01-01", "value": "310.1", "realtime_start": "2026-02-12"},
                {"date": "2025-12-01", "value": ".", "realtime_start": "2026-01-14"},
                {"date": "2025-11-01", "value": "308.9", "realtime_start": "2025-12-10"},
            ]
        }

    feed = FredMacro(api_key="k", cache_dir=tmp_path, fetch=fake_fetch)
    obs = feed.observations("cpiaucsl", D(2026, 2, 20), 5)
    assert [o.value for o in obs] == [310.1, 308.9]  # "." dropped
    p = calls[0]
    assert p["series_id"] == "CPIAUCSL"
    assert p["realtime_start"] == p["realtime_end"] == p["observation_end"] == "2026-02-20"

    feed.observations("CPIAUCSL", D(2026, 2, 20), 5)
    assert len(calls) == 1  # second call served from cache
    assert list(tmp_path.glob("CPIAUCSL_2026-02-20_5.json"))


def test_fred_requires_key(monkeypatch, tmp_path):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FRED_API_KEY"):
        FredMacro(cache_dir=tmp_path)


# ---- SecFilerFilings ----------------------------------------------------------


def test_secfiler_filters_on_filing_date_not_report_date(secfiler_dir):
    feed = SecFilerFilings(secfiler_dir)
    # FY2025 period ended 2025-09-27 but was filed 2025-10-31: not visible on Oct 1
    s = feed.summary("aapl", "7", D(2025, 10, 1))
    assert s is not None and s.period_label == "FY2024" and s.filed == D(2024, 11, 1)
    s2 = feed.summary("AAPL", "7", D(2025, 11, 1))
    assert s2 is not None and s2.period_label == "FY2025" and "FY2025" in s2.summary
    assert feed.summary("AAPL", "7", D(2024, 1, 1)) is None


def test_secfiler_signals_take_latest_visible_per_feature(secfiler_dir):
    feed = SecFilerFilings(secfiler_dir)
    early = {s.feature: s for s in feed.signals("AAPL", D(2025, 6, 1))}
    assert early["drift_1A"].value == 0.02 and "risk_headings" not in early
    late = {s.feature: s for s in feed.signals("AAPL", D(2026, 1, 1))}
    assert late["drift_1A"].value == 0.05 and late["risk_headings"].value == 30


def test_secfiler_risk_headings_pick_latest_filed_year(secfiler_dir):
    feed = SecFilerFilings(secfiler_dir)
    assert [r.heading for r in feed.risk_headings("AAPL", D(2025, 6, 1), 10)] == ["Old risk"]
    rows = feed.risk_headings("AAPL", D(2026, 1, 1), 10)
    assert [r.heading for r in rows] == ["New risk"] and rows[0].theme == "AI regulation"
    assert feed.risk_headings("AAPL", D(2020, 1, 1), 10) == []


def test_secfiler_missing_dir_is_clear(monkeypatch, tmp_path):
    monkeypatch.delenv("SECFILER_DATA_DIR", raising=False)
    with pytest.raises(ValueError, match="SecFiler data dir"):
        SecFilerFilings()
    with pytest.raises(ValueError):
        SecFilerFilings(tmp_path / "nope")


# ---- point-in-time contract, run over every feed built above -------------------


def test_point_in_time_contract(macro_csv, news_jsonl, secfiler_dir):
    macro, news, filings = FixtureMacro(macro_csv), FixtureNews(news_jsonl), SecFilerFilings(secfiler_dir)
    for as_of in (D(2024, 6, 1), D(2025, 10, 15), D(2026, 1, 8), D(2026, 3, 1)):
        assert all(o.released <= as_of and o.day <= as_of for o in macro.observations("CPIAUCSL", as_of, 10))
        assert all(h.day <= as_of for h in news.headlines("", as_of, 10))
        assert all(s.filed <= as_of for s in filings.signals("AAPL", as_of))
        summ = filings.summary("AAPL", "7", as_of)
        assert summ is None or summ.filed <= as_of


# ---- real SecFiler data, only if present on this machine -----------------------


@pytest.mark.skipif(
    not Path("~/PycharmProjects/SecFiler/data/analysis/features.csv").expanduser().is_file(),
    reason="real SecFiler data not present",
)
def test_real_secfiler_apple_signals_are_point_in_time():
    feed = SecFilerFilings(Path("~/PycharmProjects/SecFiler/data").expanduser())
    # Apple's FY2025 10-K was filed 2025-10-31: invisible the day before, visible the day after.
    before = feed.signals("AAPL", D(2025, 10, 30))
    after = {s.feature: s for s in feed.signals("AAPL", D(2025, 11, 1))}
    assert before and all(s.period_label <= "FY2024" for s in before)
    assert after["drift_1A"].period_label == "FY2025" and after["drift_1A"].filed == D(2025, 10, 31)
    s = feed.summary("AAPL", "7", D(2025, 11, 1))
    assert s is not None and s.period_label == "FY2025" and len(s.summary) > 100
