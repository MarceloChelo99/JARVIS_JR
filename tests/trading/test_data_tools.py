import json
from datetime import date

from jarvis_jr.trading.broker_memory import InMemoryBroker
from jarvis_jr.trading.data_tools import DataTools
from jarvis_jr.trading.feed import Headline, MacroObservation

DAYS = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]


class StubMacro:
    def observations(self, series, as_of, n):
        return [MacroObservation(series, as_of, 4.1, as_of)][:n]


class StubNews:
    def __init__(self):
        self.calls = []

    def headlines(self, query, as_of, n):
        self.calls.append((query, as_of, n))
        return [Headline(as_of, f"news for {query}")]


def test_only_configured_feeds_are_exposed():
    broker = InMemoryBroker({"A": [1.0, 2.0, 3.0]}, DAYS, 100)
    names = {s["name"] for s in DataTools(clock=lambda: DAYS[0], broker=broker).schemas}
    assert names == {"get_price_history"}
    names = {s["name"] for s in DataTools(clock=lambda: DAYS[0], news=StubNews()).schemas}
    assert names == {"get_news"}
    assert DataTools(clock=lambda: DAYS[0]).dispatch("get_news", {"query": "x"}).startswith("ERROR")


def test_price_history_and_window_change():
    broker = InMemoryBroker({"A": [100.0, 110.0, 121.0]}, DAYS, 100)
    tools = DataTools(clock=lambda: broker.portfolio().day, broker=broker)
    broker.advance()
    broker.advance()
    out = json.loads(tools.dispatch("get_price_history", {"symbol": "a", "n": 2}))
    assert out["closes"] == [["2026-01-06", 110.0], ["2026-01-07", 121.0]]
    assert out["window_change_pct"] == 10.0


def test_same_day_repeat_is_deduped_and_resets_when_clock_moves():
    news = StubNews()
    today = {"d": DAYS[0]}
    tools = DataTools(clock=lambda: today["d"], news=news)
    first = tools.dispatch("get_news", {"query": "AAPL"})
    again = tools.dispatch("get_news", {"query": "AAPL"})
    assert "news for AAPL" in first and again.startswith("ALREADY RETRIEVED TODAY")
    assert len(news.calls) == 1  # feed not hit twice
    tools.dispatch("get_news", {"query": "MSFT"})  # different args -> real call
    assert len(news.calls) == 2
    today["d"] = DAYS[1]
    assert "news for AAPL" in tools.dispatch("get_news", {"query": "AAPL"})
    assert len(news.calls) == 3


def test_clock_is_injected_not_taken_from_args():
    news = StubNews()
    today = {"d": DAYS[1]}
    tools = DataTools(clock=lambda: today["d"], news=news, macro=StubMacro())
    tools.dispatch("get_news", {"query": "AAPL", "n": 99})
    assert news.calls == [("AAPL", DAYS[1], 5)]  # n clamped, as_of from clock
    out = json.loads(tools.dispatch("get_macro", {"series": "UNRATE"}))
    assert out["as_of"] == "2026-01-06" and out["observations"] == [["2026-01-06", 4.1]]
