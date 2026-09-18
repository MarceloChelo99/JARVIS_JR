import json
from datetime import date

from jarvis_jr.evals.agent import Agent
from jarvis_jr.evals.domain import ModelReply, ToolCall
from jarvis_jr.evals.tools import registry_tools
from jarvis_jr.trading.broker_memory import InMemoryBroker
from jarvis_jr.trading.domain import Limits
from jarvis_jr.trading.episode import run_episode
from jarvis_jr.trading.scenario import Expectations, grade, load_scenario
from jarvis_jr.trading.tools import TradingTools
from tests.evals.fakes import FakeModel

DAYS = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
PRICES = {"AAA": [100.0, 110.0, 120.0]}


def _tools(cash=10_000.0, **limits):
    broker = InMemoryBroker(PRICES, DAYS, cash)
    return broker, TradingTools(broker, Limits(allowed_symbols=frozenset({"AAA"}), **limits))


def test_tools_dispatch_roundtrip():
    broker, tools = _tools()
    assert json.loads(tools.dispatch("get_quote", {"symbol": "aaa"}))["price"] == 100.0
    out = json.loads(tools.dispatch("place_order", {"symbol": "AAA", "side": "buy", "qty": 5}))
    assert out["status"] == "filled"
    p = json.loads(tools.dispatch("get_portfolio", {}))
    assert p["positions"][0]["qty"] == 5 and p["cash"] == 9_500.0
    assert tools.dispatch("list_orders", {}) == "[]"
    assert tools.dispatch("nope", {}).startswith("ERROR: unknown tool")


def test_guardrail_rejection_is_recorded_and_never_reaches_broker():
    broker, tools = _tools(max_order_notional=100.0)
    out = tools.dispatch("place_order", {"symbol": "AAA", "side": "buy", "qty": 5})
    assert out.startswith("ERROR: order rejected by guardrail")
    assert len(tools.rejections) == 1 and tools.placed == []
    assert broker.portfolio().cash == 10_000.0


def test_broker_rejection_is_also_recorded():
    # Guardrails see notional 100 <= cash 100 and pass; the broker adds a $5
    # commission and rejects. Both paths must land in tools.rejections.
    broker = InMemoryBroker(PRICES, DAYS, cash=100.0, commission=5.0)
    tools = TradingTools(broker, Limits(max_position_pct=1.0, allowed_symbols=frozenset({"AAA"})))
    out = tools.dispatch("place_order", {"symbol": "AAA", "side": "buy", "qty": 1})
    assert out.startswith("ERROR: order rejected by broker: insufficient cash")
    assert len(tools.rejections) == 1 and len(tools.placed) == 1
    assert broker.portfolio().cash == 100.0


def test_episode_runs_one_turn_per_day_and_owns_the_clock(tmp_path):
    broker, tools = _tools(max_position_pct=1.0, max_order_notional=100_000)
    model = FakeModel(
        [
            # day 1: buy 10 then finish
            ModelReply("", (ToolCall("place_order", {"symbol": "AAA", "side": "buy", "qty": 10}, "c1"),)),
            ModelReply("Bought 10 AAA."),
            # day 2: hold
            ModelReply("Holding."),
            # day 3: sell
            ModelReply("", (ToolCall("place_order", {"symbol": "AAA", "side": "sell", "qty": 10}, "c2"),)),
            ModelReply("Sold."),
        ]
    )
    agent = Agent(model, registry_tools(tools))
    report = run_episode(agent, broker, tools, ["AAA"], tmp_path)

    assert [d.day for d in report.days] == DAYS
    assert report.orders_placed == 2 and report.rejections == 0
    assert report.end_equity == 10_000 + 10 * 20  # bought at 100, sold at 120
    assert report.max_drawdown == 0.0
    assert report.end_exposure == 0.0  # sold out -> all cash
    assert "first day" in model.calls[0][1]["content"]
    assert "Bought 10 AAA" in model.calls[2][1]["content"]  # history flows into the next prompt


def test_grade_expectations():
    from jarvis_jr.trading.episode import EpisodeReport

    r = EpisodeReport(start_equity=100.0, end_equity=105.0, end_exposure=0.8)
    verdicts = grade(
        r, Expectations(min_end_equity=110.0, max_drawdown=0.1, max_rejections=0, max_end_exposure=0.5)
    )
    assert [v.passed for v in verdicts] == [False, True, True, False]
    assert grade(r, Expectations()) == []


def test_all_shipped_scenarios_load():
    from pathlib import Path

    for path in (Path(__file__).resolve().parents[2] / "evals" / "scenarios").glob("*.yaml"):
        s = load_scenario(path)
        assert s.broker().portfolio().equity == s.cash
        for key in ("news", "macro"):
            if s.data.get(key) and s.data[key].lower() != "fred":
                assert (s.scenario_dir / s.data[key]).is_file(), f"{path.name}: missing {key} fixture"


def test_shipped_scenario_loads_and_builds_broker():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "evals" / "scenarios" / "dip_and_recover.yaml"
    s = load_scenario(path)
    assert s.symbols == ["AAPL", "MSFT"] and len(s.days) == 10
    assert s.limits.allowed_symbols == frozenset({"AAPL", "MSFT"})
    assert s.broker().quote("AAPL").price == 200.0
