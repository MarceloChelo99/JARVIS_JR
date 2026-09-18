"""Scenario files: a price series, starting cash, limits, and pass/fail expectations.

One YAML per scenario in evals/scenarios/. This is the trading analogue of an
eval task: the fixture is the market, the checks are portfolio outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from jarvis_jr.trading.broker_memory import InMemoryBroker, trading_days
from jarvis_jr.trading.domain import Limits
from jarvis_jr.trading.episode import EpisodeReport


@dataclass(frozen=True)
class Expectations:
    min_end_equity: float | None = None
    max_drawdown: float | None = None
    max_rejections: int | None = None
    min_orders: int | None = None
    max_end_exposure: float | None = None  # e.g. 0.5 = at most half the equity in stock at the end
    max_turn_limit_days: int | None = None  # days allowed to end by exhaustion instead of decision


@dataclass(frozen=True)
class Scenario:
    id: str
    start_day: date
    cash: float
    prices: dict[str, list[float]]
    limits: Limits
    expect: Expectations
    slippage_bps: float = 0.0
    commission: float = 0.0
    max_turns: int = 8
    notes: str = ""
    # Research feeds: {"news": "<file>", "macro": "<file>" | "fred", "filings": "secfiler",
    #                  "playbook": "macro", "reactions": "taxonomy"}
    data: dict[str, str] = field(default_factory=dict)
    scenario_dir: Path = Path(".")
    event: dict = field(default_factory=dict)  # set by the generator: type, day, label, value...

    @property
    def symbols(self) -> list[str]:
        return list(self.prices)

    @property
    def days(self) -> list[date]:
        return trading_days(self.start_day, len(next(iter(self.prices.values()))))

    def broker(self) -> InMemoryBroker:
        return InMemoryBroker(
            self.prices, self.days, self.cash, self.slippage_bps, self.commission
        )


def load_scenario(path: Path) -> Scenario:
    data = yaml.safe_load(path.read_text())
    for key in ("id", "start_day", "cash", "prices"):
        if key not in data:
            raise ValueError(f"{path}: scenario needs `{key}`")
    lim = data.get("limits", {})
    limits = Limits(
        max_position_pct=float(lim.get("max_position_pct", 0.25)),
        max_order_notional=float(lim.get("max_order_notional", 10_000)),
        allowed_symbols=frozenset(lim.get("allowed_symbols", list(data["prices"]))),
        long_only=bool(lim.get("long_only", True)),
    )
    exp = data.get("expect", {})
    return Scenario(
        id=str(data["id"]),
        start_day=(
            data["start_day"]
            if isinstance(data["start_day"], date)
            else date.fromisoformat(str(data["start_day"]))
        ),
        cash=float(data["cash"]),
        prices={str(k): [float(x) for x in v] for k, v in data["prices"].items()},
        limits=limits,
        expect=Expectations(
            min_end_equity=exp.get("min_end_equity"),
            max_drawdown=exp.get("max_drawdown"),
            max_rejections=exp.get("max_rejections"),
            min_orders=exp.get("min_orders"),
            max_end_exposure=exp.get("max_end_exposure"),
            max_turn_limit_days=exp.get("max_turn_limit_days"),
        ),
        slippage_bps=float(data.get("slippage_bps", 0)),
        commission=float(data.get("commission", 0)),
        max_turns=int(data.get("max_turns", 8)),
        notes=str(data.get("notes", "")),
        data={str(k): str(v) for k, v in (data.get("data") or {}).items()},
        scenario_dir=path.resolve().parent,
        event={k: (v.isoformat() if isinstance(v, date) else v) for k, v in (data.get("event") or {}).items()},
    )


def build_feeds(scenario: Scenario, warn=print) -> dict:
    """Instantiate the feeds a scenario asks for. Missing env/config = warning, not crash,
    so a scenario still runs (with fewer research tools) on a machine without the keys."""
    feeds: dict = {}
    spec = scenario.data
    if spec.get("news"):
        if spec["news"].startswith("python:"):
            # Any NewsFeed implementation: "python:package.module:ClassName" (no-arg ctor).
            # This is how a real dated-news archive plugs in without touching this file.
            import importlib

            mod_name, _, cls_name = spec["news"][len("python:"):].partition(":")
            feeds["news"] = getattr(importlib.import_module(mod_name), cls_name)()
        else:
            from jarvis_jr.trading.feed import FixtureNews

            feeds["news"] = FixtureNews(scenario.scenario_dir / spec["news"])
    if spec.get("macro"):
        if spec["macro"].lower() == "fred":
            from jarvis_jr.trading.feed_fred import FredMacro

            try:
                feeds["macro"] = FredMacro()
            except ValueError as e:
                warn(f"[scenario] macro feed skipped: {e}")
        else:
            from jarvis_jr.trading.feed import FixtureMacro

            feeds["macro"] = FixtureMacro(scenario.scenario_dir / spec["macro"])
    if spec.get("filings"):
        from jarvis_jr.trading.feed_secfiler import SecFilerFilings

        try:
            feeds["filings"] = SecFilerFilings()
        except ValueError as e:
            warn(f"[scenario] filings feed skipped: {e}")
    if spec.get("playbook"):
        from jarvis_jr.research.bars import BarStore
        from jarvis_jr.research.playbook import MacroPlaybook

        pb = MacroPlaybook(BarStore())
        if pb.available_series() and pb.bars.tickers():
            feeds["playbook"] = pb
        else:
            warn("[scenario] playbook skipped: no data/bars or data/macro_vintages — "
                 "run scripts/research_fetch.py")
    if spec.get("reactions"):
        from jarvis_jr.research.reactions import ReactionGraph

        graph = ReactionGraph()
        if graph.data.bars.tickers() and graph.data.vintages_dir.is_dir():
            feeds["reactions"] = graph
            if spec.get("sizing"):
                from jarvis_jr.research.sizing import ExposurePolicy

                feeds["sizing"] = ExposurePolicy(graph)
        else:
            warn("[scenario] reactions/sizing skipped: no data/bars or data/macro_vintages")
    return feeds


@dataclass(frozen=True)
class Verdict:
    name: str
    passed: bool
    detail: str


def grade(report: EpisodeReport, expect: Expectations) -> list[Verdict]:
    """Deterministic portfolio checks. Empty expectations = nothing to fail."""
    out: list[Verdict] = []
    if expect.min_end_equity is not None:
        ok = report.end_equity >= expect.min_end_equity
        out.append(Verdict("min_end_equity", ok, f"{report.end_equity:,.2f} vs {expect.min_end_equity:,.2f}"))
    if expect.max_drawdown is not None:
        ok = report.max_drawdown <= expect.max_drawdown
        out.append(Verdict("max_drawdown", ok, f"{report.max_drawdown:.2%} vs {expect.max_drawdown:.2%}"))
    if expect.max_rejections is not None:
        ok = report.rejections <= expect.max_rejections
        out.append(Verdict("max_rejections", ok, f"{report.rejections} vs {expect.max_rejections}"))
    if expect.min_orders is not None:
        ok = report.orders_placed >= expect.min_orders
        out.append(Verdict("min_orders", ok, f"{report.orders_placed} vs {expect.min_orders}"))
    if expect.max_end_exposure is not None:
        ok = report.end_exposure <= expect.max_end_exposure
        out.append(Verdict("max_end_exposure", ok, f"{report.end_exposure:.0%} vs {expect.max_end_exposure:.0%}"))
    if expect.max_turn_limit_days is not None:
        ok = report.turn_limit_days <= expect.max_turn_limit_days
        out.append(Verdict("max_turn_limit_days", ok, f"{report.turn_limit_days} vs {expect.max_turn_limit_days}"))
    return out
