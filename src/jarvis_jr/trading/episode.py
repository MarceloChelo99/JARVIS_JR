"""The daily episode loop: prompt → one Agent.run → advance the clock → repeat.

The runner owns time. The agent gets one turn per trading day with fresh
context (today's quotes, the portfolio, a short history of its own recent
actions) and may place orders through the trading tools. Metrics are
computed from the equity curve after the last day.

# GROWTH: the planner/executor split goes here — an "analyze" call with a
# smart model producing a plan, then the executor turn with tools. Add it
# only once a single-model daily strategy has a baseline in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from jarvis_jr.evals.agent import Agent
from jarvis_jr.evals.domain import Trajectory
from jarvis_jr.trading.domain import Broker, Limits
from jarvis_jr.trading.tools import TradingTools

DAILY_PROMPT = """Today is {day}. You manage a long-only stock portfolio and get ONE decision
turn per trading day. Research first with whatever tools you have (price history, macro
data, news, SEC filing signals and summaries — all as known today), then place any
orders you want for today (market orders fill now; limit orders wait for the price).
Doing nothing is a valid decision.

Budget: you have {max_turns} turns today. Batch your research calls into the first 2-3
turns (one turn may contain many tool calls), then place orders, then STOP: reply
WITHOUT tool calls with a one-paragraph summary of what you learned, what you did, and
why. Never re-request data you already have.

Tradeable symbols: {symbols}
Guardrails: max {max_pos:.0%} of equity per symbol, max {max_notional:,.0f} per order, no shorting.

Portfolio right now: cash {cash:,.2f}, equity {equity:,.2f}, positions: {positions}
Today's prices: {quotes}
Your recent days:
{history}"""


@dataclass
class DayRecord:
    day: date
    equity_before: float
    equity_after: float
    trajectory: Trajectory
    orders_placed: int
    rejections: int

    @property
    def hit_turn_limit(self) -> bool:
        return self.trajectory.hit_turn_limit


@dataclass
class EpisodeReport:
    days: list[DayRecord] = field(default_factory=list)
    start_equity: float = 0.0
    end_equity: float = 0.0  # marked to market on the day AFTER the last decision
    end_exposure: float = 0.0  # value of positions / equity at the end (0 = all cash)

    @property
    def pnl(self) -> float:
        return self.end_equity - self.start_equity

    @property
    def return_pct(self) -> float:
        return self.pnl / self.start_equity if self.start_equity else 0.0

    @property
    def max_drawdown(self) -> float:
        """Largest peak-to-trough fall in equity, as a fraction of the peak."""
        peak, worst = self.start_equity, 0.0
        curve = [d.equity_after for d in self.days] + [self.end_equity]
        for equity in curve:
            peak = max(peak, equity)
            if peak > 0:
                worst = max(worst, (peak - equity) / peak)
        return worst

    @property
    def orders_placed(self) -> int:
        return sum(d.orders_placed for d in self.days)

    @property
    def rejections(self) -> int:
        return sum(d.rejections for d in self.days)

    @property
    def turn_limit_days(self) -> int:
        """Days where the agent ran out of turns before giving a final answer —
        its 'decisions' on those days may just be exhaustion."""
        return sum(1 for d in self.days if d.hit_turn_limit)

    def summary(self) -> dict:
        return {
            "days": len(self.days),
            "start_equity": round(self.start_equity, 2),
            "end_equity": round(self.end_equity, 2),
            "pnl": round(self.pnl, 2),
            "return_pct": round(self.return_pct, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "end_exposure": round(self.end_exposure, 4),
            "orders_placed": self.orders_placed,
            "rejections": self.rejections,
            "turn_limit_days": self.turn_limit_days,
        }


def daily_prompt(
    broker: Broker, limits: Limits, symbols: list[str], history: list[str], max_turns: int = 8
) -> str:
    p = broker.portfolio()
    quotes = ", ".join(f"{s} {broker.quote(s).price:.2f}" for s in symbols)
    positions = (
        ", ".join(f"{pos.symbol} x{pos.qty:g} @ {pos.avg_cost:.2f}" for pos in p.positions)
        or "none"
    )
    return DAILY_PROMPT.format(
        max_turns=max_turns,
        day=p.day.isoformat(),
        symbols=", ".join(symbols),
        max_pos=limits.max_position_pct,
        max_notional=limits.max_order_notional,
        cash=p.cash,
        equity=p.equity,
        positions=positions,
        quotes=quotes,
        history="\n".join(history[-5:]) or "(first day)",
    )


def run_episode(
    agent: Agent,
    broker: Broker,
    tools: TradingTools,
    symbols: list[str],
    workspace: Path,
    max_days: int | None = None,
    max_turns: int = 8,
    on_day=None,
) -> EpisodeReport:
    report = EpisodeReport(start_equity=broker.portfolio().equity)
    history: list[str] = []
    day_no = 0
    while max_days is None or day_no < max_days:
        day_no += 1
        before = broker.portfolio().equity
        placed_before, rejected_before = len(tools.placed), len(tools.rejections)

        prompt = daily_prompt(broker, tools.limits, symbols, history, max_turns)
        trajectory = agent.run(prompt, workspace, max_turns)

        placed = len(tools.placed) - placed_before
        rejected = len(tools.rejections) - rejected_before
        after = broker.portfolio().equity
        record = DayRecord(broker.portfolio().day, before, after, trajectory, placed, rejected)
        report.days.append(record)
        history.append(
            f"- {record.day}: equity {after:,.0f}, {placed} order(s), {rejected} rejected — "
            f"{trajectory.final_answer[:160]}"
        )
        if on_day:
            on_day(record)
        if broker.advance() is None:
            break
    final = broker.portfolio()
    report.end_equity = final.equity
    report.end_exposure = (final.equity - final.cash) / final.equity if final.equity else 0.0
    return report
