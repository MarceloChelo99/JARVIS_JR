"""Agent-driven trading against any Broker (in-memory simulator today).

domain      — DTOs + the Broker protocol (the contract a simulator must meet)
guardrails  — pure policy checks in front of every order
broker_memory — v1 exchange: daily price replay, market + limit fills
tools       — get_quote / get_portfolio / place_order / cancel_order / list_orders
episode     — the daily loop; the runner owns the clock
scenario    — YAML scenarios with portfolio-level expectations
"""

from jarvis_jr.trading.broker_memory import InMemoryBroker, trading_days
from jarvis_jr.trading.domain import (
    Broker,
    Limits,
    Order,
    OrderRequest,
    Portfolio,
    Position,
    Quote,
)
from jarvis_jr.trading.episode import EpisodeReport, run_episode
from jarvis_jr.trading.scenario import Scenario, grade, load_scenario
from jarvis_jr.trading.tools import TradingTools

__all__ = [
    "Broker",
    "EpisodeReport",
    "InMemoryBroker",
    "Limits",
    "Order",
    "OrderRequest",
    "Portfolio",
    "Position",
    "Quote",
    "Scenario",
    "TradingTools",
    "grade",
    "load_scenario",
    "run_episode",
    "trading_days",
]
