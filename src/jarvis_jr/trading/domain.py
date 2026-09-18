"""Trading vocabulary: the shapes that cross the broker boundary.

Imports nothing from the package. Every broker (in-memory simulator today,
your exchange simulator next, a paper/live broker eventually) speaks exactly
these types, so the agent tools and the episode runner never change.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

Side = Literal["buy", "sell"]
Kind = Literal["market", "limit"]
Status = Literal["filled", "pending", "rejected", "cancelled"]


@dataclass(frozen=True)
class Quote:
    symbol: str
    price: float
    day: date


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    avg_cost: float


@dataclass(frozen=True)
class Portfolio:
    cash: float
    positions: tuple[Position, ...]
    equity: float  # cash + sum(qty * current price)
    day: date

    def position(self, symbol: str) -> Position | None:
        return next((p for p in self.positions if p.symbol == symbol), None)

    def held(self, symbol: str) -> float:
        pos = self.position(symbol)
        return pos.qty if pos else 0.0


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: Side
    qty: float
    kind: Kind = "market"
    limit_price: float | None = None


@dataclass(frozen=True)
class Order:
    id: str
    request: OrderRequest
    status: Status
    fill_price: float | None = None
    day: date | None = None
    reason: str = ""  # why rejected / cancelled


@dataclass(frozen=True)
class Limits:
    """Guardrails enforced by the tool layer before an order reaches any broker."""

    max_position_pct: float = 0.25  # one symbol's value / equity, after the order
    max_order_notional: float = 10_000.0
    allowed_symbols: frozenset[str] = frozenset()  # empty = any symbol
    long_only: bool = True


class Broker(Protocol):
    """What every market implementation must provide.

    `advance` is for the runner only — it is never exposed as an agent tool.
    The clock belongs to the harness, not the model.
    """

    def quote(self, symbol: str) -> Quote: ...
    def history(self, symbol: str, n: int) -> list[Quote]: ...  # last n days up to today, oldest first
    def portfolio(self) -> Portfolio: ...
    def place_order(self, req: OrderRequest) -> Order: ...
    def cancel_order(self, order_id: str) -> Order: ...
    def open_orders(self) -> list[Order]: ...
    def advance(self) -> date | None: ...  # next trading day, or None when the scenario ends
