"""InMemoryBroker: the v1 exchange. Replays a daily price series.

Market orders fill immediately at today's price (plus slippage/commission).
Limit orders sit pending and fill on a later day whose price crosses the
limit. Physical constraints (cash, holdings) are enforced here because they
are properties of the market; policy limits live in guardrails.py.

This is also the reference implementation for the Broker contract tests in
tests/trading/broker_contract.py — a future simulator passes those and
plugs in unchanged.
"""

from __future__ import annotations

from datetime import date, timedelta

from jarvis_jr.trading.domain import Order, OrderRequest, Portfolio, Position, Quote


def trading_days(start: date, n: int) -> list[date]:
    """n consecutive weekdays from `start` (inclusive)."""
    days: list[date] = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


class InMemoryBroker:
    def __init__(
        self,
        prices: dict[str, list[float]],
        days: list[date],
        cash: float,
        slippage_bps: float = 0.0,
        commission: float = 0.0,
    ):
        lengths = {len(series) for series in prices.values()}
        if not prices or lengths != {len(days)}:
            raise ValueError("every price series must have exactly one price per day")
        self._prices = prices
        self._days = days
        self._i = 0
        self._cash = float(cash)
        self._slip = slippage_bps / 10_000.0
        self._commission = commission
        self._positions: dict[str, tuple[float, float]] = {}  # symbol -> (qty, avg_cost)
        self._orders: dict[str, Order] = {}
        self._seq = 0

    # ---- Broker protocol --------------------------------------------------

    @property
    def day(self) -> date:
        return self._days[self._i]

    def quote(self, symbol: str) -> Quote:
        if symbol not in self._prices:
            raise ValueError(f"unknown symbol {symbol!r}")
        return Quote(symbol, self._prices[symbol][self._i], self.day)

    def history(self, symbol: str, n: int) -> list[Quote]:
        if symbol not in self._prices:
            raise ValueError(f"unknown symbol {symbol!r}")
        start = max(0, self._i + 1 - n)
        series = self._prices[symbol]
        return [Quote(symbol, series[i], self._days[i]) for i in range(start, self._i + 1)]

    def portfolio(self) -> Portfolio:
        positions = tuple(
            Position(sym, qty, cost) for sym, (qty, cost) in self._positions.items() if qty > 0
        )
        equity = self._cash + sum(p.qty * self.quote(p.symbol).price for p in positions)
        return Portfolio(cash=self._cash, positions=positions, equity=equity, day=self.day)

    def place_order(self, req: OrderRequest) -> Order:
        self._seq += 1
        order_id = f"o{self._seq}"
        if req.symbol not in self._prices:
            return self._record(Order(order_id, req, "rejected", reason=f"unknown symbol {req.symbol}"))
        if req.qty <= 0:
            return self._record(Order(order_id, req, "rejected", reason="qty must be positive"))
        price = self.quote(req.symbol).price
        if req.kind == "market":
            return self._record(self._fill(order_id, req, price))
        if req.limit_price is None or req.limit_price <= 0:
            return self._record(Order(order_id, req, "rejected", reason="limit_price required"))
        if self._crosses(req, price):
            return self._record(self._fill(order_id, req, price))
        return self._record(Order(order_id, req, "pending", day=self.day))

    def cancel_order(self, order_id: str) -> Order:
        order = self._orders.get(order_id)
        if order is None:
            return Order(order_id, OrderRequest("?", "buy", 0), "rejected", reason="no such order")
        if order.status != "pending":
            return order
        cancelled = Order(order.id, order.request, "cancelled", day=self.day, reason="cancelled")
        return self._record(cancelled)

    def open_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if o.status == "pending"]

    def advance(self) -> date | None:
        if self._i + 1 >= len(self._days):
            return None
        self._i += 1
        for order in self.open_orders():
            price = self.quote(order.request.symbol).price
            if self._crosses(order.request, price):
                self._record(self._fill(order.id, order.request, price))
        return self.day

    # ---- internals ----------------------------------------------------------

    @staticmethod
    def _crosses(req: OrderRequest, price: float) -> bool:
        assert req.limit_price is not None
        return price <= req.limit_price if req.side == "buy" else price >= req.limit_price

    def _fill(self, order_id: str, req: OrderRequest, price: float) -> Order:
        fill = price * (1 + self._slip) if req.side == "buy" else price * (1 - self._slip)
        qty, cost = self._positions.get(req.symbol, (0.0, 0.0))
        if req.side == "buy":
            total = req.qty * fill + self._commission
            if total > self._cash + 1e-9:
                return Order(order_id, req, "rejected", reason="insufficient cash")
            self._cash -= total
            new_qty = qty + req.qty
            new_cost = (qty * cost + req.qty * fill) / new_qty
            self._positions[req.symbol] = (new_qty, new_cost)
        else:
            if req.qty > qty + 1e-9:
                return Order(order_id, req, "rejected", reason=f"only {qty} {req.symbol} held")
            self._cash += req.qty * fill - self._commission
            self._positions[req.symbol] = (qty - req.qty, cost)
        return Order(order_id, req, "filled", fill_price=fill, day=self.day)

    def _record(self, order: Order) -> Order:
        self._orders[order.id] = order
        return order
