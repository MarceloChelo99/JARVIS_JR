"""Pure order guardrails: (request, portfolio, price, limits) -> rejection reason or None.

No broker, no I/O — so the rules are testable in isolation and identical
across simulator, paper, and live brokers. The tool layer calls this before
forwarding any order.
"""

from __future__ import annotations

from jarvis_jr.trading.domain import Limits, OrderRequest, Portfolio


def check(req: OrderRequest, portfolio: Portfolio, price: float, limits: Limits) -> str | None:
    """Return a human-readable reason to reject `req`, or None if it passes."""
    if req.qty <= 0:
        return "qty must be positive"
    if limits.allowed_symbols and req.symbol not in limits.allowed_symbols:
        return f"{req.symbol} is not in the allowed symbols {sorted(limits.allowed_symbols)}"
    if req.kind == "limit" and (req.limit_price is None or req.limit_price <= 0):
        return "limit orders need a positive limit_price"

    ref_price = req.limit_price if req.kind == "limit" and req.limit_price else price
    notional = req.qty * ref_price
    if notional > limits.max_order_notional:
        return f"order notional {notional:,.2f} exceeds max {limits.max_order_notional:,.2f}"

    held = portfolio.held(req.symbol)
    if req.side == "sell":
        if limits.long_only and req.qty > held + 1e-9:
            return f"long-only: cannot sell {req.qty} {req.symbol}, only {held} held"
        return None

    # buy
    if limits.long_only and notional > portfolio.cash + 1e-9:
        return f"insufficient cash: need {notional:,.2f}, have {portfolio.cash:,.2f}"
    if portfolio.equity > 0:
        value_after = (held * price) + notional
        if value_after > limits.max_position_pct * portfolio.equity + 1e-9:
            return (
                f"{req.symbol} would be {value_after / portfolio.equity:.0%} of equity, "
                f"max is {limits.max_position_pct:.0%}"
            )
    return None
