from datetime import date

from jarvis_jr.trading.domain import Limits, OrderRequest, Portfolio, Position
from jarvis_jr.trading.guardrails import check

DAY = date(2026, 1, 5)
LIMITS = Limits(max_position_pct=0.25, max_order_notional=5_000, allowed_symbols=frozenset({"AAA"}))
EMPTY = Portfolio(cash=10_000, positions=(), equity=10_000, day=DAY)


def test_clean_order_passes():
    assert check(OrderRequest("AAA", "buy", 10), EMPTY, 100.0, LIMITS) is None


def test_symbol_whitelist():
    assert "not in the allowed" in check(OrderRequest("BBB", "buy", 1), EMPTY, 1.0, LIMITS)


def test_order_notional_cap():
    assert "exceeds max" in check(OrderRequest("AAA", "buy", 60), EMPTY, 100.0, LIMITS)


def test_position_cap_counts_existing_holdings():
    held = Portfolio(cash=8_000, positions=(Position("AAA", 20, 100.0),), equity=10_000, day=DAY)
    # 20 held = 2000 = 20%; buying 10 more -> 30% > 25%
    assert "of equity" in check(OrderRequest("AAA", "buy", 10), held, 100.0, LIMITS)
    assert check(OrderRequest("AAA", "buy", 4), held, 100.0, LIMITS) is None


def test_long_only_blocks_overselling_and_overspending():
    assert "long-only" in check(OrderRequest("AAA", "sell", 1), EMPTY, 100.0, LIMITS)
    poor = Portfolio(cash=100, positions=(), equity=100, day=DAY)
    assert "insufficient cash" in check(OrderRequest("AAA", "buy", 2), poor, 100.0, LIMITS)


def test_limit_orders_use_limit_price_for_notional():
    # 40 shares at limit 100 = 4000 <= 5000 cap, even though 40 x market 200 = 8000 would not
    loose = Limits(max_position_pct=1.0, max_order_notional=5_000, allowed_symbols=frozenset({"AAA"}))
    assert check(OrderRequest("AAA", "buy", 40, "limit", 100.0), EMPTY, 200.0, loose) is None
    assert "exceeds max" in check(OrderRequest("AAA", "buy", 40), EMPTY, 200.0, loose)
    assert "limit_price" in check(OrderRequest("AAA", "buy", 1, "limit", None), EMPTY, 1.0, LIMITS)


def test_bad_qty():
    assert "positive" in check(OrderRequest("AAA", "buy", 0), EMPTY, 1.0, LIMITS)
