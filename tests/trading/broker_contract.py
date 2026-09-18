"""The Broker contract: behaviors every implementation must exhibit.

To certify a new broker (your exchange simulator, a paper-trading adapter),
subclass BrokerContract in a test module and implement `make_broker`. The
in-memory broker is certified in test_broker_memory.py; do the same for yours.

Conventions the contract pins down:
- prices/days are given per symbol, one price per day; day 0 is "today".
- market orders fill immediately; limit orders fill on advance() when the
  price crosses; advance() returns None at the end of the series.
- physical constraints (cash, holdings) reject with status "rejected" and a
  reason — they never raise.
"""

from __future__ import annotations

from datetime import date

from jarvis_jr.trading.domain import Broker, OrderRequest

PRICES = {"AAA": [100.0, 90.0, 110.0], "BBB": [50.0, 50.0, 50.0]}
DAYS = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
CASH = 10_000.0


class BrokerContract:
    def make_broker(self, prices=PRICES, days=DAYS, cash=CASH) -> Broker:
        raise NotImplementedError

    # ---- quotes & portfolio ---------------------------------------------

    def test_quote_reflects_current_day(self):
        b = self.make_broker()
        assert b.quote("AAA").price == 100.0 and b.quote("AAA").day == DAYS[0]
        assert b.advance() == DAYS[1]
        assert b.quote("AAA").price == 90.0

    def test_history_never_includes_the_future(self):
        b = self.make_broker()
        assert [q.price for q in b.history("AAA", 5)] == [100.0]  # day 0: only today
        b.advance()
        hist = b.history("AAA", 5)
        assert [q.price for q in hist] == [100.0, 90.0]
        assert hist[-1].day == DAYS[1]
        assert [q.price for q in b.history("AAA", 1)] == [90.0]

    def test_empty_portfolio_equity_is_cash(self):
        p = self.make_broker().portfolio()
        assert p.cash == CASH and p.equity == CASH and p.positions == ()

    # ---- market orders ----------------------------------------------------

    def test_market_buy_then_sell_roundtrip(self):
        b = self.make_broker()
        buy = b.place_order(OrderRequest("AAA", "buy", 10))
        assert buy.status == "filled" and buy.fill_price is not None
        p = b.portfolio()
        assert p.held("AAA") == 10 and p.cash < CASH
        assert abs(p.equity - CASH) < 5  # only slippage/commission moved equity

        sell = b.place_order(OrderRequest("AAA", "sell", 10))
        assert sell.status == "filled"
        assert b.portfolio().held("AAA") == 0

    def test_equity_tracks_price_moves(self):
        b = self.make_broker()
        b.place_order(OrderRequest("AAA", "buy", 10))
        b.advance()  # AAA 100 -> 90
        assert b.portfolio().equity < CASH - 90  # lost ~100 on 10 shares

    def test_insufficient_cash_is_rejected_not_raised(self):
        b = self.make_broker(cash=500.0)
        o = b.place_order(OrderRequest("AAA", "buy", 10))
        assert o.status == "rejected" and o.reason
        assert b.portfolio().cash == 500.0

    def test_selling_more_than_held_is_rejected(self):
        b = self.make_broker()
        o = b.place_order(OrderRequest("AAA", "sell", 1))
        assert o.status == "rejected"

    def test_unknown_symbol_is_rejected(self):
        o = self.make_broker().place_order(OrderRequest("ZZZ", "buy", 1))
        assert o.status == "rejected"

    # ---- limit orders -----------------------------------------------------

    def test_limit_buy_waits_then_fills_when_price_crosses(self):
        b = self.make_broker()
        o = b.place_order(OrderRequest("AAA", "buy", 10, kind="limit", limit_price=95.0))
        assert o.status == "pending"
        assert [x.id for x in b.open_orders()] == [o.id]
        b.advance()  # 90 <= 95 -> fills
        assert b.open_orders() == []
        assert b.portfolio().held("AAA") == 10

    def test_limit_that_never_crosses_stays_open(self):
        b = self.make_broker()
        b.place_order(OrderRequest("BBB", "buy", 1, kind="limit", limit_price=40.0))
        while b.advance() is not None:
            pass
        assert len(b.open_orders()) == 1

    def test_cancel_pending_order(self):
        b = self.make_broker()
        o = b.place_order(OrderRequest("AAA", "buy", 1, kind="limit", limit_price=1.0))
        assert b.cancel_order(o.id).status == "cancelled"
        assert b.open_orders() == []

    # ---- clock --------------------------------------------------------------

    def test_advance_returns_none_at_end(self):
        b = self.make_broker()
        assert b.advance() == DAYS[1]
        assert b.advance() == DAYS[2]
        assert b.advance() is None
        assert b.quote("AAA").day == DAYS[2]  # stays on the last day
