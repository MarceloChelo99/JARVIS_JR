from datetime import date

import pytest

from jarvis_jr.trading.broker_memory import InMemoryBroker, trading_days
from jarvis_jr.trading.domain import OrderRequest
from tests.trading.broker_contract import BrokerContract


class TestInMemoryBroker(BrokerContract):
    def make_broker(self, prices=None, days=None, cash=None):
        from tests.trading.broker_contract import CASH, DAYS, PRICES

        return InMemoryBroker(
            prices if prices is not None else PRICES,
            days if days is not None else DAYS,
            cash if cash is not None else CASH,
        )


def test_slippage_and_commission_apply():
    b = InMemoryBroker({"A": [100.0]}, [date(2026, 1, 5)], 10_000, slippage_bps=100, commission=2)
    o = b.place_order(OrderRequest("A", "buy", 10))
    assert o.fill_price == pytest.approx(101.0)
    assert b.portfolio().cash == pytest.approx(10_000 - 1010 - 2)


def test_mismatched_series_rejected():
    with pytest.raises(ValueError):
        InMemoryBroker({"A": [1.0, 2.0], "B": [1.0]}, [date(2026, 1, 5), date(2026, 1, 6)], 1)


def test_trading_days_skips_weekends():
    days = trading_days(date(2026, 1, 2), 3)  # Fri, (skip Sat/Sun), Mon, Tue
    assert days == [date(2026, 1, 2), date(2026, 1, 5), date(2026, 1, 6)]
