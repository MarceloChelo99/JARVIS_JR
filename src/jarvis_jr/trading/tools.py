"""Agent-facing trading tools over any Broker, with guardrails in front.

Same duck-typed shape as ToolRegistry / CoderTools (`schemas`, `dispatch`,
`describe`), so the assistant's LLM clients and the eval shim
(`jarvis_jr.evals.tools.registry_tools`) both consume it unchanged.

place_order is the only write; every rejection — by guardrail or by the
broker — is recorded so an episode can report them.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from jarvis_jr.trading import guardrails
from jarvis_jr.trading.domain import Broker, Limits, Order, OrderRequest

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_quote",
        "description": "Current price for a symbol on the current trading day.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_portfolio",
        "description": "Cash, positions (qty, avg cost, current value) and total equity.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "place_order",
        "description": (
            "Place a buy or sell order. Market orders fill now; limit orders fill on a "
            "later day if the price crosses limit_price. Rejected orders return an ERROR "
            "with the reason (guardrails: max position size, max order size, allowed "
            "symbols, long-only)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "side": {"type": "string", "enum": ["buy", "sell"]},
                "qty": {"type": "number", "description": "Number of shares, > 0"},
                "kind": {"type": "string", "enum": ["market", "limit"], "default": "market"},
                "limit_price": {"type": "number", "description": "Required for limit orders"},
            },
            "required": ["symbol", "side", "qty"],
        },
    },
    {
        "name": "cancel_order",
        "description": "Cancel a pending limit order by id.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "list_orders",
        "description": "List pending (unfilled) limit orders.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def _order_json(order: Order) -> str:
    return json.dumps(
        {
            "id": order.id,
            "status": order.status,
            "symbol": order.request.symbol,
            "side": order.request.side,
            "qty": order.request.qty,
            "kind": order.request.kind,
            "limit_price": order.request.limit_price,
            "fill_price": order.fill_price,
            "reason": order.reason,
        }
    )


class TradingTools:
    def __init__(self, broker: Broker, limits: Limits):
        self.broker = broker
        self.limits = limits
        self.rejections: list[Order] = []  # every rejected order this session, for reports
        self.placed: list[Order] = []

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return list(TOOL_SCHEMAS)

    def describe(self, name: str, args: dict[str, Any]) -> str:
        if name == "place_order":
            return (
                f"Place a {args.get('kind', 'market')} {args.get('side')} order for "
                f"{args.get('qty')} {args.get('symbol')}."
            )
        return f"Run {name} with {args}."

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        handler = {
            "get_quote": self._quote,
            "get_portfolio": self._portfolio,
            "place_order": self._place_order,
            "cancel_order": self._cancel,
            "list_orders": self._list_orders,
        }.get(name)
        if handler is None:
            return f"ERROR: unknown tool '{name}'."
        try:
            return handler(args)
        except Exception as e:  # surface to the model, never crash the loop
            return f"ERROR: {type(e).__name__}: {e}"

    # ---- handlers -----------------------------------------------------------

    def _quote(self, args: dict[str, Any]) -> str:
        q = self.broker.quote(str(args["symbol"]).upper())
        return json.dumps({"symbol": q.symbol, "price": q.price, "day": q.day.isoformat()})

    def _portfolio(self, args: dict[str, Any]) -> str:
        p = self.broker.portfolio()
        return json.dumps(
            {
                "day": p.day.isoformat(),
                "cash": round(p.cash, 2),
                "equity": round(p.equity, 2),
                "positions": [
                    {
                        **asdict(pos),
                        "value": round(pos.qty * self.broker.quote(pos.symbol).price, 2),
                    }
                    for pos in p.positions
                ],
            }
        )

    def _place_order(self, args: dict[str, Any]) -> str:
        req = OrderRequest(
            symbol=str(args["symbol"]).upper(),
            side=args["side"],
            qty=float(args["qty"]),
            kind=args.get("kind", "market"),
            limit_price=float(args["limit_price"]) if args.get("limit_price") is not None else None,
        )
        price = self.broker.quote(req.symbol).price
        reason = guardrails.check(req, self.broker.portfolio(), price, self.limits)
        if reason:
            self.rejections.append(Order("guardrail", req, "rejected", reason=reason))
            return f"ERROR: order rejected by guardrail: {reason}"
        order = self.broker.place_order(req)
        self.placed.append(order)
        if order.status == "rejected":
            self.rejections.append(order)
            return f"ERROR: order rejected by broker: {order.reason}"
        return _order_json(order)

    def _cancel(self, args: dict[str, Any]) -> str:
        return _order_json(self.broker.cancel_order(str(args["order_id"])))

    def _list_orders(self, args: dict[str, Any]) -> str:
        orders = self.broker.open_orders()
        return json.dumps([json.loads(_order_json(o)) for o in orders]) if orders else "[]"
