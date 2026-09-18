"""Research tools for the agent: price history, macro, news, filings.

Registry-shaped (`schemas` / `dispatch` / `describe`) like TradingTools. Only
the tools whose feed is configured appear in `schemas`, so the agent never
sees a research tool it cannot use. Every call reads `as_of` from the clock
the runner supplies — the model cannot ask about the future.

Outputs are deliberately compact: the local model runs on an 8K context and
research must leave room for reasoning and the trading turn.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from typing import Any

from jarvis_jr.trading.domain import Broker
from jarvis_jr.trading.feed import FilingsFeed, MacroFeed, NewsFeed, PlaybookFeed, ReactionFeed, SizingFeed

MAX_HISTORY, MAX_MACRO, MAX_NEWS, MAX_HEADINGS = 30, 12, 5, 15

_SCHEMAS: dict[str, dict[str, Any]] = {
    "get_price_history": {
        "name": "get_price_history",
        "description": f"Daily closing prices for a symbol, oldest first, up to {MAX_HISTORY} days ending today, with the % change over the window.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "n": {"type": "integer", "description": f"Days (1-{MAX_HISTORY}), default 10"},
            },
            "required": ["symbol"],
        },
    },
    "get_macro": {
        "name": "get_macro",
        "description": (
            "Latest values of a macro series as known today (FRED id, e.g. CPIAUCSL, UNRATE, "
            "FEDFUNDS, DGS10, DGS2, T10Y2Y, VIXCLS, PAYEMS), newest first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "series": {"type": "string"},
                "n": {"type": "integer", "description": f"Observations (1-{MAX_MACRO}), default 6"},
            },
            "required": ["series"],
        },
    },
    "get_news": {
        "name": "get_news",
        "description": f"Recent headlines mentioning a symbol or keyword, newest first, up to {MAX_NEWS}.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Ticker or keyword; empty for all"},
                "n": {"type": "integer", "description": f"1-{MAX_NEWS}, default 5"},
            },
            "required": ["query"],
        },
    },
    "get_filing_signals": {
        "name": "get_filing_signals",
        "description": (
            "Compact signals from a company's latest SEC filings available today: year-over-year "
            "drift of Business (drift_1), Risk Factors (drift_1A) and MD&A (drift_7) text "
            "(0 = unchanged, higher = more change), count of risk headings, risk theme counts, "
            "new risk clusters, and sector cluster."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
    "get_filing_summary": {
        "name": "get_filing_summary",
        "description": (
            "A ~250-word summary of one section of the company's latest filing available today. "
            "item: '1' Business, '1A' Risk Factors, '7' MD&A."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "item": {"type": "string", "enum": ["1", "1A", "7"]},
            },
            "required": ["ticker", "item"],
        },
    },
    "get_event_playbook": {
        "name": "get_event_playbook",
        "description": (
            "How markets reacted to the largest surprises of a macro release type over the "
            "last few years (before today only): mean/median return, up-rate, worst/best over "
            "1 and 5 trading days for SPY, QQQ, XLP (staples), TLT (bonds). series = FRED id "
            "(CPIAUCSL, UNRATE, PAYEMS, FEDFUNDS...). direction = 'above' (hot print), 'below', "
            "or 'all'. Check n_events — small samples are anecdotes. Use it to size a reaction, "
            "not to predict one."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "series": {"type": "string"},
                "direction": {"type": "string", "enum": ["above", "below", "all"]},
                "lookback_years": {"type": "integer", "description": "1-15, default 3"},
            },
            "required": ["series", "direction"],
        },
    },
    "get_reaction_profile": {
        "name": "get_reaction_profile",
        "description": (
            "How the market has typically reacted to an event type over the last few years "
            "(events before today only), as a distribution over reaction archetypes: "
            "inflation_shock (stocks down, bonds down), growth_scare (stocks down, bonds up), "
            "goldilocks (both up), reflation (stocks up, bonds down), muted. Also mean returns "
            "and the defensives-minus-growth spread. event_type examples: cpi_release, "
            "payrolls_release, fomc_decision, vix_spike, pce_release, jobless_claims_release. "
            "label: 'above'/'below' (vs expectation), 'above/large', 'hike/75bp', 'risk_off', or 'all'. "
            "Check n; small samples are anecdotes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "event_type": {"type": "string"},
                "label": {"type": "string", "default": "all"},
                "horizon_days": {"type": "integer", "description": "1 or 5, default 1"},
            },
            "required": ["event_type"],
        },
    },
    "get_exposure_recommendation": {
        "name": "get_exposure_recommendation",
        "description": (
            "Risk-sizing advice after an event: the exposure level (0, 0.3, 0.6 or 1.0 of "
            "equity in stocks) that was mean-variance optimal after similar events in the "
            "same VIX regime over the last few years (events before today only). Returns n, "
            "the mean and volatility of 5-day SPY returns it is based on, and whether it fell "
            "back to the default because the sample was too small. It sizes risk; it does not "
            "pick stocks. Same event_type/label vocabulary as get_reaction_profile."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"event_type": {"type": "string"}, "label": {"type": "string", "default": "all"}},
            "required": ["event_type"],
        },
    },
    "get_risk_headings": {
        "name": "get_risk_headings",
        "description": f"Risk-factor headings (with theme labels) from the latest 10-K available today, up to {MAX_HEADINGS}.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "n": {"type": "integer", "description": f"1-{MAX_HEADINGS}, default 10"},
            },
            "required": ["ticker"],
        },
    },
}


def _clamp(args: dict, key: str, default: int, cap: int) -> int:
    try:
        return max(1, min(int(args.get(key, default)), cap))
    except (TypeError, ValueError):
        return default


class DataTools:
    def __init__(
        self,
        clock: Callable[[], date],
        broker: Broker | None = None,
        macro: MacroFeed | None = None,
        news: NewsFeed | None = None,
        filings: FilingsFeed | None = None,
        playbook: PlaybookFeed | None = None,
        reactions: ReactionFeed | None = None,
        sizing: SizingFeed | None = None,
    ):
        self.clock = clock
        self.broker = broker
        self.macro = macro
        self.news = news
        self.filings = filings
        self.playbook = playbook
        self.reactions = reactions
        self.sizing = sizing
        # Same-day dedupe: a repeated identical call gets a short reminder instead of
        # the data again. Stops research loops without touching the prompt, and keeps
        # the context small. Cleared automatically when the clock moves.
        self._seen_day: date | None = None
        self._seen: dict[tuple[str, str], str] = {}
        self._handlers: dict[str, Callable[[dict[str, Any]], str]] = {}
        if broker is not None:
            self._handlers["get_price_history"] = self._history
        if macro is not None:
            self._handlers["get_macro"] = self._macro
        if news is not None:
            self._handlers["get_news"] = self._news
        if filings is not None:
            self._handlers["get_filing_signals"] = self._signals
            self._handlers["get_filing_summary"] = self._summary
            self._handlers["get_risk_headings"] = self._headings
        if playbook is not None:
            self._handlers["get_event_playbook"] = self._playbook
        if reactions is not None:
            self._handlers["get_reaction_profile"] = self._reaction
        if sizing is not None:
            self._handlers["get_exposure_recommendation"] = self._sizing

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [_SCHEMAS[name] for name in self._handlers]

    def describe(self, name: str, args: dict[str, Any]) -> str:
        return f"Look up {name} with {args}."

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        handler = self._handlers.get(name)
        if handler is None:
            return f"ERROR: unknown tool '{name}'."
        today = self.clock()
        if today != self._seen_day:
            self._seen_day, self._seen = today, {}
        key = (name, json.dumps(args, sort_keys=True, default=str))
        if key in self._seen:
            return (
                f"ALREADY RETRIEVED TODAY ({name} {args}) — the result is above in this "
                "conversation. Do not call it again; use what you have and decide."
            )
        try:
            out = handler(args)
        except Exception as e:  # surface to the model, never crash the loop
            return f"ERROR: {type(e).__name__}: {e}"
        self._seen[key] = out
        return out

    # ---- handlers -----------------------------------------------------------

    def _history(self, args: dict[str, Any]) -> str:
        assert self.broker is not None
        quotes = self.broker.history(str(args["symbol"]).upper(), _clamp(args, "n", 10, MAX_HISTORY))
        if not quotes:
            return "no history"
        change = (quotes[-1].price / quotes[0].price - 1) if quotes[0].price else 0.0
        return json.dumps(
            {
                "symbol": quotes[-1].symbol,
                "window_change_pct": round(change * 100, 2),
                "closes": [[q.day.isoformat(), q.price] for q in quotes],
            }
        )

    def _macro(self, args: dict[str, Any]) -> str:
        assert self.macro is not None
        obs = self.macro.observations(str(args["series"]), self.clock(), _clamp(args, "n", 6, MAX_MACRO))
        if not obs:
            return f"no observations for {args['series']} as of {self.clock()}"
        return json.dumps(
            {
                "series": obs[0].series,
                "as_of": self.clock().isoformat(),
                "observations": [[o.day.isoformat(), o.value] for o in obs],
            }
        )

    def _news(self, args: dict[str, Any]) -> str:
        assert self.news is not None
        items = self.news.headlines(str(args.get("query", "")), self.clock(), _clamp(args, "n", 5, MAX_NEWS))
        if not items:
            return "no matching headlines"
        return json.dumps(
            [
                {"date": h.day.isoformat(), "title": h.title, "source": h.source, "summary": h.summary}
                for h in items
            ]
        )

    def _signals(self, args: dict[str, Any]) -> str:
        assert self.filings is not None
        sigs = self.filings.signals(str(args["ticker"]), self.clock())
        if not sigs:
            return f"no filing signals for {args['ticker']} as of {self.clock()}"
        return json.dumps(
            {
                "ticker": str(args["ticker"]).upper(),
                "latest_filing": max(s.filed for s in sigs).isoformat(),
                "signals": {s.feature: {"value": s.value, "period": s.period_label, **({"note": s.note} if s.note else {})} for s in sigs},
            }
        )

    def _summary(self, args: dict[str, Any]) -> str:
        assert self.filings is not None
        s = self.filings.summary(str(args["ticker"]), str(args["item"]), self.clock())
        if s is None:
            return f"no filing summary for {args['ticker']} item {args['item']} as of {self.clock()}"
        return json.dumps(
            {
                "ticker": s.ticker,
                "form": s.form,
                "period": s.period_label,
                "filed": s.filed.isoformat(),
                "item": f"{s.item} {s.title}",
                "summary": s.summary,
            }
        )

    def _playbook(self, args: dict[str, Any]) -> str:
        assert self.playbook is not None
        series = str(args["series"]).upper()
        if series not in self.playbook.available_series():
            return f"no playbook for {series}; available: {self.playbook.available_series()}"
        years = _clamp(args, "lookback_years", 3, 15)
        entry = self.playbook.lookup(series, str(args.get("direction", "all")), self.clock(), years)
        if entry.n_events == 0:
            return f"no completed {series} events in the {years}y before {self.clock()}"
        return entry.to_json()

    def _reaction(self, args: dict[str, Any]) -> str:
        assert self.reactions is not None
        et = str(args["event_type"]).lower()
        if et not in self.reactions.available_types():
            return f"unknown event_type {et}; available: {self.reactions.available_types()}"
        label = str(args.get("label", "all") or "all")
        horizon = 5 if _clamp(args, "horizon_days", 1, 5) >= 3 else 1
        prof = self.reactions.profile(et, label, self.clock(), horizon=horizon)
        if prof.n == 0:
            return (
                f"no completed {et} events with label {label!r} before {self.clock()}; "
                f"labels seen: {self.reactions.labels_for(et)[:10]}"
            )
        return prof.to_json()

    def _sizing(self, args: dict[str, Any]) -> str:
        assert self.sizing is not None
        try:
            advice = self.sizing.advise(str(args["event_type"]).lower(), str(args.get("label", "all") or "all"), self.clock())
        except KeyError as e:
            return f"unknown event_type {e}"
        return advice.to_json()

    def _headings(self, args: dict[str, Any]) -> str:
        assert self.filings is not None
        rows = self.filings.risk_headings(str(args["ticker"]), self.clock(), _clamp(args, "n", 10, MAX_HEADINGS))
        if not rows:
            return f"no risk headings for {args['ticker']} as of {self.clock()}"
        return json.dumps([{"heading": r.heading, "theme": r.theme or r.cluster_label} for r in rows])
