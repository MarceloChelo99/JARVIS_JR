"""Tool schemas (Anthropic format) + runtime dispatcher."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from jarvis_jr.tools.calendar import GoogleCalendar
from jarvis_jr.tools.mcp_client import MCPManager
from jarvis_jr.tools.timer import TimerManager
from jarvis_jr.tools.web import WebTools


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "create_event",
        "description": (
            "Create a calendar event on the user's primary calendar. "
            "Resolve relative times (e.g. 'tomorrow at 3pm') to absolute ISO 8601 "
            "in the user's local timezone before calling."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title"},
                "start": {
                    "type": "string",
                    "description": "ISO 8601 start time, e.g. 2026-05-21T15:00:00-07:00",
                },
                "end": {
                    "type": "string",
                    "description": (
                        "ISO 8601 end time. Optional; defaults to 30 minutes after start."
                    ),
                },
                "location": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["title", "start"],
        },
    },
    {
        "name": "list_events",
        "description": "List calendar events in a time range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": "ISO 8601 start of range"},
                "end": {"type": "string", "description": "ISO 8601 end of range"},
                "max_results": {"type": "integer", "default": 10},
            },
            "required": ["start", "end"],
        },
    },
    {
        "name": "set_timer",
        "description": (
            "Set a timer. When the timer goes off, the assistant will speak "
            "a notification mentioning the label."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "duration_seconds": {"type": "integer"},
                "label": {"type": "string", "description": "What the timer is for"},
            },
            "required": ["duration_seconds", "label"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web (DuckDuckGo). Returns a JSON list of results with "
            "title, url, and snippet. Use for current events, facts you're "
            "unsure about, prices, weather, anything after your training data. "
            "Follow up with fetch_page on a result URL if the snippets aren't enough."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_page",
        "description": (
            "Fetch a web page and return its readable text content (truncated). "
            "Use after web_search when you need more than the snippet."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to fetch"},
            },
            "required": ["url"],
        },
    },
]


class ToolRegistry:
    """Dispatch tool calls coming back from the LLM to their Python implementations."""

    def __init__(
        self,
        calendar: GoogleCalendar | None,
        timer_manager: TimerManager,
        web: WebTools | None = None,
        mcp: MCPManager | None = None,
    ):
        self.calendar = calendar
        self.timer_manager = timer_manager
        self.web = web if web is not None else WebTools()
        self.mcp = mcp
        self._handlers: dict[str, Callable[[dict[str, Any]], str]] = {
            "create_event": self._create_event,
            "list_events": self._list_events,
            "set_timer": self._set_timer,
            "web_search": self._web_search,
            "fetch_page": self._fetch_page,
        }

    @property
    def schemas(self) -> list[dict[str, Any]]:
        if self.mcp is not None and self.mcp.schemas:
            return TOOL_SCHEMAS + self.mcp.schemas
        return TOOL_SCHEMAS

    def describe(self, name: str, args: dict[str, Any]) -> str:
        """A human-readable fallback proposal for confirmation prompts."""
        if name == "create_event":
            return (
                f"Create event '{args.get('title', '?')}' at {args.get('start', '?')}"
                + (f" until {args['end']}" if args.get("end") else "")
                + "."
            )
        if name == "list_events":
            return f"List events from {args.get('start')} to {args.get('end')}."
        if name == "set_timer":
            return (
                f"Set a {args.get('duration_seconds', '?')}s timer for "
                f"'{args.get('label', '?')}'."
            )
        if name == "web_search":
            return f"Search the web for '{args.get('query', '?')}'."
        if name == "fetch_page":
            return f"Fetch and read {args.get('url', '?')}."
        return f"Run {name} with {args}."

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        handler = self._handlers.get(name)
        if handler is None:
            if self.mcp is not None and self.mcp.owns(name):
                try:
                    return self.mcp.call(name, args)
                except Exception as e:
                    return f"ERROR: {type(e).__name__}: {e}"
            return f"ERROR: unknown tool '{name}'."
        try:
            return handler(args)
        except Exception as e:  # surface errors back to the LLM
            return f"ERROR: {type(e).__name__}: {e}"

    # ---- handlers ----------------------------------------------------------

    def _create_event(self, args: dict[str, Any]) -> str:
        if self.calendar is None:
            return "ERROR: calendar not configured."
        ev = self.calendar.create_event(
            title=args["title"],
            start=args["start"],
            end=args.get("end"),
            location=args.get("location"),
            description=args.get("description"),
        )
        return json.dumps(
            {
                "id": ev.get("id"),
                "summary": ev.get("summary"),
                "start": ev.get("start"),
                "end": ev.get("end"),
                "htmlLink": ev.get("htmlLink"),
            }
        )

    def _list_events(self, args: dict[str, Any]) -> str:
        if self.calendar is None:
            return "ERROR: calendar not configured."
        events = self.calendar.list_events(
            start=args["start"],
            end=args["end"],
            max_results=args.get("max_results", 10),
        )
        slim = [
            {
                "id": e.get("id"),
                "summary": e.get("summary"),
                "start": e.get("start"),
                "end": e.get("end"),
                "location": e.get("location"),
            }
            for e in events
        ]
        return json.dumps(slim) if slim else "No events in range."

    def _web_search(self, args: dict[str, Any]) -> str:
        return self.web.search(args["query"], args.get("max_results"))

    def _fetch_page(self, args: dict[str, Any]) -> str:
        return self.web.fetch_page(args["url"])

    def _set_timer(self, args: dict[str, Any]) -> str:
        handle = self.timer_manager.set_timer(
            duration_seconds=int(args["duration_seconds"]),
            label=args["label"],
        )
        return json.dumps(
            {
                "id": handle.id,
                "label": handle.label,
                "duration_seconds": handle.duration_seconds,
            }
        )
