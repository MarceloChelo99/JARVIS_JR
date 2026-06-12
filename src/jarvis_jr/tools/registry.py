"""Tool schemas (Anthropic format) + runtime dispatcher."""

from __future__ import annotations

import json
from collections.abc import Callable
from fnmatch import fnmatch
from typing import Any

from pathlib import Path

from jarvis_jr.tools.calendar import GoogleCalendar
from jarvis_jr.tools.knowledge import CountryKnowledge
from jarvis_jr.tools.macos import MacControl
from jarvis_jr.tools.mcp_client import MCPManager
from jarvis_jr.tools.memory import MemoryStore
from jarvis_jr.tools.timer import TimerManager
from jarvis_jr.tools.web import WebTools

_DEFAULT_MEMORY_PATH = Path(__file__).resolve().parents[3] / "data" / "memory.json"


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
        "name": "remember",
        "description": (
            "Save a fact to persistent memory so it survives across sessions. "
            "Use when the user tells you something worth keeping: preferences, "
            "codes, names, routines, allergies. Store it as a complete sentence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The fact, as one sentence"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall",
        "description": (
            "Search persistent memory. Call this when the user references "
            "something they may have told you before ('what's my gym code?'). "
            "Empty query returns the most recent memories."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search text; empty for recent"},
            },
            "required": [],
        },
    },
    {
        "name": "forget",
        "description": "Delete memories matching an id or text. Use only when the user asks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "id_or_query": {"type": "string"},
            },
            "required": ["id_or_query"],
        },
    },
    {
        "name": "open_app",
        "description": "Open a macOS application by name, e.g. 'Safari', 'Notes', 'Spotify'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Application name"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "control_music",
        "description": "Control the Apple Music app: play, pause, next, or previous track.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["play", "pause", "next", "previous"],
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "set_volume",
        "description": "Set the system output volume (0-100).",
        "input_schema": {
            "type": "object",
            "properties": {
                "level": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": ["level"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web (DuckDuckGo). Returns a JSON list of results with "
            "title, url, and snippet. Use for facts, prices, weather, how-tos — "
            "anything you're unsure about. For news/sports/current events use "
            "web_news instead. Set recency to limit result age. Follow up with "
            "fetch_page on a result URL if the snippets aren't enough."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 5},
                "recency": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year"],
                    "description": "Only results from this period. Use 'day'/'week' for time-sensitive questions.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_news",
        "description": (
            "Search recent news (DuckDuckGo News). Returns dated, sourced "
            "articles as JSON. Always prefer this over web_search for news, "
            "sports results, and 'what happened' questions — check the date "
            "field and lead with the newest relevant item."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "News search query"},
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
    {
        "name": "lookup_country",
        "description": (
            "Authoritative offline reference for every country in the world "
            "(ISO 3166-1). Returns official name, ISO codes (alpha-2, alpha-3, "
            "numeric), continent, and flag emoji. Prefer this over guessing or "
            "web_search for country facts. Pass a country name or code to look one "
            "up; leave query empty to list all countries; pass a continent to list "
            "just that region (Africa, Asia, Europe, North America, South America, "
            "Oceania)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Country name or ISO code; empty to list all.",
                },
                "continent": {
                    "type": "string",
                    "description": "Optional filter when listing, e.g. 'Europe'.",
                },
            },
            "required": [],
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
        memory: MemoryStore | None = None,
        mac: MacControl | None = None,
        knowledge: CountryKnowledge | None = None,
        enabled_patterns: list[str] | None = None,
    ):
        self.calendar = calendar
        self.timer_manager = timer_manager
        self.web = web if web is not None else WebTools()
        self.mcp = mcp
        self.memory = memory if memory is not None else MemoryStore(_DEFAULT_MEMORY_PATH)
        self.mac = mac if mac is not None else MacControl()
        self.knowledge = knowledge if knowledge is not None else CountryKnowledge()
        self.enabled_patterns = enabled_patterns  # None = everything
        self._handlers: dict[str, Callable[[dict[str, Any]], str]] = {
            "create_event": self._create_event,
            "list_events": self._list_events,
            "set_timer": self._set_timer,
            "web_search": self._web_search,
            "web_news": self._web_news,
            "fetch_page": self._fetch_page,
            "remember": lambda a: self.memory.remember(a["content"]),
            "recall": lambda a: self.memory.recall(a.get("query", "")),
            "forget": lambda a: self.memory.forget(a["id_or_query"]),
            "open_app": lambda a: self.mac.open_app(a["name"]),
            "control_music": lambda a: self.mac.control_music(a["action"]),
            "set_volume": lambda a: self.mac.set_volume(a["level"]),
            "lookup_country": lambda a: self.knowledge.lookup(
                a.get("query", ""), a.get("continent")
            ),
        }

    def _enabled(self, name: str) -> bool:
        # Calendar tools are pointless noise when no calendar is configured.
        if self.calendar is None and name in ("create_event", "list_events"):
            return False
        if self.enabled_patterns is None:
            return True
        return any(fnmatch(name, pat) for pat in self.enabled_patterns)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        all_schemas = list(TOOL_SCHEMAS)
        if self.mcp is not None:
            all_schemas += self.mcp.schemas
        return [s for s in all_schemas if self._enabled(s["name"])]

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
        if not self._enabled(name):
            return f"ERROR: tool '{name}' is not enabled in the current profile."
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
        return self.web.search(args["query"], args.get("max_results"), args.get("recency"))

    def _web_news(self, args: dict[str, Any]) -> str:
        return self.web.news(args["query"], args.get("max_results"))

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
