"""Interactive text chat with JARVIS Jr. — no mic, no TTS, no camera.

Same brain and tools as the voice assistant, multi-turn, typed:

    uv run python scripts/chat.py                     # voice tool profile
    uv run python scripts/chat.py --profile desk      # everything incl. MCP/EDGAR
    uv run python scripts/chat.py --provider google   # different backend

Commands inside the chat:
    /reset   clear conversation history
    /tools   list the tools the model can see
    /quit    exit (also Ctrl+C / Ctrl+D)

Sensitive tools (calendar writes, forget) confirm via y/n in the terminal.
"""

from __future__ import annotations

import argparse
import sys

from jarvis_jr.confirm import stdin_confirmer
from jarvis_jr.llm import build_llm_client
from jarvis_jr.llm.prompts import build_system_prompt
from jarvis_jr.settings import load_settings
from jarvis_jr.tools.calendar import CalendarError, GoogleCalendar
from jarvis_jr.tools.mcp_client import MCPManager
from jarvis_jr.tools.registry import ToolRegistry
from jarvis_jr.tools.timer import TimerManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Text chat with JARVIS Jr.")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "google", "ollama"],
        help="Override llm.provider for this run.",
    )
    parser.add_argument("--model", help="Override the LLM model ID for this run.")
    parser.add_argument("--profile", help="Tool profile (e.g. 'voice', 'desk').")
    args = parser.parse_args()

    settings = load_settings()
    if args.provider:
        settings.llm.provider = args.provider
        per_provider = getattr(settings.llm, args.provider, {}) or {}
        if not args.model and per_provider.get("model"):
            settings.llm.model = per_provider["model"]
    if args.model:
        settings.llm.model = args.model
    if args.profile:
        settings.tools.profile = args.profile
    patterns = settings.tools.active_patterns()

    calendar: GoogleCalendar | None = None
    try:
        calendar = GoogleCalendar(
            calendar_id=settings.calendar.default_calendar_id,
            default_duration_minutes=settings.calendar.default_duration_minutes,
            default_timezone=settings.calendar.default_timezone,
        )
    except CalendarError:
        pass  # calendar tools auto-hide when unconfigured

    mcp: MCPManager | None = None
    from fnmatch import fnmatch

    wanted = [
        srv
        for srv in settings.mcp.servers
        if any(
            fnmatch(f"{srv['name']}_x", pat) or pat.startswith(srv["name"]) or pat == "*"
            for pat in patterns
        )
    ]
    if wanted:
        print(f"[chat] starting {len(wanted)} MCP server(s)…")
        mcp = MCPManager(servers=wanted, call_timeout_s=settings.mcp.call_timeout_s)
        mcp.start()

    timer_manager = TimerManager()  # fires print-only notifications
    registry = ToolRegistry(
        calendar=calendar,
        timer_manager=timer_manager,
        mcp=mcp,
        enabled_patterns=patterns,
    )

    original_dispatch = registry.dispatch

    def traced_dispatch(name, tool_args):
        print(f"  🔧 {name}({tool_args})")
        return original_dispatch(name, tool_args)

    registry.dispatch = traced_dispatch  # type: ignore[method-assign]

    extras = {
        k: v
        for k, v in (getattr(settings.llm, settings.llm.provider, {}) or {}).items()
        if k != "model"
    }
    client = build_llm_client(
        provider=settings.llm.provider,
        model=settings.llm.model,
        tools=registry,
        confirmer=stdin_confirmer,
        system_prompt=build_system_prompt(),
        require_confirmation_for=settings.confirmation.require_for,
        max_tokens=settings.llm.max_tokens,
        **extras,
    )

    tool_names = [s["name"] for s in registry.schemas]
    print(
        f"\nJARVIS Jr. text chat — {settings.llm.provider} / {settings.llm.model}\n"
        f"Profile '{settings.tools.profile}': {len(tool_names)} tool(s). "
        "/tools to list, /reset to clear, /quit to exit.\n"
    )

    try:
        while True:
            try:
                user_text = input("you ❯ ").strip()
            except EOFError:
                break
            if not user_text:
                continue
            if user_text in ("/quit", "/exit", "/q"):
                break
            if user_text == "/reset":
                client.reset()
                print("(history cleared)\n")
                continue
            if user_text == "/tools":
                print("  " + "\n  ".join(tool_names) + "\n")
                continue
            try:
                reply = client.submit(user_text)
            except Exception as e:
                print(f"[chat] error: {type(e).__name__}: {e}\n")
                continue
            print(f"\njarvis ❯ {reply}\n")
    except KeyboardInterrupt:
        pass
    finally:
        timer_manager.cancel_all()
        if mcp is not None:
            mcp.stop()
    print("\nGoodbye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
