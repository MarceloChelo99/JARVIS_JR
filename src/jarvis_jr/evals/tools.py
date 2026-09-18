"""Eval-side tools: sandboxed file ops, web fetch, Gmail (IMAP) — plus a shim
that exposes any JARVIS tool registry (ToolRegistry, CoderTools, MCP tools)
to the eval agent under the same contract.

Every tool follows the same contract (the Tool protocol) and returns the one
ToolResult shape, so the agent loop treats them uniformly.

Visibility rule: a tool declares the env vars it needs in `requires_env`;
`available_tools()` only hands the agent tools whose vars are all set. A tool
with an empty `requires_env` is always available. This keeps the agent from
ever seeing a tool that cannot work.

# GROWTH: split into a tools/ package when this file passes ~12 native tools —
# not before. Shell exec is NOT added here: use CoderTools via registry_tools().
"""
from __future__ import annotations

import email
import email.header
import imaplib
import os
from pathlib import Path
from typing import Protocol

import requests

from jarvis_jr.evals.domain import ToolResult

MAX_OUTPUT_CHARS = 20_000


class Tool(Protocol):
    name: str
    description: str
    parameters: dict  # JSON schema for the arguments
    requires_env: tuple[str, ...]  # env vars that must be set for the tool to be visible

    def run(self, args: dict, workspace: Path) -> ToolResult: ...


def openai_schema(tool: Tool) -> dict:
    """Render a Tool as an OpenAI-style function definition."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _resolve(workspace: Path, rel_path: str) -> Path:
    """Resolve a path inside the workspace; refuse anything that escapes it."""
    candidate = (workspace / rel_path).resolve()
    if not candidate.is_relative_to(workspace.resolve()):
        raise PermissionError(f"path escapes the workspace: {rel_path}")
    return candidate


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(text)} chars total]"


# --------------------------------------------------------------------------
# File tools (always available; sandboxed to the task workspace)
# --------------------------------------------------------------------------


class ReadFile:
    name = "read_file"
    description = "Read a text file from the workspace. Path is relative to the workspace root."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Relative file path"}},
        "required": ["path"],
    }
    requires_env: tuple[str, ...] = ()

    def run(self, args: dict, workspace: Path) -> ToolResult:
        try:
            target = _resolve(workspace, args["path"])
            return ToolResult.success(_truncate(target.read_text()))
        except FileNotFoundError:
            return ToolResult.failure(f"no such file: {args.get('path')}")
        except (PermissionError, KeyError, OSError, UnicodeDecodeError) as exc:
            return ToolResult.failure(str(exc))


class WriteFile:
    name = "write_file"
    description = "Create or overwrite a text file in the workspace."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative file path"},
            "content": {"type": "string", "description": "Full file content"},
        },
        "required": ["path", "content"],
    }
    requires_env: tuple[str, ...] = ()

    def run(self, args: dict, workspace: Path) -> ToolResult:
        try:
            target = _resolve(workspace, args["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(args["content"])
            return ToolResult.success(f"wrote {len(args['content'])} chars to {args['path']}")
        except (PermissionError, KeyError, OSError) as exc:
            return ToolResult.failure(str(exc))


class EditFile:
    name = "edit_file"
    description = (
        "Replace text in an existing file. `old` must occur exactly once; "
        "it is replaced with `new`."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative file path"},
            "old": {"type": "string", "description": "Exact text to replace (must be unique)"},
            "new": {"type": "string", "description": "Replacement text"},
        },
        "required": ["path", "old", "new"],
    }
    requires_env: tuple[str, ...] = ()

    def run(self, args: dict, workspace: Path) -> ToolResult:
        try:
            target = _resolve(workspace, args["path"])
            text = target.read_text()
            count = text.count(args["old"])
            if count != 1:
                return ToolResult.failure(
                    f"`old` occurs {count} times in {args['path']}; it must occur exactly once"
                )
            target.write_text(text.replace(args["old"], args["new"], 1))
            return ToolResult.success(f"edited {args['path']}")
        except FileNotFoundError:
            return ToolResult.failure(f"no such file: {args.get('path')}")
        except (PermissionError, KeyError, OSError) as exc:
            return ToolResult.failure(str(exc))


class ListFiles:
    name = "list_files"
    description = "List all files in the workspace, as relative paths."
    parameters = {"type": "object", "properties": {}}
    requires_env: tuple[str, ...] = ()

    def run(self, args: dict, workspace: Path) -> ToolResult:
        paths = sorted(
            str(p.relative_to(workspace)) for p in workspace.rglob("*") if p.is_file()
        )
        return ToolResult.success("\n".join(paths) if paths else "(workspace is empty)")


# --------------------------------------------------------------------------
# Web tools
# --------------------------------------------------------------------------


class FetchUrl:
    name = "fetch_url"
    description = "Fetch a URL over HTTP GET and return the response body as text."
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "Full http(s) URL"}},
        "required": ["url"],
    }
    requires_env: tuple[str, ...] = ()
    timeout = 20

    def run(self, args: dict, workspace: Path) -> ToolResult:
        url = args.get("url", "")
        if not url.startswith(("http://", "https://")):
            return ToolResult.failure(f"not an http(s) URL: {url}")
        try:
            resp = requests.get(
                url, timeout=self.timeout, headers={"User-Agent": "agent-harness/0.1"}
            )
            resp.raise_for_status()
            return ToolResult.success(_truncate(resp.text))
        except requests.RequestException as exc:
            return ToolResult.failure(f"fetch failed: {exc}")


# --------------------------------------------------------------------------
# Gmail tools (read-only IMAP; visible only when the GMAIL_* env vars are set)
#
# Auth: an app password (myaccount.google.com/apppasswords, requires 2-Step
# Verification) in GMAIL_APP_PASSWORD, plus GMAIL_ADDRESS. Read-only by
# construction: the mailbox is opened readonly and all fetches use BODY.PEEK,
# so nothing is ever sent, deleted, or even marked read.
# --------------------------------------------------------------------------

IMAP_HOST = "imap.gmail.com"
MAX_EMAIL_RESULTS = 10
MAX_EMAIL_BODY_CHARS = 8_000
GMAIL_ENV = ("GMAIL_ADDRESS", "GMAIL_APP_PASSWORD")


def _gmail_connect():
    conn = imaplib.IMAP4_SSL(IMAP_HOST)
    conn.login(os.environ["GMAIL_ADDRESS"], os.environ["GMAIL_APP_PASSWORD"])
    conn.select("INBOX", readonly=True)
    return conn


def _decode_header(value) -> str:
    if value is None:
        return ""
    out = []
    for text, charset in email.header.decode_header(str(value)):
        out.append(text.decode(charset or "utf-8", "replace") if isinstance(text, bytes) else text)
    return "".join(out)


def _summary_line(uid: str, msg) -> str:
    return (
        f"[id {uid}] {_decode_header(msg.get('Date'))} | "
        f"from: {_decode_header(msg.get('From'))} | "
        f"subject: {_decode_header(msg.get('Subject'))}"
    )


def _body_text(msg) -> str:
    """Extract the plain-text body of a parsed email message."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and not part.get("Content-Disposition"):
                payload = part.get_payload(decode=True)
                if payload is not None:
                    return payload.decode(part.get_content_charset() or "utf-8", "replace")
        return "(no plain-text part found)"
    payload = msg.get_payload(decode=True)
    if payload is None:
        return str(msg.get_payload())
    return payload.decode(msg.get_content_charset() or "utf-8", "replace")


def _fetch_summaries(conn, uids: list[bytes]) -> list[str]:
    lines = []
    for uid in uids:
        status, data = conn.uid("fetch", uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])")
        if status == "OK" and data and data[0]:
            msg = email.message_from_bytes(data[0][1])
            lines.append(_summary_line(uid.decode(), msg))
    return lines


class _GmailTool:
    """Shared plumbing: env guard, connect, run, always log out."""

    requires_env: tuple[str, ...] = GMAIL_ENV

    def __init__(self, connect=_gmail_connect):
        self._connect = connect

    def run(self, args: dict, workspace: Path) -> ToolResult:
        missing = [v for v in self.requires_env if not os.environ.get(v)]
        if missing:
            return ToolResult.failure(f"missing env vars: {', '.join(missing)}")
        try:
            conn = self._connect()
        except (imaplib.IMAP4.error, OSError, KeyError) as exc:
            return ToolResult.failure(f"Gmail login failed: {exc}")
        try:
            return self._run(conn, args)
        except (imaplib.IMAP4.error, OSError) as exc:
            return ToolResult.failure(f"Gmail error: {exc}")
        finally:
            try:
                conn.logout()
            except Exception:
                pass


class SearchEmail(_GmailTool):
    name = "search_email"
    description = (
        "Search the Gmail inbox using Gmail search syntax "
        "(e.g. 'from:alice subject:invoice newer_than:7d'). "
        f"Returns up to {MAX_EMAIL_RESULTS} matches with their ids."
    )
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Gmail search query"}},
        "required": ["query"],
    }

    def _run(self, conn, args: dict) -> ToolResult:
        query = args.get("query", "").replace('"', "")
        status, data = conn.uid("search", None, "X-GM-RAW", f'"{query}"')
        if status != "OK":
            return ToolResult.failure(f"search failed: {status}")
        uids = data[0].split()[-MAX_EMAIL_RESULTS:]
        if not uids:
            return ToolResult.success("no matching emails")
        return ToolResult.success("\n".join(_fetch_summaries(conn, uids)))


class ListRecentEmails(_GmailTool):
    name = "list_recent_emails"
    description = f"List the most recent emails in the inbox (up to {MAX_EMAIL_RESULTS})."
    parameters = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "description": f"How many (1-{MAX_EMAIL_RESULTS}), default 5"}
        },
    }

    def _run(self, conn, args: dict) -> ToolResult:
        n = max(1, min(int(args.get("n", 5)), MAX_EMAIL_RESULTS))
        status, data = conn.uid("search", None, "ALL")
        if status != "OK":
            return ToolResult.failure(f"listing failed: {status}")
        uids = data[0].split()[-n:]
        if not uids:
            return ToolResult.success("inbox is empty")
        return ToolResult.success("\n".join(reversed(_fetch_summaries(conn, uids))))


class ReadEmail(_GmailTool):
    name = "read_email"
    description = "Read one email by the id shown in search/list results. Returns headers and the plain-text body."
    parameters = {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "Email id from search results"}},
        "required": ["id"],
    }

    def _run(self, conn, args: dict) -> ToolResult:
        uid = str(args.get("id", "")).strip()
        if not uid.isdigit():
            return ToolResult.failure(f"not a valid email id: {uid!r}")
        status, data = conn.uid("fetch", uid.encode(), "(BODY.PEEK[])")
        if status != "OK" or not data or not data[0]:
            return ToolResult.failure(f"no email with id {uid}")
        msg = email.message_from_bytes(data[0][1])
        body = _body_text(msg)
        if len(body) > MAX_EMAIL_BODY_CHARS:
            body = body[:MAX_EMAIL_BODY_CHARS] + f"\n... [truncated, {len(body)} chars total]"
        return ToolResult.success(f"{_summary_line(uid, msg)}\n\n{body}")


# --------------------------------------------------------------------------
# The tool registry: one flat list + one env filter
# --------------------------------------------------------------------------

ALL_TOOLS: list[Tool] = [
    ReadFile(),
    WriteFile(),
    EditFile(),
    ListFiles(),
    FetchUrl(),
    SearchEmail(),
    ListRecentEmails(),
    ReadEmail(),
]


def available_tools(env=None) -> list[Tool]:
    """Every native eval tool whose required env vars are all set (and non-empty)."""
    env = os.environ if env is None else env
    return [t for t in ALL_TOOLS if all(env.get(v) for v in t.requires_env)]


# --------------------------------------------------------------------------
# Shim: JARVIS tool registries (ToolRegistry, CoderTools, MCP) as eval Tools
#
# Anything with `.schemas` (Anthropic-format list) and `.dispatch(name, args)
# -> str` qualifies. Registry errors come back as "ERROR: ..." strings; the
# shim turns those into ToolResult.failure so graders see them uniformly.
# --------------------------------------------------------------------------


class RegistryTool:
    requires_env: tuple[str, ...] = ()

    def __init__(self, registry, schema: dict):
        self._registry = registry
        self.name: str = schema["name"]
        self.description: str = schema["description"]
        self.parameters: dict = schema["input_schema"]

    def run(self, args: dict, workspace: Path) -> ToolResult:
        # Registry tools scope themselves (CoderTools to its repo_root, etc.);
        # the eval workspace is not their concern.
        out = self._registry.dispatch(self.name, args)
        if out.startswith("ERROR:"):
            return ToolResult.failure(out[len("ERROR:") :].strip())
        return ToolResult.success(_truncate(out))


def registry_tools(registry) -> list[Tool]:
    """Expose every enabled tool of a JARVIS registry to the eval agent."""
    return [RegistryTool(registry, schema) for schema in registry.schemas]
