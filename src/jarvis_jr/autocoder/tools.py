"""Repo-scoped filesystem + bash tools for the coder agent.

Implements the same duck-typed interface as `ToolRegistry` (schemas / dispatch /
describe) so it can be plugged into any `LLMClient` via `build_llm_client`.

Safety rails:
- All paths resolve against `repo_root`; access outside it is rejected.
- Bash commands run with cwd=repo_root.
- Network-y commands are blocked by default (opt in with `allow_network=True`).
"""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": (
            "Read a UTF-8 text file from the repo. Returns the contents, or an "
            "ERROR string if the file doesn't exist."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repo-relative path."},
                "offset": {
                    "type": "integer",
                    "description": "1-indexed starting line. Optional.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines to return. Optional.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create or overwrite a file with the given content. Prefer edit_file "
            "for modifications to existing files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace one unique occurrence of `old_string` with `new_string` in "
            "`path`. Fails (no edit applied) if old_string is missing or appears "
            "more than once — include surrounding context to make it unique."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "run_bash",
        "description": (
            "Run a shell command in the repo root. Returns exit code + stdout + "
            "stderr. Network commands (curl/wget/uv add/etc.) are blocked unless "
            "the runner was started with --allow-network."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_sec": {
                    "type": "integer",
                    "description": "Max seconds before the process is killed. Default 60.",
                },
            },
            "required": ["command"],
        },
    },
]


# Substrings in a bash command that should be blocked when allow_network=False.
_NETWORK_FORBIDDEN = (
    "curl ", "wget ", "uv add ", "uv remove ", "uv sync",
    "pip install", "pip3 install", "npm install", "npm i ", "yarn add",
    "brew install", "brew uninstall", "git push", "git pull", "git fetch",
)


class PathOutsideRepo(ValueError):
    pass


class CoderTools:
    """Repo-scoped tool registry for the autocoder."""

    def __init__(
        self,
        repo_root: Path,
        on_call: Callable[[str, dict, str], None] | None = None,
        allow_network: bool = False,
        default_bash_timeout: int = 60,
    ):
        self.repo_root = repo_root.resolve()
        self.on_call = on_call
        self.allow_network = allow_network
        self.default_bash_timeout = default_bash_timeout
        self._handlers: dict[str, Callable[[dict[str, Any]], str]] = {
            "read_file": self._read,
            "write_file": self._write,
            "edit_file": self._edit,
            "run_bash": self._bash,
        }

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def describe(self, name: str, args: dict[str, Any]) -> str:
        if name == "read_file":
            return f"read_file({args.get('path')!r})"
        if name == "write_file":
            content = args.get("content", "")
            return f"write_file({args.get('path')!r}, content[{len(content)} chars])"
        if name == "edit_file":
            return f"edit_file({args.get('path')!r}, ...)"
        if name == "run_bash":
            cmd = args.get("command", "")
            short = cmd if len(cmd) <= 80 else cmd[:77] + "..."
            return f"run_bash({short!r})"
        return f"{name}({args})"

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        handler = self._handlers.get(name)
        if handler is None:
            result = f"ERROR: unknown tool '{name}'"
        else:
            try:
                result = handler(args)
            except PathOutsideRepo as e:
                result = f"ERROR: {e}"
            except Exception as e:  # noqa: BLE001
                result = f"ERROR: {type(e).__name__}: {e}"
        if self.on_call is not None:
            try:
                self.on_call(name, args, result)
            except Exception:  # logging must not crash the agent
                pass
        return result

    # ---- handlers ----------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        p = (self.repo_root / path).resolve()
        try:
            p.relative_to(self.repo_root)
        except ValueError as exc:
            raise PathOutsideRepo(
                f"path {path!r} resolves outside repo root {self.repo_root}"
            ) from exc
        return p

    def _read(self, args: dict[str, Any]) -> str:
        path = self._resolve(args["path"])
        if not path.exists():
            return f"ERROR: file not found: {args['path']}"
        if not path.is_file():
            return f"ERROR: not a file: {args['path']}"
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"ERROR: file is not UTF-8 text: {args['path']}"
        offset = args.get("offset")
        limit = args.get("limit")
        if offset is None and limit is None:
            return text
        lines = text.splitlines(keepends=True)
        start = max(0, (offset or 1) - 1)
        end = start + limit if limit else len(lines)
        return "".join(lines[start:end])

    def _write(self, args: dict[str, Any]) -> str:
        path = self._resolve(args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        content = args["content"]
        path.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {args['path']}"

    def _edit(self, args: dict[str, Any]) -> str:
        path = self._resolve(args["path"])
        if not path.exists():
            return f"ERROR: file not found: {args['path']}"
        old = args["old_string"]
        new = args["new_string"]
        if old == new:
            return "ERROR: old_string equals new_string; nothing to do"
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            return f"ERROR: old_string not found in {args['path']}"
        if count > 1:
            return (
                f"ERROR: old_string appears {count} times in {args['path']}; "
                "add more surrounding context to make it unique"
            )
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"edited {args['path']}"

    def _bash(self, args: dict[str, Any]) -> str:
        command: str = args["command"]
        timeout = int(args.get("timeout_sec", self.default_bash_timeout))
        if not self.allow_network:
            lowered = command.lower()
            for needle in _NETWORK_FORBIDDEN:
                if needle in lowered:
                    return (
                        f"ERROR: command contains blocked substring {needle!r}. "
                        "Re-run the autocoder with --allow-network if this is intentional."
                    )
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: command timed out after {timeout}s: {command}"
        # Trim very large outputs.
        out = _truncate(proc.stdout, 12000)
        err = _truncate(proc.stderr, 4000)
        return json.dumps(
            {"exit": proc.returncode, "stdout": out, "stderr": err},
            ensure_ascii=False,
        )

    # Helper for shlex-style argv if the agent ever wants it.
    @staticmethod
    def _argv(command: str) -> list[str]:
        return shlex.split(command)


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... [truncated; {len(s) - limit} more chars]"
