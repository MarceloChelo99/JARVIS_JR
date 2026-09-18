"""Test doubles. FakeModel proves the agent/graders never need a real server;
FakeRegistry proves the registry shim needs only `schemas` + `dispatch`."""

from __future__ import annotations

from jarvis_jr.evals.domain import ModelReply


class FakeModel:
    """Replays a scripted list of ModelReply objects, one per complete() call."""

    def __init__(self, replies: list[ModelReply]):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []  # messages passed to each complete()

    def complete(self, messages, tools=None) -> ModelReply:
        self.calls.append(list(messages))
        if not self.replies:
            raise AssertionError("FakeModel ran out of scripted replies")
        return self.replies.pop(0)


class FakeRegistry:
    """Minimal stand-in for ToolRegistry / CoderTools."""

    schemas = [
        {
            "name": "echo",
            "description": "Echo the input back.",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }
    ]

    def dispatch(self, name: str, args: dict) -> str:
        if name != "echo":
            return f"ERROR: unknown tool '{name}'."
        if "text" not in args:
            return "ERROR: KeyError: 'text'"
        return args["text"]
