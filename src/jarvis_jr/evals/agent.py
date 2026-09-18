"""The agent loop: prompt in, tools executed, Trajectory out.

The loop is deliberately plain — call the model, run whatever tools it asked
for, feed results back, stop when it answers without tool calls or hits the
turn limit. All cleverness lives in the model; all record-keeping lives here.
"""
from __future__ import annotations

from pathlib import Path

from jarvis_jr.evals.domain import Step, ToolResult, Trajectory
from jarvis_jr.evals.tools import Tool, openai_schema

SYSTEM_PROMPT = """You are an autonomous agent working inside a sandboxed workspace directory.
Use the available tools to complete the user's task. File paths are always
relative to the workspace root.

Work step by step. When the task is complete, reply WITHOUT any tool call,
stating what you did and the final answer."""


class Agent:
    def __init__(self, model, tools: list[Tool], system_prompt: str = SYSTEM_PROMPT):
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.schemas = [openai_schema(t) for t in tools]
        self.system_prompt = system_prompt

    def run(self, prompt: str, workspace: Path, max_turns: int = 12) -> Trajectory:
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        trajectory = Trajectory()

        for turn in range(max_turns):
            last_turn = turn == max_turns - 1
            # On the final turn the model gets no tools: it must answer. The
            # trajectory still records that the limit was hit, so exhaustion
            # stays measurable — but the answer is the model's, not a placeholder.
            reply = self.model.complete(messages, tools=None if last_turn else self.schemas)

            if not reply.tool_calls:
                trajectory.final_answer = reply.content
                trajectory.hit_turn_limit = last_turn
                return trajectory

            messages.append(reply.raw_message or _assistant_message(reply))
            step = Step(thought=reply.content, tool_calls=list(reply.tool_calls))
            for call in reply.tool_calls:
                result = self._execute(call.tool, call.args, workspace)
                step.tool_results.append(result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": result.as_text(),
                    }
                )
            trajectory.steps.append(step)

        # Only reachable if the tool-less final turn still produced tool calls.
        trajectory.hit_turn_limit = True
        trajectory.final_answer = reply.content or "(turn limit reached before a final answer)"
        return trajectory

    def _execute(self, name: str, args: dict, workspace: Path) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult.failure(f"unknown tool: {name}")
        try:
            return tool.run(args, workspace)
        except NotImplementedError as exc:
            return ToolResult.failure(f"tool not available: {exc}")


def _assistant_message(reply) -> dict:
    """Reconstruct an OpenAI-format assistant message when raw_message is absent
    (e.g. fake models in tests)."""
    import json

    return {
        "role": "assistant",
        "content": reply.content or None,
        "tool_calls": [
            {
                "id": c.call_id or f"call_{i}",
                "type": "function",
                "function": {"name": c.tool, "arguments": json.dumps(c.args)},
            }
            for i, c in enumerate(reply.tool_calls)
        ],
    }
