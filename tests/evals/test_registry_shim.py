from jarvis_jr.evals.agent import Agent
from jarvis_jr.evals.domain import ModelReply, ToolCall
from jarvis_jr.evals.tools import openai_schema, registry_tools
from tests.evals.fakes import FakeModel, FakeRegistry


def test_shim_renders_schema_and_dispatches(tmp_path):
    (tool,) = registry_tools(FakeRegistry())
    assert tool.name == "echo"
    assert openai_schema(tool)["function"]["parameters"]["required"] == ["text"]

    ok = tool.run({"text": "hi"}, tmp_path)
    assert ok.ok and ok.output == "hi"

    bad = tool.run({}, tmp_path)
    assert not bad.ok and "KeyError" in bad.error


def test_agent_can_drive_registry_tools(tmp_path):
    model = FakeModel(
        [
            ModelReply(content="", tool_calls=(ToolCall("echo", {"text": "ping"}, "c1"),)),
            ModelReply(content="got ping"),
        ]
    )
    trajectory = Agent(model, registry_tools(FakeRegistry())).run("say ping", tmp_path)
    assert trajectory.tool_sequence() == ["echo"]
    assert trajectory.steps[0].tool_results[0].output == "ping"


def test_real_registry_lookup_country_through_shim(tmp_path):
    """The assistant's offline country tool, graded like any eval tool."""
    from jarvis_jr.tools.registry import ToolRegistry
    from jarvis_jr.tools.timer import TimerManager

    registry = ToolRegistry(
        calendar=None, timer_manager=TimerManager(), enabled_patterns=["lookup_country"]
    )
    tools = registry_tools(registry)
    assert [t.name for t in tools] == ["lookup_country"]
    result = tools[0].run({"query": "Peru"}, tmp_path)
    assert result.ok and '"alpha_3": "PER"' in result.output
