from tests.evals.fakes import FakeModel

from jarvis_jr.evals.agent import Agent
from jarvis_jr.evals.domain import ModelReply, ToolCall
from jarvis_jr.evals.tools import available_tools


def test_agent_executes_tool_calls_and_records_trajectory(tmp_path):
    model = FakeModel(
        [
            ModelReply(
                content="I'll create the file.",
                tool_calls=(ToolCall("write_file", {"path": "out.txt", "content": "done"}, "c1"),),
            ),
            ModelReply(content="Created out.txt containing 'done'."),
        ]
    )
    trajectory = Agent(model, available_tools()).run("make out.txt", tmp_path)

    assert (tmp_path / "out.txt").read_text() == "done"
    assert trajectory.tool_sequence() == ["write_file"]
    assert trajectory.steps[0].tool_results[0].ok
    assert "out.txt" in trajectory.final_answer
    assert not trajectory.hit_turn_limit
    # the tool result was fed back to the model on the second call
    assert model.calls[1][-1]["role"] == "tool"


def test_agent_reports_unknown_tool_instead_of_crashing(tmp_path):
    model = FakeModel(
        [
            ModelReply(content="", tool_calls=(ToolCall("no_such_tool", {}, "c1"),)),
            ModelReply(content="giving up"),
        ]
    )
    trajectory = Agent(model, available_tools()).run("task", tmp_path)
    assert not trajectory.steps[0].tool_results[0].ok
    assert "unknown tool" in trajectory.steps[0].tool_results[0].error


def test_agent_forces_a_final_answer_on_the_last_turn(tmp_path):
    looping = ModelReply(content="", tool_calls=(ToolCall("list_files", {}, "c1"),))
    model = FakeModel([looping, looping, ModelReply(content="forced summary")])
    trajectory = Agent(model, available_tools()).run("task", tmp_path, max_turns=3)
    assert trajectory.hit_turn_limit  # exhaustion is still recorded
    assert trajectory.final_answer == "forced summary"
    assert len(trajectory.steps) == 2
    assert model.calls[2] is not None
    # the last call carried no tool schemas


def test_last_turn_call_has_no_tools(tmp_path):
    class Spy(FakeModel):
        def complete(self, messages, tools=None):
            self.tools_seen = getattr(self, "tools_seen", []) + [tools is not None]
            return super().complete(messages, tools)

    looping = ModelReply(content="", tool_calls=(ToolCall("list_files", {}, "c1"),))
    model = Spy([looping, ModelReply(content="done")])
    Agent(model, available_tools()).run("task", tmp_path, max_turns=2)
    assert model.tools_seen == [True, False]
