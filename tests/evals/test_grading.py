from tests.evals.fakes import FakeModel

from jarvis_jr.evals.domain import Check, ModelReply, Step, TaskSpec, ToolCall, Trajectory
from jarvis_jr.evals.grading import ChecksGrader, JudgeGrader, TrajectoryGrader, _ordered_matches


def _trajectory(tools: list[str], answer: str = "done") -> Trajectory:
    steps = [Step(thought="", tool_calls=[ToolCall(t, {})], tool_results=[]) for t in tools]
    return Trajectory(steps=steps, final_answer=answer)


def test_checks_grader_scores_each_check(tmp_path):
    (tmp_path / "hello.txt").write_text("hello harness")
    task = TaskSpec(
        id="t",
        prompt="p",
        checks=(
            Check("file_exists", path="hello.txt"),
            Check("file_contains", path="hello.txt", expect="harness"),
            Check("file_matches", path="hello.txt", expect=r"hello \w+"),
            Check("answer_contains", expect="nope"),
        ),
    )
    verdict = ChecksGrader().grade(task, _trajectory([], answer="done"), tmp_path)
    assert verdict.score == 0.75 and not verdict.passed
    assert "FAIL answer_contains" in verdict.detail


def test_trajectory_grader_subsequence_and_forbidden(tmp_path):
    task = TaskSpec(
        id="t", prompt="p", expected_tools=("read_file", "edit_file"), forbidden_tools=("write_file",)
    )
    good = TrajectoryGrader().grade(task, _trajectory(["list_files", "read_file", "edit_file"]), tmp_path)
    assert good.passed and good.score == 1.0

    out_of_order = TrajectoryGrader().grade(task, _trajectory(["edit_file", "read_file"]), tmp_path)
    assert not out_of_order.passed and out_of_order.score == 0.5

    forbidden = TrajectoryGrader().grade(
        task, _trajectory(["read_file", "edit_file", "write_file"]), tmp_path
    )
    assert not forbidden.passed and forbidden.score == 0.0


def test_ordered_matches():
    assert _ordered_matches(["a", "b"], ["x", "a", "y", "b"]) == 2
    assert _ordered_matches(["a", "b"], ["b", "a"]) == 1
    assert _ordered_matches([], ["a"]) == 0


def test_judge_grader_parses_score(tmp_path):
    judge_model = FakeModel(
        [ModelReply(content='Sure. {"score": 0.9, "reasoning": "clean work"}')]
    )
    task = TaskSpec(id="t", prompt="p", rubric="be correct")
    verdict = JudgeGrader(judge_model).grade(task, _trajectory(["write_file"]), tmp_path)
    assert verdict.passed and verdict.score == 0.9 and verdict.detail == "clean work"


def test_judge_grader_skips_without_rubric_and_survives_garbage(tmp_path):
    task_no_rubric = TaskSpec(id="t", prompt="p")
    verdict = JudgeGrader(FakeModel([])).grade(task_no_rubric, _trajectory([]), tmp_path)
    assert verdict.passed and "skipped" in verdict.detail

    task = TaskSpec(id="t", prompt="p", rubric="r")
    garbage = JudgeGrader(FakeModel([ModelReply(content="no json here")]))
    verdict = garbage.grade(task, _trajectory([]), tmp_path)
    assert not verdict.passed and verdict.score == 0.0
