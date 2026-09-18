import pytest

from jarvis_jr.evals.tasks import load_task, load_tasks

GOOD = """
id: sample
prompt: do the thing
expected_tools: [read_file]
checks:
  - kind: file_exists
    path: out.txt
rubric: graded well
"""


def test_load_task_builds_spec(tmp_path):
    path = tmp_path / "sample.yaml"
    path.write_text(GOOD)
    task = load_task(path)
    assert task.id == "sample"
    assert task.expected_tools == ("read_file",)
    assert task.checks[0].kind == "file_exists"
    assert task.max_turns == 12


def test_load_task_rejects_unknown_check_kind(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("id: x\nprompt: y\nchecks:\n  - kind: file_is_pretty\n")
    with pytest.raises(ValueError, match="unknown check kind"):
        load_task(path)


def test_load_task_resolves_fixture_dir(tmp_path):
    (tmp_path / "fixtures" / "x").mkdir(parents=True)
    path = tmp_path / "t.yaml"
    path.write_text("id: x\nprompt: y\nfixture_dir: fixtures/x\n")
    assert load_task(path).fixture_dir == str(tmp_path / "fixtures" / "x")


def test_load_tasks_requires_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tasks(tmp_path)


def test_shipped_example_tasks_load():
    from pathlib import Path

    tasks_dir = Path(__file__).resolve().parents[2] / "evals" / "tasks"
    tasks = load_tasks(tasks_dir)
    assert {t.id for t in tasks} == {"hello_file", "edit_note", "web_lookup", "country_lookup"}
