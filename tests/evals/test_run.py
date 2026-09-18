import pytest

from jarvis_jr.evals.grading import ChecksGrader, JudgeGrader, TrajectoryGrader
from jarvis_jr.evals.run import build_graders, build_tools
from jarvis_jr.settings import Settings


def test_single_grader_selection():
    graders = build_graders("checks", "http://x", "m")
    assert len(graders) == 1 and isinstance(graders[0], ChecksGrader)


def test_all_three_by_name():
    graders = build_graders("checks, trajectory, judge", "http://x", "m")
    assert [type(g) for g in graders] == [ChecksGrader, TrajectoryGrader, JudgeGrader]


def test_unknown_or_empty_rejected():
    with pytest.raises(ValueError, match="unknown grader"):
        build_graders("vibes", "http://x", "m")
    with pytest.raises(ValueError, match="at least one"):
        build_graders("", "http://x", "m")


def test_build_tools_without_profile_is_native_only():
    names = {t.name for t in build_tools(Settings(), profile=None)}
    assert "read_file" in names and "lookup_country" not in names


def test_build_tools_unknown_profile():
    with pytest.raises(ValueError, match="unknown tool profile"):
        build_tools(Settings(), profile="nope")


def test_build_tools_with_profile_adds_assistant_tools():
    settings = Settings()
    settings.tools.profiles["evals"] = ["lookup_country", "web_search"]
    names = {t.name for t in build_tools(settings, profile="evals")}
    assert {"read_file", "lookup_country", "web_search"} <= names
    assert "create_event" not in names  # not in the profile
    assert "open_app" not in names
