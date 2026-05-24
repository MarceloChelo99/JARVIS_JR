import pytest
from pathlib import Path
from src.jarvis_jr.autocoder.tools import CoderTools

@pytest.fixture
def coder_tools(tmp_path):
    return CoderTools(repo_root=tmp_path, allow_network=False)

@pytest.fixture
def coder_tools_network_allowed(tmp_path):
    return CoderTools(repo_root=tmp_path, allow_network=True)

def test_path_safety(coder_tools, tmp_path):
    evil_path = "../evil.txt"
    (tmp_path.parent / "evil.txt").write_text("evil content")
    result = coder_tools.dispatch("read_file", {"path": evil_path})
    assert "ERROR: path '../evil.txt' resolves outside repo root" in result

def test_edit_file_multiple_occurrences(coder_tools, tmp_path):
    test_file = tmp_path / "test_edit.txt"
    test_file.write_text("line1\nline2\nline1")
    result = coder_tools.dispatch(
        "edit_file", {"path": "test_edit.txt", "old_string": "line1", "new_string": "newline"}
    )
    assert "ERROR: old_string appears 2 times in test_edit.txt; add more surrounding context to make it unique" in result

def test_run_bash_network_forbidden(coder_tools):
    result = coder_tools.dispatch("run_bash", {"command": "curl example.com"})
    assert "ERROR: command contains blocked substring 'curl '. Re-run the autocoder with --allow-network if this is intentional." in result

def test_run_bash_network_allowed(coder_tools_network_allowed):
    # This test doesn't actually hit the network, just verifies the check is bypassed.
    # We use a command that will likely succeed and return an empty string for stdout/stderr.
    result = coder_tools_network_allowed.dispatch("run_bash", {"command": "curl --version"})
    # Check for a successful exit code (0) and some curl version info in stdout or stderr
    # We can't guarantee 'curl' is installed, so we check for the error string *not* being there
    assert "ERROR: command contains blocked substring 'curl '" not in result
    assert "exit" in result
    assert "stdout" in result
    assert "stderr" in result
