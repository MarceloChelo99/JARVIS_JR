from jarvis_jr.evals.tools import EditFile, ListFiles, ReadFile, WriteFile, available_tools, openai_schema


def test_write_then_read_roundtrip(tmp_path):
    write = WriteFile().run({"path": "a/b.txt", "content": "hi"}, tmp_path)
    assert write.ok
    read = ReadFile().run({"path": "a/b.txt"}, tmp_path)
    assert read.ok and read.output == "hi"


def test_edit_requires_unique_match(tmp_path):
    (tmp_path / "f.txt").write_text("x x")
    result = EditFile().run({"path": "f.txt", "old": "x", "new": "y"}, tmp_path)
    assert not result.ok and "2 times" in result.error

    (tmp_path / "g.txt").write_text("hello world")
    result = EditFile().run({"path": "g.txt", "old": "world", "new": "harness"}, tmp_path)
    assert result.ok and (tmp_path / "g.txt").read_text() == "hello harness"


def test_path_escape_is_blocked(tmp_path):
    for args in ({"path": "../evil.txt", "content": "x"},):
        result = WriteFile().run(args, tmp_path)
        assert not result.ok and "escapes" in result.error
    result = ReadFile().run({"path": "../../etc/hostname"}, tmp_path)
    assert not result.ok


def test_list_files(tmp_path):
    (tmp_path / "z.txt").write_text("1")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text("2")
    result = ListFiles().run({}, tmp_path)
    assert result.ok and result.output.splitlines() == ["sub/a.txt", "z.txt"]


def test_every_default_tool_renders_an_openai_schema():
    for tool in available_tools():
        schema = openai_schema(tool)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == tool.name
        assert schema["function"]["parameters"]["type"] == "object"
