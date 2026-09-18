import email

import pytest

from jarvis_jr.evals.tools import (
    ListRecentEmails,
    ReadEmail,
    SearchEmail,
    _body_text,
    _decode_header,
    available_tools,
    openai_schema,
)

RAW_EMAIL = b"""From: Alice <alice@example.com>\r
Subject: =?utf-8?q?Invoice_=2342?=\r
Date: Mon, 13 Jul 2026 10:00:00 -0700\r
Content-Type: text/plain; charset=utf-8\r
\r
Please find invoice 42 attached.\r
"""


class FakeIMAP:
    """Stands in for imaplib's connection: uid() + logout()."""

    def __init__(self):
        self.mailbox = {b"7": RAW_EMAIL}

    def uid(self, cmd, *args):
        if cmd == "search":
            return "OK", [b" ".join(self.mailbox.keys())]
        uid = args[0] if isinstance(args[0], bytes) else args[0].encode()
        raw = self.mailbox.get(uid)
        return ("OK", [(b"7 (BODY[])", raw)]) if raw else ("OK", [None])

    def logout(self):
        return "BYE", []


@pytest.fixture(autouse=True)
def creds(monkeypatch):
    monkeypatch.setenv("GMAIL_ADDRESS", "test@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "xxxx")


def test_env_gating_of_available_tools():
    with_creds = {"GMAIL_ADDRESS": "a", "GMAIL_APP_PASSWORD": "b"}
    names = {t.name for t in available_tools(with_creds)}
    assert {"search_email", "list_recent_emails", "read_email"} <= names

    names_without = {t.name for t in available_tools({})}
    assert "search_email" not in names_without
    assert "read_file" in names_without  # env-free tools always present


def test_unconfigured_returns_failure_not_crash(monkeypatch, tmp_path):
    monkeypatch.delenv("GMAIL_ADDRESS")
    result = SearchEmail().run({"query": "x"}, tmp_path)
    assert not result.ok and "missing env vars" in result.error


def test_search_returns_summaries(tmp_path):
    result = SearchEmail(connect=FakeIMAP).run({"query": "invoice"}, tmp_path)
    assert result.ok
    assert "[id 7]" in result.output
    assert "alice@example.com" in result.output
    assert "Invoice #42" in result.output  # encoded header was decoded


def test_read_email_returns_body(tmp_path):
    result = ReadEmail(connect=FakeIMAP).run({"id": "7"}, tmp_path)
    assert result.ok and "invoice 42 attached" in result.output

    bad = ReadEmail(connect=FakeIMAP).run({"id": "DROP TABLE"}, tmp_path)
    assert not bad.ok


def test_list_recent(tmp_path):
    result = ListRecentEmails(connect=FakeIMAP).run({"n": 3}, tmp_path)
    assert result.ok and "[id 7]" in result.output


def test_body_and_header_helpers():
    msg = email.message_from_bytes(RAW_EMAIL)
    assert "invoice 42" in _body_text(msg)
    assert _decode_header(msg.get("Subject")) == "Invoice #42"
    assert _decode_header(None) == ""


def test_gmail_tools_render_openai_schemas(tmp_path):
    for tool in (SearchEmail(FakeIMAP), ListRecentEmails(FakeIMAP), ReadEmail(FakeIMAP)):
        schema = openai_schema(tool)
        assert schema["function"]["name"] == tool.name
