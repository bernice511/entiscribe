import json
import subprocess

import pytest
from pydantic import BaseModel

from src.llm.claude_cli import ClaudeCLIError, run_claude, run_claude_structured


class _FakeCompleted:
    def __init__(self, stdout: str, returncode: int = 0, stderr: str = ""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _envelope(**overrides) -> str:
    base = {"is_error": False, "result": "hello"}
    base.update(overrides)
    return json.dumps(base)


def test_run_claude_returns_result_text(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompleted(_envelope(result="hi there")))

    assert run_claude("say hi") == "hi there"


def test_run_claude_builds_expected_command(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return _FakeCompleted(_envelope(structured_output={}))

    monkeypatch.setattr(subprocess, "run", fake_run)

    run_claude("say hi", json_schema={"type": "object"}, model="sonnet")

    assert captured["command"] == [
        "claude",
        "-p",
        "say hi",
        "--output-format",
        "json",
        "--json-schema",
        json.dumps({"type": "object"}),
        "--model",
        "sonnet",
    ]


def test_run_claude_returns_structured_output_when_schema_given(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _FakeCompleted(_envelope(structured_output={"name": "Ann"})),
    )

    assert run_claude("extract", json_schema={"type": "object"}) == {"name": "Ann"}


def test_run_claude_raises_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _FakeCompleted("", returncode=1, stderr="boom")
    )

    with pytest.raises(ClaudeCLIError, match="boom"):
        run_claude("say hi")


def test_run_claude_raises_on_non_json_stdout(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompleted("not json"))

    with pytest.raises(ClaudeCLIError):
        run_claude("say hi")


def test_run_claude_raises_when_envelope_reports_error(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _FakeCompleted(_envelope(is_error=True, result="denied"))
    )

    with pytest.raises(ClaudeCLIError, match="denied"):
        run_claude("say hi")


def test_run_claude_raises_on_timeout(monkeypatch):
    def fake_run(*a, **k):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=1)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ClaudeCLIError, match="timed out"):
        run_claude("say hi", timeout=1)


class _Person(BaseModel):
    name: str
    city: str


def test_run_claude_structured_validates_into_pydantic_model(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _FakeCompleted(_envelope(structured_output={"name": "John", "city": "Boston"})),
    )

    person = run_claude_structured("extract", _Person)

    assert person == _Person(name="John", city="Boston")
