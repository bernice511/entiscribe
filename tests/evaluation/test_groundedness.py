import re

from src.evaluation import groundedness
from src.evaluation.groundedness import GroundednessBatch, GroundednessItem, score_extraction_results
from src.llm.claude_cli import ClaudeCLIError


def _requested_indices(prompt: str) -> list[int]:
    return [int(m) for m in re.findall(r"^(\d+)\. entity type:", prompt, re.MULTILINE)]


def _fake_run_claude_structured(prompt, schema_model, *, model=None, timeout=120):
    return GroundednessBatch(
        items=[GroundednessItem(index=i, score=0.9, reasoning="supported by context") for i in _requested_indices(prompt)]
    )


def test_score_extraction_results_produces_one_row_per_value(monkeypatch):
    monkeypatch.setattr(groundedness, "run_claude_structured", _fake_run_claude_structured)

    extraction_results = {"a.pdf": {"Person": ["John", "Mary"]}}
    contexts = {"a.pdf": "John and Mary work at Acme."}

    rows = score_extraction_results(extraction_results, contexts)

    assert len(rows) == 2
    assert {r["value"] for r in rows} == {"John", "Mary"}
    assert all(r["file_name"] == "a.pdf" and r["entity_name"] == "Person" for r in rows)
    assert all(r["groundedness_score"] == 0.9 for r in rows)


def test_score_extraction_results_skips_entities_with_no_values(monkeypatch):
    monkeypatch.setattr(groundedness, "run_claude_structured", _fake_run_claude_structured)

    rows = score_extraction_results({"a.pdf": {"Person": []}}, {"a.pdf": "no people here"})

    assert rows == []


def test_score_extraction_results_uses_a_single_batched_call_per_file(monkeypatch):
    calls = []

    def counting_fake(prompt, schema_model, *, model=None, timeout=120):
        calls.append(prompt)
        return _fake_run_claude_structured(prompt, schema_model, model=model, timeout=timeout)

    monkeypatch.setattr(groundedness, "run_claude_structured", counting_fake)

    extraction_results = {"a.pdf": {"Person": ["John", "Mary"], "Location": ["Boston"]}}
    rows = score_extraction_results(extraction_results, {"a.pdf": "context"})

    assert len(calls) == 1
    assert len(rows) == 3


def test_score_extraction_results_isolates_a_single_file_failure(monkeypatch):
    def flaky_run_claude_structured(prompt, schema_model, *, model=None, timeout=120):
        if "b-context" in prompt:
            raise ClaudeCLIError("claude CLI not found on PATH")
        return _fake_run_claude_structured(prompt, schema_model, model=model, timeout=timeout)

    monkeypatch.setattr(groundedness, "run_claude_structured", flaky_run_claude_structured)

    extraction_results = {"a.pdf": {"Person": ["John"]}, "b.pdf": {"Person": ["Mary"]}}
    contexts = {"a.pdf": "a-context", "b.pdf": "b-context"}

    rows = score_extraction_results(extraction_results, contexts)

    by_file = {r["file_name"]: r for r in rows}
    assert by_file["a.pdf"]["groundedness_score"] == 0.9
    assert by_file["b.pdf"]["groundedness_score"] is None
    assert "claude CLI not found on PATH" in by_file["b.pdf"]["groundedness_reasoning"]
