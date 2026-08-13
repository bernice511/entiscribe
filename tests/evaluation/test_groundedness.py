from src.evaluation import groundedness
from src.evaluation.groundedness import GroundednessScore, score_extraction_results


def _fake_run_claude_structured(prompt, schema_model, *, model=None, timeout=120):
    return GroundednessScore(score=0.9, reasoning="supported by context")


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
