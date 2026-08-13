import hashlib

import pandas as pd

from src.evaluation import groundedness
from src.evaluation.groundedness import GroundednessScore
from src.ingestion.vector_store import DocumentStore
from src.pipeline.evaluate_graph import evaluate


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()[:8]
        return [b / 255 for b in digest]


def _fake_run_claude_structured(prompt, schema_model, *, model=None, timeout=120):
    return GroundednessScore(score=1.0, reasoning="ok")


def _make_store(tmp_path, name: str) -> DocumentStore:
    store = DocumentStore(
        persist_directory=tmp_path / "chroma",
        collection_name=name,
        embeddings=FakeEmbeddings(),
    )
    store.add_file("a.pdf", "Acme Corp invoice for John.")
    return store


def test_evaluate_merges_groundedness_and_gold_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(groundedness, "run_claude_structured", _fake_run_claude_structured)
    store = _make_store(tmp_path, "evaluate-test-one")
    extraction_results = {"a.pdf": {"Person": ["John"]}}
    gold_df = pd.DataFrame([{"file_name": "a.pdf", "Person": "John"}])

    report = evaluate(store, {"Person": "person desc"}, extraction_results, gold_df=gold_df)

    assert report == [
        {
            "file_name": "a.pdf",
            "entity_name": "Person",
            "value": "John",
            "groundedness_score": 1.0,
            "groundedness_reasoning": "ok",
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
    ]


def test_evaluate_without_gold_leaves_metrics_none(tmp_path, monkeypatch):
    monkeypatch.setattr(groundedness, "run_claude_structured", _fake_run_claude_structured)
    store = _make_store(tmp_path, "evaluate-test-two")
    extraction_results = {"a.pdf": {"Person": ["John"]}}

    report = evaluate(store, {"Person": "person desc"}, extraction_results, gold_df=None)

    assert report[0]["precision"] is None
    assert report[0]["groundedness_score"] == 1.0
