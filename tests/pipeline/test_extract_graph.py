import hashlib

from src.ingestion.vector_store import DocumentStore
from src.llm.claude_cli import ClaudeCLIError
from src.pipeline import extract_graph
from src.pipeline.extract_graph import extract_entities, extract_entities_for_file


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()[:8]
        return [b / 255 for b in digest]


def _fake_run_claude_structured(prompt, schema_model, *, model=None, timeout=120):
    return schema_model(**{name: [name] for name in schema_model.model_fields})


def _make_store(tmp_path, name: str) -> DocumentStore:
    return DocumentStore(
        persist_directory=tmp_path / "chroma",
        collection_name=name,
        embeddings=FakeEmbeddings(),
    )


def test_extract_entities_for_file_maps_result_back_to_entity_names(tmp_path, monkeypatch):
    monkeypatch.setattr(extract_graph, "run_claude_structured", _fake_run_claude_structured)
    store = _make_store(tmp_path, "extract-test-one")
    store.add_file("a.pdf", "Acme Corp is located in Boston.")

    result = extract_entities_for_file(store, "a.pdf", {"Organization": "org", "Location": "loc"})

    assert result == {"Organization": ["organization"], "Location": ["location"]}


def test_extract_entities_runs_for_every_stored_file(tmp_path, monkeypatch):
    monkeypatch.setattr(extract_graph, "run_claude_structured", _fake_run_claude_structured)
    store = _make_store(tmp_path, "extract-test-two")
    store.add_file("a.pdf", "content a")
    store.add_file("b.pdf", "content b")

    result, errors = extract_entities(store, {"Person": "person desc"})

    assert set(result.keys()) == {"a.pdf", "b.pdf"}
    assert result["a.pdf"] == {"Person": ["person"]}
    assert errors == {}


def test_extract_entities_isolates_a_single_file_failure(tmp_path, monkeypatch):
    def flaky_run_claude_structured(prompt, schema_model, *, model=None, timeout=120):
        if "b.pdf" in prompt or "content b" in prompt:
            raise ClaudeCLIError("claude CLI not found on PATH")
        return schema_model(**{name: [name] for name in schema_model.model_fields})

    monkeypatch.setattr(extract_graph, "run_claude_structured", flaky_run_claude_structured)
    store = _make_store(tmp_path, "extract-test-three")
    store.add_file("a.pdf", "content a")
    store.add_file("b.pdf", "content b")

    result, errors = extract_entities(store, {"Person": "person desc"})

    assert result == {"a.pdf": {"Person": ["person"]}}
    assert errors == {"b.pdf": "claude CLI not found on PATH"}
