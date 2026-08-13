import hashlib

from src.ingestion.vector_store import DocumentStore


class FakeEmbeddings:
    """Deterministic, no-download stand-in for a real embedding model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()[:8]
        return [b / 255 for b in digest]


def _make_store(tmp_path, name: str) -> DocumentStore:
    return DocumentStore(
        persist_directory=tmp_path / "chroma",
        collection_name=name,
        embeddings=FakeEmbeddings(),
    )


def test_add_file_registers_it_in_list_files(tmp_path):
    store = _make_store(tmp_path, "col-one")

    store.add_file("invoice.pdf", "Total due: $42.00 from Acme Corp.")

    assert store.list_files() == ["invoice.pdf"]


def test_add_file_is_additive_not_destructive(tmp_path):
    store = _make_store(tmp_path, "col-two")

    store.add_file("a.pdf", "content a")
    store.add_file("b.pdf", "content b")

    assert store.list_files() == ["a.pdf", "b.pdf"]


def test_query_is_scoped_to_requested_file(tmp_path):
    store = _make_store(tmp_path, "col-three")
    store.add_file("a.pdf", "Acme Corp invoice")
    store.add_file("b.pdf", "Globex invoice")

    results = store.query("invoice", file_name="a.pdf", k=5)

    assert results == ["Acme Corp invoice"]


def test_clear_all_empties_store_and_file_list(tmp_path):
    store = _make_store(tmp_path, "col-four")
    store.add_file("a.pdf", "some content")

    store.clear_all()

    assert store.list_files() == []
    assert store.query("content", file_name="a.pdf") == []


def test_reopening_store_recovers_previously_added_files(tmp_path):
    persist_dir = tmp_path / "chroma"
    DocumentStore(persist_directory=persist_dir, collection_name="col-five", embeddings=FakeEmbeddings()).add_file(
        "a.pdf", "some content"
    )

    reopened = DocumentStore(persist_directory=persist_dir, collection_name="col-five", embeddings=FakeEmbeddings())

    assert reopened.list_files() == ["a.pdf"]
