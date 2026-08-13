import hashlib

from src.ingestion.vector_store import DocumentStore
from src.pipeline.ingest_graph import ingest_pdf


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()[:8]
        return [b / 255 for b in digest]


def test_ingest_pdf_loads_chunks_and_stores_file(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.ingest_graph.extract_text", lambda path: "Acme Corp invoice text")

    store = DocumentStore(
        persist_directory=tmp_path / "chroma",
        collection_name="ingest-graph-test",
        embeddings=FakeEmbeddings(),
    )

    ingest_pdf(store, file_name="invoice.pdf", file_path="unused.pdf")

    assert store.list_files() == ["invoice.pdf"]
    assert store.query("invoice", file_name="invoice.pdf") == ["Acme Corp invoice text"]
