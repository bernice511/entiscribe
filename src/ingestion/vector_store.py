from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_PERSIST_DIRECTORY = "chroma_db"
DEFAULT_COLLECTION_NAME = "ner_documents"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)


def build_default_embeddings() -> Embeddings:
    return HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)


def chunk_text(text: str) -> list[str]:
    return _SPLITTER.split_text(text)


class DocumentStore:
    """Additive vector store: uploading a new file never removes existing ones."""

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embeddings: Embeddings | None = None,
    ):
        self._chroma = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings or build_default_embeddings(),
            persist_directory=str(persist_directory),
        )
        self._file_names: set[str] = self._scan_file_names()

    def _scan_file_names(self) -> set[str]:
        records = self._chroma.get(include=["metadatas"])
        return {meta["file_name"] for meta in records["metadatas"] if "file_name" in meta}

    def add_file(self, file_name: str, text: str) -> None:
        self.add_chunks(file_name, chunk_text(text))

    def add_chunks(self, file_name: str, chunks: list[str]) -> None:
        documents = [Document(page_content=chunk, metadata={"file_name": file_name}) for chunk in chunks]
        if documents:
            self._chroma.add_documents(documents)
        self._file_names.add(file_name)

    def list_files(self) -> list[str]:
        return sorted(self._file_names)

    def query(self, query_text: str, file_name: str, k: int = 5) -> list[str]:
        results = self._chroma.similarity_search(query_text, k=k, filter={"file_name": file_name})
        return [doc.page_content for doc in results]

    def clear_all(self) -> None:
        self._chroma.reset_collection()
        self._file_names = set()
