from langgraph.graph import END, START, StateGraph

from src.ingestion.pdf import extract_text
from src.ingestion.vector_store import DocumentStore, chunk_text
from src.pipeline.state import IngestState


def build_ingest_graph(store: DocumentStore):
    def load_pdf(state: IngestState) -> IngestState:
        return {"text": extract_text(state["file_path"])}

    def chunk(state: IngestState) -> IngestState:
        return {"chunks": chunk_text(state["text"])}

    def embed_store(state: IngestState) -> IngestState:
        store.add_chunks(state["file_name"], state["chunks"])
        return {}

    graph = StateGraph(IngestState)
    graph.add_node("load_pdf", load_pdf)
    graph.add_node("chunk", chunk)
    graph.add_node("embed_store", embed_store)
    graph.add_edge(START, "load_pdf")
    graph.add_edge("load_pdf", "chunk")
    graph.add_edge("chunk", "embed_store")
    graph.add_edge("embed_store", END)
    return graph.compile()


def ingest_pdf(store: DocumentStore, file_name: str, file_path: str) -> None:
    graph = build_ingest_graph(store)
    graph.invoke({"file_name": file_name, "file_path": file_path})
