import concurrent.futures

from langgraph.graph import END, START, StateGraph

from src.entities.schema import build_entity_model
from src.ingestion.vector_store import DocumentStore
from src.llm.claude_cli import ClaudeCLIError, run_claude_structured
from src.pipeline.state import ExtractState

EXTRACTION_INSTRUCTIONS = (
    "You are extracting structured information from a document excerpt. "
    "For each entity type below, list every distinct value that appears in the context. "
    "If a value is not present in the context, leave its list empty. Do not invent values "
    "that are not supported by the context."
)


def build_context(store: DocumentStore, file_name: str, entities: dict[str, str], k: int = 5) -> str:
    query = " ".join(entities.values())
    chunks = store.query(query, file_name=file_name, k=k)
    return "\n\n".join(chunks)


def build_extract_graph(store: DocumentStore, entities: dict[str, str], model: str | None = None):
    entity_model, field_to_entity = build_entity_model(entities)
    entity_descriptions = "\n".join(f"- {name}: {desc}" for name, desc in entities.items())

    def retrieve(state: ExtractState) -> ExtractState:
        return {"context": build_context(store, state["file_name"], entities)}

    def extract(state: ExtractState) -> ExtractState:
        prompt = (
            f"{EXTRACTION_INSTRUCTIONS}\n\nEntity types:\n{entity_descriptions}\n\n"
            f"Context:\n{state['context']}"
        )
        extracted = run_claude_structured(prompt, entity_model, model=model)
        result = {entity_name: getattr(extracted, field) for field, entity_name in field_to_entity.items()}
        return {"result": result}

    graph = StateGraph(ExtractState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("extract", extract)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", END)
    return graph.compile()


def extract_entities_for_file(
    store: DocumentStore,
    file_name: str,
    entities: dict[str, str],
    model: str | None = None,
) -> dict[str, list[str]]:
    graph = build_extract_graph(store, entities, model=model)
    final_state = graph.invoke({"file_name": file_name})
    return final_state["result"]


def extract_entities(
    store: DocumentStore,
    entities: dict[str, str],
    model: str | None = None,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, str]]:
    """Extracts entities for every stored file, run concurrently across files. A failure on
    one file is isolated: it's reported in the returned errors dict instead of aborting
    extraction for the rest."""
    file_names = store.list_files()
    if not file_names:
        return {}, {}

    results: dict[str, dict[str, list[str]]] = {}
    errors: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(file_names))) as executor:
        future_to_file = {
            executor.submit(extract_entities_for_file, store, file_name, entities, model=model): file_name
            for file_name in file_names
        }
        for future in concurrent.futures.as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                results[file_name] = future.result()
            except ClaudeCLIError as exc:
                errors[file_name] = str(exc)
    return results, errors
