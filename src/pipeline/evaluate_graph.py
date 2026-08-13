import pandas as pd
from langgraph.graph import END, START, StateGraph

from src.evaluation.groundedness import score_extraction_results
from src.evaluation.metrics import compare_to_gold
from src.ingestion.vector_store import DocumentStore
from src.pipeline.extract_graph import build_context
from src.pipeline.state import EvaluateState


def build_evaluate_graph(store: DocumentStore, entities: dict[str, str], model: str | None = None):
    def score_groundedness(state: EvaluateState) -> EvaluateState:
        contexts = {
            file_name: build_context(store, file_name, entities) for file_name in state["extraction_results"]
        }
        groundedness = score_extraction_results(state["extraction_results"], contexts, model=model)
        return {"groundedness": groundedness}

    def compare_gold(state: EvaluateState) -> EvaluateState:
        gold_df = state.get("gold_standard")
        if gold_df is None:
            return {"metrics": {}}
        entity_metrics = compare_to_gold(state["extraction_results"], gold_df)
        return {"metrics": {m.entity_name: m for m in entity_metrics}}

    def merge(state: EvaluateState) -> EvaluateState:
        metrics = state.get("metrics") or {}
        report = []
        for row in state["groundedness"]:
            entity_metric = metrics.get(row["entity_name"])
            report.append(
                {
                    **row,
                    "precision": entity_metric.precision if entity_metric else None,
                    "recall": entity_metric.recall if entity_metric else None,
                    "f1": entity_metric.f1 if entity_metric else None,
                }
            )
        return {"report": report}

    graph = StateGraph(EvaluateState)
    graph.add_node("score_groundedness", score_groundedness)
    graph.add_node("compare_gold", compare_gold)
    graph.add_node("merge", merge)
    graph.add_edge(START, "score_groundedness")
    graph.add_edge("score_groundedness", "compare_gold")
    graph.add_edge("compare_gold", "merge")
    graph.add_edge("merge", END)
    return graph.compile()


def evaluate(
    store: DocumentStore,
    entities: dict[str, str],
    extraction_results: dict[str, dict[str, list[str]]],
    gold_df: pd.DataFrame | None = None,
    model: str | None = None,
) -> list[dict]:
    graph = build_evaluate_graph(store, entities, model=model)
    final_state = graph.invoke({"extraction_results": extraction_results, "gold_standard": gold_df})
    return final_state["report"]
