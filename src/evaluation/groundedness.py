import concurrent.futures

from pydantic import BaseModel, Field

from src.llm.claude_cli import ClaudeCLIError, run_claude_structured

GROUNDEDNESS_INSTRUCTIONS = (
    "You are checking whether extracted values are actually supported by a document excerpt. "
    "Score how well-grounded each value is in the context, from 0.0 (not supported at all, "
    "likely hallucinated) to 1.0 (clearly and directly supported). Score every item below "
    "independently and return exactly one entry per index."
)


class GroundednessItem(BaseModel):
    index: int = Field(description="Index of the item being scored, matching the numbered input list")
    score: float = Field(ge=0.0, le=1.0, description="Groundedness score from 0.0 to 1.0")
    reasoning: str = Field(description="One short sentence explaining the score")


class GroundednessBatch(BaseModel):
    items: list[GroundednessItem] = Field(description="One entry per input item, covering every index")


def score_groundedness_batch(
    items: list[tuple[str, str]], context: str, model: str | None = None
) -> dict[int, GroundednessItem]:
    numbered = "\n".join(
        f"{i}. entity type: {entity_name} | value: {value}" for i, (entity_name, value) in enumerate(items)
    )
    prompt = f"{GROUNDEDNESS_INSTRUCTIONS}\n\nItems:\n{numbered}\n\nContext:\n{context}"
    batch = run_claude_structured(prompt, GroundednessBatch, model=model)
    return {item.index: item for item in batch.items}


def _score_file(file_name: str, entities: dict[str, list[str]], context: str, model: str | None) -> list[dict]:
    items = [(entity_name, value) for entity_name, values in entities.items() for value in values]
    if not items:
        return []

    batch_error = None
    scored: dict[int, GroundednessItem] = {}
    try:
        scored = score_groundedness_batch(items, context, model=model)
    except ClaudeCLIError as exc:
        batch_error = str(exc)

    rows = []
    for i, (entity_name, value) in enumerate(items):
        result = scored.get(i)
        if result is not None:
            score, reasoning = result.score, result.reasoning
        else:
            score, reasoning = None, batch_error or "Model did not return a score for this item."
        rows.append(
            {
                "file_name": file_name,
                "entity_name": entity_name,
                "value": value,
                "groundedness_score": score,
                "groundedness_reasoning": reasoning,
            }
        )
    return rows


def score_extraction_results(
    extraction_results: dict[str, dict[str, list[str]]],
    contexts: dict[str, str],
    model: str | None = None,
) -> list[dict]:
    """Scores every extracted value's groundedness, one batched LLM call per file (run
    concurrently across files) instead of one call per value."""
    if not extraction_results:
        return []

    rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(extraction_results))) as executor:
        futures = [
            executor.submit(_score_file, file_name, entities, contexts.get(file_name, ""), model)
            for file_name, entities in extraction_results.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            rows.extend(future.result())
    return rows
