from pydantic import BaseModel, Field

from src.llm.claude_cli import run_claude_structured

GROUNDEDNESS_INSTRUCTIONS = (
    "You are checking whether an extracted value is actually supported by a document excerpt. "
    "Score how well-grounded the value is in the context, from 0.0 (not supported at all, "
    "likely hallucinated) to 1.0 (clearly and directly supported)."
)


class GroundednessScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Groundedness score from 0.0 to 1.0")
    reasoning: str = Field(description="One short sentence explaining the score")


def score_groundedness(
    entity_name: str, value: str, context: str, model: str | None = None
) -> GroundednessScore:
    prompt = (
        f"{GROUNDEDNESS_INSTRUCTIONS}\n\n"
        f"Entity type: {entity_name}\nExtracted value: {value}\n\nContext:\n{context}"
    )
    return run_claude_structured(prompt, GroundednessScore, model=model)


def score_extraction_results(
    extraction_results: dict[str, dict[str, list[str]]],
    contexts: dict[str, str],
    model: str | None = None,
) -> list[dict]:
    rows = []
    for file_name, entities in extraction_results.items():
        context = contexts.get(file_name, "")
        for entity_name, values in entities.items():
            for value in values:
                result = score_groundedness(entity_name, value, context, model=model)
                rows.append(
                    {
                        "file_name": file_name,
                        "entity_name": entity_name,
                        "value": value,
                        "groundedness_score": result.score,
                        "groundedness_reasoning": result.reasoning,
                    }
                )
    return rows
