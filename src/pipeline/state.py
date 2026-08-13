from typing import TypedDict


class IngestState(TypedDict, total=False):
    file_name: str
    file_path: str
    text: str
    chunks: list[str]


class ExtractState(TypedDict, total=False):
    file_name: str
    entities: dict[str, str]
    context: str
    result: dict


class EvaluateState(TypedDict, total=False):
    extraction_results: list[dict]
    gold_standard: list[dict]
    metrics: dict
    groundedness: list[dict]
    report: list[dict]
