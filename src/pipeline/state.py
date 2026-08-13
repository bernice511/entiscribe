from typing import TypedDict

import pandas as pd


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
    extraction_results: dict[str, dict[str, list[str]]]
    gold_standard: pd.DataFrame | None
    metrics: dict
    groundedness: list[dict]
    report: list[dict]
