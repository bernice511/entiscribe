from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from src.ingestion.vector_store import DocumentStore

_STATE_KEY = "ner_app_state"


@dataclass
class SessionState:
    store: DocumentStore
    entities: dict[str, str] = field(default_factory=dict)
    model_alias: str = "sonnet"
    extraction_results: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    evaluation_report: list[dict] = field(default_factory=list)
    gold_df: pd.DataFrame | None = None


def get_state() -> SessionState:
    if _STATE_KEY not in st.session_state:
        st.session_state[_STATE_KEY] = SessionState(store=DocumentStore())
    return st.session_state[_STATE_KEY]
