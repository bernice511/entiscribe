import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.pipeline.extract_graph import extract_entities
from src.pipeline.ingest_graph import ingest_pdf
from src.session import get_state


def render_upload_tab() -> None:
    state = get_state()

    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("Add to document store"):
        already_stored = set(state.store.list_files())
        added = 0
        for uploaded in uploaded_files:
            if uploaded.name in already_stored:
                continue
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            try:
                ingest_pdf(state.store, file_name=uploaded.name, file_path=tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            added += 1
        st.success(f"Stored {added} new file(s).")

    stored_files = state.store.list_files()
    st.write(f"Documents in store: {len(stored_files)}")
    if stored_files:
        st.write(", ".join(stored_files))

    if not state.entities:
        st.info("Define at least one entity type in the sidebar to run extraction.")
    elif stored_files and st.button("Run extraction"):
        with st.spinner("Extracting entities..."):
            state.extraction_results, state.extraction_errors = extract_entities(
                state.store, state.entities, model=state.model_alias
            )

    _render_results_per_file(state.extraction_results, state.extraction_errors)


def _render_results_per_file(
    extraction_results: dict[str, dict[str, list[str]]],
    extraction_errors: dict[str, str],
) -> None:
    file_names = sorted(set(extraction_results) | set(extraction_errors))
    if not file_names:
        return

    st.subheader("Extraction results")
    for file_name, tab in zip(file_names, st.tabs(file_names)):
        with tab:
            if file_name in extraction_errors:
                st.error(extraction_errors[file_name])
                continue
            rows = [
                {"entity": entity_name, "values": ", ".join(values)}
                for entity_name, values in extraction_results[file_name].items()
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch")
