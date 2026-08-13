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
            state.extraction_results = extract_entities(state.store, state.entities, model=state.model_alias)

    if state.extraction_results:
        rows = [
            {"file_name": file_name, "entity": entity_name, "values": ", ".join(values)}
            for file_name, entities in state.extraction_results.items()
            for entity_name, values in entities.items()
        ]
        st.dataframe(pd.DataFrame(rows))
