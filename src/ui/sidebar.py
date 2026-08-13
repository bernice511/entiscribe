import streamlit as st

from src.entities.schema import PRESET_ENTITIES
from src.session import get_state

MODEL_ALIASES = ["sonnet", "opus", "haiku"]


def render_sidebar() -> None:
    state = get_state()

    with st.sidebar:
        st.header("Entity types")
        selected_presets = st.multiselect(
            "Quick-pick presets",
            options=list(PRESET_ENTITIES),
            default=[name for name in state.entities if name in PRESET_ENTITIES],
        )
        for name in selected_presets:
            state.entities.setdefault(name, PRESET_ENTITIES[name])
        for name in list(state.entities):
            if name in PRESET_ENTITIES and name not in selected_presets:
                del state.entities[name]

        st.subheader("Custom entities")
        custom_name = st.text_input("Entity name", key="custom_entity_name")
        custom_description = st.text_input("Description", key="custom_entity_description")
        if st.button("Add custom entity") and custom_name:
            state.entities[custom_name] = custom_description or custom_name

        if state.entities:
            st.caption("Active entities: " + ", ".join(state.entities))

        st.divider()
        st.header("Settings")
        state.model_alias = st.selectbox(
            "Claude model", MODEL_ALIASES, index=MODEL_ALIASES.index(state.model_alias)
        )

        st.divider()
        if st.button("Clear all documents"):
            state.store.clear_all()
            state.extraction_results = {}
            state.evaluation_report = []
