import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline.evaluate_graph import evaluate
from src.session import get_state


def render_evaluate_tab() -> None:
    state = get_state()

    if not state.extraction_results:
        st.info("Run extraction in the Upload & Extract tab first.")
        return

    gold_file = st.file_uploader(
        "Optional gold-standard CSV (a file_name column plus one column per entity type)", type="csv"
    )
    if gold_file is not None:
        state.gold_df = pd.read_csv(gold_file)

    if st.button("Run evaluation"):
        with st.spinner("Scoring extraction quality..."):
            state.evaluation_report = evaluate(
                state.store,
                state.entities,
                state.extraction_results,
                gold_df=state.gold_df,
                model=state.model_alias,
            )

    if not state.evaluation_report:
        return

    report_df = pd.DataFrame(state.evaluation_report)
    st.dataframe(report_df)

    st.plotly_chart(
        px.bar(
            report_df.groupby("entity_name", as_index=False)["groundedness_score"].mean(),
            x="entity_name",
            y="groundedness_score",
            title="Average groundedness by entity type",
        )
    )

    if report_df["precision"].notna().any():
        gold_summary = report_df.groupby("entity_name", as_index=False)[["precision", "recall", "f1"]].first()
        st.plotly_chart(
            px.bar(
                gold_summary.melt(id_vars="entity_name", var_name="metric", value_name="score"),
                x="entity_name",
                y="score",
                color="metric",
                barmode="group",
                title="Gold-standard precision / recall / F1 by entity type",
            )
        )
