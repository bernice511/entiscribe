import streamlit as st

from src.ui.evaluate_tab import render_evaluate_tab
from src.ui.sidebar import render_sidebar
from src.ui.upload_tab import render_upload_tab

st.set_page_config(page_title="PDF Named Entity Extractor", layout="wide")
st.title("PDF Named Entity Extractor")

render_sidebar()

upload_tab, evaluate_tab = st.tabs(["Upload & Extract", "Evaluate"])
with upload_tab:
    render_upload_tab()
with evaluate_tab:
    render_evaluate_tab()
