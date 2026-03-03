import streamlit as st
from ui.text_ui import render_text_ui
from ui.youtube_ui import render_youtube_ui

st.set_page_config(
    page_title="Cognitive Load Estimator",
    layout="wide"
)

st.title("Cognitive Load Heatmap")

mode = st.sidebar.radio(
    "Select Mode",
    ["Text Heatmap", "YouTube Timeline Heatmap"]
)

if mode == "Text Heatmap":
    render_text_ui()

elif mode == "YouTube Timeline Heatmap":
    render_youtube_ui()
