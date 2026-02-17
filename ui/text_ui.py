import streamlit as st
from scripts.pipeline import analyze_text
from visualization.heatmap import generate_text_heatmap

def render_text_ui():

    st.header("Text Cognitive Load Heatmap")

    text_input = st.text_area(
        "Paste your text here:",
        height=250
    )

    if st.button("Analyze Text"):

        if not text_input.strip():
            st.warning("Please enter text.")
            return

        with st.spinner("Analyzing cognitive load..."):
            results = analyze_text(text_input)
            html_output = generate_text_heatmap(results)

        st.markdown("### Heatmap Result")
        st.components.v1.html(
            html_output,
            height=500,
            scrolling=True
        )
