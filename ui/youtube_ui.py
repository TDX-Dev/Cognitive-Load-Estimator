import streamlit as st
from scripts.youtube_pipeline import analyze_video
from visualization.heatmap1 import generate_text_heatmap, generate_timeline_heatmap

def render_youtube_ui():

    st.header("YouTube Cognitive Load Timeline")

    video_url = st.text_input("Paste YouTube URL")

    if st.button("Analyze Video"):

        if not video_url:
            st.warning("Enter a YouTube URL")
            return

        with st.spinner("Analyzing video transcript..."):

            results = analyze_video(video_url)

            if not results:
                st.error("No transcript available.")
                return

            total_duration = max(r["end"] for r in results)

            timeline_html = generate_timeline_heatmap(results, total_duration)
            transcript_html = generate_text_heatmap(results)

        st.video(video_url)

        st.markdown("### Timeline Cognitive Load")
        st.components.v1.html(timeline_html, height=60)

        st.markdown("### Transcript Heatmap")
        st.components.v1.html(transcript_html, height=500, scrolling=True)