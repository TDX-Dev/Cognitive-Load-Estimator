import streamlit as st

def render_youtube_ui():

    st.header("YouTube Cognitive Load Timeline")

    video_url = st.text_input("Paste YouTube URL")

    if st.button("Analyze Video"):

        if not video_url:
            st.warning("Enter a YouTube URL")
            return

        st.info("Transcript + timeline analysis will appear here.")

        # Future:
        # 1. Extract transcript with timestamps
        # 2. Chunk by time windows
        # 3. Run analyze_text on each chunk
        # 4. Plot timeline heatmap
