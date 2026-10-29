import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from speech_analytics import (
    transcribe_audio, analyze_transcript, load_sample_dataset,
    summarize_transcript, extract_keywords
)

st.set_page_config(page_title="Speech Analytics Workbench", page_icon="🎙️", layout="wide")
st.title("🎙️ Speech Analytics Workbench")
st.caption("Local-first demo: transcribe audio, inspect sentiment and topics, and analyze a sample call dataset.")

with st.sidebar:
    st.header("Settings")
    model_size = st.selectbox("Whisper model size", ["tiny", "base", "small"], index=0,
                              help="tiny is fastest; larger models may be more accurate but use more resources.")
    st.info("First audio transcription may download the selected model. For best results, use clear audio.")
    st.markdown("**Privacy note:** Audio is processed on your machine by default. Do not upload sensitive recordings unless you are authorized.")

tab1, tab2, tab3 = st.tabs(["Analyze audio", "Analyze transcript", "Explore sample dataset"])

with tab1:
    st.subheader("Upload a call recording")
    audio = st.file_uploader("Choose audio file", type=["wav", "mp3", "m4a", "ogg", "flac", "mp4"])
    if audio is not None:
        st.audio(audio)
        if st.button("Transcribe and analyze", type="primary"):
            suffix = Path(audio.name).suffix or ".wav"
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio.getbuffer())
                    tmp_path = tmp.name
                with st.spinner("Transcribing audio. The first run may take longer while the model downloads..."):
                    transcript = transcribe_audio(tmp_path, model_size=model_size)
                st.session_state["latest_transcript"] = transcript
                st.success("Transcription complete.")
                st.text_area("Transcript", transcript, height=220)
                result = analyze_transcript(transcript)
                c1, c2, c3 = st.columns(3)
                c1.metric("Sentiment", result["sentiment"])
                c2.metric("Positive words", result["positive_count"])
                c3.metric("Negative words", result["negative_count"])
                st.markdown("**Key phrases**")
                st.write(", ".join(extract_keywords(transcript)) or "No key phrases detected.")
                st.markdown("**AI-generated summary**")
                st.write(summarize_transcript(transcript))
            except Exception as e:
                st.error(f"Could not process this audio: {e}")
                st.markdown("Check the installation steps in README.md. Audio transcription requires FFmpeg and the faster-whisper package.")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

with tab2:
    st.subheader("Analyze text directly")
    default_text = st.session_state.get("latest_transcript", "I have been waiting for three days and nobody has resolved my issue.")
    transcript_input = st.text_area("Paste a transcript", value=default_text, height=180)
    if st.button("Analyze transcript"):
        if not transcript_input.strip():
            st.warning("Enter a transcript first.")
        else:
            result = analyze_transcript(transcript_input)
            c1, c2, c3 = st.columns(3)
            c1.metric("Rule-based sentiment", result["sentiment"])
            c2.metric("Positive word hits", result["positive_count"])
            c3.metric("Negative word hits", result["negative_count"])
            st.write("**Key phrases:**", ", ".join(extract_keywords(transcript_input)) or "None detected")
            st.write("**Summary:**", summarize_transcript(transcript_input))
            st.caption("Sentiment and summary are baseline heuristics, not validated production models.")

with tab3:
    st.subheader("Included sample call-transcript dataset")
    df = load_sample_dataset()
    c1, c2, c3 = st.columns(3)
    c1.metric("Calls", len(df))
    c2.metric("Average duration", f"{df['duration_seconds'].mean():.0f} sec")
    c3.metric("Resolution rate", f"{100 * df['resolved'].mean():.0f}%")
    sentiment_filter = st.multiselect("Filter by sentiment", sorted(df["sentiment"].unique()), default=sorted(df["sentiment"].unique()))
    filtered = df[df["sentiment"].isin(sentiment_filter)]
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"), "speech_analytics_sample.csv", "text/csv")
    st.markdown("**Sentiment distribution**")
    st.bar_chart(filtered["sentiment"].value_counts())
    st.markdown("**Common call intents**")
    st.bar_chart(filtered["intent"].value_counts())
    st.caption("All included records are synthetic examples created for learning; they are not real customer data.")
