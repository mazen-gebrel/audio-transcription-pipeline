import streamlit as st
import whisper
import os
import tempfile
import pandas as pd

# --- Page Configuration ---
st.set_page_config(page_title="AI Audio Transcriber", page_icon="🎙️", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    .stAudio { margin-top: 1rem; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# --- Dynamic Model Loader ---
# We pass the model_size as an argument so Streamlit caches each size separately
@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size):
    return whisper.load_model(model_size)

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("⚙️ Engine Settings")
    st.markdown("Adjust the AI parameters below to balance speed and accuracy.")
    
    st.markdown("### 1. Model Size")
    model_choice = st.selectbox(
        "Select Whisper Model:", 
        ["base", "small", "medium", "large"],
        index=0,
        help="Base is fast but inaccurate for non-English. Medium/Large are highly accurate for Arabic but require more RAM and take longer."
    )
    
    st.markdown("### 2. Language Override")
    lang_choice = st.selectbox(
        "Force Language Detection:",
        ["Auto-Detect", "Arabic", "English", "French", "Spanish"],
        help="If the AI is hallucinating or translating poorly, forcing the language helps it lock onto the correct vocabulary."
    )

st.title("🎙️ Multilingual Audio Transcription")
st.markdown("Upload an audio file. Use the sidebar to increase model size for complex languages like Arabic.")

# Load model based on sidebar selection
with st.spinner(f"Loading '{model_choice}' model into memory..."):
    model = load_whisper_model(model_choice)

# --- File Uploader ---
st.divider()
uploaded_file = st.file_uploader("Upload an Audio File", type=["mp3", "wav", "m4a", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    # Task selection
    task = st.radio("Select Task:", ["Transcribe (Keep Original Language)", "Translate (Convert to English)"])
    
    if st.button("🚀 Process Audio", type="primary"):
        with st.spinner(f"Analyzing audio with '{model_choice}' model... this may take a while for large models."):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # --- Advanced Transcription Logic ---
                # Set up the arguments dictionary based on user sidebar choices
                transcribe_args = {"fp16": False} # fp16=False prevents warnings on machines without dedicated GPUs
                
                if task == "Translate (Convert to English)":
                    transcribe_args["task"] = "translate"
                    
                # Map the UI language choice to Whisper's internal language codes
                lang_map = {"Arabic": "ar", "English": "en", "French": "fr", "Spanish": "es"}
                if lang_choice != "Auto-Detect":
                    transcribe_args["language"] = lang_map[lang_choice]

                # Run the model with the dynamic arguments
                result = model.transcribe(tmp_file_path, **transcribe_args)
                
                # --- Display Results ---
                st.success("✅ Processing Complete!")
                
                detected_lang = lang_choice if lang_choice != "Auto-Detect" else result.get("language", "unknown").upper()
                st.markdown(f"**Language Processed As:** `{detected_lang}`")
                
                st.markdown("### 📝 Final Text")
                st.info(result["text"])
                
                if "segments" in result:
                    st.markdown("### ⏱️ Timestamp Breakdown")
                    df = pd.DataFrame([
                        {"Start": f"{s['start']:.2f}s", "End": f"{s['end']:.2f}s", "Text": s['text']} 
                        for s in result["segments"]
                    ])
                    st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")
            
            finally:
                os.remove(tmp_file_path)