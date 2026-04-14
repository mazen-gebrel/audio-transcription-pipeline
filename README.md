# 🎙️ Multilingual AI Audio Transcription Pipeline

An end-to-end Data Science application that leverages deep learning to transcribe and translate multilingual audio files. 

## 🎯 Business Value
Unstructured audio data (customer support calls, user interviews, meeting recordings) is notoriously difficult to analyze. This pipeline automates the extraction of structured text from raw audio, enabling downstream NLP tasks like sentiment analysis, topic modeling, and keyword extraction, while seamlessly handling cross-border language translation.

## 🏗️ Architecture & Features

* **The Engine:** Powered by OpenAI's open-source `Whisper` model, running entirely locally.
* **Dynamic Resource Allocation:** The pipeline features a dynamic model loader (Base to Large) allowing users to trade compute time for accuracy depending on the linguistic complexity of the audio.
* **Linguistic Overrides:** Includes forced language routing to heavily improve transcription accuracy on morphologically complex languages (like Arabic) by bypassing auto-detect hallucinations.
* **Interactive UI (`app.py`):** Built with `Streamlit`, featuring a native audio player, caching for the heavy ML weights, and structured Pandas DataFrames for timestamp breakdown.

## ⚙️ The Tech Stack
* **Language:** Python 3.10
* **Deep Learning:** `openai-whisper`, `torch`
* **Data Processing:** `pandas`
* **Frontend UI:** `streamlit`

## 🚀 Running It Locally

**1. Clone the repository:**
```bash
git clone [https://github.com/mazen-gebrel/audio-transcription-pipeline.git](https://github.com/mazen-gebrel/audio-transcription-pipeline.git)
cd audio-transcription-pipeline
```
**2. Install dependencies:**
(Note: Requires ffmpeg installed on your system path).
```Bash
pip install openai-whisper streamlit torch pandas
```
**3. Launch the application:**

```Bash
streamlit run app.py
```
