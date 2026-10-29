# Speech Analytics Workbench

A beginner-friendly, local-first speech analytics project for learning and portfolio demonstrations.

## What it does
- Upload an audio recording and transcribe speech with Whisper via `faster-whisper`.
- Analyze a pasted or generated transcript.
- Run a transparent baseline sentiment score (positive/negative keyword counts).
- Extract frequent keywords and generate a simple extractive summary.
- Explore and download an included sample call-transcript dataset.
- View basic call metrics, sentiment distribution, and call-intent distribution.

## Dataset
`data/sample_call_transcripts.csv` contains 30 **synthetic** call records. It includes:
- `call_id`: synthetic call identifier
- `date`: example call date
- `agent_id`: synthetic agent label
- `duration_seconds`: call duration
- `transcript`: sample conversation transcript
- `sentiment`: illustrative label
- `intent`: call reason
- `resolved`: whether the example call is marked resolved
- `csat_score`: illustrative 1–5 score

The data is fictional and must not be represented as real customer data. The labels are illustrative and are not ground-truth model annotations.

## Requirements
- Windows 10/11
- Python 3.11–3.13 recommended
- Internet access for installing packages and the first Whisper model download
- FFmpeg is recommended for broad audio-format support. If a format fails, convert it to WAV first.

## Setup on Windows PowerShell

Open PowerShell in this project folder.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.13 is not installed, use an installed supported Python version, for example:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once for the current user, then reopen PowerShell:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Run the application

```powershell
streamlit run app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`. Open it in your browser.

## First audio run
Choose `tiny` model first. The Whisper model is downloaded the first time and cached locally. Larger models can improve transcription quality but need more time, RAM, and compute. CPU transcription can be slow.

## Project architecture

1. Audio upload: Streamlit accepts a recording.
2. Temporary file: the uploaded bytes are written to a temporary file.
3. ASR (automatic speech recognition): `faster-whisper` converts audio to text.
4. Text processing: normalizes words and extracts frequent terms.
5. Baseline sentiment: compares positive and negative word counts.
6. Summary: selects representative sentences using keyword overlap.
7. Dashboard: displays transcript, sentiment, keywords, sample data, and aggregate metrics.

## Important limitations
- Sentiment is a transparent keyword baseline, not a trained or validated sentiment model. Sarcasm, negation, accents, domain language, and mixed sentiment can be misclassified.
- The summary is extractive and heuristic; it is not an LLM-generated factual summary.
- Speaker diarization, emotion recognition, interruption detection, compliance scoring, and agent/customer separation are not included in this starter version.
- Do not use the output for employee evaluation, customer-impacting decisions, or compliance claims without validation, consent, privacy review, and human oversight.
- Only process recordings you have permission and a lawful basis to use. Follow applicable call-recording and data-protection requirements.

## Suggested next improvements
1. Add diarization with a suitable model and evaluate speaker attribution.
2. Replace keyword sentiment with a validated model and report precision/recall/F1 on held-out labelled data.
3. Add word-level timestamps and searchable transcript segments.
4. Add issue taxonomy, call resolution, and CSAT analysis.
5. Add a database, user authentication, and role-based access control before multi-user deployment.
