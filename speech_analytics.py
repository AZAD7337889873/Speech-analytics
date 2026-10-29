import re
from pathlib import Path
import pandas as pd

POSITIVE_WORDS = {
    "good", "great", "excellent", "helpful", "thanks", "thank", "resolved", "happy",
    "satisfied", "appreciate", "quick", "friendly", "perfect", "wonderful", "easy"
}
NEGATIVE_WORDS = {
    "bad", "poor", "angry", "upset", "waiting", "late", "broken", "issue", "problem",
    "unresolved", "frustrated", "terrible", "disappointed", "cancel", "charged", "refund",
    "never", "failed", "worst", "confusing"
}

def _tokens(text):
    return re.findall(r"\b[a-zA-Z']+\b", (text or "").lower())

def analyze_transcript(text):
    tokens = _tokens(text)
    pos = sum(token in POSITIVE_WORDS for token in tokens)
    neg = sum(token in NEGATIVE_WORDS for token in tokens)
    if pos > neg:
        sentiment = "Positive"
    elif neg > pos:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return {"sentiment": sentiment, "positive_count": pos, "negative_count": neg}

def extract_keywords(text, top_n=8):
    stop_words = {
        "the", "and", "for", "that", "this", "with", "you", "your", "have", "has",
        "was", "were", "are", "but", "not", "can", "could", "would", "from", "they",
        "them", "then", "there", "here", "just", "very", "into", "about", "been",
        "our", "their", "its", "she", "his", "her", "what", "when", "where", "how",
        "all", "also", "will", "shall", "should", "please", "hello", "yeah", "okay"
    }
    tokens = [t for t in _tokens(text) if len(t) > 3 and t not in stop_words]
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return [word for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]]

def summarize_transcript(text, max_sentences=3):
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return "No transcript available."
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= max_sentences:
        return text
    # Simple extractive baseline: choose sentences containing the most frequent non-stopword terms.
    keywords = set(extract_keywords(text, top_n=10))
    scored = []
    for idx, sentence in enumerate(sentences):
        terms = set(_tokens(sentence))
        scored.append((len(terms & keywords), -idx, sentence))
    selected = sorted(scored, reverse=True)[:max_sentences]
    return " ".join(item[2] for item in sorted(selected, key=lambda x: -x[1]))

def transcribe_audio(audio_path, model_size="tiny"):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is not installed. Run: pip install -r requirements.txt") from exc
    # CPU/int8 is the most accessible default for a Windows laptop.
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, vad_filter=True)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    if not text:
        raise RuntimeError("No speech was detected. Try a clearer recording or another model size.")
    return text

def load_sample_dataset():
    path = Path(__file__).parent / "data" / "sample_call_transcripts.csv"
    return pd.read_csv(path)
