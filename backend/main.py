import io
import os
import tempfile
import time
import numpy as np
import soundfile as sf
import whisper
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Dysarthria Speech Aid API",
    description="ASR backend using OpenAI Whisper for speech-impaired users",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Whisper Model (loads once at startup) ─────────────────────────────────
# Use "tiny" for fast demo (~39MB). Upgrade to "small" or "medium" for better accuracy.
MODEL_SIZE = "tiny"   # Options: tiny | base | small | medium | large
print(f"⏳ Loading Whisper model ({MODEL_SIZE})...")
model = whisper.load_model(MODEL_SIZE)
print("✅ Whisper model loaded!")


# ── Health Check ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "whisper-small",
        "description": "Dysarthria Speech Aid API is running",
    }


# ── Transcription Endpoint ─────────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Accepts an audio file (WAV/MP3/WEBM/OGG) and returns the Whisper transcription.
    """
    # Read raw audio bytes
    contents = await audio.read()

    try:
        start = time.time()

        # Load WAV audio into numpy array (no ffmpeg needed!)
        audio_buffer = io.BytesIO(contents)
        audio_array, sample_rate = sf.read(audio_buffer, dtype="float32")

        # Convert stereo to mono if needed
        if audio_array.ndim == 2:
            audio_array = audio_array.mean(axis=1)

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            import scipy.signal as signal
            num_samples = int(len(audio_array) * 16000 / sample_rate)
            audio_array = signal.resample(audio_array, num_samples)

        # Run Whisper inference directly on numpy array (no ffmpeg!)
        result = model.transcribe(
            audio_array,
            task="transcribe",
            language=None,
            fp16=False,
            verbose=False,
        )

        elapsed = round(time.time() - start, 2)
        text = result.get("text", "").strip()
        language = result.get("language", "unknown")

        # Build segments summary
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip(),
                "confidence": round(1 - seg.get("no_speech_prob", 0), 3),
            })

        return JSONResponse({
            "success": True,
            "text": text,
            "language": language,
            "duration_seconds": elapsed,
            "segments": segments,
            "word_count": len(text.split()) if text else 0,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
