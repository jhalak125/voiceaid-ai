import io
import os
import time
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Dysarthria Speech Aid API",
    description="ASR backend using Faster-Whisper for speech-impaired users",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Faster-Whisper Model ──────────────────────────────────────────────────
# faster-whisper: no torch, no ffmpeg, no pkg_resources issues!
# Uses CTranslate2 under the hood — 4x faster, 2x less RAM than openai-whisper
MODEL_SIZE = os.getenv("MODEL_SIZE", "tiny")   # tiny=~75MB RAM | small=~240MB
print(f"⏳ Loading Faster-Whisper model ({MODEL_SIZE})...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("✅ Faster-Whisper model loaded!")


# ── Health Check ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": f"faster-whisper-{MODEL_SIZE}",
        "description": "Dysarthria Speech Aid API is running",
    }


# ── Transcription Endpoint ─────────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Accepts a WAV audio file and returns the Whisper transcription.
    No ffmpeg needed — browser sends 16kHz mono WAV.
    """
    contents = await audio.read()

    try:
        start = time.time()

        # Load WAV → numpy array (no ffmpeg!)
        audio_buffer = io.BytesIO(contents)
        audio_array, sample_rate = sf.read(audio_buffer, dtype="float32")

        # Stereo → mono
        if audio_array.ndim == 2:
            audio_array = audio_array.mean(axis=1)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import scipy.signal as signal
            num_samples = int(len(audio_array) * 16000 / sample_rate)
            audio_array = signal.resample(audio_array, num_samples)

        # Save to temp WAV for faster-whisper (it needs a file path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import soundfile as sf2
            sf2.write(tmp.name, audio_array, 16000)
            tmp_path = tmp.name

        # Run faster-whisper inference
        segments_gen, info = model.transcribe(
            tmp_path,
            task="transcribe",
            language=None,       # auto-detect
            beam_size=5,
            vad_filter=True,     # skip silent parts
        )

        # Clean up temp file
        os.remove(tmp_path)

        # Collect results
        elapsed = round(time.time() - start, 2)
        segments = []
        full_text = []

        for seg in segments_gen:
            full_text.append(seg.text.strip())
            segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "confidence": round(seg.avg_logprob + 1, 3),  # normalize logprob
            })

        text = " ".join(full_text).strip()
        language = info.language if info else "unknown"

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
