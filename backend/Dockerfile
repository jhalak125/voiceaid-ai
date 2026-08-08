---
title: VoiceAid AI - Dysarthria Speech Aid
emoji: 🎤
colorFrom: purple
colorTo: cyan
sdk: docker
app_port: 7860
---

FROM python:3.9-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Whisper model during build (avoids cold-start delay)
RUN python -c "import whisper; whisper.load_model('small')"

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
