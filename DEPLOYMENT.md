# Deployment Guide — VoiceAid AI

## Architecture Overview
```
Frontend (GitHub Pages / Vercel) ──→ Backend (Hugging Face Spaces / Railway)
     static HTML/CSS/JS                  FastAPI + Whisper model
```

---

## Option A: Hugging Face Spaces (Recommended for ML hackathon — FREE)

### Deploy Backend to HF Spaces

1. Go to https://huggingface.co/new-space
2. SDK: Docker | Visibility: Public | Name: voiceaid-ai-backend

3. Push backend folder:
```bash
cd backend/
git init
git add .
git commit -m "VoiceAid backend"
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/voiceaid-ai-backend
git push space main
```

4. Your API will be live at:
   https://YOUR_HF_USERNAME-voiceaid-ai-backend.hf.space

### Deploy Frontend to GitHub Pages

1. In your GitHub repo → Settings → Pages
2. Source: main branch, /frontend folder
3. Your frontend: https://YOUR_USERNAME.github.io/voiceaid-ai/frontend/

4. Update API_BASE in frontend/app.js:
```js
const API_BASE = "https://YOUR_HF_USERNAME-voiceaid-ai-backend.hf.space";
```

---

## Option B: Railway (Easiest full-stack — FREE tier)

1. Go to https://railway.app
2. New Project → Deploy from GitHub repo
3. Select your voiceaid-ai repo
4. Set root directory to: backend/
5. Add env variable: PORT=8001
6. Railway auto-detects Python and deploys!
7. Get your URL: https://voiceaid-ai.up.railway.app

---

## Option C: Render.com (Free — may be slow on free tier)

1. Go to https://render.com → New Web Service
2. Connect GitHub repo
3. Root directory: backend/
4. Build command: pip install -r requirements.txt
5. Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
6. Instance: Free (512MB RAM — may OOM with larger Whisper models)
   → Use MODEL_SIZE=tiny for free tier!

---

## Option D: Google Colab + ngrok (Best for live demo day — FREE)

Run this in a Colab notebook for instant public URL:

```python
!pip install fastapi uvicorn openai-whisper soundfile scipy pyngrok

from pyngrok import ngrok
import subprocess, threading

def run():
    subprocess.run(["uvicorn", "main:app", "--port", "8001"])

threading.Thread(target=run).start()

tunnel = ngrok.connect(8001)
print("Public URL:", tunnel.public_url)
# Update this URL in your frontend app.js
```

This gives you a public URL in 30 seconds — perfect for hackathon demo day!

---

## Quick Summary

| Platform | Cost | RAM | Setup Time | Best For |
|----------|------|-----|-----------|----------|
| HF Spaces | Free | 16GB | 10 min | ML projects ✅ |
| Railway | Free tier | 512MB | 5 min | Quick deploy |
| Render | Free tier | 512MB | 5 min | Simple apps |
| Google Colab | Free | 12GB | 2 min | Demo day 🏆 |
| VPS (DigitalOcean) | $6/mo | 1GB+ | 30 min | Production |
