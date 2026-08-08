/* ─────────────────────────────────────────────────────────────
   DYSARTHRIA SPEECH AID — app.js
   Handles: mic recording, waveform, API calls, history, UI state
───────────────────────────────────────────────────────────── */

// Resolve API URL: Priority -> localStorage > window.VOICEAID_API_URL > localhost
let API_BASE = localStorage.getItem("voiceaid_api_url") || window.VOICEAID_API_URL || "http://localhost:8001";


// ── DOM References ─────────────────────────────────────────────
const recordBtn       = document.getElementById("record-btn");
const recordBtnIcon   = document.getElementById("record-btn-icon");
const recordBtnLabel  = document.getElementById("record-btn-label");
const recordHint      = document.getElementById("record-hint");
const recordingMeta   = document.getElementById("recording-meta");
const recordTimer     = document.getElementById("record-timer");
const levelBar        = document.getElementById("level-bar");
const waveformCanvas  = document.getElementById("waveform-canvas");
const waveformIdle    = document.getElementById("waveform-idle");
const resultLoading   = document.getElementById("result-loading");
const resultEmpty     = document.getElementById("result-empty");
const resultTextWrap  = document.getElementById("result-text-wrap");
const resultText      = document.getElementById("result-text");
const resultMeta      = document.getElementById("result-meta");
const resultActions   = document.getElementById("result-actions");
const segmentsWrap    = document.getElementById("segments-wrap");
const segmentsList    = document.getElementById("segments-list");
const metaLang        = document.getElementById("meta-lang");
const metaTime        = document.getElementById("meta-time");
const metaWords       = document.getElementById("meta-words");
const copyBtn         = document.getElementById("copy-btn");
const speakBtn        = document.getElementById("speak-btn");
const historySection  = document.getElementById("history-section");
const historyList     = document.getElementById("history-list");
const clearHistoryBtn = document.getElementById("clear-history-btn");
const toast           = document.getElementById("toast");
const statusDot       = document.getElementById("status-dot");
const statusText      = document.getElementById("status-text");

// ── State ──────────────────────────────────────────────────────
let mediaRecorder    = null;
let audioChunks      = [];
let audioContext     = null;
let analyser         = null;
let animationId      = null;
let timerInterval    = null;
let recordingSeconds = 0;
let isRecording      = false;
let sessionHistory   = [];

const ctx = waveformCanvas.getContext("2d");

// ── Health Check with Render Cold-Start Resilience ────────────
let healthCheckTimer = null;

async function checkHealth() {
  statusText.textContent = "Connecting...";
  try {
    // 30-second timeout for Render free-tier cold starts
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(30000) });
    if (res.ok) {
      statusDot.classList.add("online");
      statusText.textContent = "API Online";
    } else {
      throw new Error("Non-OK response");
    }
  } catch {
    statusDot.classList.remove("online");
    statusText.textContent = "API Offline";
  }
}

// ── Timer ──────────────────────────────────────────────────────
function startTimer() {
  recordingSeconds = 0;
  recordTimer.textContent = "00:00";
  timerInterval = setInterval(() => {
    recordingSeconds++;
    const m = String(Math.floor(recordingSeconds / 60)).padStart(2, "0");
    const s = String(recordingSeconds % 60).padStart(2, "0");
    recordTimer.textContent = `${m}:${s}`;
  }, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
}

// ── Waveform Visualizer ─────────────────────────────────────────
function drawWaveform() {
  if (!analyser) return;
  const bufferLength = analyser.fftSize;
  const dataArray    = new Uint8Array(bufferLength);

  function draw() {
    animationId = requestAnimationFrame(draw);
    analyser.getByteTimeDomainData(dataArray);

    const W = waveformCanvas.width;
    const H = waveformCanvas.height;

    ctx.clearRect(0, 0, W, H);

    // Gradient stroke
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0,   "#7c3aed");
    grad.addColorStop(0.5, "#06b6d4");
    grad.addColorStop(1,   "#10b981");

    ctx.lineWidth   = 2.5;
    ctx.strokeStyle = grad;
    ctx.shadowBlur  = 12;
    ctx.shadowColor = "#7c3aed";
    ctx.beginPath();

    const sliceWidth = W / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * H) / 2;
      if (i === 0) ctx.moveTo(x, y);
      else         ctx.lineTo(x, y);
      x += sliceWidth;
    }

    ctx.lineTo(W, H / 2);
    ctx.stroke();

    // Level bar
    let sum = 0;
    for (let i = 0; i < bufferLength; i++) sum += Math.abs(dataArray[i] - 128);
    const avg = sum / bufferLength;
    const pct = Math.min(100, (avg / 50) * 100);
    levelBar.style.width = `${pct}%`;
  }

  draw();
}

function stopWaveform() {
  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
  ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
  levelBar.style.width = "0%";
}

// ── Recording: Start ───────────────────────────────────────────
async function startRecording() {
  if (isRecording) return;

  // Request mic permission
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    showToast("❌ Microphone access denied. Please allow it in browser settings.", "error");
    return;
  }

  isRecording = true;
  audioChunks = [];

  // UI → recording state
  recordBtn.classList.add("recording");
  recordBtn.setAttribute("aria-pressed", "true");
  recordBtnIcon.textContent = "⏹️";
  recordBtnLabel.textContent = "Recording...";
  recordHint.textContent = "Release to transcribe";
  recordingMeta.style.display = "flex";
  waveformIdle.style.display  = "none";
  waveformCanvas.style.display = "block";

  startTimer();

  // Web Audio analyser
  audioContext = new AudioContext();
  const source  = audioContext.createMediaStreamSource(stream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  source.connect(analyser);
  drawWaveform();

  // MediaRecorder
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
  mediaRecorder.onstop = () => {
    stream.getTracks().forEach(t => t.stop());
    const blob = new Blob(audioChunks, { type: "audio/webm" });
    sendToAPI(blob);
  };

  mediaRecorder.start();
}

// ── Recording: Stop ────────────────────────────────────────────
function stopRecording() {
  if (!isRecording || !mediaRecorder) return;
  isRecording = false;

  mediaRecorder.stop();
  stopTimer();
  stopWaveform();

  if (audioContext) {
    audioContext.close();
    audioContext = null;
    analyser = null;
  }

  // UI → idle
  recordBtn.classList.remove("recording");
  recordBtn.setAttribute("aria-pressed", "false");
  recordBtnIcon.textContent  = "🎤";
  recordBtnLabel.textContent = "Hold to Speak";
  recordHint.textContent     = "Hold the button while you speak, then release to transcribe";
  recordingMeta.style.display = "none";
  waveformCanvas.style.display = "none";
  waveformIdle.style.display   = "flex";
}

// ── API Call ───────────────────────────────────────────────────
async function sendToAPI(blob) {
  // Show loading
  resultEmpty.style.display    = "none";
  resultTextWrap.style.display = "none";
  resultMeta.style.display     = "none";
  resultActions.style.display  = "none";
  segmentsWrap.style.display   = "none";
  resultLoading.style.display  = "flex";

  try {
    // Convert WebM blob → WAV (no ffmpeg needed on server!)
    const wavBlob = await convertToWav(blob);

    const formData = new FormData();
    formData.append("audio", wavBlob, "recording.wav");

    const res = await fetch(`${API_BASE}/transcribe`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Transcription failed");
    }

    const data = await res.json();
    displayResult(data);

  } catch (err) {
    resultLoading.style.display = "none";
    resultEmpty.style.display   = "flex";
    showToast(`❌ ${err.message}`, "error");
  }
}

// ── Convert any audio blob → WAV using AudioContext (no ffmpeg!) ──
async function convertToWav(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const audioCtx    = new AudioContext({ sampleRate: 16000 });
  const decoded     = await audioCtx.decodeAudioData(arrayBuffer);
  await audioCtx.close();

  // Mix down to mono
  const numChannels = decoded.numberOfChannels;
  const length      = decoded.length;
  const mono        = new Float32Array(length);

  for (let c = 0; c < numChannels; c++) {
    const channel = decoded.getChannelData(c);
    for (let i = 0; i < length; i++) mono[i] += channel[i] / numChannels;
  }

  // Encode as 16-bit PCM WAV
  const wavBuffer = encodeWav(mono, 16000);
  return new Blob([wavBuffer], { type: "audio/wav" });
}

// ── WAV encoder (pure JS, no libraries) ───────────────────────
function encodeWav(samples, sampleRate) {
  const buffer     = new ArrayBuffer(44 + samples.length * 2);
  const view       = new DataView(buffer);
  const writeStr   = (offset, str) => { for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i)); };

  writeStr(0,  "RIFF");
  view.setUint32(4,  36 + samples.length * 2, true);
  writeStr(8,  "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);              // chunk size
  view.setUint16(20, 1,  true);              // PCM format
  view.setUint16(22, 1,  true);              // mono
  view.setUint32(24, sampleRate, true);      // sample rate
  view.setUint32(28, sampleRate * 2, true);  // byte rate
  view.setUint16(32, 2,  true);              // block align
  view.setUint16(34, 16, true);              // bits per sample
  writeStr(36, "data");
  view.setUint32(40, samples.length * 2, true);

  // Convert float32 → int16
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return buffer;
}

// ── Display Result ─────────────────────────────────────────────
function displayResult(data) {
  resultLoading.style.display = "none";

  const text = data.text || "(No speech detected)";

  // Main text
  resultText.textContent = text;
  resultTextWrap.style.display = "block";

  // Meta
  const langMap = { en:"🇬🇧 English", hi:"🇮🇳 Hindi", unknown:"🌐 Unknown" };
  metaLang.textContent  = langMap[data.language] || `🌐 ${data.language}`;
  metaTime.textContent  = `⏱ ${data.duration_seconds}s`;
  metaWords.textContent = `📊 ${data.word_count} words`;
  resultMeta.style.display = "flex";

  // Actions
  resultActions.style.display = "flex";

  // Segments
  if (data.segments && data.segments.length > 0) {
    segmentsList.innerHTML = "";
    data.segments.forEach(seg => {
      const item = document.createElement("div");
      item.className = "segment-item";
      const confPct = Math.round(seg.confidence * 100);
      item.innerHTML = `
        <span class="segment-time">${seg.start}s → ${seg.end}s</span>
        <span class="segment-text">${seg.text}</span>
        <span class="segment-conf">${confPct}%</span>
      `;
      segmentsList.appendChild(item);
    });
    segmentsWrap.style.display = "block";
  }

  // Add to history
  addToHistory(text);
  showToast("✅ Transcription complete!", "success");
}

// ── History ────────────────────────────────────────────────────
function addToHistory(text) {
  const now    = new Date();
  const timeStr = now.toLocaleTimeString("en-IN", { hour:"2-digit", minute:"2-digit" });

  sessionHistory.unshift({ time: timeStr, text });

  const item = document.createElement("div");
  item.className = "history-item";
  item.innerHTML = `
    <span class="history-time">${timeStr}</span>
    <span class="history-text">${text}</span>
  `;
  historyList.prepend(item);
  historySection.style.display = "block";
}

clearHistoryBtn.addEventListener("click", () => {
  historyList.innerHTML = "";
  sessionHistory = [];
  historySection.style.display = "none";
  showToast("🗑️ History cleared");
});

// ── Copy to Clipboard ──────────────────────────────────────────
copyBtn.addEventListener("click", async () => {
  const text = resultText.textContent;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    showToast("📋 Copied to clipboard!", "success");
    copyBtn.textContent = "✅";
    setTimeout(() => { copyBtn.textContent = "📋"; }, 2000);
  } catch {
    showToast("❌ Copy failed", "error");
  }
});

// ── Text-to-Speech ─────────────────────────────────────────────
speakBtn.addEventListener("click", () => {
  const text = resultText.textContent;
  if (!text || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 0.9;
  window.speechSynthesis.speak(utter);
  showToast("🔊 Reading aloud...");
});

// ── Toast ──────────────────────────────────────────────────────
let toastTimer = null;
function showToast(msg, type = "") {
  toast.textContent = msg;
  toast.className = `toast ${type} show`;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.classList.remove("show"); }, 3000);
}

// ── Record Button Events ───────────────────────────────────────
// Mouse
recordBtn.addEventListener("mousedown",  startRecording);
recordBtn.addEventListener("mouseup",    stopRecording);
recordBtn.addEventListener("mouseleave", () => { if (isRecording) stopRecording(); });

// Touch (mobile)
recordBtn.addEventListener("touchstart", (e) => { e.preventDefault(); startRecording(); });
recordBtn.addEventListener("touchend",   (e) => { e.preventDefault(); stopRecording(); });

// Keyboard accessibility (Space to toggle)
recordBtn.addEventListener("keydown", (e) => {
  if (e.code === "Space") { e.preventDefault(); if (!isRecording) startRecording(); }
});
recordBtn.addEventListener("keyup", (e) => {
  if (e.code === "Space") { e.preventDefault(); stopRecording(); }
});

// ── Smooth Nav Active Link ─────────────────────────────────────
const sections = document.querySelectorAll("section[id]");
const navLinks  = document.querySelectorAll(".nav-link");

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      navLinks.forEach(link => link.classList.remove("active"));
      const active = document.querySelector(`.nav-link[href="#${entry.target.id}"]`);
      if (active) active.classList.add("active");
    }
  });
}, { threshold: 0.4 });

sections.forEach(sec => observer.observe(sec));

// ── Animate WER Bars on Scroll ─────────────────────────────────
const werBars = document.querySelectorAll(".wer-bar");
const werObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      // Width is already set inline, CSS transition will fire on visibility
      entry.target.style.opacity = "1";
    }
  });
}, { threshold: 0.3 });

werBars.forEach(bar => {
  bar.style.width = "0%";
  bar.style.opacity = "0";
  werObserver.observe(bar);
});

// Trigger on page load (if already visible)
setTimeout(() => {
  werBars.forEach(bar => {
    const w = bar.getAttribute("style").match(/width:(\d+)%/);
    const finalWidth = bar.classList.contains("bad") ? "78%"
                     : bar.classList.contains("good") ? "38%"
                     : "22%";
    bar.style.width   = finalWidth;
    bar.style.opacity = "1";
    bar.style.transition = "width 1.5s cubic-bezier(0.4,0,0.2,1), opacity 0.5s";
  });
}, 500);

// ── Init ───────────────────────────────────────────────────────
checkHealth();
setInterval(checkHealth, 30000); // Re-check every 30s
