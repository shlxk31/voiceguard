"""
api/main.py

Minimal FastAPI backend. Two endpoints:
  POST /analyze          - upload a full audio file, get one DetectionResult
  WS   /stream            - stream 16kHz PCM chunks, get a DetectionResult per
                             chunk plus a mid-call alarm event when the
                             CallAccumulator fires

Run with:  uvicorn src.api.main:app --reload

TODO (Week 9):
  1. Wire up `get_detector()` to actually load your trained/pretrained model
     (start with LFCCGMMDetector to get the endpoint working end-to-end,
     swap to AasistLDetector once that's validated).
  2. Add basic input validation (sample rate, mono/stereo, max duration).
  3. Decide your real chunking strategy for the WS endpoint (client sends
     raw PCM frames of a fixed size matching your chosen window_size).
"""

from fastapi import FastAPI, UploadFile, WebSocket
import librosa
import numpy as np
import soundfile as sf
import io
import json
from pathlib import Path

from ..detector.aasist_l import AasistLDetector
from ..streaming.accumulator import CallAccumulator, AccumulatorConfig

app = FastAPI(title="VoiceGuard — Streaming Deepfake Detection")

_detector = None  # lazy-loaded singleton


def get_detector():
    # AasistLDetector is pretrained (clovaai/aasist checkpoint) -- no fitting
    # needed, unlike LFCCGMMDetector which requires .fit() on ASVspoof data.
    global _detector
    if _detector is None:
        _detector = AasistLDetector()
        # Use the threshold frozen on the ASVspoof19 LA dev split, so the demo
        # reports the same operating point the paper's numbers are measured at.
        baseline = Path(__file__).resolve().parent.parent.parent / "results" / "aasist_baseline.json"
        if baseline.exists():
            with open(baseline) as f:
                _detector.threshold = json.load(f)["threshold"]
    return _detector


@app.post("/analyze")
async def analyze(file: UploadFile):
    audio_bytes = await file.read()
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # crude mono downmix; document if you keep this
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    detector = get_detector()
    result = detector.score(audio, sample_rate=sr)
    return {
        "is_spoof": result.is_spoof,
        "score": result.score,
        "confidence": result.confidence,
        "latency_ms": result.latency_ms,
    }


@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    detector = get_detector()
    accumulator = CallAccumulator(AccumulatorConfig(), chunk_duration_ms=500)
    try:
        while True:
            chunk_bytes = await websocket.receive_bytes()
            chunk = np.frombuffer(chunk_bytes, dtype=np.float32)
            result = detector.score(chunk, sample_rate=16000)
            alarm = accumulator.update(result)
            await websocket.send_json({
                "is_spoof": result.is_spoof,
                "score": result.score,
                "latency_ms": result.latency_ms,
                "alarm_raised": alarm,
                "time_to_detection_ms": accumulator.time_to_detection_ms,
            })
    except Exception:
        await websocket.close()
