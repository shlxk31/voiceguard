# VoiceGuard — Streaming Deepfake/Scam-Call Detection

Detects AI-cloned/synthetic speech in **live, degraded VoIP audio** (as opposed to clean
studio recordings), using a sliding-window classifier + cross-chunk confidence
accumulation to flag a call mid-conversation with a measured latency cost.

## Why this project (one line)
Commercial detectors hit 90%+ accuracy on clean audio but drop to 60–75% on live
compressed VoIP calls. Nobody publishes a streaming (chunked) evaluation with
per-chunk latency AND an accumulation-based mid-call alarm — that's the gap this fills.

## Project phases
1. **Baseline reproduction** — pretrained AASIST-L on ASVspoof19 LA / 21 DF / In-the-Wild
2. **VoIP degradation harness** — Opus/AMR encode-decode + Gilbert-Elliott packet loss
3. **Streaming reformulation** — sliding-window (250ms/500ms/1s) classification + latency measurement
4. **Accumulation rule** — N-of-M window voting, time-to-detection metric
5. **Modern-clone stress test** — F5-TTS/XTTS-v2/Chatterbox generated attacks
6. **FastAPI wrapper** — WebSocket streaming endpoint
7. **Paper write-up** — target: arXiv preprint -> IEEE SLT / SPSC workshop

## Repo layout
- `src/detector/`   — the core, swappable detection model (AASIST-L wrapper, LFCC-GMM baseline)
- `src/voip_sim/`    — codec + packet-loss degradation pipeline
- `src/streaming/`   — sliding-window buffer + accumulation decision rule
- `src/api/`         — FastAPI app (upload + WebSocket streaming endpoints)
- `data/raw/`        — ASVspoof / In-the-Wild (not committed — see data/README)
- `data/degraded/`   — generated VoIP-degraded test sets
- `notebooks/`       — exploration, EER/latency plots
- `tests/`           — unit tests + evaluation scripts
