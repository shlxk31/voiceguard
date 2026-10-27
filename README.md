# VoiceGuard — Voice Anti-Spoofing for Voice-Authorized Transactions

VoiceGuard detects AI-cloned / synthetic speech in voice-authorized transactions
(order placement, money transfers, account changes). It is built as a **layered
defense**: instead of asking only "is this the enrolled user's voice?" it also
asks "is this a real human, or an AI-generated fake?", "did this come from a
live microphone or a recording?", and "is this request normal for this user?".

Each layer answers a different question, covers a different threat, and produces
a calibrated log-likelihood ratio. A cost-derived decision policy (based on the
t-DCF / Bayes risk framework) fuses the layers into an allow / challenge / deny
verdict that scales with the transaction amount.

---

## What is implemented

| Layer | Question | Model | Threat it catches | Status |
|-------|----------|-------|-------------------|--------|
| **L1 Voiceprint** | Is this the enrolled user's voice? | ECAPA-TDNN (192-d embedding, cosine similarity) | Impostor (different human) | ✅ Implemented |
| **L2 Liveness** | Did this come from a mouth or a loudspeaker? | Air-conduction replay detector (logistic regression on MFCC/spectral features) | Replay of a recording | ⚠️ Implemented, narrow scope (15 pairs, one speaker, one room) |
| **L3 Anti-spoofing** | Is this a real human or AI-generated? | AASIST-L (raw waveform, graph attention) | TTS / voice-conversion clone | ✅ Implemented, benchmarked |
| **L4 Context** | Is this request normal for this user? | Contextual risk scorer (injection + semantic + behavioural) | Stolen device, coerced user | ⚠️ Implemented, evaluated on synthetic corpus |
| **L5 Decision policy** | Allow / challenge / deny? | Cost-derived thresholds (t-DCF / Bayes risk) | — | ✅ Implemented |

Layers 1–4 are evidence sources. Layer 5 is the decision policy that fuses them
using a **weakest-link (conjunctive) rule**: an attacker must defeat every
layer, so adding a layer can only lower the fused score (never open a hole).

---

## Verified results (reproducible)

These numbers come from the evaluation scripts in `tests/` and are recorded in
`results/`:

| Result | Value | Source |
|--------|-------|--------|
| AASIST-L eval EER (ASVspoof2019 LA) | **0.99%** (71,237 clips) | `tests/run_aasist_baseline.py` → `results/aasist_baseline.json` |
| LFCC-GMM eval EER (ASVspoof2019 LA) | 18.75% | `tests/run_lfcc_baseline.py` |
| Spoof attacks passing L1 voice gate | 54.0% (of 400 spoof trials) | `tests/eval_spoof_vs_speaker_verification.py` |
| Impostors passing L1 voice gate | 0.0% | same |
| Spoofs caught by L3 (AASIST-L) | 87.0% | same |
| Genuine users flagged by L3 | 0.0% | same |
| VoIP degradation: AASIST-L advantage vs LFCC-GMM | 15.6× (clean) → 1.9× (8 kbps / 15% loss) | `tests/eval_degraded.py` → `results/degradation_eer_table.csv` |
| Streaming: chunk-level EER at 250/500/1000 ms | 36.4% / 33.3% / 24.6% (clean) | `tests/eval_streaming.py` → `results/streaming_*.csv` |

See `results/` for the full tables. All thresholds are tuned on a development
split and frozen before evaluation (no re-tuning on test data).

---

## Current limitations (honest)

1. **L2 liveness is narrow in scope.** Trained on 15 live/playback pairs from one
   speaker, one room, one playback device (laptop speakers). Not tested against
   high-fidelity loudspeakers, other rooms, or other voices. The `recordings/`
   directory contains an earlier confounded experiment (iPhone vs MacBook) that
   should not be used as evidence.

2. **L4 context is evaluated on synthetic data.** The command corpus
   (`data/command_corpus.json`) is template-generated, and behavioural histories
   are synthetically seeded. Results involving L4 must be reported as resting on
   simulated behaviour, not real deployment data.

3. **No real end-to-end authorization data.** The SQLite database
   (`data/securevoice.db`) contains demo/synthetic transactions only, not real
   user observations.

4. **Streaming is naive chunking.** Short audio chunks are tile-repeated to the
   model's fixed ~4-second input length. True window-aware retraining is future
   work.

5. **AASIST-L is benchmarked on studio audio.** Its 0.99% EER is on ASVspoof2019
   LA (studio recordings). Performance on real phone audio degrades (see the
   domain-gap finding documented in `PAPER_OUTLINE.md` and `PROJECT_V2_PLAN.md`).

6. **The confidence formula in `aasist_l.py` is a placeholder** (`abs(score −
   threshold) × 2`). It should be replaced with calibrated LLRs from
   `models/calibrators.json`.

---

## New research direction

A separate feasibility study is under `research/audio_injection_generalization/`.
It investigates whether **VoiceSHIELD-Small** (a pretrained malicious-speech
detector) generalizes to **concurrent audio prompt injection** attacks from the
**AudioAgentSecurity** dataset, compared against **AASIST-L** as a separate
anti-spoofing baseline. This is preliminary feasibility work only — no model
training or large downloads have been performed. See
`research/audio_injection_generalization/FEASIBILITY_REPORT.md`.

---

## How to run

### Prerequisites

```bash
# Create a virtual environment (Python 3.10+)
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

**Dataset:** ASVspoof 2019 LA is required for the baseline evaluations. It is
not committed to the repository. Download it from
https://datashare.ed.ac.uk/handle/10283/3336 and place it at
`data/raw/asvspoof2019_la/` (see `data/README.md`).

### Run the demos

```bash
# Terminal demo (3 scenarios: genuine, spoof, imposter)
python demo_two_layer.py --scenario genuine
python demo_two_layer.py --scenario spoof
python demo_two_layer.py --scenario imposter

# Browser demo (two-layer enroll / order / deny)
.venv/bin/uvicorn src.api.demo_app:app --port 8900
# Open http://127.0.0.1:8900

# Full SecureVoice app (4 layers + cost-derived policy)
.venv/bin/uvicorn src.api.secure_voice_app:app --port 8901
# Open http://127.0.0.1:8901
```

### Run the evaluations

```bash
# AASIST-L baseline (reproduces 0.99% EER)
python tests/run_aasist_baseline.py --data-root data/raw/asvspoof2019_la

# LFCC-GMM baseline (18.75% EER)
python tests/run_lfcc_baseline.py --data-root data/raw/asvspoof2019_la

# Spoof vs speaker verification (800 trials)
python tests/eval_spoof_vs_speaker_verification.py

# VoIP degradation EER table
python tests/build_degraded_eval_set.py      # generates data/degraded/ (~83 min)
python tests/eval_degraded.py

# Streaming chunk EER vs latency
python tests/eval_streaming.py --split eval

# Mid-call alarm / time-to-detection tuning
python tests/tune_accumulator.py
```

### Fit calibrators (required for the full app)

```bash
python tools/fit_calibrators.py          # L1 + L3 calibrators
python tools/fit_l4_calibrator.py        # L4 calibrator (requires Groq API key)
python tools/train_l2_liveness.py        # L2 calibrator (requires captured pairs)
```

Calibrators are saved to `models/calibrators.json`.

---

## Repository layout (source preserved)

```
src/detector/      — SpoofDetector interface, AASIST-L, LFCC-GMM, liveness, metrics
src/api/           — FastAPI apps: demo_app.py, main.py, secure_voice_app.py
src/risk/          — calibration, engine, policy, pricing, stake (L5 decision)
src/context/       — L4 context layer, user profiles (SQLite)
src/nlu/           — command understanding / amount extraction
src/assistant/     — Groq client (firewalled), passphrase, response composer
src/streaming/     — CallAccumulator (mid-call alarm, time-to-detection)
src/voip_sim/      — Opus codec + Gilbert-Elliott packet-loss degradation
tests/             — evaluation scripts (baselines, degradation, streaming, etc.)
tools/             — calibrator fitting, ablation, figure building
models/            — calibrators.json; checkpoints are NOT committed (*.pth gitignored)
data/              — datasets and databases (NOT committed, gitignored)
results/           — measured results (JSON/CSV); cache/ is gitignored
research/          — new research work (audio_injection_generalization/)
docs/              — project status and project documentation
```

---

## Security notes

- `.env` contains a Groq API key. It is **gitignored and not tracked**. Rotate
  the key at https://console.groq.com/keys before sharing this repository.
- The Groq API key is used only by the L4/LLM components (prompt-injection
  scoring, intent extraction, spoken replies). Authorization works without it
  (those components degrade gracefully).
- Never commit `.env`, model checkpoints (`*.pth`), or personal recordings
  (`recordings/`, `*.wav`, `*.m4a`, `*.webm`).

---

## Citation

If you use this work, please cite ASVspoof 2019 (Todisco et al., Interspeech
2019) for the dataset, and the AASIST paper (Jung et al., ICASSP 2022) for the
anti-spoofing model. The cost-derived threshold is based on Kinnunen et al.,
"t-DCF: a Detection Cost Function…", Odyssey 2018 (arXiv:1804.09618).
