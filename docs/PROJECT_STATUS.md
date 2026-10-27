# VoiceGuard — Project Status

**Purpose:** Single source of truth for what is implemented, what is verified,
what is unverified/experimental, and what remains to be done. This file is the
honest baseline for a final-year major project and research paper.

**Status date:** 2026-10-27
**Branch:** `main`
**Remote:** `https://github.com/shlxk31/voiceguard.git`

---

## 1. Implemented (code exists and runs)

### 1.1 Source code (`src/`)

| Module | Files | What it does | Verified? |
|--------|-------|--------------|-----------|
| `src/detector/base.py` | `SpoofDetector` ABC + `DetectionResult` | Common interface for all detectors | ✅ |
| `src/detector/aasist_l.py` | `AasistLDetector` | AASIST-L anti-spoofing (pretrained checkpoint) | ✅ (benchmarked) |
| `src/detector/lfcc_gmm.py` | `LFCCGMMDetector` | Classical LFCC + GMM baseline | ✅ (benchmarked) |
| `src/detector/liveness_l2.py` | `LivenessDetector` | Replay / liveness detector | ⚠️ (narrow scope) |
| `src/detector/metrics.py` | `compute_eer()` | Equal Error Rate computation | ✅ |
| `src/api/demo_app.py` | FastAPI app | Two-layer browser demo (enroll / order / deny) | ✅ |
| `src/api/main.py` | FastAPI app | `/analyze` + `/stream` WebSocket endpoints | ✅ (skeleton) |
| `src/api/secure_voice_app.py` | FastAPI app | Full 4-layer + cost-policy authorization app | ✅ |
| `src/risk/calibration.py` | `ScoreCalibrator` | Platt scaling → calibrated LLRs | ✅ |
| `src/risk/policy.py` | `RiskPolicy` | Cost-derived allow/challenge/deny thresholds | ✅ |
| `src/risk/engine.py` | `DecisionEngine` | Fuses calibrated LLRs (weakest-link rule) | ✅ |
| `src/risk/pricing.py` | `PriceEstimator` | LLM-based item price estimation | ⚠️ (knowledge-based, not live lookup) |
| `src/risk/stake.py` | `estimate_stake()` | Monetary + consequence stake estimation | ✅ |
| `src/context/l4.py` | `ContextLayer` | L4 context: injection + semantic + behavioural | ⚠️ (synthetic data) |
| `src/context/profile.py` | `ProfileStore` | SQLite user profiles + transaction history | ✅ |
| `src/nlu/understand.py` | `CommandUnderstanding` | ASR (faster-whisper) + amount extraction | ✅ |
| `src/assistant/groq_client.py` | `GroqAssistant` | Firewalled LLM client (injection + chat + TTS) | ⚠️ (requires API key) |
| `src/assistant/passphrase.py` | `verify()` | Fuzzy passphrase verification | ✅ |
| `src/assistant/respond.py` | `ResponseComposer` | Turns decisions into spoken replies | ✅ |
| `src/streaming/accumulator.py` | `CallAccumulator` | N-of-M mid-call alarm + time-to-detection | ✅ |
| `src/voip_sim/degrade.py` | `degrade()` | Opus codec + Gilbert-Elliott packet loss | ✅ |

### 1.2 Root-level scripts

| Script | Purpose | Verified? |
|--------|---------|-----------|
| `demo_two_layer.py` | Terminal demo (genuine / spoof / imposter) | ✅ |
| `liveness_detector.py` | Standalone liveness research prototype | ✅ (prototype) |

### 1.3 Tests (`tests/`) — reproducible evaluations

| Script | Purpose | Result |
|--------|---------|--------|
| `run_aasist_baseline.py` | AASIST-L on ASVspoof2019 LA eval | **0.99% EER** (reproduces published) |
| `run_lfcc_baseline.py` | LFCC-GMM on ASVspoof2019 LA eval | **18.75% EER** |
| `eval_spoof_vs_speaker_verification.py` | 800 trials: genuine / spoof / imposter | 54% spoofs pass L1; 87% caught by L3 |
| `eval_degraded.py` | Clean vs VoIP-degraded EER table | Advantage 15.6× → 1.9× |
| `eval_streaming.py` | Chunk EER vs latency at 250/500/1000 ms | 36.4% / 33.3% / 24.6% (clean) |
| `tune_accumulator.py` | Mid-call alarm tuning + time-to-detection | Configurable |
| `build_degraded_eval_set.py` | Generates `data/degraded/` (~83 min) | Generates cached sets |

### 1.4 Tools (`tools/`)

| Script | Purpose |
|--------|---------|
| `fit_calibrators.py` | Fit L1 + L3 calibrators → `models/calibrators.json` |
| `fit_l4_calibrator.py` | Fit L4 calibrator (requires Groq API key) |
| `train_l2_liveness.py` | Train L2 liveness model from captured pairs |
| `run_ablation.py` | Layer ablation study (FAR per threat) |
| `sensitivity_analysis.py` | Cost-parameter sensitivity |
| `build_command_corpus.py` | Generate synthetic command corpus |
| `build_figures.py` | Generate paper figures |
| `build_paper.py`, `build_slides_*.py` | Build paper / slides |

### 1.5 Results (`results/`) — committed, measured

| File | Content |
|------|---------|
| `aasist_baseline.json` | AASIST-L: 0.99% EER, threshold 0.9157 |
| `degradation_eer_table.csv` | VoIP degradation EER table (2 detectors × 11 conditions) |
| `spoof_vs_speaker_verification.json` | 800-trial spoof vs SV results |
| `streaming_dev_clean.csv`, `streaming_eval_clean.csv`, `streaming_eval_opus8kbps_loss15pct_plcon.csv` | Streaming chunk EER + latency |
| `ablation.csv`, `sensitivity.csv` | Ablation and sensitivity tables |
| `l2_scores.json`, `l4_scores.json` | L2 / L4 out-of-fold scores |
| `calibration_split.json` | Dev/eval split indices (stratified) |
| `cache/` | **gitignored** — cached eval arrays |

---

## 2. Unverified (implemented but claims not yet backed by full evidence)

| Item | What is claimed | What is actually verified | Gap |
|------|----------------|---------------------------|-----|
| **L2 liveness** | 100% LOO accuracy on 15 pairs | Verified on 15 pairs, one speaker, one room, laptop speakers | Not tested on high-fidelity speakers, other rooms, other voices |
| **L4 context** | EER 13.34% on eval half | Verified on synthetic template corpus (240 commands) | Not a deployment estimate; behaviour histories are synthetic |
| **Domain adaptation** | Claimed 8/8 → 2/8 false positives | No evaluation script or result file exists | `tests/eval_robust.py` referenced but missing |
| **Confidence formula** | `abs(score − threshold) × 2` | Placeholder; not calibrated | Should use calibrated LLRs from `models/calibrators.json` |
| **Groq-dependent features** | Prompt-injection scoring, intent extraction, TTS | Code exists; requires API key | Cannot run without `GROQ_API_KEY` |

---

## 3. Experimental (new research direction, not yet validated)

Located entirely under `research/audio_injection_generalization/`:

| File | Purpose | Status |
|------|---------|--------|
| `FEASIBILITY_REPORT.md` | Feasibility of VoiceSHIELD-Small + AudioAgentSecurity vs AASIST-L | Written; no experiments run |
| `smoke_test.py` | Loads VoiceSHIELD-Small, runs inference on audio | Not yet run (requires `transformers` install + dataset download) |

This is **feasibility-only** work. No model training, no large dataset downloads, no claimed results.

---

## 4. Remaining to be done

### 4.1 High priority (short deadline)

1. **Fix the confidence formula** in `src/detector/aasist_l.py` — replace
   `abs(score − threshold) × 2` with calibrated LLRs from `models/calibrators.json`.
   (~0.5 day)

2. **Validate or retract the domain-adaptation claim** (8/8 → 2/8). Either write
   `tests/eval_robust.py` to evaluate `models/aasist_l_domain_adapted.pth` on
   held-out real clips, or remove the unsupported claim. (1–2 days)

3. **Re-record L2 liveness data on one device/session** — same mic, same room,
   live + its own playback back-to-back. Re-run `tools/train_l2_liveness.py`.
   Report with stated scope. (1 afternoon)

4. **Rotate the Groq API key** — `.env` contains a key that was on disk. Rotate
   at https://console.groq.com/keys. (5 minutes)

### 4.2 Medium priority

5. **Run all verified evaluations end-to-end** to confirm reproducibility:
   - `run_aasist_baseline.py`, `run_lfcc_baseline.py`
   - `eval_spoof_vs_speaker_verification.py`
   - `eval_degraded.py` (requires building degraded sets first)
   - `eval_streaming.py`
   - `tune_accumulator.py`

6. **Add unit tests** — currently `tests/` are evaluation scripts, not unit
   tests. Add `pytest`-based tests for core modules (calibration, policy,
   engine, accumulator).

7. **Complete the feasibility smoke test** — install `transformers`,
   `safetensors`, `accelerate`; manually accept AudioAgentSecurity HF access;
   download a few samples; run `research/audio_injection_generalization/smoke_test.py`.

### 4.3 Longer term (paper / project completion)

8. **Write the research paper** — based on `PAPER_OUTLINE.md`, focused on the
   domain-gap finding (studio-trained models fail on real phone audio).

9. **Run the full ablation** — `tools/run_ablation.py` on held-out data only.

10. **Address ALL threats to validity** listed in `ASSISTANT_LAYER_RESULTS.md §6`.

---

## 5. What is NOT in this repository (and should not be added)

| Item | Reason |
|------|--------|
| ASVspoof 2019 LA dataset | Large; not committed; download separately to `data/raw/` |
| `.env` with Groq API key | Secret; gitignored; never commit |
| `.venv/` | Python environment; gitignored |
| Model checkpoints (`*.pth`) | Large; regenerated by finetune scripts; gitignored |
| `recordings/` (`.wav`, `.m4a`, `.webm`) | Personal recordings; gitignored |
| `data/` (SQLite DB, captures, command corpus) | Local data; gitignored |
| `results/cache/` | Large cached arrays; gitignored |
| `external/aasist/` | Cloned AASIST reference repo; gitignored |
| `*.pptx`, `*.zip`, `report_phase3/` | Binary artifacts; gitignored |

---

## 6. Known gaps between documentation and code

| Document claim | Code reality | Action |
|---------------|-------------|--------|
| "Zero false alarms on real users" (anti-spoofing) | Contradicted by domain-gap finding: 8/8 real recordings flagged as fake | Retract or qualify in paper |
| L2 liveness is a working layer | Code exists but data is confounded / narrow | Re-record or scope down |
| L4 evaluated on real behaviour | Evaluated on synthetic corpus only | Label as simulated in paper |
| Domain adaptation fix validated | No eval script or result file | Write eval or retract claim |

---

## 7. Decision log

| Decision | Rationale | Date |
|----------|-----------|------|
| 4 layers + 1 policy (not 5 layers) | L5 is a policy, not a detector; avoids category error | 2026-09 |
| Weakest-link fusion (min) instead of sum | Sum lets correct L3 evidence cancel correct L1 evidence (impostor FAR 0% → 47.5%) | 2026-09 |
| Cost-derived thresholds (t-DCF) | Replaces arbitrary tiers (₹500/₹2000/₹10000) with principled derivation | 2026-09 |
| Research kept under `research/audio_injection_generalization/` | Separates new research direction from existing application | 2026-10 |
| No destructive file moves | Preserve import paths and existing application | 2026-10 |
