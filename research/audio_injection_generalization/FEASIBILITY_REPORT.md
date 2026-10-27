# Feasibility Report — Audio Injection Generalization Study

**Folder:** `research/audio_injection_generalization/`
**Status:** Preliminary feasibility assessment (no experiments run yet)
**Date:** 2026-10-27

---

## 1. Research Question

Can **VoiceSHIELD-Small** (a pretrained malicious-speech detector) generalize to detect
**concurrent audio prompt injection** attacks from the **AudioAgentSecurity** dataset,
and how does it compare against **AASIST-L** (the existing anti-spoofing baseline in
this project) as a separate audio security baseline?

In plain terms: VoiceSHIELD-Small was trained to flag "malicious" speech content
(prompt injection, social engineering, credential-harvesting). AudioAgentSecurity
contains benign user speech with a concealed malicious instruction mixed in.
The open question is whether a malicious-speech detector trained on one kind of
adversarial audio will flag AudioAgentSecurity's injection samples, and whether
AASIST-L (trained to detect synthetic vs. real human voice) has any signal on
this different threat at all.

---

## 2. Components Under Investigation

### 2.1 Dataset — AudioAgentSecurity

| Property | Value |
|----------|-------|
| Hugging Face URL | https://huggingface.co/datasets/Limax11/AudioAgentSecurity |
| Paper | arXiv:2607.28165 — "Piggybacking on Perception: Stealthy Concurrent Audio Prompt Injections against Multimodal LLM Agents" |
| GitHub | https://github.com/Limax666/AudioAgentSecurity |
| Access | Public, but **requires HF login + agreeing to dataset access conditions** before files can be downloaded |
| Size (metadata) | 5.75 kB reported on HF; actual audio download size not stated |
| Samples | 2,160 generated attack audio samples |
| Attack patterns | 10: dialect, dolphin, foreign, high_freq, pulse, spectral_inversion, spectral_scramble, speed, texture, whisper |
| Task scenarios | 8: development, finance, iot, media_creative, navigation, personal_assistant, productivity, system_control |
| Format | WAV audio |
| Layout | `benchmark_dataset_mixed/benign/` (clean user speech), `benchmark_dataset_mixed/mixed/<pattern>/` (concurrent user + attack) |
| Labels | `metadata.json` (record-level: scenario, pattern, pairing, labels) |
| Benign source | Public speech corpora (no real user credentials or personal data) |
| Malicious content | Synthetic research artifact instruction texts |
| License | Not explicitly stated on HF dataset card; paper is arXiv preprint |
| Intended use | Defensive security research and education (not for attacking production systems) |

**Access note:** The HF dataset page states "You need to agree to share your Hugging Face information to access this dataset." This means a programmatic download via `huggingface-cli` or `datasets.load_dataset()` may be blocked until the user manually accepts the conditions on the HF website. This is a practical blocker for unattended runs.

### 2.2 Candidate Baseline — VoiceSHIELD-Small

| Property | Value |
|----------|-------|
| Hugging Face URL | https://huggingface.co/Emvo-ai/voiceSHIELD-small |
| Paper | arXiv:2603.07708 — "VoiceSHIELD-Small: Real-Time Malicious Speech Detection and Transcription" |
| License | MIT |
| Base model | openai/whisper-small (~88M trainable parameters) |
| Model size | ~354 MB (safetensors) |
| Architecture | Whisper encoder + mean-pool + MLP classification head (safe/malicious) + Whisper decoder (transcript) |
| Input | 16 kHz mono audio (WAV, MP3, FLAC) |
| Output | Label (`safe` / `malicious`), confidence score, transcript |
| Inference command | `pip install torch torchaudio transformers safetensors` then `from transformers import pipeline; pipe = pipeline("audio-classification", model="Emvo-ai/voiceSHIELD-small", trust_remote_code=True)` |
| Hardware | 90–120 ms per clip on mid-tier GPU; CPU possible but slower |
| Reported test metrics | 947-sample hold-out test set: Accuracy 99.16%, F1 0.9865, Precision 0.9966, Recall 0.9767, ROC-AUC 0.9948, FNR 2.33%, FPR 0.15% |
| Default threshold | 0.2 (F1-optimal); use ≤0.2 for security-focused deployments to keep FNR < 5% |
| Training data | 6,310 samples (4,416 train / 947 val / 947 test); "malicious" = prompt injection, social engineering, safety-bypass, credential-harvesting speech |
| Known limitations | English only; controlled recording conditions; not validated on accents, telephony compression, or novel attack patterns; 2.33% FNR; distribution shift on unseen attack strategies |

**Key dependency:** `trust_remote_code=True` is required, meaning the model loads and executes code from the Hugging Face Hub. This is a supply-chain consideration.

### 2.3 Separate Baseline — AASIST-L (existing project)

| Property | Value |
|----------|-------|
| Location | `external/aasist/models/weights/AASIST-L.pth` (already present in repo) |
| Paper | AASIST (Jung et al., ICASSP 2022) |
| Task | Anti-spoofing: bonafide (real human) vs. spoof (synthetic/TTS/VC) |
| Reported EER | 0.99% on ASVspoof2019 LA eval (71,237 clips) — reproduced in this project at `results/aasist_baseline.json` |
| Threat model | Synthetic vs. real human voice — **different** from AudioAgentSecurity's concurrent injection |
| Relevance | Included as a "separate audio anti-spoofing baseline" per the task, but expected to have limited signal on prompt-injection audio because it answers "is this voice synthetic?" not "does this speech contain a malicious instruction?" |

---

## 3. Distinctness from Existing Work

AudioAgentSecurity's paper (arXiv:2607.28165) evaluates 11 state-of-the-art **multimodal LLM agents** (Gemini 3 Pro, GPT-4o-audio, etc.) and reports average Attack Success Rate of 69.10% against Gemini 3 Pro. Its proposed defense, CADV, is based on acoustic source separation and cross-modal consistency analysis.

This feasibility study is distinct because:

1. **VoiceSHIELD-Small is an audio-only malicious-speech classifier**, not a multimodal agent. Evaluating it on AudioAgentSecurity tests whether a lightweight audio model can detect injected malicious intent without access to the agent's internal state or multimodal context.

2. **Unseen attack patterns:** AudioAgentSecurity's 10 attack patterns (spectral inversion, scrambling, dolphin carriers, etc.) are unlikely to have been in VoiceSHIELD-Small's training data (which covers generic prompt injection / social engineering speech). This is a genuine out-of-distribution test.

3. **AASIST-L comparison:** AASIST-L addresses a different threat (spoofing/synthetic voice). Comparing it on AudioAgentSecurity establishes whether anti-spoofing detectors have any transferable signal to injection attacks — expected to be low, which is itself a useful negative result.

The proposed experiment is **not** a replication of the AudioAgentSecurity paper's agent-evaluation. It is a detector-generalization study: does an audio classifier built for one threat flag audio built for a different, related threat?

---

## 4. Proposed Experiment

### 4.1 Data Split

Use AudioAgentSecurity's `metadata.json` to stratify samples. Proposed split (subject to verifying actual label distribution in metadata):

| Split | Purpose | Samples (approx.) |
|-------|---------|-------------------|
| Train | Not used (zero-shot evaluation only) | — |
| Validation | Threshold tuning on VoiceSHIELD-Small outputs | ~500 mixed samples |
| Test | Held-out generalization measurement | ~1,000 mixed samples + benign controls |

**No training is performed.** Both VoiceSHIELD-Small and AASIST-L are used as zero-shot (pretrained) baselines. This avoids data leakage and keeps the study feasible within a short deadline.

### 4.2 Evaluation Procedure

1. Download AudioAgentSecurity (requires manual HF agreement).
2. Load VoiceSHIELD-Small via `transformers.pipeline` with `trust_remote_code=True`.
3. For each mixed audio sample, record: predicted label, confidence, transcript.
4. For each benign control sample, record the same (to measure false-positive rate).
5. Load AASIST-L from `external/aasist/` and score the same samples (bonafide vs. spoof score).
6. Compute metrics (below) per attack pattern and overall.

### 4.3 Metrics

| Metric | Definition |
|--------|------------|
| Detection rate (recall) | % of attack samples flagged `malicious` by VoiceSHIELD-Small |
| False-positive rate | % of benign control samples flagged `malicious` |
| AASIST-L spoof score distribution | Mean/median P(spoof) on attack vs. benign samples (separation check) |
| ROC-AUC | VoiceSHIELD-Small confidence as a continuous detector |
| Per-pattern breakdown | Detection rate for each of the 10 attack patterns |

### 4.4 Train/Validation/Test Separation

No model training occurs. The split is used only to:
- **Validation:** pick a VoiceSHIELD-Small decision threshold (default 0.2) that balances detection vs. false alarms on held-out data.
- **Test:** report final detection rate and FPR on unseen samples.

This separation prevents threshold overfitting and mirrors the discipline used in the existing project (dev-tuned, eval-frozen thresholds).

---

## 5. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| AudioAgentSecurity requires manual HF agreement to download | High | Plan for manual download step; document it clearly; do not assume unattended pipeline |
| VoiceSHIELD-Small requires `trust_remote_code=True` (executes remote code) | Medium | Review the modeling code from the HF repo before running; run in an isolated environment |
| VoiceSHIELD-Small is ~354 MB download | Low | Acceptable for a one-time feasibility check |
| Different threat models may yield near-random results | Medium | Expected and acceptable — a negative result ("VoiceSHIELD-Small does not generalize") is still a valid research finding |
| English-only model; AudioAgentSecurity may include non-English attack carriers | Low | Check metadata for language; report any non-English samples separately |
| AASIST-L expected to have no signal on injection audio | Low | Frame as a control/negative baseline, not a failure |
| No ground-truth labels beyond metadata.json | Low | Verify metadata.json contains usable labels before committing to experiment |

---

## 6. Go / No-Go Recommendation

**GO — with conditions.**

### Conditions:
1. **Manual HF dataset access:** The user must manually accept AudioAgentSecurity's access conditions on Hugging Face before the dataset can be downloaded. This is not blocker for the feasibility study itself, but it means the smoke test and any evaluation require a manual step.

2. **Install `transformers`:** The current project's `requirements.txt` does not include `transformers`. It must be added (with `safetensors` and `accelerate`) to run VoiceSHIELD-Small. This is a safe, additive dependency change.

3. **Zero-shot only:** Do not fine-tune either model. Use pretrained checkpoints only. This keeps the experiment short and avoids data-leakage concerns.

4. **Smoke test first:** Run VoiceSHIELD-Small on a small number of actual AudioAgentSecurity samples (once downloaded) to confirm the model loads and produces sensible output before scaling to the full evaluation.

### Not recommended:
- Do not claim VoiceSHIELD-Small "solves" audio prompt injection. Its 2.33% FNR on its own test set means it will miss threats.
- Do not compare VoiceSHIELD-Small's reported 99.16% accuracy (on its own training-domain test set) directly against AudioAgentSecurity results — different data distributions.
- Do not train large models or download the full dataset until the smoke test succeeds on a few samples.

---

## 7. References

1. AudioAgentSecurity dataset (Hugging Face): https://huggingface.co/datasets/Limax11/AudioAgentSecurity
2. AudioAgentSecurity paper (arXiv): https://arxiv.org/abs/2607.28165
3. AudioAgentSecurity GitHub: https://github.com/Limax666/AudioAgentSecurity
4. VoiceSHIELD-Small model (Hugging Face): https://huggingface.co/Emvo-ai/voiceSHIELD-small
5. VoiceSHIELD-Small paper (arXiv): https://arxiv.org/abs/2603.07708
6. AASIST-L (existing project): `external/aasist/`, `results/aasist_baseline.json` (0.99% EER on ASVspoof2019 LA eval)

---

## 8. Next Steps (if GO)

1. Add `transformers`, `safetensors`, `accelerate` to `requirements.txt`.
2. Manually accept AudioAgentSecurity access conditions on Hugging Face.
3. Download a small subset of AudioAgentSecurity audio.
4. Run the smoke test (`smoke_test.py`) on those samples.
5. If the smoke test loads the model and returns labels, proceed to the full evaluation described in §4.
