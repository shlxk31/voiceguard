# SecureVoice ↔ VoiceGuard — Integration & Paper Plan

## The situation in one paragraph
Your team presented **SecureVoice: Multi-Layer Continuous Authentication for
Voice Assistants** — a working React + FastAPI + PostgreSQL app with voice
enrollment, similarity scoring, trust scoring, amount-tiered authorization,
and audit logging. Meanwhile this repo (VoiceGuard) built **rigorously
benchmarked anti-spoofing**: AASIST-L reproducing its published 0.99% EER on
ASVspoof2019 LA, plus a channel-degradation robustness study. The two fit
together almost exactly: SecureVoice is a deployment vehicle with no
validated ML; VoiceGuard is validated ML with no deployment vehicle.

---

## Discrepancies to fix before the paper (be upfront about these)

### 1. ECAPA-TDNN is claimed but not implemented
Slides 5, 7, 11, 12, 13 all state SpeechBrain ECAPA-TDNN with 192-dim
speaker embeddings. The actual code in `backend/voice_processing.py` is:

```
overall_similarity = 0.4*mfcc_cosine + 0.2*pitch + 0.2*formant + 0.2*spectral_centroid
```

`speechbrain` appears nowhere in `backend/` or `backend/requirements.txt`.
**Do not put the ECAPA claim in a paper until the code matches it.** The fix
is cheap — `speechbrain` is already installed in this repo's venv and its
`EncoderClassifier` ECAPA model is a ~10-line drop-in. Doing it also gives
you a *real* speaker-verification EER to report, which you currently don't
have.

### 2. Spoof + liveness detection are unvalidated heuristics
`advanced_voice_processing.py` decides spoofing with hardcoded thresholds:
`pitch_std < 5.0 → +0.25`, `jitter < 0.001 → +0.3`, `spectral_flatness > 0.8
→ +0.2`, etc. These magic numbers were never fit to data or measured against
a benchmark. A reviewer will ask "what is its EER?" and there is no answer.

### 3. The PPT reports zero quantitative results
Every number on the slides is a *threshold* or a *design parameter* (0.75
similarity, trust ≥ 85, ₹10,000 tier) — none is a measured performance
result. This is the single biggest gap between the current project and a
publishable paper, and it is exactly what VoiceGuard's work supplies.

---

## The core vulnerability that makes this a real paper

**Speaker verification alone is defeated by voice cloning.** If an attacker
clones the enrolled user's voice, cosine similarity goes *up*, not down —
the clone sounds like the target, so SecureVoice's trust score rises and the
transaction is approved. Speaker verification answers "is this the right
voice?" but not "is this a real human?"

That gives you a concrete, demonstrable, publishable experiment:

> Run ASVspoof2019 LA voice-conversion/TTS attacks through both layers.
> Show they pass the speaker-similarity check while the AASIST-L
> anti-spoofing layer catches them at 0.99% EER. Report the
> authorization outcome (approved vs. denied) with and without the
> anti-spoofing layer, across the existing ₹ risk tiers.

That single experiment turns SecureVoice from "a system we built" into
"a system whose security property we measured and improved."

---

## Architecture mapping — where VoiceGuard slots in

| SecureVoice layer (from PPT) | Status | VoiceGuard contribution |
|---|---|---|
| L1 Voice capture & preprocessing | Works | Reuse; add 16kHz mono normalization to match detector input contract |
| L2 Speaker verification | **Heuristic, not ECAPA** | Swap in real SpeechBrain ECAPA-TDNN; measure its EER |
| **L2.5 Anti-spoofing (NEW)** | **Missing** | `AasistLDetector` — 0.99% EER, benchmarked, drop-in via `.score(audio, sr)` |
| L3 Trust scoring engine | Works | Add `spoof_score` as an input + a hard-deny gate, mirroring the existing `similarity < 0.50 → deny` rule |
| L4 Decision + audit logging | Works | Log spoof score and anti-spoofing verdict per attempt (paper needs this trail anyway) |

The interface already lines up: `AasistLDetector.score()` returns
`DetectionResult(score, is_spoof, confidence, latency_ms)` and takes a
16kHz mono numpy array — the same shape SecureVoice's backend already
produces after `librosa` loading.

---

## Mapping the PPT's own stated limitations to work already done

| PPT limitation (slide 14) | What VoiceGuard already has |
|---|---|
| #4 "Voice conversion attack resilience needs dedicated testing" | **Done.** ASVspoof2019 LA *is* TTS + VC attacks. AASIST-L: 0.99% eval EER; LFCC-GMM baseline: 18.75%. Full dev/eval threshold discipline. |
| #3 "Noisy environment robustness not yet evaluated" | **Methodology done, needs a re-aimed channel.** The Opus + packet-loss harness measures exactly this kind of degradation (AASIST-L 0.99% → 14.00% EER). For a smart speaker, add a noise/reverb axis instead of VoIP codec — cheap, reuses `src/voip_sim/degrade.py`'s structure. |
| #1 "Liveness detection is AC-signal-only" | Partially explored in `liveness_detector.py`. **Currently confounded** (recording device differed between classes) — see `LIVENESS_DETECTOR_RESULTS.md`. Needs same-device re-recording before any claim. |
| #2 "Behavioral biometrics limited to session-level" | Out of scope for VoiceGuard; leave as future work. |

---

## Recommended paper framing

**Title direction:** *Measuring and Closing the Synthetic-Speech Gap in
Voice-Authenticated Transactions*

**Claim:** Speaker-verification-based voice authentication systems are
structurally vulnerable to synthetic and converted speech; we quantify that
vulnerability in a working transaction-authorization system and show a
benchmarked anti-spoofing layer closes it, including under degraded
acoustic conditions.

**Contributions:**
1. Benchmarked anti-spoofing integrated into a live authorization pipeline
   (AASIST-L, 0.99% EER reproduced on ASVspoof2019 LA eval).
2. Demonstration that speaker similarity alone approves cloned-voice
   transactions, with the approval rate measured across the existing risk
   tiers, before vs. after adding the anti-spoofing layer.
3. Degradation study: how the anti-spoofing layer's EER moves under
   channel/acoustic degradation (advantage over a classical baseline
   collapses from 15.6× to 1.9× at the worst condition).
4. Streaming/latency characterization for continuous (not one-shot)
   authentication — chunk-level EER vs. window size and per-chunk latency.

Note contribution 4 is what makes "**continuous** authentication" in your
title actually true — right now the system authenticates once per request.

---

## Concrete next steps, in order

1. **Swap in real ECAPA-TDNN** for speaker verification (makes the slide
   honest, gives you an SV EER). ~10 lines + a model download.
2. **Wire `AasistLDetector` into the SecureVoice backend** as a new layer,
   with the frozen threshold from `results/aasist_baseline.json`.
3. **Run the headline experiment**: cloned/TTS audio vs. both layers →
   approval rate with and without anti-spoofing, per risk tier.
4. **Add a noise/reverb degradation axis** to address PPT limitation #3 in
   the terms it was framed.
5. **Re-record liveness data on one device** to de-confound that side
   experiment — or cut it from this paper and keep it as future work.
6. Fold results into `PAPER_OUTLINE.md` (already drafted, needs reframing
   from the VoIP-call framing to this voice-assistant framing).

## What to tell your mentor
Lead with: "We found our verification layer can be defeated by cloned
speech, so we added a benchmarked anti-spoofing layer and measured it —
0.99% EER, matching published results, and here's how it degrades under
realistic conditions." That's a stronger story than more system features,
because it's a *measured* security finding rather than an engineering
description.
