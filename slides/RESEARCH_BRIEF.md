# SecureVoice — context brief for a research assistant

Paste this whole file into a new Claude chat, then ask for the research you want.

---

## 1. What the project is

**SecureVoice: Multi-Layer Continuous Authentication for Voice Assistants.**
A TE Computer Engineering major project at Sardar Patel Institute of
Technology (Atharva Kadge, Damodar Kamath, Shlok Kanvinde).

A voice assistant that can place orders and move money. Before it acts, it
decides: **allow, challenge, or deny**.

## 2. The five layers (this is the current design)

| Layer | Model / method | Question it answers | Threat it stops |
|---|---|---|---|
| L1 Voiceprint | SpeechBrain ECAPA-TDNN, 192-d embedding, cosine similarity | Is this the enrolled user? | A different person |
| L2 Liveness | Logistic model on spectral/cepstral features | A mouth, or a loudspeaker? | Replay of a recording |
| L3 Anti-spoofing | AASIST-L (raw waveform, graph attention) | Human throat, or computer? | TTS / voice-conversion clone |
| L4 Context | Injection classifier + LLM-extracted intent features + user history | Is this request normal? | Manipulated / injected command |
| L5 Risk policy | Bayes-derived threshold | How much proof is needed? | *Not a detector — it decides* |

**How they combine:** each raw score is calibrated (Platt scaling) into a
log-likelihood ratio, then fused by **weakest link** (`min`), not by adding.
The result is compared against a threshold derived from cost.

## 3. Measured results (all from our own experiments)

| Metric | Value |
|---|---|
| AASIST-L, ASVspoof 2019 LA eval EER | **0.99%** (71,237 files) — matches published |
| LFCC-GMM baseline eval EER | 18.75% |
| Impostor false acceptance (full system) | 0.25% |
| Replay false acceptance — with L2 / without | **1.25% / 74.0%** |
| Clone false acceptance — with L3 / without | 1.25% / 43.25% |
| Injection false acceptance — with L4 / without | 10.0% / 23.5% |
| Genuine users: allowed / challenged / refused | 61.25% / 26.75% / 12.0% |
| Naive-Bayes summation instead of weakest link | impostor FAR 0.25% → **52.25%** |
| Speaker verification alone lets clones through | 54% |

## 4. Things that are NOT true (common mistakes)

- **We do not use Gemini.** Transcription is local Whisper; the LLM is
  gpt-oss-120b via Groq. Audio never leaves the device.
- **The 0.75 / 0.85 / 0.50 similarity cutoffs are obsolete.** They were the
  Phase-2 design. They are replaced by a cost-derived threshold. Do not
  present them as our method.
- **The trust score 0–100 is obsolete.** We use calibrated log-likelihood
  ratios.
- **We do not use bone conduction.** It needs hardware we do not have.

## 5. Known limitations (state these; do not hide them)

- L2 liveness: only 15 live/replay pairs, one speaker, one room, laptop
  speakers. Good loudspeakers untested.
- L4 context: calibrated on a template-generated command corpus, not real
  adversarial commands.
- Ablation trials pair independently measured conditions; assumes layer
  errors are independent.
- All live audio is from one speaker — no demographic/fairness evaluation.
- Voice conversion is under-tested relative to text-to-speech.
- Threshold constants are assumptions; sensitivity analysis shows the
  **attack prior** dominates.

## 6. References we have already verified as real

- Kinnunen et al., *t-DCF: A Detection Cost Function…*, Odyssey 2018 — this
  is the citation for our threshold derivation
- Desplanques et al., *ECAPA-TDNN*, Interspeech 2020
- Jung et al., *AASIST*, ICASSP 2022
- Todisco et al., *ASVspoof 2019*, Interspeech 2019
- Wu et al., *Spoofing and countermeasures… A survey*, Speech Comm. 2015
- Ahmed et al., *Void: voice liveness detection*, USENIX Security 2020
- Müller et al., *Does audio deepfake detection generalize?*, Interspeech 2022
- Greshake et al., *Indirect prompt injection*, AISec 2023
- Cheng & Roedig, *Personal voice assistant security and privacy — a survey*,
  Proceedings of the IEEE 2022

## 7. Rules for any research you do

1. **Never fabricate a citation, DOI, statistic, or news story.** If you
   cannot verify it, say so and label it unverified.
2. Prefer sources from the **last 5 years**, plus foundational work.
3. Use only IEEE / ACM / Springer / Elsevier / USENIX or equivalent.
4. If a number is not measured by us, label it clearly as external.
5. Give the full citation so it can be checked.

## 8. What we want researched

*(edit this line before sending)*

For example: recent voice-assistant fraud statistics for India; newer
anti-spoofing models to compare against AASIST-L; published false
acceptance/rejection rates for commercial voice authentication; work on
LLM prompt injection through speech; datasets of real adversarial voice
commands; demographic bias in speaker verification.
