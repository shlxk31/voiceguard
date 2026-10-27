# SecureVoice — Project Guide in Simple Language

A plain-English explanation of what this project is, what we measured, and
what still needs doing to turn it into a research paper.

---

## 1. The problem, in one paragraph

Voice assistants now place orders and move money. To check who is speaking,
systems use **speaker verification** — it compares your voice to a stored
sample and asks *"is this the right person?"* That works well against a
stranger. It does **not** work well against AI voice cloning, because a clone
is built to sound like you. When the fake sounds like you, the system's
confidence goes **up**, and the order gets approved.

So the missing question is: **"is this a real human, or a machine-generated
voice?"** That is a different question, and it needs a different model.

---

## 2. What our system does

Audio comes in and passes through two independent checks:

| Layer | Question it answers | Model used |
|---|---|---|
| **Layer 2** — Speaker verification | Is this the enrolled user's voice? | ECAPA-TDNN (192-dim voice embeddings, cosine similarity) |
| **Layer 2.5** — Anti-spoofing *(our addition)* | Is this a real human or an AI fake? | AASIST-L (pretrained, 85k parameters) |

Both must pass before a transaction is approved. Layer 2.5 is the new part —
the original SecureVoice design did not have a validated version of it.

---

## 3. The main results

### 3.1 The vulnerability we found
800 test trials, 20 enrolled speakers, real ASVspoof2019 data.

| What we tested | Voice similarity score | How often it passed Layer 2 |
|---|---|---|
| The real enrolled user | 0.726 | 99.5% (correct — should pass) |
| **An AI clone of that user** | **0.476** | **54.0% (bad — should fail)** |
| A different real person | 0.124 | 0.0% (correct — should fail) |

**In plain words:** speaker verification is *perfect* at spotting a different
human — not one imposter got through. But against AI clones of the enrolled
user, **more than half got through**. The layer simply cannot tell "real you"
from "fake you."

### 3.2 What our new layer does about it

| Audio type | How often our anti-spoofing layer flagged it |
|---|---|
| Real enrolled user | 0.0% (no false alarms) |
| **AI clone** | **87.0% caught** |
| Different real person | 0.0% (no false alarms) |

### 3.3 Effect on actual order approvals

| Order amount | Attacks approved — before | Attacks approved — after |
|---|---|---|
| < Rs.500 | 54.0% | **7.5%** |
| Rs.500–2,000 | 54.0% | **7.5%** |
| Rs.2,000–10,000 | 11.0% | **1.8%** |
| > Rs.10,000 | 0.0% | 0.0% |

### 3.4 Proof our anti-spoofing model is trustworthy
Before claiming anything, we reproduced the published benchmark result.

| Model | Error rate (EER) on full test set of 71,237 clips |
|---|---|
| Classical baseline (LFCC-GMM) | 18.75% |
| **AASIST-L (ours)** | **0.99%** |

0.99% matches the published AASIST-L figure almost exactly. That is what makes
every other number here believable.

### 3.5 How it holds up in bad audio conditions
We simulated a real phone/VoIP call — audio compression plus dropped packets.

| Condition | Classical baseline | AASIST-L |
|---|---|---|
| Clean audio | 18.30% | 1.17% |
| Opus 16 kbps, 5% packet loss | 21.03% | 5.23% |
| Opus 8 kbps, 15% packet loss | 26.83% | 14.00% |

**In plain words:** on clean audio the good model is ~15× better than the weak
one. On a bad connection that advantage shrinks to under 2×. Clean-audio
benchmarks make these models look better than they are in practice.

### 3.6 Why "continuous" authentication is hard
We tried scoring short audio chunks instead of whole recordings.

| Chunk length | Error rate (clean) | Error rate (bad connection) | Speed |
|---|---|---|---|
| Whole recording | 0.51% | — | — |
| 1000 ms | 24.57% | 35.62% | 55 ms |
| 500 ms | 33.28% | 37.87% | 55 ms |
| 250 ms | 36.43% | 38.70% | 55 ms |

**In plain words:** it is fast enough for real time, but the accuracy falls
apart. The model needs about 4 seconds of context to work; feeding it a
quarter-second gives it almost nothing to judge. Combine short chunks with a
bad connection and it reaches ~39% — near coin-flip (50%). The classical
baseline actually goes *past* coin-flip to 54.5%, meaning its answers become
anti-correlated with the truth. So you cannot make a system "continuous" just
by chopping audio into small pieces; that needs a model retrained for short
inputs. Measuring this is itself a useful finding.

### 3.7 The most important result: it fails on real audio
We pointed the model at 8 real voice recordings made on an iPhone.

| Audio | Flagged as synthetic |
|---|---|
| 8 real human recordings (iPhone, normal room) | **8 / 8 = 100%** |

Every one scored above 0.98 "probability synthetic." **A model with 0.99%
error on the benchmark has a 100% false-positive rate on real phone audio.**

Why: ASVspoof2019's "real" examples are studio recordings from a
sound-treated chamber. Measured comparison:

| Band | Lab "real" | Lab fake | Real iPhone audio |
|---|---|---|---|
| 0–300 Hz | 39.9% | 46.8% | **53.8%** |
| 6–8 kHz | 0.38% | 0.48% | **0.04%** |
| Noise floor | 0.0043 | 0.0186 | **0.0702** |

The model never learned "human vs. synthetic." It learned **"studio-clean vs.
everything else."** We ruled out simpler explanations by testing compression,
resampling, volume changes and mild noise on lab audio — none of those trigger
the failure. See `PROJECT_V2_PLAN.md`, which restructures the project around
this finding.

---

## 4. How to run everything

| What | Command |
|---|---|
| **Browser test UI** (record with your mic) | `.venv/bin/uvicorn src.api.demo_app:app --port 8900` then open `http://127.0.0.1:8900` |
| Terminal demo (3 canned scenarios) | `python demo_two_layer.py --scenario genuine \| spoof \| imposter` |
| The 800-trial attack experiment | `python tests/eval_spoof_vs_speaker_verification.py` |
| Anti-spoofing benchmark | `python tests/run_aasist_baseline.py` |
| Classical baseline | `python tests/run_lfcc_baseline.py` |
| Build degraded audio sets | `python tests/build_degraded_eval_set.py` |
| Clean vs degraded comparison | `python tests/eval_degraded.py` |
| Streaming chunk experiment | `python tests/eval_streaming.py --split eval` |

---

## 5. Honest limitations (put these in the paper — they help, not hurt)

1. **The attacks are from 2019.** ASVspoof2019 predates today's voice cloning
   tools. Our 54% figure is probably an *underestimate* of current risk.
2. **The bad-connection test is simulated,** not recorded from real calls.
3. **One dataset, one architecture family.**
4. **Short-chunk results use naive chunking** — we repeat a short clip to fill
   the model's 4-second input rather than retraining it properly.
5. **The liveness side-experiment is not valid yet.** Live and playback samples
   were recorded on different devices (iPhone vs MacBook), so the classifier
   learned "which device" instead of "live vs playback." Needs re-recording on
   one device, or leave it out of the paper.

---

## 6. Things to fix in the existing SecureVoice code

1. The slides say **SpeechBrain ECAPA-TDNN**, but `backend/voice_processing.py`
   actually uses a hand-weighted blend of MFCC, pitch and formant similarity,
   and `speechbrain` is not in `backend/requirements.txt`. All results in this
   document use *real* ECAPA-TDNN, so the code needs to catch up to the slide.
2. The existing spoof/liveness detection is hardcoded guesswork
   (`if pitch_std < 5.0: score += 0.25`) with no measured accuracy.
3. **The risk tiers are miscalibrated.** With real similarity scores only 43.5%
   of *genuine* users reach trust ≥ 70, and almost none reach ≥ 85 — so the
   higher-value tiers would reject most real customers.

---

## 7. Tasks to turn this into a research paper

**Already done — nothing more needed**
- [x] Reproduce a published benchmark (AASIST-L, 0.99% EER) to prove the setup is sound
- [x] Classical baseline for comparison (LFCC-GMM, 18.75%)
- [x] Measure the vulnerability (800 trials, 20 speakers)
- [x] Measure the fix (87% caught, 0% false alarms)
- [x] Robustness study under degraded audio
- [x] Streaming/latency study
- [x] Working demo (browser UI + terminal)
- [x] Proper method discipline (thresholds fixed on dev data, test data touched once)

**Still to do — writing**
- [ ] Write Related Work (ASVspoof challenge, AASIST, ECAPA-TDNN, replay vs synthetic attacks)
- [ ] Draw the system architecture figure (the 2-layer flow)
- [ ] Plot: similarity distributions for genuine / clone / imposter — this single
      figure carries the paper's main argument
- [ ] Plot: EER vs. degradation level, and EER vs. chunk length
- [ ] Write the Limitations section using §5 above
- [ ] Write the Abstract last, once the numbers are final

**Still to do — experiments (optional, strengthens the paper)**
- [ ] Repeat the attack test with a *modern* cloning tool to update the 54% figure
- [ ] Add a background-noise / room-echo condition (matches the "noisy environment"
      limitation from your Phase-1 slides better than VoIP codec does)
- [ ] Recalibrate the trust tiers against the measured score distributions
- [ ] Run the accumulator tuning to get time-to-detection numbers

**Housekeeping**
- [ ] Cite the dataset: Todisco et al., *ASVspoof 2019*, Interspeech 2019
      (required by the dataset licence, not optional)
- [ ] Freeze package versions — already captured in `env_snapshot.txt`

---

## 8. Where every number lives
| Result | File |
|---|---|
| Attack experiment (§3.1–3.3) | `results/spoof_vs_speaker_verification.{csv,json}` |
| Anti-spoofing benchmark (§3.4) | `results/aasist_baseline.json`, `WEEK2_RESULTS.md` |
| Classical baseline | `WEEK1_RESULTS.md` |
| Degradation study (§3.5) | `results/degradation_eer_table.csv` |
| Streaming study (§3.6) | `results/streaming_*.csv` |
| Meeting script | `MENTOR_DEMO.md` |
| Paper structure | `PAPER_OUTLINE.md` |
| Integration plan | `INTEGRATION_PLAN.md` |
