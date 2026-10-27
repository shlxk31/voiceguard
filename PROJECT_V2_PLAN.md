# Project v2 — restructuring around the real finding

## What changed
We tried to deploy a state-of-the-art anti-spoofing model (AASIST-L, 0.99%
EER — matching its published result) in a working voice-authorisation system,
and pointed it at real audio recorded on a phone.

**It classified 8 out of 8 real human recordings as synthetic**, every one
with >0.98 confidence.

That is not a bug we should quietly fix. It is the most useful thing this
project has produced, and it should become the project.

---

## Why the old framing was weak (you were right)

The system as designed is *assembly*, not research:
- `trust_score = 100 x similarity` — an arbitrary linear mapping
- Risk tiers (₹500 / ₹2,000 / ₹10,000, trust ≥ 50/70/85) — invented numbers,
  and we showed they're miscalibrated (only 43.5% of *genuine* users reach 70)
- Two pretrained models AND-ed together
- No claim in it that a reviewer couldn't have predicted

A reviewer's question — *"what did you learn that I didn't already know?"* —
had no good answer. Now it does.

---

## The new framing

> **Published anti-spoofing models are benchmarked on studio audio and fail
> catastrophically on real recordings. We quantify the failure, diagnose its
> cause, and measure what actually fixes it.**

This is worth doing because essentially every anti-spoofing paper reports an
ASVspoof EER, very few report what happens on audio recorded on a real phone
in a real room, and the gap we measured is not a small degradation — it is
total inversion of the decision.

### The evidence so far

| Finding | Number |
|---|---|
| AASIST-L on ASVspoof2019 LA eval (71,237 clips) | **0.99% EER** |
| Same model, same threshold, on real iPhone recordings | **100% false-positive rate (8/8)** |
| Under simulated VoIP degradation | 0.99% → 14.00% EER |
| Under 250 ms streaming chunks | 0.51% → 36.43% EER |

### The diagnosed cause

ASVspoof2019 LA's "bonafide" class is VCTK — studio recordings from a
hemi-anechoic chamber. Measured spectral comparison:

| Band | Lab bonafide | Lab spoof | Real iPhone audio |
|---|---|---|---|
| 0–300 Hz | 39.9% | 46.8% | **53.8%** |
| 3–6 kHz | 1.50% | 3.41% | **0.32%** |
| 6–8 kHz | 0.38% | 0.48% | **0.04%** |
| Noise floor | 0.0043 | 0.0186 | **0.0702** |

The model never learned "human vs. synthetic." It learned
**"studio-clean vs. everything else."** Real phone audio is 16× noisier and
has ~10× less high-frequency energy than what it was taught to call real.

Control experiment (already run): applying AAC compression, 48k↔16k
resampling, gain changes, or mild noise to lab audio does **not** trigger the
failure — so it isn't a trivial codec/format artifact. The gap is the
recording environment and microphone chain itself.

---

## Phase A — Establish the failure properly *(mostly done, needs scale)*

- [x] Reproduce the published benchmark (credibility anchor)
- [x] Show the failure on real audio (n=8, one device, one speaker)
- [x] Diagnose the spectral/noise-floor cause
- [x] Rule out codec/resampling/gain as the cause
- [ ] **Scale it up** — n=8 from one phone and one speaker is an anecdote, not
      a result. Need ~10 speakers × 2–3 devices × 2–3 rooms.

**Effort:** one afternoon with classmates. Highest value per hour in the
whole project.

---

## Phase B — Build the test set that doesn't exist *(the real contribution)*

Nobody has a public real-world anti-spoofing test set for *your* setting:
Indian speakers, consumer phones, ordinary rooms. Building a small, careful
one is a genuine, citable artifact — and it's exactly the kind of work an
undergraduate project can do better than a lab with no access to those
speakers.

**Bonafide side:** ~10 volunteers × ~10 utterances × 2 devices × 2 rooms
(quiet / normal). Log device, room, distance for every clip — the metadata is
what makes it useful for analysis.

**Attack side:** clone those same volunteers using *modern* tools, not
2019-era ones. Free/cheap options: Coqui XTTS-v2, F5-TTS, or a free tier of a
commercial cloning service. This matters — our 54% attack-success figure came
from 2019 attacks and is almost certainly an underestimate today.

**Consent:** get written consent for voice collection and cloning. Say so in
the paper; reviewers care, and it's the right thing to do.

**Effort:** the biggest time cost in the plan, and the thing most worth doing.

---

## Phase C — Test what actually fixes it *(the measurable core)*

Try each fix, measure on the Phase B set, report honestly what works:

1. **Threshold recalibration.** Cheapest possible fix: re-pick the threshold
   on real-domain data instead of ASVspoof dev.
   *Critical caveat:* if real human audio scores *higher* on spoofiness than
   actual attacks do (we're measuring this now), no threshold can work and
   this fix is dead on arrival — which is itself a strong result worth
   reporting.

2. **Augmentation fine-tuning.** Retrain AASIST-L on ASVspoof + augmented
   copies (room reverb, background noise, phone-mic frequency response, level
   variation) so it learns invariance to recording conditions rather than
   memorising studio cleanliness. AASIST-L is only 85k parameters, so this is
   genuinely feasible on your M1 — probably the single most likely fix to
   work.

3. **Score normalisation.** Normalise each score against a per-device
   reference distribution.

4. **Train on diverse data.** Evaluate against the public "In-the-Wild"
   dataset (Müller et al.) as an external check — no new collection needed.

**Success criterion, stated in advance:** get real-world false-positive rate
from 100% down to under 10% without pushing attack-detection below 80%.
Committing to the number before running the experiment is what makes it
science rather than tuning until it looks good.

---

## Phase D — Make the system logic non-arbitrary

Fixes your "it's just rules" objection:

- Replace `trust = 100 x similarity` with **calibrated probabilities**
  (fit a logistic calibration on dev data, so a score of 0.9 actually means
  90% likely).
- Fuse the two layers as a **likelihood ratio**, not a boolean AND.
- Add a **third outcome: CHALLENGE**. Binary allow/deny throws away
  uncertainty; a real system re-prompts when it isn't sure. Then report the
  operating curve (fraud caught vs. legitimate users inconvenienced), which is
  the metric a deployed system is actually judged on.
- Recalibrate the ₹ tiers against measured score distributions instead of
  invented thresholds.

---

## What the paper becomes

**Title direction:** *Studio-Clean Is Not Real: Domain Failure of
State-of-the-Art Voice Anti-Spoofing*

**Contributions:**
1. A documented, reproducible failure — 0.99% benchmark EER, 100% real-world
   false-positive rate, same model and threshold.
2. A diagnosis with controls that rule out the easy explanations.
3. A small real-world evaluation set for an under-represented speaker
   population and realistic devices.
4. A measured comparison of mitigations, with a success criterion fixed in
   advance.

This is an honest "reality check" paper. That genre is genuinely valued,
very achievable at your scale, and far more defensible than claiming a novel
architecture.

---

## Ordered next steps

| # | Task | Effort |
|---|---|---|
| 1 | Finish the "is any threshold viable?" check *(running now)* | minutes |
| 2 | Collect Phase B bonafide data from classmates | one afternoon |
| 3 | Generate modern clones of those voices | 1–2 days |
| 4 | Re-run the failure measurement at scale | hours |
| 5 | Augmentation fine-tuning of AASIST-L | 2–3 days |
| 6 | Rebuild decision logic with calibration + CHALLENGE tier | 1 day |
| 7 | Write up | ongoing |

**Keep for the paper regardless:** the benchmark reproduction (0.99%), the
degradation study, and the streaming study. They're solid, already done, and
they establish that the pipeline is correct — which is precisely what makes
the failure finding believable rather than looking like a bug in our code.
