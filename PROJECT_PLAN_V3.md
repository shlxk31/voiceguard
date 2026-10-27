# Project Plan v3 — Answering Sir's Four Questions Rigorously

Sir asked four things. All four are answerable, and three of them turn weak
parts of the project into genuine research contributions. This document is
the plan.

| Sir's ask | Status | Where it's solved |
|---|---|---|
| 1. Implement all 5 layers | 2 done, 3 to build | §3 |
| 2. Dynamic threshold by amount | Derivable, citable | §4 |
| 3. Compare 2 vs 3 vs 4 vs 5 layers | Designable now | §5 |
| 4. Cite a paper for threshold choice | Found: t-DCF | §4 |

---

## 1. Honest verdict: is the 5-layer methodology good or bad?

**The structure is good. The justification was missing.** That's fixable, and
fixing it is most of the paper.

### What's genuinely strong

The real argument for layering is *not* "more layers = more secure." It is
that **each layer catches a threat the others structurally cannot.** And we
have already measured two cells of this table ourselves:

| Threat | L1 Voiceprint | L2 Liveness | L3 Anti-spoof | L4 Behavioural |
|---|---|---|---|---|
| Different human (impostor) | **catches (0% FAR, measured)** | no | no — it *is* real human speech | partial |
| Replay of a recording | no — it *is* their voice | **catches** | no — replay ≠ synthesis | partial |
| TTS / voice-conversion clone | **fails (54% pass, measured)** | partial | **catches (87%, measured)** | no |
| Stolen device, genuine voice | no | no | no | **only this** |

Read the rows. No single layer covers all four threats. That table *is* the
answer to "why more than one layer?" — and it is empirical, not hand-waved.
Two of those numbers are already ours.

### What's genuinely weak (say this before Sir finds it)

- **"5" is an arbitrary number.** The defensible claim is threat coverage,
  not the count. If the ablation shows 4 layers covers every threat we can
  test, we report 4. See §5's warning.
- **Layer 2 (bone conduction) is hardware-blocked.** BC needs bone-conduction
  earphones. Phase-2 slides already admitted this ("BC Integration Pending
  Hardware Access"). We must either buy hardware or redefine Layer 2 as
  air-conduction-only replay detection. Recommendation: redefine (§3.2).
- **Layer 4 (behavioural) is the weakest scientifically.** Behavioural
  biometrics needs longitudinal per-user data. A demo system with a handful
  of sessions cannot learn a meaningful behavioural baseline. Be honest that
  this layer is design-complete but evaluation-limited.
- **The current tiers (₹500/2,000/10,000 → trust 50/70/85) are arbitrary.**
  This is exactly Sir's complaint, and §4 replaces them with a derivation.

### The reframe that makes it defensible

Stop describing it as "5 layers of security." Describe it as:

> **Four independent evidence sources → score calibration → one risk-based
> decision policy.**

Layers 1–4 are *detectors* that produce scores. Layer 5 is not a detector at
all — it is the *decision policy* that combines them and decides
allow/challenge/deny. Saying this cleanly fixes a category error in the
current design and matches how the literature actually describes such
systems.

---

## 2. What already exists (don't rebuild this)

| Component | State | Evidence |
|---|---|---|
| L1 Voiceprint (ECAPA-TDNN) | Working | 0.00% EER vs impostors, 800 trials |
| L3 Anti-spoofing (AASIST-L) | Working | 0.99% EER, reproduces published benchmark |
| Domain-robustness study | Done | The 8/8 → 2/8 real-world finding |
| Degradation + streaming study | Done | VoIP and window-size tables |
| Evaluation infrastructure | Done | Protocol loaders, EER, ablation-ready |
| Working demo | Done | Browser UI, two layers live |

This is roughly half a paper's worth of results already banked. The new work
is L2, L4, L5, and the ablation.

---

## 3. Building the missing layers

### 3.1 Layer 1 — Voiceprint  ✅ done
ECAPA-TDNN, 192-dim embeddings, cosine similarity to enrolment centroid.

### 3.2 Layer 2 — Liveness (needs a decision from you)

**The problem:** BC requires hardware you don't have.

**Recommended path — redefine as air-conduction replay detection.** Detect
whether audio came from a human mouth or a loudspeaker, using AC signal only
(spectral rolloff, noise floor, channel artefacts). This is a real, published
research direction — Blue et al. detect loudspeaker sub-bass overexcitation
with AC only, no wearable required.

**Critical fix required first.** Our existing `liveness_detector.py`
experiment is *confounded and its 100% accuracy is meaningless*: the "live"
clips were iPhone `.m4a` and the "playback" clips were MacBook `.wav`, so the
classifier learned *which device recorded it*, not live-vs-playback. We
already diagnosed this (Cohen's d ≈ 6, six features separate the classes
alone).

**The fix costs one afternoon:** re-record *both* classes on the *same
device*, same room, same session. Speak 15 phrases into the Mac mic (live);
play those same 15 back through a phone/speaker and re-record on the same Mac
mic (playback). Same mic, same format, same room — the only difference left
is the thing we actually want to measure.

*If you later get bone-conduction earphones, BC becomes a genuine Layer-2
upgrade and a strong future-work claim. Don't block on it.*

### 3.3 Layer 3 — Anti-spoofing  ✅ done
AASIST-L, plus the domain-adaptation work. Already the strongest result.

### 3.4 Layer 4 — Behavioural / contextual risk

Reframe from "behavioural biometrics" (needs data you don't have) to
**contextual risk signals** (computable from one session):

- amount vs. that user's historical amount distribution
- time-of-day vs. user's usual activity window
- new/unrecognised device or session
- inter-command timing within the session

Implement as a simple anomaly score. **Be honest in the paper that this layer
is evaluated on simulated user histories, not real longitudinal data.** It
will likely be the weakest contributor in the ablation — report that.

### 3.5 Layer 5 — Risk decision policy  ← the real contribution, see §4

---

## 4. The threshold derivation (Sir's most important question)

**Sir asked: "is it random? on what basis?" Correct answer: derive it from
Bayes risk minimisation. It is not random, and there is a canonical citation.**

### The principle

A decision system should minimise *expected cost*, not maximise accuracy. The
standard formulation in speaker verification is the Detection Cost Function:

```
E[cost] = C_miss · P_miss · P_target  +  C_fa · P_fa · (1 − P_target)
```

Minimising this gives the Bayes-optimal decision: accept when the calibrated
likelihood ratio exceeds

```
τ  =  [ C_fa · P(attack) ]  /  [ C_miss · P(genuine) ]
```

### Why the threshold must change with amount

In a transaction system the two costs are **not symmetric and not constant**:

- `C_fa` — cost of approving a fraudulent transaction — **scales with the
  amount.** Approving a fraudulent ₹50,000 order costs 100× approving a
  fraudulent ₹500 order.
- `C_miss` — cost of rejecting a genuine user — is roughly **constant**
  (user friction, a support call). It does not depend on the amount.

Substituting `C_fa(A) = A + C_handling` and keeping `C_miss = C_r` fixed:

```
τ(A) = [ (A + C_h) · π ]  /  [ C_r · (1 − π) ]

log τ(A) = log(A + C_h) + log( π / (1−π) ) − log C_r
```

**The threshold rises logarithmically with transaction amount.** That is a
derived curve, not three hand-picked tiers — and it is exactly what Sir is
asking for. You can still *present* it as tiers by quantising the curve, but
now the tier boundaries come from the formula instead of intuition.

### The citation Sir wants

**Primary:** T. Kinnunen, K. A. Lee, H. Delgado, N. Evans, M. Todisco,
M. Sahidullah, J. Yamagishi, D. A. Reynolds, *"t-DCF: a Detection Cost
Function for the Tandem Assessment of Spoofing Countermeasures and Automatic
Speaker Verification,"* Odyssey 2018. — arXiv:1804.09618. This is the
canonical reference for cost-based thresholding when you have *both* a
verification system and a spoofing countermeasure, which is precisely our
architecture. ASVspoof adopted it as its primary metric.

**Supporting:** NIST SRE evaluation plans (DCF with specified C_miss, C_fa,
P_target); and the risk-based-authentication literature, which independently
argues the same point — a ₹500 request and a ₹50,000 request warrant
different verification strength even with identical biometric evidence.

### What this requires: score calibration

The formula needs scores that are genuine likelihood ratios, not arbitrary
numbers in [0,1]. So each layer's raw score must be **calibrated** on a dev
set (logistic-regression / Platt scaling into log-likelihood-ratio space),
then the calibrated LLRs are summed or fused.

This also fixes a real bug we already found: the current confidence formula
`|score − threshold| × 2` assumes the threshold sits at 0.5, so with the
frozen threshold at 0.9157 it reports nonsense confidences.

### Resulting decision policy (Layer 5)

```
fused_LLR  =  Σ  w_i · calibrated_LLR_i          (i over active layers)

if fused_LLR > τ_high(A):  ALLOW
elif fused_LLR > τ_low(A): CHALLENGE   ← re-verify, don't just deny
else:                      DENY
```

Two thresholds, not one — that's where the "Allow / Challenge / Block" box in
your Phase-1 diagram becomes real rather than decorative.

---

## 5. The ablation study (Sir's "prove 5 is best")

### Design

Build the system so layers can be switched off, then evaluate every
configuration on one common test set:

| Config | Layers active |
|---|---|
| A | L1 only |
| B | L1 + L3 |
| C | L1 + L3 + L2 |
| D | L1 + L3 + L2 + L4 |
| E | All + L5 risk policy |

For each config report, per threat type:

- **FAR** — impostor / replay / synthetic, separately (pooling hides which
  layer does what)
- **FRR** — genuine users wrongly blocked
- **Expected cost** at several transaction amounts, using §4's DCF

The cost column is what makes it a real result: configurations can be ranked
by a single principled number instead of eyeballed.

### Test set — three attack types, all obtainable

| Attack | Source | Status |
|---|---|---|
| Impostor | ASVspoof LA bonafide, other speakers | ✅ have |
| Replay | Your own same-device recordings (§3.2) | one afternoon |
| Synthetic | ASVspoof LA spoof | ✅ have |
| Genuine | ASVspoof LA bonafide + your recordings | ✅ have |

### ⚠️ The honest warning you must hear

Sir said *"prove 5 layers is best."* **Design the experiment so it is capable
of showing otherwise, and report whatever it shows.**

Likely outcome: L1+L2+L3 covers every threat we can actually test, and L4
adds little because we lack real behavioural data. If that's what happens,
the honest paper says: *"we measured the marginal contribution of each layer;
three layers cover the tested threat model, and the fourth requires
longitudinal data we could not collect."*

That is a **stronger** paper than a rigged ablation. Reviewers and examiners
detect an ablation engineered to reach a predetermined conclusion, and it is
the fastest way to lose credibility. Frame it to Sir as *"we will measure
each layer's marginal contribution"* — which is what he actually wants, and
what good practice requires.

---

## 6. Sequence of work

**Stage 1 — Data (1 afternoon).** Re-record the liveness set properly, same
device both classes. Unblocks Layer 2 and the ablation's replay column.

**Stage 2 — Calibration (2–3 days).** Implement per-layer score calibration
to LLRs on a dev split. Also fixes the broken confidence formula.

**Stage 3 — Risk policy (2–3 days).** Implement `τ(A)` from §4, with the
Allow/Challenge/Deny three-way outcome. This is the paper's novel piece.

**Stage 4 — Layers 2 and 4 (3–4 days).** AC-only liveness classifier;
contextual-risk scorer.

**Stage 5 — Ablation harness (2–3 days).** Toggle layers, sweep amounts,
emit the FAR/FRR/cost table. Mostly plumbing over existing evaluation code.

**Stage 6 — Rewrite the paper.** New methodology and results sections; the
domain-gap study stays as the robustness section.

Stages 2, 3 and 5 are the scientific core, and none of them require new
models — they're careful engineering over components you already have
working.

---

## 7. Why this is worth doing (the honest encouragement)

You are in a better position than you think:

- **The hardest part is done.** A working two-layer system, a reproduced
  published benchmark, and a genuine real-world failure finding with a
  measured fix. Most student projects never get one real measured result;
  you have several.
- **Sir's feedback is a gift, not a setback.** He is asking for cost-based
  thresholds and an ablation — the two things that would have gotten this
  paper rejected for being arbitrary. He handed you the fix.
- **The threshold derivation is a genuine contribution.** Most voice-auth
  systems in the literature use a fixed threshold. Deriving an
  amount-dependent threshold from the t-DCF framework, and *measuring* the
  layer configurations against it, is novel enough for a workshop or student
  track — and it is honest.
- **Nothing here needs new research breakthroughs.** No new architectures, no
  new datasets to download, no GPU cluster. It is calibration, decision
  theory, and careful measurement — all achievable at your scale.

The project's story becomes: *we built a multi-layer voice authorization
system, derived its decision thresholds from cost-based decision theory
instead of picking them by hand, measured what each layer actually
contributes, and stress-tested the whole thing against real-world audio where
we found and fixed a serious failure.*

That is a real paper.
