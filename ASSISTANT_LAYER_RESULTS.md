# Layer 4, the conversational assistant, and the ablation study

Date: 2026-09-08. All numbers below are reproducible from the scripts named
in each section.

---

## 1. What was added

| Piece | File | What it does |
|---|---|---|
| Groq client | `src/assistant/groq_client.py` | LLM, injection classifier, TTS — behind one firewall rule |
| Spoken response | `src/assistant/respond.py` | Turns a *finished* decision into speech |
| L4 context layer | `src/context/l4.py` | Three sub-signals → one calibrated normality score |
| User accounts | `src/context/profile.py` | Per-user voiceprints + transaction history (SQLite) |
| Command corpus | `tools/build_command_corpus.py` | 240 labelled commands for L4 calibration |
| L4 calibration | `tools/fit_l4_calibrator.py` | Fits L4 on a dev half, reports on the held-out half |
| Ablation study | `tools/run_ablation.py` | Answers "why five layers" with measurement |

### The firewall rule

The LLM is an **attack surface, not an authority**. It may emit *features*
(which pass through the same calibrated-LLR pipeline as ECAPA and AASIST) and
it may *verbalise* a decision already made. It can never produce a verdict, a
threshold, or an LLR. Every response has a deterministic template fallback,
and if the model's phrasing contradicts a DENY the template is used instead.

This is what makes adding a chatbot defensible rather than decorative: the
conversational surface introduces a genuinely new threat — voice prompt
injection — and the *architecture*, not the model's good behaviour, contains it.

---

## 2. Layer 4 results

Calibrated on 120 dev commands, reported on 120 held-out. **The corpus is
template-generated** (`build_command_corpus.py`) — see §5.

```
dev  EER          : 16.67%   (threshold 0.0816, frozen)
eval EER          : 13.34%
eval acc @ frozen : 85.83%
```

Per-family accuracy on the held-out half:

| Family | n | Correct |
|---|---|---|
| genuine | 59 | 89.8% |
| prompt injection | 26 | 92.3% |
| social engineering | 35 | 74.3% |

### Why L4 fuses three sub-signals instead of trusting one

Attack detection by each sub-signal **alone**, on held-out attacks:

| Sub-signal | Attacks caught |
|---|---|
| Prompt Guard 2 (86M) | **24.6%** |
| LLM semantic markers | **91.7%** |
| Behavioural fit | 67.2% |

Prompt Guard is excellent on literal instruction-override phrasing
("ignore all previous instructions" → P = 0.9995) and weak on everything
else. The semantic signal covers the inverse. Neither alone is adequate —
the layering argument reproduced *inside* a single layer.

---

## 3. Two bugs found and fixed on the way

### 3a. The amount was being penalised twice

L4's behavioural signal scored a genuine ₹50,000 payment identically to an
attack, because a large amount is already handled by τ(A) in the risk policy.
Counting it again in L4 double-charged the same fact. The amount deviation is
now floored (`AMOUNT_FLOOR = 0.5`) so it can cost at most a factor of two —
the real work is done by the *derived* threshold, not a learned one.

### 3b. Calibrators were fitted on all 800 trials, with no held-out set

`fit_calibrators.py` claimed in its docstring to use "DEVELOPMENT data" but
fitted L1 and L3 on the full CSV. The ablation would have tested every
calibrator against its own training data. Now: a stratified 50/50 split,
seed 42, written to `results/calibration_split.json`, and `run_ablation.py`
is only permitted to read `eval_index`.

---

## 4. The fusion rule was wrong — and the ablation is what caught it

The engine originally summed calibrated LLRs (standard naive-Bayes fusion).
The first ablation run produced an impossible result: **impostor FAR rose from
0.0% with L1 alone to 47.5% with L1+L3.** Adding a working layer made the
system worse.

The cause, reproducible in three lines:

```
An IMPOSTOR is a real human speaking live.
  L1 (sim=0.12, wrong speaker)  LLR = -2.63
  L3 (P(syn)=0.001, real human) LLR = +2.72
  sum = +0.09  ->  ALLOW
```

L3 is *correct* — an impostor really is a live human. But addition assumes
every layer scores the same hypothesis, and ours do not: L1's "genuine" means
*the enrolled speaker*, L3's means *not a synthesiser*. Summing lets a correct
statement about one proposition cancel a correct statement about another.

**Fix:** defence in depth is a conjunction, not a vote. An attacker must
defeat *every* layer, so the fused score is the weakest link:

```
fused = min over available layers of ( weight × LLR )
```

This has the property the security argument requires: **adding a layer can
only ever lower the fused score**, so a new layer can never open a hole — it
can only cost false rejections, which is precisely the trade the ablation
measures. `fusion_rule="sum"` is retained only as the paper's baseline.

---

## 5. The ablation study — "why five layers?"

`tools/run_ablation.py --fusion both`. 400 composite trials per cell, on
held-out data only. FAR = % of attacks wrongly ALLOWED.

### Conjunctive rule (the operating system)

| Subset | Impostor | TTS clone | Injection | Social eng. | Genuine FRR |
|---|---|---|---|---|---|
| L1 only | 0.0% | 41.5% | 27.5% | 22.0% | 26.2% |
| L1 + L3 | 0.0% | 1.5% | **27.5%** | **22.0%** | 26.2% |
| L1 + L4 | 0.0% | **36.5%** | 11.2% | 11.0% | 38.0% |
| L3 + L4 | **65.0%** | 2.5% | 11.2% | 11.0% | 35.0% |
| **L1+L3+L4 (full)** | **0.0%** | **1.5%** | **11.2%** | **11.0%** | 38.0% |

**Every subset is blind to something, and the blindness is structural:**
drop L1 and impostors walk in (65%); drop L3 and clones walk in (36.5%);
drop L4 and injection/social-engineering walk in (27.5% / 22.0%). The full
system is best or tied-best on every threat column. That is the answer to
"prove five is better than four."

### Naive-Bayes sum (baseline — shown to be invalid)

| Subset | Impostor | TTS clone | Injection | Social eng. | Genuine FRR |
|---|---|---|---|---|---|
| L1 + L3 | 49.5% | 2.2% | 87.8% | 80.5% | 4.0% |
| L1+L3+L4 (full) | 64.5% | 9.2% | 59.8% | 54.8% | 2.0% |

Non-monotonic and worse almost everywhere. Reported as a negative result.

### The cost of being strict

The full system's 38% FRR is not 38% of customers turned away:

| Outcome for genuine users | Share |
|---|---|
| ALLOW | 62.0% |
| CHALLENGE (recoverable by voice) | 27.8% |
| DENY | **10.25%** |

Only ~10% are refused outright; the rest are *asked again*, and the spoken
challenge loop recovers them. This is what the three-outcome policy buys, and
it is why C_challenge ≪ C_reject in the cost model.

---

## 5b. Three faults found by using the system (2026-09-08)

All three were caught from real session data in `data/securevoice.db`, not
from unit tests.

### (a) Genuine live voice scored P(synthetic) = 1.0

Three browser recordings produced a fused score of exactly **-3.2906**, which
is L3's floor at P(synthetic) = 1.0. The same speaker's iPhone `.m4a` clips
score 0.000000.

Opus compression was ruled out (re-encoding a clip to WebM/Opus left the score
at 0.000000). Simulating the browser's default DSP moved it in the right
direction but did not reproduce saturation:

| Condition | P(synthetic) |
|---|---|
| Voice Memos `.m4a` | 0.000000 |
| + simulated noise suppression | 0.000388 |
| + simulated NS and AGC | 0.029236 |

So the browser's DSP contributes (a 75x push toward a false "synthetic"
verdict on genuine speech) but is not the whole cause. The likelier dominant
factor is that the domain adaptation was trained on iPhone `.m4a` clips, and
browser capture is a different microphone *and* a different processing chain
-- a third domain.

**This is a finding, not only a bug: domain adaptation did not transfer
across capture devices.** Mitigation applied: `getUserMedia` now requests
`echoCancellation:false, noiseSuppression:false, autoGainControl:false`,
which is correct regardless -- L3 asks whether a real vocal tract produced a
waveform, so it must not be given a spectrally reshaped signal. Full fix
requires extending the adaptation set with browser-captured audio.

### (b) A bare number was treated as money

`"buy me a Samsung Galaxy Fold 8 Ultra"` was authorised as an **Rs.8**
transaction. The extractor took the largest number when no currency cue was
present, so a model number set the threshold. Because tau(A) is monotonic in
the amount, silently under-estimating the stake is the worst available
failure: it assigns the *lowest* verification bar to a request whose value is
actually unknown.

Fixed in `nlu/understand.py`: a bare number is never an amount. A candidate
needs an adjacent currency cue, a monetary scale word, or a pricing/payment
word within two tokens, and is rejected outright if a unit or product-tier
word follows. Rejected candidates are returned and shown in the UI with the
reason. 12/12 on a regression set including all three observed failures.

### (c) The assistant promised a passphrase that did not exist

The spoken reply said "please repeat your passphrase" while `/api/challenge`
simply re-scored any second utterance. Now implemented
(`assistant/passphrase.py`): the enrolled phrase must come back before the
second utterance's voice evidence counts at all. Verified end to end --
wrong phrase gives DENY regardless of voice quality; correct phrase
accumulates evidence 1.36 -> 2.55 and reaches ALLOW.

**Stated limitation:** the phrase is fixed, so any recording of the user
saying it satisfies the text check. It raises the bar against an attacker who
lacks the phrase and does nothing against one holding a recording. A
system-chosen random phrase would close this, since no pre-recorded clip can
answer an unpredictable prompt -- and would give replay resistance without L2.

---

## 5c. Risk is now set by stake, not by the rupee amount

`risk/stake.py`. An amount-only policy gives A = 0 -- the *lowest* threshold
the system can issue -- to every request carrying no rupee figure:
"unlock the front door", "forward the OTP", "delete all the backups".

The Bayes derivation is untouched. tau needs C_fa, the cost of wrongly
approving; a rupee amount is one way to estimate it, not its definition:

    stake = max( monetary amount , consequence of the action )

Both are in rupees, so tau(stake) is the same formula with a better input,
and the Odyssey 2018 citation still covers it. Measured effect:

| Request | Stake | tau_allow |
|---|---|---|
| "what is the weather" | Rs.0 | -1.38 |
| "buy a Galaxy Fold" (no amount heard) | Rs.1,000 | -0.28 |
| "buy MacBook for 75,000 rupees" | Rs.75,000 | 3.64 |
| "unlock the front door" | Rs.100,000 | 3.93 |
| "forward the OTP" | Rs.100,000 | 3.93 |
| "use my flatmate's card without telling him" | Rs.500,000 | 5.53 |

Unlocking a door now demands more evidence than a Rs.75,000 purchase.

The severity anchors are **declared policy constants**, like
`CostModel.handling_cost` -- business judgements stated openly so a reviewer
can dispute a specific number rather than an opaque score. They are not
measured or learned and the paper must present them as assumptions. What is
defensible is the structure: consequence is monotonic in severity and
irreversibility, and an unrecognised action is priced as non-trivial rather
than free. The LLM only selects a bucket from a fixed enum; it never emits a
rupee figure and never sees the anchor table -- if it could name its own
stake, an attacker controlling the transcript would control the threshold.

---

## 5d. Making ordinary commands cheap and expensive ones expensive

Three faults, found by asking what "turn off the lights" actually costs:

| Command | Category assigned | Stake before | tau before |
|---|---|---|---|
| Turn off the lights | `physical_access` | Rs.50,000 | **3.24** |
| What is the weather | `informational` | Rs.1,500 | 0.01 |
| Order a packet of biscuits | `purchase` | Rs.1,500 | 0.01 |
| Buy me a new MacBook Pro | `purchase` | Rs.1,500 | 0.01 |

1. **A light switch cost as much as a door lock.** The classifier was not
   wrong; the enum was too coarse. `device_control` (Rs.100, trivially
   reversible) is now separate from `physical_access` (defeating a lock,
   safe, or alarm).
2. **The Rs.1,500 fallback floored everything.** A constant cannot be about
   the request, so a weather question demanded exactly the evidence of a
   Rs.1,500 payment. Removed entirely -- there is no numeric fallback now.
3. **Every unpriced purchase was identical.** Biscuits and a MacBook both
   Rs.1,000.

### Price estimation (`risk/pricing.py`)

The stake for a purchase now includes an estimate of what the item costs.

**This is the language model's parametric knowledge, not a live web lookup.**
Groq's search-augmented models are unavailable on this account -- they return
`Request Entity Too Large` once retrieved pages enter context -- so no
live-price path exists here. Spot-checks are good (iPhone 15 Pro Max ->
Rs.139,900, the real Indian launch price) but the figures are frozen at the
model's training cutoff and will go stale. The paper must call this a
knowledge-based estimate, never a market lookup.

**Why a fallible estimate is nonetheless safe to use.** It enters through a
maximum:

    stake = max( spoken amount , category anchor , estimated price )

An *underestimate* -- the dangerous direction, since tau is monotonic in
stake -- is absorbed by the category anchor and changes nothing. An
*overestimate* costs only friction. A hallucinated price can inconvenience a
user but cannot open a hole, which is the property required of a component
this unreliable. For the same reason the model's HIGH-end figure is used
rather than its typical one: between two guesses, take the one demanding
more verification.

Estimates are cached on disk by normalised item text, because the free tier's
8,000 tokens/minute is shared with intent extraction and the spoken reply.

### Result

| Command | Category | Price est. | Stake | tau_allow |
|---|---|---|---|---|
| What is the weather today? | informational | — | Rs.0 | **-1.38** |
| Turn off the lights | device_control | — | Rs.100 | **-1.19** |
| Get me a coffee | reversible_low | — | Rs.200 | -1.04 |
| Order a packet of biscuits | purchase | Rs.200 | Rs.1,000 | -0.28 |
| Buy an iPhone 15 Pro Max | purchase | Rs.179,900 | Rs.179,900 | 4.51 |
| Buy a Royal Enfield Classic 350 | purchase | Rs.180,000 | Rs.180,000 | 4.51 |
| Buy me a new MacBook Pro | purchase | Rs.350,000 | Rs.350,000 | **5.18** |
| Transfer 2.5 lakh to my landlord | account_change | — | Rs.250,000 | 4.84 |
| Forward the OTP to my friend | credential | — | Rs.500,000 | 5.53 |

The bar now spans nearly seven log-odds between a light switch and a laptop,
derived throughout from the same cost formula.

---

## 5e. L2 (liveness) implemented — and L3's browser problem was the DSP after all

50 clips captured through the demo UI on the MacBook: 20 genuine, plus 15
live/playback pairs.

### Correction: the L3 domain gap was the browser's DSP, not the device

Section 5b(a) concluded that browser DSP was "a contributor, not the whole
story", and leaned toward a microphone/device mismatch. **That was wrong.**
Clips captured after `getUserMedia` was switched to
`echoCancellation:false, noiseSuppression:false, autoGainControl:false`:

| Set | n | mean P(synthetic) | flagged (>0.5) |
|---|---|---|---|
| browser, adaptation clips | 20 | 0.066 | **1/20** |
| browser, live clips | 15 | 0.096 | **1/15** |
| iPhone `.m4a` (reference) | 8 | 0.294 | 2/8 |

Browser capture with DSP disabled is now *better* than the phone reference.
The original failure was **8/8 = 100%** false positives; it is now ~5%. No
retraining was required -- my simulation of the browser's noise suppression
was simply far gentler than the real WebRTC processing, which is why the
effect looked partial. **The lesson stands and is stronger: forensic audio
layers must be given the unprocessed capture.**

### L2 liveness (`tools/train_l2_liveness.py`)

Leave-one-pair-out CV over 15 pairs: **100% accuracy, 0% EER**.

That number is exactly what the earlier discredited liveness experiment
reported, so it was not accepted until it survived checks the old one could
not:

1. **The device confound is gone by construction.** The UI records the live
   clip and its own replay back to back through one microphone, seconds
   apart. Previously "live" was iPhone `.m4a` and "playback" was MacBook
   `.wav`, so the classifier separated *file formats*.
2. **Length was equalised.** Replays were recorded 1s longer to avoid
   clipping the start, which would have been perfectly separating on its own.
   All clips are cropped to 5.5s.
3. **Held-out generalisation test.** The 20 adaptation clips are genuine live
   speech that L2 never trained on: **20/20 classified live**, mean
   P(live) = 0.950. The model is not memorising its training pairs.
4. **The discriminating features are physically right.** The strongest are
   low-order MFCC means (mfcc2: 0.87 live vs -30.7 playback) -- the spectral
   envelope reshaping expected from an extra loudspeaker-and-room
   convolution. The old experiment's top features were device and codec
   artifacts instead.

**Scope, which must be quoted with the number:** 15 pairs, one speaker, one
room, one playback device -- the laptop's own speakers, which are poor
reproducers and leave an obvious signature. A high-quality loudspeaker is
untested and should be expected to be considerably harder.

### Live end-to-end: the same utterance, spoken and replayed

| Layer | Spoken live | Replayed |
|---|---|---|
| L1 Voiceprint | +0.48 | +0.21 (passes -- it *is* the right voice) |
| L2 Liveness | **+1.37** | **-1.35** (P(playback) = 0.968) |
| L3 Anti-spoofing | +2.72 | +2.70 (passes -- it *is* a real human) |
| L4 Context | +1.93 | +2.11 |
| **Outcome** | **ALLOW** | **CHALLENGE** |

L1 and L3 both pass the replay, and are right to: it is genuinely the user's
voice and genuinely a human vocal tract. Only L2 objects. This is the
layering argument in a single pair of recordings.

---

## 5f. The complete ablation, all four evidence layers

`tools/run_ablation.py`. Held-out data only. FAR = % of attacks wrongly ALLOWED.

| Subset | Impostor | **Replay** | TTS clone | Injection | Social eng. | Genuine FRR |
|---|---|---|---|---|---|---|
| L1 only | 0.2% | 74.0% | 43.2% | 23.5% | 19.5% | 26.0% |
| L1 + L3 | 0.2% | **74.0%** | 1.5% | 23.5% | 19.5% | 26.0% |
| L1 + L2 + L3 | 0.2% | 1.2% | 1.5% | **23.5%** | **19.5%** | 27.0% |
| L1 + L3 + L4 | 0.2% | **62.0%** | 1.2% | 10.0% | 7.2% | 38.0% |
| **All four (full)** | **0.2%** | **1.2%** | **1.2%** | **10.0%** | **7.2%** | 38.8% |

Every subset is blind to something, and the full system is best or tied-best
on every threat column. The replay column is the clearest single argument for
layering: **74% of replays are approved without L2 and 1.2% with it**, while
L1 and L3 -- both working perfectly -- cannot see the attack at all.

The cost is false rejection: 38.8% of genuine users are not waved straight
through. Only ~10% are refused outright; the remainder are challenged, and
the spoken passphrase loop recovers them.

---

## 6. Threats to validity — must appear in the paper

1. **L4's corpus is template-generated.** Reporting its *separation* is
   honest; reporting its EER as a deployment estimate is not. A template
   corpus cannot say how often real users phrase ordinary requests in ways
   that look adversarial.
2. **Behavioural histories are partly synthetic.** L1 and L3 rest on
   ASVspoof2019 LA; L4 does not.
3. **Composite trials assume independent layer errors.** No corpus carries
   all conditions on one utterance, so trials pair independently measured
   audio and command conditions. Valid for evaluating the *fusion policy*;
   not a claim about a real joint distribution. Correlated failures are
   possible in deployment.
4. **The stake anchors are assumptions, not measurements** (§5c), and the
   action category is assigned by an LLM, so category errors propagate to
   the threshold. Category assignment is not perfectly stable run to run.
5. **Price estimates are model knowledge, not market data** (§5d): frozen at
   the training cutoff, and unvalidated against any price feed. The
   max-combination bounds the damage but does not make them accurate.
6. **The passphrase is fixed**, giving no replay resistance (§5b(c)).
7. **Live testing is a single speaker.** Every live-audio result in this
   project comes from one person's voice. FRR estimates from it do not
   generalise.
8. **L2 (liveness/replay) is not implemented.** The replay threat is
   uncovered by every subset above. It is omitted rather than scored as
   zero. Unblocking it needs ~15 live + ~15 playback clips recorded on the
   **same device, same room, same session** — the confound that invalidated
   the earlier 100% result.

---

## 7. Operational notes

- Groq free tier is **8,000 tokens/minute**. Intent extraction is capped at
  300 max_tokens for this reason; a full 240-command calibration run takes
  ~11 minutes and is rate-limited, not compute-bound.
- The API key lives in `.env` (gitignored). **Rotate it** at
  console.groq.com/keys before the repo is shared.
- Web search (`groq/compound`) is unavailable on the free tier: it returns
  `Request Entity Too Large` once retrieved pages enter context. Not used.
