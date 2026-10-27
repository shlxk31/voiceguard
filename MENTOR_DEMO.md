# Demo Guide & Mentor Pitch

Everything below is verified working as of the last run.

---

## Part 1 — How to test the demo

### Start it
```bash
cd ~/Documents/voiceguard
.venv/bin/uvicorn src.api.demo_app:app --port 8900
```
Open **http://127.0.0.1:8900** in a browser. First launch takes ~10 s while
the two models load.

### Test A — the system works (30 seconds)
1. Click **Record enrolment clip** three times, speaking normally each time.
2. Click **Enrol (3 clips)**.
3. Leave the amount at **Rs.1,500**, click **Record & check**, and say anything.
4. Expect: high similarity, low P(synthetic), **ORDER APPROVED** on both rows.

### Test B — the attack that beats speaker verification (the money shot)
Upload a synthetic attack clip instead of recording:
```
data/raw/asvspoof2019_la/ASVspoof2019_LA_eval/flac/LA_E_8339197.flac
```
Verified result:

| | |
|---|---|
| Speaker similarity | 0.662 — **passes** the 0.50 gate |
| P(synthetic) | 0.9995 |
| Speaker verification only | **ORDER APPROVED** |
| With anti-spoofing | **ORDER DENIED** |

The UI highlights this case explicitly: *"passed speaker verification and was
stopped only by the anti-spoofing layer."*

### Test C — the domain-gap failure and the fix
Upload one of your own recordings:
```
recordings/live/live5.m4a
```
Verified result:

| Model | P(synthetic) | Verdict |
|---|---|---|
| Original AASIST-L (0.99% benchmark EER) | 0.9865 | **SYNTHETIC** — wrong |
| Domain-adapted (ours) | 0.0000 | human — correct |

This is a real human recording. The benchmark-grade model is confidently
wrong; the adapted one is right.

### Test D — the fix didn't break attack detection
Re-upload the spoof file from Test B and look at the domain-adapted row:
P(synthetic) = **0.99999 → SYNTHETIC**. The fix corrected false alarms on real
audio *without* losing the ability to catch actual attacks. Show this
immediately after Test C — it pre-empts the obvious objection.

### Terminal alternative (no browser)
```bash
.venv/bin/python demo_two_layer.py --scenario genuine
.venv/bin/python demo_two_layer.py --scenario spoof
.venv/bin/python demo_two_layer.py --scenario imposter
```

### If something breaks
- Port busy: `pkill -f "uvicorn src.api.demo_app"` then restart.
- Mic not found: `.venv/bin/python liveness_detector.py list-devices`; grant
  the browser microphone permission when prompted.
- Slow first request: models load lazily on first call — that's expected.

---

## Part 2 — How to pitch it

### The one sentence
> "We built the system, tested it honestly, and found that a model with 0.99%
> published error rate calls 100% of our real voices fake. We diagnosed why
> and fixed most of it."

### Five-minute structure

**1. The gap (30 s).** Speaker verification asks *"is this the right voice?"*
It never asks *"is this a real human?"* A cloned voice is built to sound like
the target, so cloning pushes similarity *up*, not down.

**2. Prove it (Test B, 1 min).** 54% of synthetic attacks clear the
verification gate across 800 trials. Zero impostors do. Same layer, same
threshold, two threat models, completely different outcomes.

**3. Establish credibility (30 s).** "Before trusting our own numbers, we
reproduced AASIST-L's published benchmark: we measured 0.99% EER on all
71,237 evaluation utterances; the paper reports 0.99%. Our pipeline is
correct." — This is the sentence that makes everything after it believable.

**4. The finding (Test C, 1.5 min).** "Then we pointed it at our own voices.
8 out of 8 flagged as fake, all above 0.98 confidence. We diagnosed it: the
benchmark's 'real' audio is studio recordings from an anechoic chamber. Our
phone audio has 16× the noise floor and a tenth of the high-frequency energy.
The model learned *studio-clean*, not *human*." Then note Müller et al.
(Interspeech 2022) found the same effect independently on different data —
that turns "our bug" into "a known field-wide problem we reproduced."

**5. The fix (Test D, 1 min).** 8/8 → 2/8 false positives, and it still
catches attacks. Cost: benchmark EER 2.00% → 8.12%. State the cost plainly —
naming the trade-off is what makes it a result rather than a sales pitch.

**6. What's next (30 s).** Scale the adaptation across more speakers, devices
and rooms; test against modern cloning tools, which likely push the 54%
higher.

### Anticipated questions

**"Isn't 2/8 still bad?"** Yes. It's a proof of concept on 8 clips from one
phone in one room — we're explicit about that in the paper's limitations.
It shows the approach works; it doesn't show it generalizes yet.

**"Did you just overfit to your own clips?"** We held two clips out entirely.
Both flipped from >0.94 to <0.05 P(synthetic) without ever being trained on.

**"Why did benchmark accuracy get worse?"** That's the trade-off being
measured, not a bug. The model gave up some studio-audio precision to stop
rejecting real users. For a deployed system, a 100% false-reject rate on real
customers is far worse than 8.12% EER.

**"Is this novel?"** The generalization gap is known — Müller et al. published
it in 2022. What we add: reproducing it on consumer-device audio for a
specific deployment, measuring the *downstream authorization consequence*
inside a working system, and quantifying a mitigation's cost. We say this
explicitly in the paper rather than overclaiming.

### One thing to raise proactively
Three of the seven references on the Phase-1 slide deck do not exist — the
Yadav & Rai, Liu et al., and Chen et al. entries return no matching
publication. If those came from an AI-assisted draft, they need removing
before any submission; fabricated citations are the fastest way to sink an
otherwise sound paper. All 20 references in the new paper were verified
against the actual published sources.
