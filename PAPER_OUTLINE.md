# VoiceGuard — Paper Outline

**Working title:** *When Does a Deepfake Detector Stop Working on a Real Call?
Streaming Anti-Spoofing Under VoIP Degradation*

**Target venue:** arXiv preprint → IEEE SLT / SPSC workshop. Workshop-scale
is the honest fit: single dataset, simulated channel, one architecture
family. Don't aim this at ICASSP main track without the extra work in
§Limitations.

---

## The one-sentence claim
State-of-the-art anti-spoofing models are benchmarked on clean audio, but
their advantage over weak classical baselines **largely disappears under
realistic VoIP conditions**, and no prior work reports the latency cost of
running them in a streaming, mid-call setting — we measure both.

## Contributions (what we can actually defend)
1. A controlled VoIP degradation harness (Opus 8/16/32 kbps ×
   Gilbert-Elliott burst packet loss 0/5/15% × PLC on/off) applied to
   ASVspoof2019 LA, with a reproducible clean-vs-degraded EER table for
   two detectors under **frozen, dev-tuned thresholds**.
2. The "advantage collapse" result: AASIST-L's ~15.6× EER advantage over an
   LFCC-GMM baseline on clean audio falls to ~1.9× at 8 kbps / 15% loss.
3. A streaming reformulation of a fixed-input architecture: chunk-level EER
   and per-chunk latency at 250/500/1000 ms windows, quantifying the
   accuracy cost of short-window operation.
4. A cross-chunk accumulation rule with a **time-to-detection vs.
   false-alarm-rate** operating curve — the "how many seconds into the call
   before you can flag it" number, which prior anti-spoofing work does not
   report.

---

## Section plan and which result goes where

### 1. Introduction
Frame: voice-cloning fraud happens on *calls*, but the field benchmarks on
clean studio audio. Cite ASVspoof2019 (Todisco et al. 2019), AASIST (Jung
et al. 2022). Gap: no published streaming (chunked) evaluation with
per-chunk latency AND an accumulation-based mid-call alarm.

### 2. Related work
ASVspoof challenge series; AASIST/RawNet2 architectures; replay (PA) vs.
logical-access (LA) distinction — **state clearly that this paper is LA/DF
only, not replay**; codec-robustness literature; RTCFake (arXiv 2604.23742)
as the closest prior work on re-transmitted modern clones — cite explicitly.

### 3. Experimental setup
- Data: ASVspoof2019 LA, official splits — train 2,580/22,800,
  dev 2,548/22,296, eval 7,355/63,882 (bonafide/spoof).
- Detectors: LFCC-GMM baseline (20 static + Δ + ΔΔ LFCC, mean-pooled,
  n_components=64 chosen on dev); pretrained AASIST-L (85,306 params).
- **Threshold discipline** (emphasize — it's a credibility point):
  all thresholds chosen on dev only, frozen before touching eval, and never
  re-tuned per degradation condition.
- Compute: Apple M1, CPU + MPS. Verified MPS/CPU score agreement to 2e-7.

### 4. Baseline reproduction  → `WEEK1_RESULTS.md`, `WEEK2_RESULTS.md`

| Detector | Dev EER | Eval EER |
|---|---|---|
| LFCC-GMM (n=64) | 9.38% | 18.75% |
| AASIST-L (pretrained) | 0.51% | **0.99%** |

AASIST-L reproduces its published ~0.99% almost exactly → validates the
whole pipeline before any degradation claims are made. Say this explicitly;
it is what makes the later numbers trustworthy.

### 5. VoIP degradation  → `WEEK2_DEGRADATION_RESULTS.md`,
`results/degradation_eer_table.csv`

Headline table (3,000-utterance class-balanced eval subsample):

| Condition | LFCC-GMM EER | AASIST-L EER | AASIST advantage |
|---|---|---|---|
| Clean | 18.30% | 1.17% | 15.6× |
| Opus 32k, 0% loss | 21.27% | 1.40% | 15.2× |
| Opus 16k, 5% loss | 21.03% | 5.23% | 4.0× |
| Opus 16k, 15% loss, PLC off | 24.07% | 12.33% | 2.0× |
| Opus 8k, 15% loss | 26.83% | **14.00%** | **1.9×** |

Supporting points: degradation is monotonic in both bitrate and loss rate
(sanity check the harness is behaving); PLC materially helps (8.97% with
vs. 12.33% without at 16k/15%) — worth its own sentence since real VoIP
stacks always run PLC and omitting it overstates degradation.

### 6. Streaming reformulation  → `results/streaming_*.csv` *(running)*
Chunk-level EER and latency at 250/500/1000 ms, clean and degraded.
Expected story (confirm against final numbers before writing):
short windows degrade accuracy substantially, because AASIST-L
tile-repeats any sub-4.04s chunk to its fixed input length. **Frame this
honestly as the naive-chunking baseline** — it motivates window-aware
fine-tuning as future work rather than claiming a solved streaming model.
Report per-chunk median latency vs. window duration → the real-time
feasibility claim (median latency < window length = keeps up with the call).

### 7. Mid-call alarm  → `results/accumulator_final.json`,
`results/accumulator_tradeoff.png` *(pending §6)*
N-of-M voting over consecutive chunks; sweep on dev, freeze, validate once
on eval. Report false-alarm rate, detection rate, and **median
time-to-detection in seconds**. Also report the fraction of spoof calls
that are *structurally undetectable* (shorter than N chunks, so quorum can
never be reached) — an honest limitation of N-of-M voting on short calls.

### 8. Limitations (write this section properly — it is what makes the
paper credible rather than oversold)
- **Simulated, not real, VoIP.** Opus round-trip + Gilbert-Elliott loss is
  a model of a call, not a call. No jitter buffer, no real network, no
  acoustic echo control.
- **2019-era attacks.** ASVspoof19 LA predates modern zero-shot cloning
  (F5-TTS, XTTS-v2, etc.). Results may not transfer to current threats.
  This is the single biggest caveat — state it early, not buried.
- **Naive chunking.** Short windows are tile-repeated to the model's fixed
  4.04s input rather than the model being retrained for short input.
- **Single architecture family, single dataset.**
- Utterance-level labels inherited by all chunks of an utterance.
- Confidence scale artifact: `confidence = |score − threshold| × 2` assumes
  a threshold at 0.5; with the frozen threshold at 0.9157 this compresses
  the reported confidence range. Affects the accumulator's `min_confidence`
  gate, which is why the sweep includes a `min_confidence = 0` ablation.

### 9. Future work
Window-aware fine-tuning for true streaming; modern-clone stress test;
real-call capture; and — as a clearly separate problem — playback/liveness
detection for smart-speaker false triggering (`liveness_detector.py`,
`LIVENESS_DETECTOR_RESULTS.md`; **currently confounded by recording device,
not yet a result**).

---

## Reproducibility appendix (a table reviewers like, and you already have it)
| Step | Command |
|---|---|
| LFCC-GMM baseline | `tests/run_lfcc_baseline.py` |
| AASIST-L baseline | `tests/run_aasist_baseline.py` |
| Build degraded sets | `tests/build_degraded_eval_set.py` |
| Clean vs degraded | `tests/eval_degraded.py` |
| Streaming windows | `tests/eval_streaming.py` |
| Alarm tuning | `tests/tune_accumulator.py` |
| Live demo | `uvicorn src.api.main:app` |

Package versions frozen in `env_snapshot.txt`. Dataset: ASVspoof2019 LA,
ODC-Attribution license — **cite Todisco et al. 2019, required by the
license, not optional.**

---

## What to do in the mentor meeting
Lead with §4 (the reproduction validates everything) → §5 (the advantage-
collapse table is the memorable result) → demo the FastAPI endpoint live →
§6/§7 as "in flight" → be upfront about §8. The limitations section is a
strength here: it shows you know what the numbers do and don't support.
