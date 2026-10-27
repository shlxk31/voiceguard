# Week 2 — AASIST-L Comparison Baseline

## Headline number
**AASIST-L, ASVspoof2019 LA eval EER: 0.99%** — matches the published
AASIST-L number (~0.99% per the original paper) almost exactly.

## Comparison with Week 1's LFCC-GMM baseline

| | LFCC-GMM | AASIST-L |
|---|---|---|
| Dev EER | 9.38% | 0.51% |
| **Eval EER** | **18.75%** | **0.99%** |
| Frozen threshold | 5.5996 (LLR) | 0.9157 (P(spoof)) |
| Eval accuracy @ frozen threshold | — | 88.87% |
| Mean latency | ~ms (CPU) | 67ms/utterance (MPS) |

AASIST-L's ~19x lower eval EER vs. LFCC-GMM is expected: it uses the full
raw waveform through a graph-attention network instead of mean-pooled
handcrafted features, and it's the architecture the original ASVspoof19 LA
challenge was designed around.

## A real bug found and fixed along the way
`AasistLDetector.score()` was reading `softmax(logits)[:, 1]` as P(spoof).
This checkpoint's own training convention is the opposite: **class index 0
= spoof, class index 1 = bonafide**. Verified by comparing EER under both
conventions on a dev subsample — index 1 gave ~100% EER (perfectly
inverted), index 0 gave ~0-1% (correct). Cross-checked against
`clovaai/aasist`'s own `main.py` scoring code to confirm the convention.
Fixed in `src/detector/aasist_l.py`. This also means the live FastAPI demo
built the same day AASIST-L was first wired in had been returning inverted
results until this fix — no separate demo code change was needed since it
shares the same `.score()` method.

## Performance note: batching vs. MPS
Tested batching multiple chunks into one forward pass (a planned speed-up)
before running the full eval — **it was 2-3x SLOWER**, not faster (177ms/item
at batch_size=1 vs. 510ms/item at batch_size=8) on this Apple M1 CPU, likely
because AASIST's graph-attention pairwise attention board scales poorly
with batch size and 4 CPU threads already saturate a single sample's
compute. Reverted. Switching to `device="mps"` (Apple GPU) instead gave a
genuine ~3.2x speedup (177ms -> ~55-68ms/item), verified numerically
identical to CPU (max diff 2e-7 on a 16-sample check) before trusting it
for the frozen run.

## Reproduce it
```
.venv/bin/python tests/run_aasist_baseline.py --data-root data/raw/asvspoof2019_la
```
Full run: ~24,844 dev + 71,237 eval utterances, ~67ms/utterance on MPS.
Real wall-clock time can be much longer than the ~1.5h of active compute if
the machine sleeps mid-run (macOS suspends the process; it resumes
correctly on wake, just with inflated elapsed time) — the reported
`mean_latency_ms_*` figures reflect only active per-call time, not wall
clock. Package versions in `env_snapshot.txt`.

## Next
`tests/eval_degraded.py` — clean vs. VoIP-degraded (Opus + packet loss) EER
comparison for both detectors, using these frozen thresholds.
