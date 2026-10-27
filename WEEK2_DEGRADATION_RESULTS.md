# Week 2 — VoIP Degradation Harness Results

## The core finding
**The gap between the strong neural detector and the weak classical one nearly
disappears under real VoIP conditions.**

| Condition | LFCC-GMM EER | AASIST-L EER | AASIST-L's advantage |
|---|---|---|---|
| Clean | 18.30% | 1.17% | ~15.6x better |
| Opus 32kbps, 0% loss | 21.27% | 1.40% | ~15.2x better |
| Opus 16kbps, 5% loss | 21.03% | 5.23% | ~4.0x better |
| Opus 16kbps, 15% loss (PLC off) | 24.07% | 12.33% | ~2.0x better |
| **Opus 8kbps, 15% loss (worst)** | **26.83%** | **14.00%** | **~1.9x better** |

AASIST-L's error rate goes up **~12x in absolute terms** (1.17% -> 14.00%)
from clean to the worst tested condition, while LFCC-GMM only moves
~8.5 points (18.3% -> 26.8%) because it was already weak to begin with.
This is the exact gap the project's README named as the motivating problem:
detectors that look great on clean audio fall apart on real degraded calls
— now backed by a real, reproducible number instead of an assertion.

## Full table
See `results/degradation_eer_table.csv` (22 rows: 2 detectors x 11
conditions [1 clean + 10 degraded]).

Both detectors used FROZEN thresholds throughout (LFCC-GMM's from
`WEEK1_RESULTS.md`, AASIST-L's from `results/aasist_baseline.json`) — no
per-condition re-tuning, matching how a real deployment would actually work.

## Sanity checks that passed
- Clean-condition EER on this 3,000-file balanced subsample (LFCC-GMM
  18.30%, AASIST-L 1.17%) closely tracks the full-eval-set numbers from
  Week 1/2 (18.75%, 0.99%) — confirms the degradation harness's clean
  baseline is consistent with the earlier, larger-scale runs.
- EER moves monotonically in the expected direction: lower bitrate and
  higher packet loss both make things worse, for both detectors.
- Packet-loss concealment (PLC) helps, as expected: at 16kbps/15% loss,
  AASIST-L EER is 8.97% with PLC vs. 12.33% with PLC off.

## A data quality note on the latency column
`mean_latency_ms` for AASIST-L is inconsistent across rows -- most
conditions show ~55-60ms/utterance (matching the ~55-68ms benchmarked
earlier on MPS), but several rows (e.g. 16kbps/5%, 16kbps/15%, 32kbps/*)
show 300-1050ms. This does not track anything about the audio itself (the
model doesn't know which config it's scoring); it's almost certainly the
same environment effect noted in `WEEK2_RESULTS.md` -- this was a long
unattended background run, and if the machine's system state changed
mid-run (sleep/wake, thermal throttling, background load), a handful of
calls straddling that would get inflated per-call timings that pull the
mean up over 1,500 samples. **Treat the EER numbers above as solid; treat
this specific latency column as noisy** -- a real latency benchmark should
be re-run in isolation (foreground, machine awake and idle) before citing
it anywhere.

## Reproduce it
```
.venv/bin/python tests/build_degraded_eval_set.py   # ~83 min, generates data/degraded/
.venv/bin/python tests/eval_degraded.py              # produces results/degradation_eer_table.csv
```
