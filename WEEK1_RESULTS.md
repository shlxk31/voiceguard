# Week 1 Baseline — Results

## Headline number
**LFCC-GMM baseline, ASVspoof2019 LA eval EER: 18.75%**

## Config (frozen on DEV, never touched after seeing eval)
- Detector: `LFCCGMMDetector` (src/detector/lfcc_gmm.py)
- Features: 20 static + 20 delta + 20 delta-delta LFCC (60-dim), mean-pooled
  over time (`extract_lfcc()`, torchaudio LFCC transform, 25ms/10ms
  window/hop, 16kHz)
- `n_components`: 64 (swept 64 / 128 / 256 on DEV; dev EER was 9.38% / 9.61%
  / 12.01% respectively — 64 won)
- Threshold: 5.5996 (LLR score cutoff, frozen from the DEV EER sweep)

## Discipline followed
1. Fit bonafide/spoof GMMs on TRAIN only.
2. Swept `n_components` and picked the threshold on DEV only.
3. Froze that config, then ran EVAL exactly once. The number above is that
   one run — not re-tuned afterward.

## Data verified
Real ASVspoof2019 LA (Edinburgh DataShare, ODC-Attribution license,
no account required — cite per data/README.md / Todisco et al. 2019 in the
paper's Related Work / references).

| split | bonafide | spoof | total |
|-------|----------|-------|-------|
| train | 2,580    | 22,800| 25,380|
| dev   | 2,548    | 22,296| 24,844|
| eval  | 7,355    | 63,882| 71,237|

(dev/eval flac folders each also contain a handful of extra `LA_*_A*`
utterances used only for the ASV/t-DCF side of the challenge — not in the
CM protocol, not used here.)

## Why 18.75% and not the ~8-9% in the ASVspoof19 paper
This is a legitimate, expected gap, not a leak or a bug:
- Mean-pooling LFCC over time discards temporal structure the official
  baseline's scoring doesn't (this is called out directly in
  `lfcc_gmm.py`'s docstring — it's exactly the gap AASIST-L closes next).
- `n_components` capped at 256 in the sweep vs. the official 512 (speed
  during iteration).

## Reproduce it
```
.venv/bin/pip install -r requirements.txt   # see env_snapshot.txt for exact versions
.venv/bin/python tests/run_lfcc_baseline.py --data-root data/raw/asvspoof2019_la
```
Full run takes ~6 minutes on an Apple M1 (CPU only, no GPU used).
Package versions frozen in `env_snapshot.txt` (generated same day as this run).

## Next (Week 2, per README)
Wire `AasistLDetector` (already confirmed loading the pretrained AASIST-L
checkpoint standalone — see src/detector/aasist_l.py) through the same
train/dev/eval protocol for a comparable EER number.
