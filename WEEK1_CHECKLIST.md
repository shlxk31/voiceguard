# Week 1 — Foundation Checklist

Goal by end of week: baseline detector runs end-to-end on real ASVspoof data,
with a trustworthy (if unremarkable) EER number you can compare everything
else against.

- [ ] `pip install -r requirements.txt` in a fresh venv
- [ ] Download ASVspoof 2019 LA (train/dev/eval) — see data/README.md
- [ ] Implement `extract_lfcc()` in src/detector/lfcc_gmm.py
- [ ] Fit LFCCGMMDetector on ASVspoof19 LA train split
- [ ] Run src/detector/metrics.py::compute_eer() on the DEV split, sweep
      n_components (64/128/256) — pick whichever gives lowest dev EER
- [ ] Freeze threshold from the DEV sweep. Do NOT look at eval-set EER yet.
- [ ] Run once on the EVAL split — this is your reproducible baseline number.
      Expect something in the single-to-low-double digits (%) — if you get
      near-0%, something's leaking between train/dev/eval, double-check splits.
- [ ] Clone https://github.com/clovaai/aasist and get their AASIST-L checkpoint
      loading standalone (outside this repo first, to isolate any issues)
- [ ] Write down today's exact package versions (`pip freeze > env_snapshot.txt`)
      — reproducibility matters once you're writing the paper

Once this is solid, move to Week 2: wire AasistLDetector into this repo's
interface and get its baseline EER on the same splits for comparison.
