"""
tools/fit_calibrators.py

Fits the L1 and L3 score calibrators on a DEVELOPMENT split and saves them to
models/calibrators.json. Run once; the demo loads the result.

WHY THERE IS A SPLIT AT ALL (this was a bug, and it mattered)
An earlier version of this script fitted on all 800 trials and kept nothing
back, while its docstring claimed to use development data. Nothing downstream
noticed until the ablation study needed an evaluation set: measuring the fused
system on the same trials the calibrators were fitted on would have tested
every calibrator against its own training data and quietly inflated the whole
results table.

The project's rule is that thresholds and calibration are chosen on
development data only and frozen before evaluation. So the 800 trials are
split 50/50, STRATIFIED BY TRIAL TYPE so that genuine, impostor, and spoof
trials are represented in both halves in the same proportion, with a fixed
seed. The calibrators see the dev half. The eval half is written to
results/calibration_split.json and is what the ablation is allowed to use.

L1 (ECAPA similarity) and L3 (AASIST P(synthetic)) are both fitted from
results/spoof_vs_speaker_verification.csv, which labels each trial as
genuine / imposter / spoof.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from risk.calibration import ScoreCalibrator, save_calibrators  # noqa: E402

SEED = 42
CSV_PATH = ROOT / "results" / "spoof_vs_speaker_verification.csv"

rows = list(csv.DictReader(open(CSV_PATH)))
sim = np.array([float(r["similarity"]) for r in rows])
spoof = np.array([float(r["spoof_score"]) for r in rows])
ttype = np.array([r["trial_type"] for r in rows])
is_genuine = ttype == "genuine"

# ---- stratified 50/50 dev/eval split ---------------------------------------
rng = np.random.default_rng(SEED)
dev_mask = np.zeros(len(rows), dtype=bool)
for t in np.unique(ttype):
    idx = np.flatnonzero(ttype == t)
    rng.shuffle(idx)
    dev_mask[idx[: len(idx) // 2]] = True
eval_mask = ~dev_mask

print(f"{len(rows)} trials -> {dev_mask.sum()} dev / {eval_mask.sum()} eval "
      f"(stratified, seed {SEED})")
for t in np.unique(ttype):
    print(f"  {t:10s} dev={int((dev_mask & (ttype == t)).sum()):3d} "
          f"eval={int((eval_mask & (ttype == t)).sum()):3d}")

# ---- fit on DEV ONLY --------------------------------------------------------
cals = {}
cals["L1"] = ScoreCalibrator.fit(sim[dev_mask], is_genuine[dev_mask],
                                 higher_is_genuine=True, layer_id="L1")
# L3 sees only the synthesis threat it is designed for; impostor trials are
# real human speech and would teach it nothing about synthetic detection.
m3 = dev_mask & (ttype != "imposter")
cals["L3"] = ScoreCalibrator.fit(spoof[m3], is_genuine[m3],
                                 higher_is_genuine=False, layer_id="L3")

for k, c in cals.items():
    print(f"\n{k}: slope={c.slope:+.3f} intercept={c.intercept:+.3f} "
          f"prior={c.train_prior:.3f} n={c.n_train}")

# Preserve any calibrator fitted elsewhere (L4 comes from its own script).
out = ROOT / "models" / "calibrators.json"
if out.exists():
    existing = json.loads(out.read_text())
    for k, v in existing.items():
        if k not in cals:
            cals[k] = ScoreCalibrator.from_dict(v)
save_calibrators(cals, out)
print(f"\nSaved -> {out}  (layers: {', '.join(sorted(cals))})")

split_path = ROOT / "results" / "calibration_split.json"
split_path.write_text(json.dumps({
    "seed": SEED, "source": CSV_PATH.name, "stratified_by": "trial_type",
    "note": "eval_index is the ONLY portion the ablation study may use.",
    "dev_index": np.flatnonzero(dev_mask).tolist(),
    "eval_index": np.flatnonzero(eval_mask).tolist(),
}, indent=2))
print(f"Saved split -> {split_path}")

print("\n--- sanity: LLR direction ---")
print(f"L1 sim=0.75 (strong match) -> {cals['L1'].llr(0.75):+.2f}  ({cals['L1'].odds_statement(0.75)})")
print(f"L1 sim=0.10 (no match)     -> {cals['L1'].llr(0.10):+.2f}  ({cals['L1'].odds_statement(0.10)})")
print(f"L3 P(syn)=0.001 (human)    -> {cals['L3'].llr(0.001):+.2f}  ({cals['L3'].odds_statement(0.001)})")
print(f"L3 P(syn)=0.999 (synthetic)-> {cals['L3'].llr(0.999):+.2f}  ({cals['L3'].odds_statement(0.999)})")
