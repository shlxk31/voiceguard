"""
tools/fit_l4_calibrator.py

Scores the labelled command corpus through Layer 4 and fits its calibrator,
so L4 speaks the same LLR language as L1 and L3 and can enter the fusion.

DISCIPLINE (unchanged from the rest of the project)
The corpus is split into DEV and EVAL. The calibrator is fitted on DEV only.
EVAL is scored once, afterwards, and never used to choose anything. This is
the same rule applied to the LFCC-GMM and AASIST-L baselines and it is not
relaxed here just because the corpus is synthetic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assistant.groq_client import GroqAssistant          # noqa: E402
from src.context.l4 import ContextLayer                       # noqa: E402
from src.context.profile import ProfileStore                  # noqa: E402
from src.risk.calibration import ScoreCalibrator, load_calibrators, save_calibrators  # noqa: E402


def compute_eer(genuine: np.ndarray, attack: np.ndarray) -> tuple[float, float]:
    """EER and its threshold, for scores where HIGHER means more genuine."""
    scores = np.concatenate([genuine, attack])
    best_gap, eer, thr = 1e9, 0.5, 0.0
    for t in np.unique(scores):
        frr = float(np.mean(genuine < t))
        far = float(np.mean(attack >= t))
        if abs(frr - far) < best_gap:
            best_gap, eer, thr = abs(frr - far), 0.5 * (frr + far), float(t)
    return eer, thr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=str(ROOT / "data" / "command_corpus.json"))
    ap.add_argument("--calibrators", default=str(ROOT / "models" / "calibrators.json"))
    ap.add_argument("--scores-out", default=str(ROOT / "results" / "l4_scores.json"))
    ap.add_argument("--dev-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    corpus = json.loads(Path(a.corpus).read_text())
    rows = corpus["rows"]

    # A reference history so the behavioural sub-signal has a baseline.
    store = ProfileStore(ROOT / "data" / "calibration_profile.db")
    store.delete_user("_calib")
    store.create_user("_calib", "calibration reference")
    store.seed_history("_calib", n=40, seed=a.seed)
    history = store.get_user("_calib").history

    layer = ContextLayer(GroqAssistant())
    t0 = time.time()
    scored = []
    for i, r in enumerate(rows, 1):
        res = layer.score(r["text"], amount=r["amount"], history=history)
        if res.normality is None:
            continue
        scored.append({**r, "normality": res.normality,
                       "s_injection": res.s_injection,
                       "s_semantic": res.s_semantic,
                       "s_behaviour": res.s_behaviour,
                       "n_signals": res.n_signals,
                       "pressure": res.intent.pressure,
                       "injection_prob": res.injection_prob})
        if i % 20 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(rows)}  {el:.0f}s elapsed, "
                  f"~{el / i * (len(rows) - i):.0f}s left", flush=True)

    Path(a.scores_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.scores_out).write_text(json.dumps(
        {"synthetic": True, "n": len(scored), "rows": scored}, indent=2))

    # ---- dev / eval split, fit on DEV only --------------------------------
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(scored))
    n_dev = int(len(scored) * a.dev_frac)
    dev = [scored[i] for i in idx[:n_dev]]
    ev = [scored[i] for i in idx[n_dev:]]

    def arrs(rs):
        s = np.array([r["normality"] for r in rs])
        g = np.array([r["is_genuine"] for r in rs], dtype=bool)
        return s, g

    s_dev, g_dev = arrs(dev)
    s_ev, g_ev = arrs(ev)

    cal = ScoreCalibrator.fit(s_dev, g_dev, higher_is_genuine=True, layer_id="L4")

    dev_eer, thr = compute_eer(s_dev[g_dev], s_dev[~g_dev])
    ev_eer, _ = compute_eer(s_ev[g_ev], s_ev[~g_ev])
    acc = float(np.mean((s_ev >= thr) == g_ev))

    print(f"\nL4 calibration ({len(scored)} scored, {len(dev)} dev / {len(ev)} eval)")
    print(f"  dev  EER          : {dev_eer * 100:.2f}%   (threshold {thr:.4f})")
    print(f"  eval EER          : {ev_eer * 100:.2f}%")
    print(f"  eval acc @ frozen : {acc * 100:.2f}%")
    print(f"  slope={cal.slope:.3f} intercept={cal.intercept:.3f} "
          f"prior={cal.train_prior:.3f}")

    # per-family breakdown on eval -- which attacks does L4 actually stop?
    print("\n  eval detection by family (at the frozen dev threshold):")
    for fam in sorted({r["family"] for r in ev}):
        sub = [r for r in ev if r["family"] == fam]
        if not sub:
            continue
        n_flag = sum(1 for r in sub if (r["normality"] < thr) != r["is_genuine"])
        correct = sum(1 for r in sub
                      if (r["normality"] >= thr) == r["is_genuine"])
        print(f"    {fam:20s} n={len(sub):3d}  correct={correct / len(sub) * 100:5.1f}%")

    # sub-signal complementarity -- the paper's argument for fusing three
    print("\n  attack detection by SUB-SIGNAL alone (eval attacks only):")
    atk = [r for r in ev if not r["is_genuine"]]
    for key, label in [("s_injection", "Prompt Guard only"),
                       ("s_semantic", "LLM semantic only"),
                       ("s_behaviour", "behavioural only")]:
        vals = [r[key] for r in atk if r.get(key) is not None]
        if vals:
            print(f"    {label:22s} mean={np.mean(vals):.3f}  "
                  f"caught<0.5: {np.mean(np.array(vals) < 0.5) * 100:.1f}%")

    cals = load_calibrators(a.calibrators) if Path(a.calibrators).exists() else {}
    cals["L4"] = cal
    save_calibrators(cals, Path(a.calibrators))
    print(f"\nwrote L4 calibrator -> {a.calibrators}  (layers now: "
          f"{', '.join(sorted(cals))})")


if __name__ == "__main__":
    main()
