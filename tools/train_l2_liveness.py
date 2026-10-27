"""
tools/train_l2_liveness.py

Trains and calibrates Layer 2 (liveness / replay detection) on the paired
corpus captured through the demo UI.

WHY THIS SCRIPT IS SO DEFENSIVE
The previous liveness experiment in this project reported ~100% accuracy and
was worthless: the "live" clips were iPhone .m4a files and the "playback"
clips were MacBook .wav files, so the classifier separated the RECORDING
DEVICE and never learned anything about liveness. Cohen's d was around 6 and
six individual features separated the classes on their own -- a result far
too good to be true, and it wasn't.

The corpus this script consumes was captured to make that specific mistake
impossible: each pair is recorded back to back through one microphone in one
session, the live clip first and its own replay seconds later. Same room,
same mic, same speaker, same everything except the thing under test.

Three further controls, because one fixed confound does not mean none remain:

  1. LENGTH. Replays were recorded for one second longer than the live clip
     to avoid clipping the start of playback. Duration alone would therefore
     be perfectly separating. Every clip is cropped to CROP_SECONDS.

  2. LEAVE-ONE-PAIR-OUT CV. A random split could put a live clip in train and
     its own replay in test, letting the model match on utterance content
     rather than channel. Both members of a pair are always held out
     together.

  3. A SINGLE-FEATURE AUDIT. If one feature separates the classes almost
     perfectly, that is the signature of a leak, not of a good detector.
     The script prints the strongest individual feature so the result can be
     doubted properly rather than accepted because the number looks nice.

HONEST LIMITS
15 pairs, one speaker, one room, one playback device. This is enough to fit
and report a layer; it is not enough to claim a general replay detector. Any
number produced here must be reported with that stated.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from liveness_detector import extract_features, SAMPLE_RATE      # noqa: E402
from src.api.secure_voice_app import _decode                      # noqa: E402
from src.risk.calibration import (ScoreCalibrator, load_calibrators,  # noqa: E402
                                  save_calibrators)

#: Every clip is cropped to this many seconds. See control (1) above.
CROP_SECONDS = 5.5


def load_clip(path: str) -> np.ndarray:
    audio = _decode(open(path, "rb").read())
    n = int(CROP_SECONDS * SAMPLE_RATE)
    if len(audio) < n:                      # pad rather than drop the clip
        audio = np.pad(audio, (0, n - len(audio)))
    return audio[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", default="adapt")
    ap.add_argument("--captures", default=str(ROOT / "data" / "captures"))
    ap.add_argument("--calibrators", default=str(ROOT / "models" / "calibrators.json"))
    ap.add_argument("--model-out", default=str(ROOT / "models" / "l2_liveness.json"))
    a = ap.parse_args()

    base = Path(a.captures) / a.user
    live = sorted(glob.glob(str(base / "live" / "*")))
    play = sorted(glob.glob(str(base / "playback" / "*")))
    n = min(len(live), len(play))
    if n < 5:
        sys.exit(f"need at least 5 pairs, found live={len(live)} playback={len(play)}")
    live, play = live[:n], play[:n]
    print(f"{n} pairs from {base}")

    X_live = np.stack([extract_features(load_clip(f), SAMPLE_RATE) for f in live])
    X_play = np.stack([extract_features(load_clip(f), SAMPLE_RATE) for f in play])
    X = np.vstack([X_live, X_play])
    # Label convention matches every other layer: True = genuine (live).
    y = np.concatenate([np.ones(n, bool), np.zeros(n, bool)])
    pair_id = np.concatenate([np.arange(n), np.arange(n)])
    print(f"feature matrix {X.shape}")

    # ---- control 3: is any single feature doing all the work? -------------
    d = []
    for j in range(X.shape[1]):
        a_, b_ = X[y, j], X[~y, j]
        sd = np.sqrt((a_.var() + b_.var()) / 2) + 1e-12
        d.append(abs(a_.mean() - b_.mean()) / sd)
    d = np.array(d)
    order = np.argsort(d)[::-1]
    print("\nsingle-feature separation (Cohen's d), strongest first:")
    for j in order[:5]:
        print(f"  feature {j:3d}  d = {d[j]:.2f}")
    if d.max() > 3.0:
        print("  !! a single feature separates the classes almost perfectly.")
        print("     Treat this as a suspected leak, not a good detector.")

    # ---- leave-one-pair-out cross-validation ------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    oof = np.zeros(len(y))
    for p in range(n):
        te = pair_id == p
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, C=0.5).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]

    pred = oof >= 0.5
    acc = float((pred == y).mean())
    tpr = float(pred[y].mean())
    tnr = float((~pred[~y]).mean())
    print(f"\nleave-one-pair-out CV over {n} pairs ({len(y)} clips)")
    print(f"  accuracy              : {acc * 100:.1f}%")
    print(f"  live  correctly kept  : {tpr * 100:.1f}%  ({int(pred[y].sum())}/{n})")
    print(f"  replay correctly caught: {tnr * 100:.1f}%  ({int((~pred[~y]).sum())}/{n})")

    # ---- EER on the out-of-fold scores ------------------------------------
    g, at = oof[y], oof[~y]
    best, eer, thr = 1e9, 0.5, 0.5
    for t in np.unique(oof):
        frr, far = float((g < t).mean()), float((at >= t).mean())
        if abs(frr - far) < best:
            best, eer, thr = abs(frr - far), 0.5 * (frr + far), float(t)
    print(f"  EER                   : {eer * 100:.1f}%  (threshold {thr:.3f})")

    # ---- fit the deployed model on everything, calibrate on OOF scores ----
    scaler = StandardScaler().fit(X)
    final = LogisticRegression(max_iter=1000, C=0.5).fit(scaler.transform(X), y)

    # The calibrator is fitted on OUT-OF-FOLD scores, never on the final
    # model's own training predictions, which would be wildly overconfident.
    # L2's raw score is P(playback), so higher means more attack.
    cal = ScoreCalibrator.fit(1.0 - oof, y, higher_is_genuine=False, layer_id="L2")

    Path(a.model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.model_out).write_text(json.dumps({
        "kind": "logistic_regression", "n_pairs": n,
        "crop_seconds": CROP_SECONDS,
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": final.coef_[0].tolist(), "intercept": float(final.intercept_[0]),
        "cv_accuracy": acc, "cv_eer": eer,
        "note": "15 pairs, one speaker, one room, one playback device. "
                "Not a general replay detector.",
    }, indent=2))
    print(f"\nwrote model -> {a.model_out}")

    # Per-clip out-of-fold scores, so the ablation can use L2 without
    # re-deriving them and without touching the final model's own training
    # predictions. Raw score convention: P(playback), higher = more attack.
    scores_path = ROOT / "results" / "l2_scores.json"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps({
        "n_pairs": n, "crop_seconds": CROP_SECONDS,
        "note": "out-of-fold leave-one-pair-out scores; P(playback)",
        "live": [float(1.0 - oof[i]) for i in range(n)],
        "playback": [float(1.0 - oof[n + i]) for i in range(n)],
    }, indent=2))
    print(f"wrote L2 scores -> {scores_path}")

    cals = load_calibrators(a.calibrators) if Path(a.calibrators).exists() else {}
    cals["L2"] = cal
    save_calibrators(cals, Path(a.calibrators))
    print(f"wrote L2 calibrator -> {a.calibrators} "
          f"(layers now: {', '.join(sorted(cals))})")


if __name__ == "__main__":
    main()
