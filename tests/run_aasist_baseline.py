"""
tests/run_aasist_baseline.py

Week 2 deliverable: give AASIST-L the same rigorous treatment the LFCC-GMM
baseline got in Week 1, so the two detectors are directly comparable.

Unlike LFCC-GMM, AASIST-L is a PRETRAINED checkpoint (clovaai/aasist,
external/aasist/models/weights/AASIST-L.pth) -- there's no fit step. The
threshold-selection discipline still applies:

  1. Score the DEV split, pick the EER-optimal threshold, freeze it.
  2. Score the EVAL split exactly once with that frozen threshold.

Runs on MPS (Apple GPU) by default -- measured ~3.2x faster than CPU on
this hardware with scores matching CPU within 2e-7 (see WEEK1_RESULTS.md /
the project's planning notes for that verification). Falls back to CPU if
MPS isn't available.

Usage:
    # Always smoke-test first -- full run takes ~1.5h on MPS.
    .venv/bin/python tests/run_aasist_baseline.py --dev-n-bonafide 200 --dev-n-spoof 200 \\
        --eval-n-bonafide 200 --eval-n-spoof 200

    # Full run (background):
    .venv/bin/python tests/run_aasist_baseline.py --data-root data/raw/asvspoof2019_la
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector.aasist_l import AasistLDetector
from detector.metrics import compute_eer
from eval_common import load_protocol, load_audio, stratified_subsample, save_scores_npz


def score_split(detector: AasistLDetector, data_root: Path, split: str,
                 n_bonafide: int | None, n_spoof: int | None, seed: int):
    entries = load_protocol(data_root, split)
    entries = stratified_subsample(entries, n_bonafide, n_spoof, seed=seed)

    utt_ids, labels, scores, latencies = [], [], [], []
    t0 = time.time()
    for i, (utt_id, label) in enumerate(entries):
        audio = load_audio(data_root, split, utt_id)
        result = detector.score(audio)
        utt_ids.append(utt_id)
        labels.append(label)
        scores.append(result.score)
        latencies.append(result.latency_ms)
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(entries) - i - 1) / rate
            print(f"  [{split}] {i + 1}/{len(entries)} ({elapsed:.1f}s elapsed, "
                  f"ETA {eta:.0f}s)")
    return utt_ids, labels, np.array(scores), np.array(latencies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/asvspoof2019_la"))
    parser.add_argument("--dev-n-bonafide", type=int, default=None)
    parser.add_argument("--dev-n-spoof", type=int, default=None)
    parser.add_argument("--eval-n-bonafide", type=int, default=None)
    parser.add_argument("--eval-n-spoof", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("results/aasist_baseline.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("results/cache"))
    args = parser.parse_args()

    print(f"Loading AasistLDetector on device={args.device}...")
    detector = AasistLDetector(device=args.device)

    print("Scoring DEV split...")
    dev_ids, dev_labels, dev_scores, dev_latencies = score_split(
        detector, args.data_root, "dev", args.dev_n_bonafide, args.dev_n_spoof, args.seed)
    dev_labels_arr = np.array(dev_labels)
    dev_bona_scores = dev_scores[dev_labels_arr == "bonafide"]
    dev_spoof_scores = dev_scores[dev_labels_arr == "spoof"]
    print(f"  dev: {len(dev_bona_scores)} bonafide, {len(dev_spoof_scores)} spoof")

    dev_eer, threshold = compute_eer(dev_bona_scores, dev_spoof_scores)
    detector.threshold = threshold
    print(f"\nDev EER: {dev_eer * 100:.2f}%  (sanity check: published in-domain "
          f"AASIST-L eval EER is ~1%, so this should be low single digits on a "
          f"full clean run -- if it's near 0% or near 50%, stop and investigate "
          f"before running eval)")
    print(f"Frozen threshold: {threshold:.4f}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    save_scores_npz(args.cache_dir / "aasist_dev_scores.npz", dev_ids, dev_labels, dev_scores,
                     latency_ms=dev_latencies)

    print("\nThreshold is now FROZEN. Scoring EVAL split ONCE.")
    eval_ids, eval_labels, eval_scores, eval_latencies = score_split(
        detector, args.data_root, "eval", args.eval_n_bonafide, args.eval_n_spoof, args.seed)
    eval_labels_arr = np.array(eval_labels)
    eval_bona_scores = eval_scores[eval_labels_arr == "bonafide"]
    eval_spoof_scores = eval_scores[eval_labels_arr == "spoof"]
    print(f"  eval: {len(eval_bona_scores)} bonafide, {len(eval_spoof_scores)} spoof")

    eval_eer, _ = compute_eer(eval_bona_scores, eval_spoof_scores)

    eval_preds_spoof = eval_scores > threshold
    eval_true_spoof = eval_labels_arr == "spoof"
    accuracy_at_frozen_threshold = float(np.mean(eval_preds_spoof == eval_true_spoof))

    save_scores_npz(args.cache_dir / "aasist_eval_scores.npz", eval_ids, eval_labels, eval_scores,
                     latency_ms=eval_latencies)

    result = {
        "dev_eer": dev_eer,
        "eval_eer": eval_eer,
        "threshold": float(threshold),
        "accuracy_at_frozen_threshold_eval": accuracy_at_frozen_threshold,
        "n_dev": len(dev_ids),
        "n_eval": len(eval_ids),
        "mean_latency_ms_dev": float(np.mean(dev_latencies)),
        "mean_latency_ms_eval": float(np.mean(eval_latencies)),
        "device": args.device,
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== AASIST-L BASELINE ===")
    print(f"threshold (frozen from dev): {threshold:.4f}")
    print(f"dev EER: {dev_eer * 100:.2f}%")
    print(f"EVAL EER: {eval_eer * 100:.2f}%")
    print(f"EVAL accuracy at frozen threshold: {accuracy_at_frozen_threshold * 100:.2f}%")
    print(f"mean latency: {result['mean_latency_ms_eval']:.1f}ms/utterance (eval)")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
