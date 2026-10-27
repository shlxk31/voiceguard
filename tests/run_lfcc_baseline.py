"""
tests/run_lfcc_baseline.py

Week 1 deliverable: reproduce a baseline LFCC-GMM EER number on real
ASVspoof 2019 LA data, following the threshold-selection discipline in
WEEK1_CHECKLIST.md:

  1. Fit bonafide/spoof GMMs on the TRAIN split.
  2. Sweep n_components on the DEV split only, pick the config with the
     lowest dev EER, and freeze its threshold.
  3. Run exactly once on the EVAL split with the frozen model+threshold.
     That number is the reproducible baseline -- do not re-tune after
     seeing it.

Usage:
    .venv/bin/python tests/run_lfcc_baseline.py --data-root data/raw/asvspoof2019_la
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from detector.lfcc_gmm import LFCCGMMDetector, extract_lfcc
from detector.metrics import compute_eer

PROTOCOL_NAMES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}


def load_protocol(data_root: Path, split: str) -> list[tuple[str, str]]:
    """Returns [(utt_id, label), ...] with label in {"bonafide", "spoof"}."""
    proto_path = data_root / "ASVspoof2019_LA_cm_protocols" / PROTOCOL_NAMES[split]
    entries = []
    with open(proto_path) as f:
        for line in f:
            fields = line.strip().split()
            # SPEAKER_ID AUDIO_FILE_NAME SYSTEM_ID- KEY  (columns vary slightly
            # across LA protocol files, but utt id is field[1] and label is field[-1])
            utt_id, label = fields[1], fields[-1]
            entries.append((utt_id, label))
    return entries


def load_audio(data_root: Path, split: str, utt_id: str) -> np.ndarray:
    flac_path = data_root / f"ASVspoof2019_LA_{split}" / "flac" / f"{utt_id}.flac"
    audio, sr = sf.read(flac_path, dtype="float32")
    assert sr == 16000, f"expected 16kHz, got {sr}Hz for {flac_path}"
    return audio


def extract_split_features(data_root: Path, split: str, limit: int | None = None):
    entries = load_protocol(data_root, split)
    if limit:
        entries = entries[:limit]
    feats, labels = [], []
    t0 = time.time()
    for i, (utt_id, label) in enumerate(entries):
        audio = load_audio(data_root, split, utt_id)
        feats.append(extract_lfcc(audio))
        labels.append(label)
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{split}] {i + 1}/{len(entries)} ({elapsed:.1f}s)")
    return np.stack(feats), np.array(labels)


def split_by_label(feats: np.ndarray, labels: np.ndarray):
    bonafide = feats[labels == "bonafide"]
    spoof = feats[labels == "spoof"]
    return bonafide, spoof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/asvspoof2019_la"))
    parser.add_argument("--n-components-sweep", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--limit", type=int, default=None,
                         help="Debug only: cap utterances per split.")
    args = parser.parse_args()

    print("Extracting TRAIN features...")
    train_feats, train_labels = extract_split_features(args.data_root, "train", args.limit)
    train_bona, train_spoof = split_by_label(train_feats, train_labels)
    print(f"  train: {len(train_bona)} bonafide, {len(train_spoof)} spoof")

    print("Extracting DEV features...")
    dev_feats, dev_labels = extract_split_features(args.data_root, "dev", args.limit)
    dev_bona, dev_spoof = split_by_label(dev_feats, dev_labels)
    print(f"  dev: {len(dev_bona)} bonafide, {len(dev_spoof)} spoof")

    best = None  # (dev_eer, n_components, detector, threshold)
    for n_components in args.n_components_sweep:
        print(f"\nFitting GMMs with n_components={n_components}...")
        detector = LFCCGMMDetector(n_components=n_components)
        detector.fit(train_bona, train_spoof)

        dev_bona_scores = detector.spoof_scores(dev_bona)
        dev_spoof_scores = detector.spoof_scores(dev_spoof)

        dev_eer, dev_threshold = compute_eer(dev_bona_scores, dev_spoof_scores)
        print(f"  n_components={n_components}: dev EER = {dev_eer * 100:.2f}%")

        if best is None or dev_eer < best[0]:
            best = (dev_eer, n_components, detector, dev_threshold)

    dev_eer, n_components, detector, threshold = best
    detector.threshold = threshold
    print(f"\nFrozen config: n_components={n_components}, dev EER={dev_eer * 100:.2f}%, "
          f"threshold={threshold:.4f}")
    print("Threshold is now FROZEN. Running eval split ONCE.")

    print("\nExtracting EVAL features...")
    eval_feats, eval_labels = extract_split_features(args.data_root, "eval", args.limit)
    eval_bona, eval_spoof = split_by_label(eval_feats, eval_labels)
    print(f"  eval: {len(eval_bona)} bonafide, {len(eval_spoof)} spoof")

    eval_bona_scores = detector.spoof_scores(eval_bona)
    eval_spoof_scores = detector.spoof_scores(eval_spoof)

    eval_eer, _ = compute_eer(eval_bona_scores, eval_spoof_scores)
    print(f"\n=== WEEK 1 BASELINE ===")
    print(f"n_components: {n_components}")
    print(f"threshold (frozen from dev): {threshold:.4f}")
    print(f"dev EER: {dev_eer * 100:.2f}%")
    print(f"EVAL EER: {eval_eer * 100:.2f}%")


if __name__ == "__main__":
    main()
