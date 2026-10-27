"""
tests/eval_degraded.py

The core motivating result for the paper: how much does each detector's
EER degrade under real VoIP conditions (Opus codec + Gilbert-Elliott packet
loss), compared to clean audio?

Both detectors use FROZEN thresholds throughout (LFCC-GMM's from
WEEK1_RESULTS.md, AASIST-L's from results/aasist_baseline.json) -- no
re-tuning per condition, matching how a real deployment would work (you
don't get to re-calibrate per call).

Requires:
  - results/aasist_baseline.json (from tests/run_aasist_baseline.py)
  - data/degraded/*/manifest.json (from tests/build_degraded_eval_set.py)

Usage:
    .venv/bin/python tests/eval_degraded.py
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector.lfcc_gmm import LFCCGMMDetector, extract_lfcc
from detector.aasist_l import AasistLDetector
from detector.metrics import compute_eer
from eval_common import load_protocol, load_audio, split_by_label


def score_utt_ids(detector, audio_by_id: dict[str, np.ndarray]) -> dict[str, float]:
    return {utt_id: detector.score(audio).score for utt_id, audio in audio_by_id.items()}


def report_condition(name: str, detector_name: str, bona_scores: np.ndarray, spoof_scores: np.ndarray,
                      frozen_threshold: float, mean_latency_ms: float, extra: dict) -> dict:
    eer, _ = compute_eer(bona_scores, spoof_scores)
    all_scores = np.concatenate([bona_scores, spoof_scores])
    all_true_spoof = np.concatenate([np.zeros(len(bona_scores), dtype=bool),
                                      np.ones(len(spoof_scores), dtype=bool)])
    preds_spoof = all_scores > frozen_threshold
    accuracy_at_frozen = float(np.mean(preds_spoof == all_true_spoof))
    row = {
        "detector": detector_name,
        "config": name,
        "n_bonafide": len(bona_scores),
        "n_spoof": len(spoof_scores),
        "eer_pct": eer * 100,
        "accuracy_at_frozen_threshold_pct": accuracy_at_frozen * 100,
        "mean_latency_ms": mean_latency_ms,
    }
    row.update(extra)
    print(f"  [{detector_name}] {name}: EER={eer * 100:.2f}%  "
          f"acc@frozen_threshold={accuracy_at_frozen * 100:.2f}%")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/asvspoof2019_la"))
    parser.add_argument("--degraded-root", type=Path, default=Path("data/degraded"))
    parser.add_argument("--aasist-baseline", type=Path, default=Path("results/aasist_baseline.json"))
    parser.add_argument("--lfcc-n-components", type=int, default=64,
                         help="Frozen choice from WEEK1_RESULTS.md")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=Path("results/degradation_eer_table.csv"))
    args = parser.parse_args()

    with open(args.aasist_baseline) as f:
        aasist_baseline = json.load(f)
    aasist_threshold = aasist_baseline["threshold"]
    print(f"Loaded frozen AASIST-L threshold: {aasist_threshold:.4f}")

    print(f"\nRefitting LFCC-GMM on TRAIN (n_components={args.lfcc_n_components})...")
    train_entries = load_protocol(args.data_root, "train")
    train_feats, train_labels = [], []
    for utt_id, label in train_entries:
        train_feats.append(extract_lfcc(load_audio(args.data_root, "train", utt_id)))
        train_labels.append(label)
    train_feats, train_labels = np.stack(train_feats), np.array(train_labels)
    train_bona, train_spoof = split_by_label(train_feats, train_labels)
    lfcc_detector = LFCCGMMDetector(n_components=args.lfcc_n_components)
    lfcc_detector.fit(train_bona, train_spoof)

    print("Re-deriving frozen LFCC-GMM threshold from DEV (matches WEEK1_RESULTS.md)...")
    dev_entries = load_protocol(args.data_root, "dev")
    dev_feats, dev_labels = [], []
    for utt_id, label in dev_entries:
        dev_feats.append(extract_lfcc(load_audio(args.data_root, "dev", utt_id)))
        dev_labels.append(label)
    dev_feats, dev_labels = np.stack(dev_feats), np.array(dev_labels)
    dev_bona, dev_spoof = split_by_label(dev_feats, dev_labels)
    _, lfcc_threshold = compute_eer(lfcc_detector.spoof_scores(dev_bona),
                                     lfcc_detector.spoof_scores(dev_spoof))
    lfcc_detector.threshold = lfcc_threshold
    print(f"  LFCC-GMM frozen threshold: {lfcc_threshold:.4f}")

    print(f"\nLoading AasistLDetector on device={args.device}...")
    aasist_detector = AasistLDetector(device=args.device)
    aasist_detector.threshold = aasist_threshold

    # Discover degraded configs + the shared source utt_id list from their manifests.
    manifests = sorted(args.degraded_root.glob("*/manifest.json"))
    if not manifests:
        raise SystemExit(f"No manifests found under {args.degraded_root} -- "
                          f"run tests/build_degraded_eval_set.py first.")
    with open(manifests[0]) as f:
        first_manifest = json.load(f)
    source_split = first_manifest["source_split"]
    source_utt_ids = first_manifest["source_utt_ids"]
    eval_entries = dict(load_protocol(args.data_root, source_split))  # utt_id -> label
    print(f"\nUsing {len(source_utt_ids)} shared utt_ids from {source_split} "
          f"(seed={first_manifest['seed']})")

    rows = []

    # --- Clean condition: score the same utt_ids fresh from the original data ---
    print("\nScoring CLEAN condition...")
    clean_audio = {uid: load_audio(args.data_root, source_split, uid) for uid in source_utt_ids}
    clean_labels = np.array([eval_entries[uid] for uid in source_utt_ids])

    t0 = time.time()
    lfcc_clean_scores = np.array([
        lfcc_detector.spoof_scores(extract_lfcc(a).reshape(1, -1))[0] for a in clean_audio.values()
    ])
    lfcc_clean_latency = (time.time() - t0) * 1000 / len(clean_audio)
    lfcc_bona, lfcc_spoof = split_by_label(lfcc_clean_scores, clean_labels)
    rows.append(report_condition("clean", "lfcc_gmm", lfcc_bona, lfcc_spoof, lfcc_threshold,
                                  lfcc_clean_latency, {"bitrate_kbps": None, "loss_rate": 0.0, "use_plc": None}))

    t0 = time.time()
    aasist_clean_scores = np.array([aasist_detector.score(a).score for a in clean_audio.values()])
    aasist_clean_latency = (time.time() - t0) * 1000 / len(clean_audio)
    aasist_bona, aasist_spoof = split_by_label(aasist_clean_scores, clean_labels)
    rows.append(report_condition("clean", "aasist_l", aasist_bona, aasist_spoof, aasist_threshold,
                                  aasist_clean_latency, {"bitrate_kbps": None, "loss_rate": 0.0, "use_plc": None}))

    # --- Degraded conditions ---
    for manifest_path in manifests:
        with open(manifest_path) as f:
            manifest = json.load(f)
        config_dir = manifest_path.parent
        name = config_dir.name
        print(f"\nScoring degraded condition: {name}...")

        degraded_audio = []
        for uid in source_utt_ids:
            audio, sr = sf.read(config_dir / f"{uid}.wav", dtype="float32")
            assert sr == 16000
            degraded_audio.append(audio)
        labels = np.array([eval_entries[uid] for uid in source_utt_ids])

        t0 = time.time()
        lfcc_scores = np.array([
            lfcc_detector.spoof_scores(extract_lfcc(a).reshape(1, -1))[0] for a in degraded_audio
        ])
        lfcc_latency = (time.time() - t0) * 1000 / len(degraded_audio)
        lfcc_bona, lfcc_spoof = split_by_label(lfcc_scores, labels)
        rows.append(report_condition(name, "lfcc_gmm", lfcc_bona, lfcc_spoof, lfcc_threshold,
                                      lfcc_latency, {"bitrate_kbps": manifest["bitrate_kbps"],
                                                      "loss_rate": manifest["packet_loss_rate"],
                                                      "use_plc": manifest["use_plc"]}))

        t0 = time.time()
        aasist_scores = np.array([aasist_detector.score(a).score for a in degraded_audio])
        aasist_latency = (time.time() - t0) * 1000 / len(degraded_audio)
        aasist_bona, aasist_spoof = split_by_label(aasist_scores, labels)
        rows.append(report_condition(name, "aasist_l", aasist_bona, aasist_spoof, aasist_threshold,
                                      aasist_latency, {"bitrate_kbps": manifest["bitrate_kbps"],
                                                        "loss_rate": manifest["packet_loss_rate"],
                                                        "use_plc": manifest["use_plc"]}))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["detector", "config", "bitrate_kbps", "loss_rate", "use_plc",
                  "n_bonafide", "n_spoof", "eer_pct", "accuracy_at_frozen_threshold_pct",
                  "mean_latency_ms"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
