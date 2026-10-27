"""
tests/eval_streaming.py

The project's claimed novel contribution, measured: what happens to
detection accuracy and latency when you stop scoring whole utterances and
start scoring short streaming chunks instead?

Produces the core EER-vs-window-size-vs-latency table, plus a per-chunk
score cache that tests/tune_accumulator.py consumes to tune the mid-call
alarm rule.

DESIGN DECISIONS (all paper-relevant, state them explicitly in the write-up):
  1. Chunks are NON-OVERLAPPING and back-to-back, matching how
     CallAccumulator and the /stream WebSocket endpoint already consume
     audio. Overlapping/strided windows are a possible later ablation.
  2. Every chunk INHERITS its parent utterance's bonafide/spoof label.
     Reasonable here because an ASVspoof utterance is single-source
     throughout (the whole clip is either genuine or synthetic).
  3. Trailing partial chunks are DROPPED, not zero-padded -- padding would
     inject artificial silence into a real class label. This shrinks
     effective sample count more at 1000ms than at 250ms.
  4. AASIST-L internally tile-repeats any chunk up to its fixed 4.04s input
     (see src/detector/aasist_l.py). So a 250ms chunk is scored as ~16
     repeats of itself. That is exactly the naive-chunking baseline the
     "streaming reformulation" work is meant to improve on -- if short
     windows degrade badly here, that IS the result motivating the next
     step, not a bug.

Usage:
    # dev split, clean -- produces the cache tune_accumulator.py tunes on
    .venv/bin/python tests/eval_streaming.py --split dev

    # eval split, clean -- headline streaming numbers
    .venv/bin/python tests/eval_streaming.py --split eval

    # eval split, worst degraded condition
    .venv/bin/python tests/eval_streaming.py --split eval \\
        --degraded-config opus8kbps_loss15pct_plcon
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
from eval_common import load_protocol, load_audio, stratified_subsample


def chunk_audio(audio: np.ndarray, window_ms: int, sr: int = 16000) -> list[np.ndarray]:
    """Non-overlapping fixed-length chunks; trailing partial chunk dropped."""
    win = int(sr * window_ms / 1000)
    n_chunks = len(audio) // win
    return [audio[i * win:(i + 1) * win] for i in range(n_chunks)]


def fit_lfcc_detector(data_root: Path, n_components: int = 64) -> LFCCGMMDetector:
    print(f"Fitting LFCC-GMM on TRAIN (n_components={n_components})...")
    feats, labels = [], []
    for utt_id, label in load_protocol(data_root, "train"):
        feats.append(extract_lfcc(load_audio(data_root, "train", utt_id)))
        labels.append(label)
    feats, labels = np.stack(feats), np.array(labels)
    det = LFCCGMMDetector(n_components=n_components)
    det.fit(feats[labels == "bonafide"], feats[labels == "spoof"])
    return det


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/asvspoof2019_la"))
    parser.add_argument("--split", default="eval", choices=["dev", "eval"])
    parser.add_argument("--degraded-root", type=Path, default=Path("data/degraded"))
    parser.add_argument("--degraded-config", default=None,
                         help="Name of a data/degraded/<config> folder; omit for clean audio.")
    parser.add_argument("--window-ms", type=int, nargs="+", default=[250, 500, 1000])
    parser.add_argument("--n-bonafide", type=int, default=300)
    parser.add_argument("--n-spoof", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--aasist-baseline", type=Path, default=Path("results/aasist_baseline.json"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--chunk-scores-out", type=Path, default=None)
    args = parser.parse_args()

    tag = args.degraded_config or "clean"
    out_csv = args.out or Path(f"results/streaming_{args.split}_{tag}.csv")
    cache_out = args.chunk_scores_out or Path(f"results/cache/streaming_chunks_{args.split}_{tag}.npz")

    # --- Assemble the utterance list + how to load each one's audio ---
    if args.degraded_config:
        manifest_path = args.degraded_root / args.degraded_config / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest["source_split"] != args.split:
            raise SystemExit(f"{args.degraded_config} was built from "
                              f"'{manifest['source_split']}', not '{args.split}'")
        label_by_id = dict(load_protocol(args.data_root, args.split))
        entries = [(uid, label_by_id[uid]) for uid in manifest["source_utt_ids"]]
        entries = stratified_subsample(entries, args.n_bonafide, args.n_spoof, seed=args.seed)
        config_dir = args.degraded_root / args.degraded_config

        def get_audio(utt_id):
            audio, sr = sf.read(config_dir / f"{utt_id}.wav", dtype="float32")
            assert sr == 16000
            return audio
    else:
        entries = load_protocol(args.data_root, args.split)
        entries = stratified_subsample(entries, args.n_bonafide, args.n_spoof, seed=args.seed)

        def get_audio(utt_id):
            return load_audio(args.data_root, args.split, utt_id)

    print(f"Streaming eval: split={args.split} condition={tag} "
          f"({len(entries)} utterances, windows={args.window_ms}ms)")

    lfcc_detector = fit_lfcc_detector(args.data_root)
    print(f"Loading AasistLDetector on device={args.device}...")
    aasist_detector = AasistLDetector(device=args.device)
    if args.aasist_baseline.exists():
        with open(args.aasist_baseline) as f:
            aasist_detector.threshold = json.load(f)["threshold"]

    detectors = {"lfcc_gmm": lfcc_detector, "aasist_l": aasist_detector}

    records = []  # one row per (detector, window, utterance, chunk)
    for window_ms in args.window_ms:
        print(f"\n--- window={window_ms}ms ---")
        t0 = time.time()
        for i, (utt_id, label) in enumerate(entries):
            audio = get_audio(utt_id)
            for chunk_index, chunk in enumerate(chunk_audio(audio, window_ms)):
                for det_name, det in detectors.items():
                    r = det.score(chunk, sample_rate=16000)
                    records.append((det_name, window_ms, utt_id, label, chunk_index,
                                     r.score, int(r.is_spoof), r.confidence, r.latency_ms))
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {i + 1}/{len(entries)} utts ({el:.0f}s, "
                      f"ETA {(len(entries) - i - 1) / ((i + 1) / el):.0f}s)")

    if not records:
        raise SystemExit("No chunks produced -- are the utterances shorter than the window?")

    det_names = np.array([r[0] for r in records])
    windows = np.array([r[1] for r in records])
    utt_ids = np.array([r[2] for r in records])
    labels = np.array([r[3] for r in records])
    chunk_idx = np.array([r[4] for r in records], dtype=np.int32)
    scores = np.array([r[5] for r in records], dtype=np.float64)
    is_spoof = np.array([r[6] for r in records], dtype=np.int8)
    confidence = np.array([r[7] for r in records], dtype=np.float64)
    latency = np.array([r[8] for r in records], dtype=np.float64)

    # Sorted by (detector, window, utt, chunk order) -- tune_accumulator.py
    # replays chunks sequentially per utterance, so order matters.
    order = np.lexsort((chunk_idx, utt_ids, windows, det_names))
    cache_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_out, detector=det_names[order], window_ms=windows[order],
             utt_id=utt_ids[order], label=labels[order], chunk_index=chunk_idx[order],
             score=scores[order], is_spoof=is_spoof[order], confidence=confidence[order],
             latency_ms=latency[order])
    print(f"\nPer-chunk cache -> {cache_out}")

    rows = []
    print(f"\n{'detector':10s}{'window':>8s}{'chunks':>9s}{'EER%':>9s}"
          f"{'lat_med':>10s}{'lat_mean':>10s}")
    for det_name in sorted(set(det_names)):
        for window_ms in args.window_ms:
            m = (det_names == det_name) & (windows == window_ms)
            if not m.any():
                continue
            bona = scores[m & (labels == "bonafide")]
            spoof = scores[m & (labels == "spoof")]
            eer, _ = compute_eer(bona, spoof)
            # Median latency alongside mean: a long unattended run can pick up
            # a few enormous outliers (system sleep/throttle) that wreck the mean.
            lat_med, lat_mean = float(np.median(latency[m])), float(np.mean(latency[m]))
            rows.append({"detector": det_name, "condition": tag, "split": args.split,
                          "window_ms": window_ms, "n_chunks": int(m.sum()),
                          "n_utterances": len(entries), "eer_pct": eer * 100,
                          "latency_ms_median": lat_med, "latency_ms_mean": lat_mean,
                          "realtime_ok": bool(lat_med < window_ms)})
            print(f"{det_name:10s}{window_ms:>8d}{int(m.sum()):>9d}{eer * 100:>9.2f}"
                  f"{lat_med:>10.1f}{lat_mean:>10.1f}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nTable -> {out_csv}")


if __name__ == "__main__":
    main()
