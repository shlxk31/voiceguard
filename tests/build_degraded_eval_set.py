"""
tests/build_degraded_eval_set.py

Generates cached VoIP-degraded copies of a class-balanced subsample of the
ASVspoof2019 LA eval split, per data/README.md's convention:
    data/degraded/<config-name>/<utt_id>.wav + manifest.json

Generated ONCE and cached to disk (not degraded on-the-fly per eval run)
because:
  - apply_packet_loss() uses unseeded randomness -- caching removes any
    ambiguity about whether two detectors see the "same" degraded audio.
  - Opus encode/decode isn't free; later scripts (eval_degraded.py, and any
    future streaming/accumulator work) reuse these files at zero extra cost.

Usage:
    .venv/bin/python tests/build_degraded_eval_set.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from voip_sim.degrade import DegradationConfig, degrade
from eval_common import load_protocol, load_audio, stratified_subsample

# (bitrate_kbps, packet_loss_rate, use_plc) -- concrete, bounded matrix.
# burst_factor stays at the DegradationConfig default (0.5) throughout;
# crossing that as a 4th axis is a later ablation, not built here.
CONFIGS = [
    (8, 0.0, True), (16, 0.0, True), (32, 0.0, True),
    (8, 0.05, True), (16, 0.05, True), (32, 0.05, True),
    (8, 0.15, True), (16, 0.15, True), (32, 0.15, True),
    (16, 0.15, False),  # PLC-off comparison at one representative combo
]


def config_name(bitrate_kbps: int, packet_loss_rate: float, use_plc: bool) -> str:
    loss_pct = int(round(packet_loss_rate * 100))
    plc_tag = "on" if use_plc else "off"
    return f"opus{bitrate_kbps}kbps_loss{loss_pct}pct_plc{plc_tag}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/asvspoof2019_la"))
    parser.add_argument("--split", default="eval")
    parser.add_argument("--n-bonafide", type=int, default=1500)
    parser.add_argument("--n-spoof", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-root", type=Path, default=Path("data/degraded"))
    parser.add_argument("--only", default=None, help="Rerun a single config by name")
    args = parser.parse_args()

    entries = load_protocol(args.data_root, args.split)
    entries = stratified_subsample(entries, args.n_bonafide, args.n_spoof, seed=args.seed)
    utt_ids = [utt_id for utt_id, _ in entries]
    print(f"Subsampled {len(entries)} utterances from {args.split} "
          f"({args.n_bonafide} bonafide + {args.n_spoof} spoof, seed={args.seed})")

    configs = CONFIGS
    if args.only:
        configs = [c for c in CONFIGS if config_name(*c) == args.only]
        if not configs:
            raise SystemExit(f"No config named {args.only!r}; choices: "
                              f"{[config_name(*c) for c in CONFIGS]}")

    for bitrate_kbps, packet_loss_rate, use_plc in configs:
        name = config_name(bitrate_kbps, packet_loss_rate, use_plc)
        out_dir = args.out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = DegradationConfig(bitrate_kbps=bitrate_kbps, packet_loss_rate=packet_loss_rate,
                                 use_plc=use_plc)

        print(f"\nGenerating {name} ({len(entries)} files)...")
        t0 = time.time()
        for i, (utt_id, label) in enumerate(entries):
            audio = load_audio(args.data_root, args.split, utt_id)
            degraded = degrade(audio, 16000, cfg)
            sf.write(out_dir / f"{utt_id}.wav", degraded, 16000)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(entries)} ({time.time() - t0:.1f}s)")

        manifest = {
            "codec": cfg.codec,
            "bitrate_kbps": cfg.bitrate_kbps,
            "packet_loss_rate": cfg.packet_loss_rate,
            "burst_factor": cfg.burst_factor,
            "frame_ms": cfg.frame_ms,
            "use_plc": cfg.use_plc,
            "seed": args.seed,
            "source_split": args.split,
            "source_utt_ids": utt_ids,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  done in {time.time() - t0:.1f}s -> {out_dir}")

    print(f"\nAll configs written under {args.out_root}")


if __name__ == "__main__":
    main()
