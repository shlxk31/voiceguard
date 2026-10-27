"""
tests/finetune_domain_adapt.py

Fast fix attempt #2, after synthetic augmentation (tests/finetune_robust.py)
failed to transfer to real audio (37.50% -> 18.33% on synthetic-augmented
dev data, but STILL 8/8 = 100% false positives on real iPhone recordings).

Direct domain adaptation: mix real recorded clips into the bonafide training
set, heavily oversampled, alongside the normal ASVspoof train split (so the
model doesn't forget what spoof sounds like). Starts fresh from the ORIGINAL
pretrained checkpoint, not the failed synthetic-augmented one, for a clean
signal.

HONEST LIMITATION, stated up front: with only 8 real clips (one phone, one
room, one speaker), holding out 2 for validation tells us only whether this
generalizes to different SENTENCES from the same phone/room/speaker -- not
to other devices, rooms, or people. A pass here is a weak but real signal;
a fail here is a strong signal the whole approach needs more real data.

Usage:
    .venv/bin/python tests/finetune_domain_adapt.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.aasist_l import AasistLDetector, _pad_or_crop, _DEFAULT_CHECKPOINT
from detector.metrics import compute_eer
from eval_common import load_protocol, load_audio, stratified_subsample
from liveness_detector import load_audio_any

SR = 16000


def jitter(a, rng):
    """Light augmentation for the real clips ONLY, so repeated exposures
    aren't byte-identical (which would just reweight, not generalize even
    within-domain) -- small gain and additive noise, nothing that changes
    the recording-chain signature we're trying to teach the model."""
    y = a * rng.uniform(0.7, 1.3)
    y = y + rng.standard_normal(len(y)).astype(np.float32) * 0.003
    peak = np.abs(y).max()
    return (y / peak * rng.uniform(0.6, 0.95)).astype(np.float32) if peak > 1e-9 else y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/asvspoof2019_la"))
    parser.add_argument("--real-dir", type=Path, default=Path("recordings/live"))
    parser.add_argument("--n-val-real", type=int, default=2,
                         help="Held out from training entirely -- the real generalization check.")
    parser.add_argument("--real-repeats", type=int, default=15,
                         help="How many (jittered) copies of each TRAIN real clip per epoch.")
    parser.add_argument("--asvspoof-bonafide", type=int, default=150)
    parser.add_argument("--asvspoof-spoof", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default="cpu")  # measured faster than MPS for training
    parser.add_argument("--out", type=Path, default=Path("models/aasist_l_domain_adapted.pth"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    real_files = sorted(args.real_dir.glob("*.m4a"))
    if len(real_files) <= args.n_val_real:
        raise SystemExit(f"Only {len(real_files)} real clips found in {args.real_dir}; "
                          f"need more than --n-val-real ({args.n_val_real}).")
    perm = rng.permutation(len(real_files))
    val_real = [real_files[i] for i in perm[:args.n_val_real]]
    train_real = [real_files[i] for i in perm[args.n_val_real:]]
    print(f"Real clips: {len(train_real)} for training (oversampled x{args.real_repeats}), "
          f"{len(val_real)} HELD OUT for validation: {[f.name for f in val_real]}")

    print(f"\nLoading pretrained AASIST-L from the ORIGINAL checkpoint (not the "
          f"failed synthetic-augmented one)...")
    det = AasistLDetector(checkpoint_path=str(_DEFAULT_CHECKPOINT), device=args.device)
    model, nb_samp = det.model, det.nb_samp
    model.train()

    asv_entries = stratified_subsample(load_protocol(args.data_root, "train"),
                                        args.asvspoof_bonafide, args.asvspoof_spoof, seed=args.seed)
    print(f"ASVspoof train mix: {args.asvspoof_bonafide} bonafide + {args.asvspoof_spoof} spoof")

    real_audio_train = [load_audio_any(f) for f in train_real]

    def build_epoch_examples():
        """(loader_fn, label) pairs for one epoch: ASVspoof entries + oversampled
        jittered copies of the real clips, all shuffled together."""
        items = []
        for utt_id, label in asv_entries:
            y = 1 if label == "bonafide" else 0
            items.append((lambda u=utt_id: load_audio(args.data_root, "train", u), y))
        for a in real_audio_train:
            for _ in range(args.real_repeats):
                items.append((lambda a=a: jitter(a, rng), 1))   # label 1 = bonafide
        rng.shuffle(items)
        return items

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    dev_entries = stratified_subsample(load_protocol(args.data_root, "dev"), 60, 60, seed=args.seed)

    @torch.no_grad()
    def eval_dev():
        model.eval()
        bona, spoof = [], []
        for utt_id, label in dev_entries:
            a = load_audio(args.data_root, "dev", utt_id)
            x = torch.from_numpy(_pad_or_crop(a, nb_samp)).float().unsqueeze(0).to(det.device)
            p = torch.softmax(model(x)[1], dim=-1)[0, 0].item()
            (bona if label == "bonafide" else spoof).append(p)
        model.train()
        return compute_eer(np.array(bona), np.array(spoof))[0]

    @torch.no_grad()
    def eval_real(files):
        model.eval()
        scores = []
        for f in files:
            a = load_audio_any(f)
            x = torch.from_numpy(_pad_or_crop(a, nb_samp)).float().unsqueeze(0).to(det.device)
            scores.append(torch.softmax(model(x)[1], dim=-1)[0, 0].item())  # P(spoof)
        model.train()
        return scores

    print(f"\nBaseline (original pretrained model, before domain adaptation):")
    print(f"  ASVspoof dev EER          : {eval_dev() * 100:.2f}%")
    print(f"  HELD-OUT real clips P(spoof): {[f'{s:.3f}' for s in eval_real(val_real)]}  "
          f"(want LOW -- these are real human speech)")
    print(f"  TRAIN real clips P(spoof)   : {[f'{s:.3f}' for s in eval_real(train_real)]}")

    for epoch in range(1, args.epochs + 1):
        examples = build_epoch_examples()
        t0 = time.time()
        losses = []
        for i, (load_fn, y) in enumerate(examples):
            x = torch.from_numpy(_pad_or_crop(load_fn(), nb_samp)).float().unsqueeze(0).to(det.device)
            yt = torch.tensor([y]).to(det.device)
            opt.zero_grad()
            loss = criterion(model(x)[1], yt)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if (i + 1) % 50 == 0:
                print(f"  epoch {epoch} {i + 1}/{len(examples)} "
                      f"loss={np.mean(losses[-50:]):.4f} ({time.time() - t0:.0f}s)", flush=True)

        dev_eer = eval_dev()
        val_scores = eval_real(val_real)
        train_scores = eval_real(train_real)
        print(f"\nepoch {epoch}: mean loss {np.mean(losses):.4f}")
        print(f"  ASVspoof dev EER            : {dev_eer * 100:.2f}%  (must NOT collapse)")
        print(f"  HELD-OUT real clips P(spoof): {[f'{s:.3f}' for s in val_scores]}  "
              f"<-- the real generalization test")
        print(f"  TRAIN real clips P(spoof)   : {[f'{s:.3f}' for s in train_scores]}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.out)
        print(f"  saved -> {args.out}\n")

    print("Done.")


if __name__ == "__main__":
    main()
