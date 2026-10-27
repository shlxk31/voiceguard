"""
tests/finetune_robust.py

THE FIX for the domain-failure finding.

Problem (measured): AASIST-L scores 0.99% EER on ASVspoof2019 LA but flags
100% of real iPhone recordings as synthetic. Diagnosis: its "bonafide" class
is VCTK studio audio, so it learned "studio-clean = real" rather than
"human = real". Real audio has a ~16x higher noise floor and ~10x less
high-frequency energy, so it falls outside the model's idea of real.

Fix: fine-tune on the SAME data, but with realistic recording conditions
applied on the fly -- room reverb, background noise, microphone frequency
response, level variation. Crucially the augmentation is applied EQUALLY to
both classes, so the model cannot simply relearn "noisy = spoof"; it has to
find cues that survive the channel.

    .venv/bin/python tests/finetune_robust.py --epochs 3

Outputs models/aasist_l_robust.pth, then evaluate with:
    .venv/bin/python tests/eval_robust.py
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

from detector.aasist_l import AasistLDetector, _pad_or_crop
from eval_common import load_protocol, load_audio, stratified_subsample

SR = 16000


def random_room_ir(rng, sr=SR, max_rt60=0.5):
    """Cheap synthetic room impulse response: exponentially decaying noise."""
    rt60 = rng.uniform(0.1, max_rt60)
    n = int(sr * rt60)
    ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    ir[0] += 1.0                      # direct path
    return (ir / (np.abs(ir).max() + 1e-9)).astype(np.float32)


def spectral_tilt(x, rng):
    """Crude microphone frequency response: random low-pass + bass lift, which
    is the direction real phone audio differs from studio audio."""
    from scipy.signal import butter, lfilter
    cutoff = rng.uniform(3000, 7500)
    b, a = butter(2, cutoff / (SR / 2), btype="low")
    y = lfilter(b, a, x)
    if rng.random() < 0.5:
        bb, ba = butter(2, 300 / (SR / 2), btype="low")
        y = y + 0.3 * lfilter(bb, ba, x)   # bass lift
    return y.astype(np.float32)


def augment(x, rng):
    """Apply a random realistic recording chain. Same distribution for both
    classes -- that symmetry is what forces the model to learn real cues."""
    from scipy.signal import fftconvolve
    y = x.copy()
    if rng.random() < 0.6:                                    # room reverb
        y = fftconvolve(y, random_room_ir(rng))[:len(x)]      # FFT conv: np.convolve
        # here is direct O(N*M) -- ~500M ops/sample on a 65k-sample clip vs an
        # 8k-sample IR, which is what made the smoke test look hung.
    if rng.random() < 0.7:                                    # mic response
        y = spectral_tilt(y, rng)
    if rng.random() < 0.8:                                    # background noise
        snr = rng.uniform(5, 30)
        noise = rng.standard_normal(len(y)).astype(np.float32)
        sig_p = np.mean(y ** 2) + 1e-12
        y = y + noise * np.sqrt(sig_p / (10 ** (snr / 10)))
    if rng.random() < 0.5:                                    # level variation
        y = y * rng.uniform(0.2, 1.5)
    peak = np.abs(y).max()
    if peak > 1e-9:
        y = y / peak * rng.uniform(0.5, 0.99)
    return y.astype(np.float32)


def batches(entries, data_root, split, nb_samp, batch_size, rng, augment_prob):
    order = rng.permutation(len(entries))
    for i in range(0, len(order) - batch_size + 1, batch_size):
        xs, ys = [], []
        for j in order[i:i + batch_size]:
            utt_id, label = entries[j]
            a = load_audio(data_root, split, utt_id)
            if rng.random() < augment_prob:
                a = augment(a, rng)
            xs.append(_pad_or_crop(a, nb_samp))
            # Checkpoint convention (verified): index 0 = spoof, 1 = bonafide.
            ys.append(1 if label == "bonafide" else 0)
        yield torch.from_numpy(np.stack(xs)).float(), torch.tensor(ys)


@torch.no_grad()
def quick_eer(model, entries, data_root, split, nb_samp, device, rng, augment_prob):
    from detector.metrics import compute_eer
    model.eval()
    bona, spoof = [], []
    for utt_id, label in entries:
        a = load_audio(data_root, split, utt_id)
        if augment_prob and rng.random() < augment_prob:
            a = augment(a, rng)
        x = torch.from_numpy(_pad_or_crop(a, nb_samp)).float().unsqueeze(0).to(device)
        p = torch.softmax(model(x)[1], dim=-1)[0, 0].item()   # P(spoof)
        (bona if label == "bonafide" else spoof).append(p)
    model.train()
    return compute_eer(np.array(bona), np.array(spoof))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/asvspoof2019_la"))
    parser.add_argument("--epochs", type=int, default=3)
    # batch_size=1 measured fastest per-sample (0.45s) vs batch_size=8 (3.92s/
    # sample) -- AASIST's graph-attention layer scales badly with batch size,
    # for training just like it did for inference (see WEEK2_RESULTS.md).
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-bonafide", type=int, default=2000)
    parser.add_argument("--train-spoof", type=int, default=2000)
    parser.add_argument("--val-bonafide", type=int, default=150)
    parser.add_argument("--val-spoof", type=int, default=150)
    parser.add_argument("--augment-prob", type=float, default=0.8)
    # CPU beats MPS for TRAINING this model (measured: ~13-34s/batch on CPU vs
    # ~38-62s/batch on MPS) -- MPS likely falls back to CPU per-op for AASIST's
    # custom graph-attention layers, adding overhead instead of saving time.
    # This is the opposite of inference, where MPS was ~3.2x faster.
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=Path("models/aasist_l_robust.pth"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading pretrained AASIST-L on {args.device}...")
    det = AasistLDetector(device=args.device)
    model, nb_samp = det.model, det.nb_samp
    model.train()

    train = stratified_subsample(load_protocol(args.data_root, "train"),
                                  args.train_bonafide, args.train_spoof, seed=args.seed)
    val = stratified_subsample(load_protocol(args.data_root, "dev"),
                                args.val_bonafide, args.val_spoof, seed=args.seed)
    print(f"train: {len(train)} utterances   val: {len(val)} utterances")
    print(f"augmentation applied to {args.augment_prob:.0%} of samples, both classes equally\n")

    print("Baseline (pretrained, before fine-tuning):")
    e_clean = quick_eer(model, val, args.data_root, "dev", nb_samp, det.device, rng, 0.0)
    e_aug = quick_eer(model, val, args.data_root, "dev", nb_samp, det.device, rng, 1.0)
    print(f"  dev EER, clean audio     : {e_clean * 100:.2f}%")
    print(f"  dev EER, realistic audio : {e_aug * 100:.2f}%   <-- the failure\n")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        t0, losses, n = time.time(), [], 0
        for xb, yb in batches(train, args.data_root, "train", nb_samp,
                               args.batch_size, rng, args.augment_prob):
            xb, yb = xb.to(det.device), yb.to(det.device)
            opt.zero_grad()
            loss = criterion(model(xb)[1], yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            n += len(yb)
            print(f"  epoch {epoch}  {n}/{len(train)}  "
                  f"loss={np.mean(losses[-5:]):.4f}  ({time.time() - t0:.0f}s)", flush=True)
        ec = quick_eer(model, val, args.data_root, "dev", nb_samp, det.device, rng, 0.0)
        ea = quick_eer(model, val, args.data_root, "dev", nb_samp, det.device, rng, 1.0)
        print(f"epoch {epoch}: mean loss {np.mean(losses):.4f} | "
              f"dev EER clean {ec * 100:.2f}% | dev EER realistic {ea * 100:.2f}%")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.out)
        print(f"  saved -> {args.out}")

    print("\nDone. Evaluate with: .venv/bin/python tests/eval_robust.py")


if __name__ == "__main__":
    main()
