"""
demo_two_layer.py

LIVE DEMO for the mentor meeting. Shows, on one screen, why speaker
verification alone is not enough to secure a voice-authorised transaction.

Runs a piece of audio through BOTH security layers and prints the decision:

  Layer 2   Speaker verification  (ECAPA-TDNN, 192-dim embeddings)
            -> "is this the enrolled user's voice?"
  Layer 2.5 Anti-spoofing         (AASIST-L, 0.99% EER on ASVspoof2019 LA)
            -> "is this a real human, or an AI-generated fake?"

...then shows the authorisation outcome WITH and WITHOUT the anti-spoofing
layer, so the value of adding it is visible in one glance.

QUICK DEMO (uses real ASVspoof2019 LA audio, nothing to prepare):

    python demo_two_layer.py --scenario genuine    # real enrolled user
    python demo_two_layer.py --scenario spoof      # AI-generated fake of them
    python demo_two_layer.py --scenario imposter   # a different real person

Run the three in that order in front of your mentor. The `spoof` case is
the point: the voice matches, but it is not a human.

YOUR OWN AUDIO:
    python demo_two_layer.py --enroll a.wav b.wav c.wav --test unknown.wav
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from detector.aasist_l import AasistLDetector

DATA_ROOT = Path("data/raw/asvspoof2019_la")
SIMILARITY_HARD_DENY = 0.50   # SecureVoice rule: below this, deny outright
# Rs.500-2,000 tier (trust >= 50). Chosen because measured genuine users reach
# it 99.5% of the time, whereas SecureVoice's higher tiers (trust >= 70 / >= 85)
# are only reached by 43.5% / ~0% of GENUINE users with real ECAPA similarity
# scores -- i.e. those tiers as published would lock out most legitimate users.
ORDER_AMOUNT = 1500
TIER_MIN_TRUST = 50

W = 66


def rule(ch="-"):
    print(ch * W)


def load_16k(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio


def pick_scenario_files(scenario, seed=42):
    """Pick real ASVspoof2019 LA files: 3 enrolment + 1 test utterance."""
    rows = []
    with open(DATA_ROOT / "ASVspoof2019_LA_cm_protocols"
              / "ASVspoof2019.LA.cm.eval.trl.txt") as f:
        for line in f:
            p = line.split()
            rows.append((p[0], p[1], p[-1]))

    bona, spoof = defaultdict(list), defaultdict(list)
    for spk, utt, label in rows:
        (bona if label == "bonafide" else spoof)[spk].append(utt)

    rng = np.random.default_rng(seed)
    speakers = sorted(s for s in bona if len(bona[s]) >= 5 and spoof[s])
    target = speakers[int(rng.integers(len(speakers)))]

    enroll = bona[target][:3]
    flac = DATA_ROOT / "ASVspoof2019_LA_eval" / "flac"
    enroll_paths = [flac / f"{u}.flac" for u in enroll]

    if scenario == "genuine":
        cands, desc = [bona[target][3]], f"a REAL recording of the enrolled user ({target})"
    elif scenario == "spoof":
        # Offer several candidates; main() picks one that actually clears the
        # speaker-verification gate, since that is the case worth showing.
        # 54% of attacks do (measured, n=400) -- see
        # results/spoof_vs_speaker_verification.json.
        cands = spoof[target][:12]
        desc = f"an AI-GENERATED fake of the enrolled user ({target})"
    else:
        other = [s for s in speakers if s != target][0]
        cands, desc = [bona[other][0]], f"a REAL recording of a DIFFERENT person ({other})"

    return (enroll_paths, [flac / f"{u}.flac" for u in cands], desc, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["genuine", "spoof", "imposter"])
    parser.add_argument("--enroll", type=Path, nargs="+")
    parser.add_argument("--test", type=Path)
    parser.add_argument("--amount", type=int, default=ORDER_AMOUNT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.scenario:
        enroll_files, candidates, description, target = pick_scenario_files(args.scenario)
    elif args.enroll and args.test:
        enroll_files, candidates = args.enroll, [args.test]
        description, target = "your own supplied audio", "enrolled-user"
    else:
        raise SystemExit("Use --scenario {genuine,spoof,imposter}, or "
                          "--enroll a.wav b.wav --test x.wav")

    print()
    rule("=")
    print("  SecureVoice  —  two-layer voice authorisation".center(W))
    rule("=")

    print("  loading models...", end="\r")
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="models/ecapa",
        run_opts={"device": args.device})
    aasist = AasistLDetector(device=args.device)
    baseline = Path("results/aasist_baseline.json")
    if baseline.exists():
        with open(baseline) as f:
            aasist.threshold = json.load(f)["threshold"]
    print(" " * 40, end="\r")

    def embed(path):
        with torch.no_grad():
            e = ecapa.encode_batch(torch.from_numpy(load_16k(path)).unsqueeze(0))
        v = e.squeeze().cpu().numpy()
        return v / (np.linalg.norm(v) + 1e-9)

    centroid = np.mean([embed(p) for p in enroll_files], axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9

    # Among the candidates, prefer one that CLEARS the speaker-verification
    # gate -- that is the case the demo exists to show. Falls back to the
    # closest one if none clears it.
    scored = [(float(np.dot(centroid, embed(p))), p) for p in candidates]
    passing = [(s, p) for s, p in scored if s >= SIMILARITY_HARD_DENY]
    similarity, test_file = max(passing or scored)

    voice_matches = similarity >= SIMILARITY_HARD_DENY
    trust = similarity * 100
    result = aasist.score(load_16k(test_file))
    is_fake = result.score > aasist.threshold

    print(f"  Enrolled user : {target}  ({len(enroll_files)} enrolment recordings)")
    print(f"  Incoming audio: {test_file.name}")
    print(f"  This is       : {description}")
    print(f"  Requested     : place an order for Rs.{args.amount:,}")
    if args.scenario == "spoof":
        print(f"  (a representative attack: 54% of the 400 attacks measured")
        print(f"   in results/spoof_vs_speaker_verification.json clear layer 2)")
    rule()
    print("  LAYER 2 — Speaker verification  (ECAPA-TDNN, 192-dim)")
    print(f"     cosine similarity to enrolled voice : {similarity:.3f}")
    print(f"     trust score (0-100)                 : {trust:.0f}")
    print(f"     verdict  : {'VOICE MATCHES the enrolled user' if voice_matches else 'NOT the enrolled user'}")
    rule()
    print("  LAYER 2.5 — Anti-spoofing  (AASIST-L, 0.99% EER benchmark)")
    print(f"     P(synthetic speech)                 : {result.score:.4f}")
    print(f"     decision threshold (frozen on dev)  : {aasist.threshold:.4f}")
    print(f"     latency                             : {result.latency_ms:.0f} ms")
    print(f"     verdict  : {'AI-GENERATED / SYNTHETIC SPEECH' if is_fake else 'genuine human speech'}")
    rule("=")

    approved_before = voice_matches and trust >= TIER_MIN_TRUST
    approved_after = approved_before and not is_fake

    def verdict(ok):
        return "ORDER APPROVED" if ok else "ORDER DENIED"

    print(f"  Rs.{args.amount:,} order requires trust >= {TIER_MIN_TRUST}")
    print()
    print(f"     SecureVoice as presented (speaker verification only)")
    print(f"        -> {verdict(approved_before)}")
    print()
    print(f"     SecureVoice + anti-spoofing layer")
    print(f"        -> {verdict(approved_after)}")
    rule("=")
    if approved_before and not approved_after:
        print("  >> The attack passed speaker verification and was stopped")
        print("     ONLY by the anti-spoofing layer.")
        rule("=")
    print()


if __name__ == "__main__":
    main()
