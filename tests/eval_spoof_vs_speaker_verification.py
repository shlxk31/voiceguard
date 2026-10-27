"""
tests/eval_spoof_vs_speaker_verification.py

THE HEADLINE EXPERIMENT for the SecureVoice paper.

Claim under test: speaker verification alone cannot stop voice cloning.
A cloned/synthesised voice is *designed* to sound like the target speaker,
so it scores HIGH on speaker similarity -- the trust score goes UP and the
transaction is approved. Speaker verification answers "is this the right
voice?", never "is this a real human?"

We measure that directly on ASVspoof2019 LA, which conveniently labels
every spoof utterance with the speaker it is impersonating (field 0 of the
CM protocol), and then show what adding a benchmarked anti-spoofing layer
(AASIST-L) does to the authorization outcome.

Three trial types per enrolled speaker:
  genuine  -- other bonafide utterances of the SAME speaker  (should pass)
  imposter -- bonafide utterances of a DIFFERENT speaker      (should fail)
  spoof    -- TTS/VC attacks TARGETING that speaker           (the attack)

Layers:
  L2   speaker verification -- SpeechBrain ECAPA-TDNN, 192-dim embeddings,
       cosine similarity against an enrolment centroid (this is what the
       SecureVoice slides describe; the shipped backend currently uses a
       hand-weighted MFCC/pitch/formant blend instead).
  L2.5 anti-spoofing -- AASIST-L with the threshold frozen on the
       ASVspoof19 LA dev split (results/aasist_baseline.json).

MODELLING ASSUMPTION (state this in the paper): SecureVoice's trust score
blends voice similarity with device, behavioural and session factors. Here
those other factors are held ideal so the voice channel is isolated --
trust = 100 x similarity. That is the quantity the attack actually moves.

Usage:
    .venv/bin/python tests/eval_spoof_vs_speaker_verification.py
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector.aasist_l import AasistLDetector
from detector.metrics import compute_eer
from eval_common import load_audio

# SecureVoice's published authorization matrix (PPT slides 11 / 13).
RISK_TIERS = [
    ("< Rs.500", 0),
    ("Rs.500-2,000", 50),
    ("Rs.2,000-10,000", 70),
    ("> Rs.10,000", 85),
]
HARD_DENY_SIMILARITY = 0.50  # SecureVoice: similarity < 0.50 -> deny regardless


def load_protocol_with_speakers(data_root: Path, split: str):
    """ASVspoof19 LA CM protocol: SPEAKER_ID UTT_ID - SYSTEM_ID KEY.
    For spoof rows the speaker id is the speaker being impersonated."""
    name = {"dev": "ASVspoof2019.LA.cm.dev.trl.txt",
            "eval": "ASVspoof2019.LA.cm.eval.trl.txt"}[split]
    rows = []
    with open(data_root / "ASVspoof2019_LA_cm_protocols" / name) as f:
        for line in f:
            p = line.split()
            rows.append((p[0], p[1], p[-1]))  # speaker, utt_id, label
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/asvspoof2019_la"))
    parser.add_argument("--split", default="eval")
    parser.add_argument("--n-speakers", type=int, default=20)
    parser.add_argument("--n-enroll", type=int, default=3)
    parser.add_argument("--n-genuine", type=int, default=10)
    parser.add_argument("--n-spoof", type=int, default=20)
    parser.add_argument("--n-imposter", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--aasist-baseline", type=Path,
                         default=Path("results/aasist_baseline.json"))
    parser.add_argument("--out", type=Path,
                         default=Path("results/spoof_vs_speaker_verification.csv"))
    parser.add_argument("--summary-out", type=Path,
                         default=Path("results/spoof_vs_speaker_verification.json"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    rows = load_protocol_with_speakers(args.data_root, args.split)
    bona_by_spk, spoof_by_spk = defaultdict(list), defaultdict(list)
    for spk, utt, label in rows:
        (bona_by_spk if label == "bonafide" else spoof_by_spk)[spk].append(utt)

    speakers = sorted(s for s in bona_by_spk
                      if len(bona_by_spk[s]) >= args.n_enroll + 2 and spoof_by_spk[s])
    speakers = list(rng.permutation(speakers)[:args.n_speakers])
    print(f"{len(speakers)} enrolled speakers from {args.split} "
          f"({args.n_enroll} enrolment utterances each)")

    print("Loading ECAPA-TDNN (speaker verification, layer 2)...")
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="models/ecapa",
        run_opts={"device": args.device})

    print("Loading AASIST-L (anti-spoofing, layer 2.5)...")
    aasist = AasistLDetector(device=args.device)
    with open(args.aasist_baseline) as f:
        aasist.threshold = json.load(f)["threshold"]
    print(f"  frozen spoof threshold = {aasist.threshold:.4f}")

    embed_cache: dict[str, np.ndarray] = {}

    def embed(utt_id):
        if utt_id not in embed_cache:
            audio = load_audio(args.data_root, args.split, utt_id)
            with torch.no_grad():
                e = ecapa.encode_batch(torch.from_numpy(audio).unsqueeze(0))
            v = e.squeeze().cpu().numpy()
            embed_cache[utt_id] = v / (np.linalg.norm(v) + 1e-9)
        return embed_cache[utt_id]

    records = []
    for i, spk in enumerate(speakers):
        bona = list(rng.permutation(bona_by_spk[spk]))
        enroll_utts, genuine_utts = bona[:args.n_enroll], bona[args.n_enroll:][:args.n_genuine]
        centroid = np.mean([embed(u) for u in enroll_utts], axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9

        spoof_utts = list(rng.permutation(spoof_by_spk[spk]))[:args.n_spoof]
        others = [s for s in speakers if s != spk]
        imposter_utts = []
        for o in rng.permutation(others)[:args.n_imposter]:
            imposter_utts.append(rng.choice(bona_by_spk[o]))

        for trial_type, utts in (("genuine", genuine_utts), ("spoof", spoof_utts),
                                  ("imposter", imposter_utts)):
            for utt in utts:
                similarity = float(np.dot(centroid, embed(utt)))
                audio = load_audio(args.data_root, args.split, utt)
                spoof_score = aasist.score(audio).score
                records.append({"speaker": spk, "utt_id": utt, "trial_type": trial_type,
                                 "similarity": similarity, "spoof_score": spoof_score,
                                 "flagged_spoof": int(spoof_score > aasist.threshold)})
        print(f"  [{i + 1}/{len(speakers)}] {spk}: {len(genuine_utts)} genuine, "
              f"{len(spoof_utts)} spoof, {len(imposter_utts)} imposter")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    sim = {t: np.array([r["similarity"] for r in records if r["trial_type"] == t])
           for t in ("genuine", "spoof", "imposter")}
    spf = {t: np.array([r["spoof_score"] for r in records if r["trial_type"] == t])
           for t in ("genuine", "spoof", "imposter")}
    flag = {t: np.array([r["flagged_spoof"] for r in records if r["trial_type"] == t])
            for t in ("genuine", "spoof", "imposter")}

    print("\n=== LAYER 2: speaker verification (ECAPA-TDNN cosine similarity) ===")
    print(f"{'trial':10s}{'n':>6s}{'mean sim':>11s}{'median':>10s}{'>=0.50':>9s}{'>=0.75':>9s}")
    for t in ("genuine", "spoof", "imposter"):
        s = sim[t]
        print(f"{t:10s}{len(s):>6d}{s.mean():>11.3f}{np.median(s):>10.3f}"
              f"{(s >= 0.50).mean() * 100:>8.1f}%{(s >= 0.75).mean() * 100:>8.1f}%")

    sv_eer_imposter, _ = compute_eer(-sim["genuine"], -sim["imposter"])
    sv_eer_spoof, _ = compute_eer(-sim["genuine"], -sim["spoof"])
    print(f"\n  SV EER, genuine vs imposter : {sv_eer_imposter * 100:.2f}%  "
          f"(this is the number a normal SV benchmark reports)")
    print(f"  SV EER, genuine vs SPOOF    : {sv_eer_spoof * 100:.2f}%  "
          f"(near 50% = spoofs are indistinguishable from genuine to this layer)")

    print("\n=== LAYER 2.5: anti-spoofing (AASIST-L) ===")
    print(f"{'trial':10s}{'mean P(spoof)':>16s}{'flagged as spoof':>19s}")
    for t in ("genuine", "spoof", "imposter"):
        print(f"{t:10s}{spf[t].mean():>16.4f}{flag[t].mean() * 100:>18.1f}%")

    print("\n=== AUTHORIZATION OUTCOME (SecureVoice risk tiers) ===")
    print("Trust = 100 x similarity, other trust factors held ideal.")
    print(f"{'tier':18s}{'min trust':>10s}{'spoof approved':>17s}{'spoof approved':>17s}")
    print(f"{'':18s}{'':>10s}{'(SV only)':>17s}{'(SV + antispoof)':>17s}")
    tier_rows = []
    for tier_name, min_trust in RISK_TIERS:
        trust = sim["spoof"] * 100
        approved_sv = (trust >= min_trust) & (sim["spoof"] >= HARD_DENY_SIMILARITY)
        approved_both = approved_sv & (flag["spoof"] == 0)
        tier_rows.append({"tier": tier_name, "min_trust": min_trust,
                           "spoof_approved_sv_only_pct": float(approved_sv.mean() * 100),
                           "spoof_approved_with_antispoof_pct": float(approved_both.mean() * 100)})
        print(f"{tier_name:18s}{min_trust:>10d}{approved_sv.mean() * 100:>16.1f}%"
              f"{approved_both.mean() * 100:>16.1f}%")

    summary = {
        "n_speakers": len(speakers), "n_trials": len(records),
        "similarity": {t: {"mean": float(sim[t].mean()), "median": float(np.median(sim[t])),
                            "frac_ge_0.50": float((sim[t] >= 0.50).mean()),
                            "frac_ge_0.75": float((sim[t] >= 0.75).mean())}
                        for t in sim},
        "spoof_score": {t: {"mean": float(spf[t].mean()),
                             "flagged_rate": float(flag[t].mean())} for t in spf},
        "sv_eer_genuine_vs_imposter": sv_eer_imposter,
        "sv_eer_genuine_vs_spoof": sv_eer_spoof,
        "aasist_threshold": aasist.threshold,
        "authorization_by_tier": tier_rows,
    }
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPer-trial CSV -> {args.out}\nSummary -> {args.summary_out}")


if __name__ == "__main__":
    main()
