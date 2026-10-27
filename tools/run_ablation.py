"""
tools/run_ablation.py

The layer-ablation study: does every layer actually earn its place?

THE QUESTION
"Why five layers? Prove that five beats four, three, and two." The only
honest answer is a measurement, and the measurement has to be per-THREAT,
because that is the entire premise of layering. An aggregate accuracy number
would hide the point: a subset can look excellent on average while being
completely blind to one attack class. Blindness is the thing we are
demonstrating, so it must be visible in the table.

WHAT IS AND IS NOT REAL HERE -- READ THIS BEFORE QUOTING ANY NUMBER
Each layer's scores come from its own measured evaluation:

  L1 (voiceprint)   real. ECAPA-TDNN on 800 ASVspoof2019 LA trials
                    (200 genuine / 200 impostor / 400 spoof).
  L3 (anti-spoof)   real. AASIST-L on those same 800 trials.
  L4 (context)      scores are real model outputs, but on a TEMPLATE-
                    GENERATED command corpus (see build_command_corpus.py).
  L2 (liveness)     real model outputs, but on a SMALL corpus captured on
                    one machine: 15 live/playback pairs, one speaker, one
                    room, one playback device (the laptop's own speakers).
                    Scores are leave-one-pair-out, never in-sample. Good
                    enough to report; not a general replay detector, and a
                    high-quality loudspeaker is untested.

COMPOSITE TRIALS
No public corpus carries all of these conditions on the same utterance, so
a trial here PAIRS an independently measured audio condition with an
independently measured command condition. This is a valid way to evaluate
the FUSION POLICY -- which is what the ablation is about -- and it is not a
claim about a real joint distribution. Pairing assumes the layers' errors
are independent; a real deployment could have correlated failures, and the
paper must say so.

We therefore report FIVE threat rows and, crucially, the genuine row, since
the cost of layering is false rejection of real users.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.risk.calibration import load_calibrators          # noqa: E402
from src.risk.engine import DecisionEngine                  # noqa: E402
from src.risk.policy import RiskPolicy, CostModel           # noqa: E402

#: Subsets evaluated. L2 is absent throughout because it is unimplemented.
SUBSETS = [
    ("L1 only",                   ["L1"]),
    ("L1 + L3",                   ["L1", "L3"]),
    ("L1 + L2 + L3",              ["L1", "L2", "L3"]),
    ("L1 + L3 + L4",              ["L1", "L3", "L4"]),
    ("L1 + L2 + L3 + L4 (full)",  ["L1", "L2", "L3", "L4"]),
]

#: Threat -> (audio condition, command condition). "genuine" is the
#: legitimate-user row and is what false-rejection is measured on.
#: Threat -> (audio condition, command condition, liveness condition).
#: "genuine" is the legitimate-user row and is what false rejection is
#: measured on. The replay row is the one L2 exists for: the attacker holds a
#: recording of the real user, so L1 sees the right voice and L3 sees a real
#: human -- both correctly pass -- and only L2 can object.
THREATS = {
    "genuine user":            ("genuine", "genuine", "live"),
    "impostor (other human)":  ("imposter", "genuine", "live"),
    "replay of a recording":   ("genuine", "genuine", "playback"),
    "TTS / VC clone":          ("spoof", "genuine", "live"),
    "prompt injection":        ("genuine", "injection", "live"),
    "social engineering":      ("genuine", "social_engineering", "live"),
}


def load_audio_scores(path: Path, split_path: Path) -> dict[str, list[dict]]:
    """
    Audio-layer scores, restricted to the EVALUATION half.

    The L1 and L3 calibrators were fitted on the development half. Scoring
    the fused system on those same trials would test each calibrator against
    its own training data; the split file exists precisely so this script
    cannot do that by accident.
    """
    rows = list(csv.DictReader(open(path)))
    keep = set(json.loads(Path(split_path).read_text())["eval_index"])
    by_cond: dict[str, list[dict]] = {}
    for i, r in enumerate(rows):
        if i not in keep:
            continue
        by_cond.setdefault(r["trial_type"], []).append(
            {"L1": float(r["similarity"]), "L3": float(r["spoof_score"])})
    return by_cond


def load_liveness_scores(path: Path) -> dict[str, list[float]]:
    """Out-of-fold L2 scores, P(playback), keyed by capture condition."""
    d = json.loads(Path(path).read_text())
    return {"live": [float(x) for x in d["live"]],
            "playback": [float(x) for x in d["playback"]]}


def load_command_scores(path: Path) -> dict[str, list[dict]]:
    """Command scores, restricted to L4's held-out half for the same reason."""
    d = json.loads(Path(path).read_text())
    by_fam: dict[str, list[dict]] = {}
    for r in d["rows"]:
        if r.get("split") != "eval":
            continue
        by_fam.setdefault(r["family"], []).append(
            {"L4": float(r["normality"]), "amount": float(r["amount"])})
    return by_fam


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", default=str(ROOT / "results" / "spoof_vs_speaker_verification.csv"))
    ap.add_argument("--split", default=str(ROOT / "results" / "calibration_split.json"))
    ap.add_argument("--commands", default=str(ROOT / "results" / "l4_scores.json"))
    ap.add_argument("--liveness", default=str(ROOT / "results" / "l2_scores.json"))
    ap.add_argument("--calibrators", default=str(ROOT / "models" / "calibrators.json"))
    ap.add_argument("--n-trials", type=int, default=400,
                    help="composite trials per threat row")
    ap.add_argument("--amount", type=float, default=None,
                    help="fix the amount; default uses each command's own")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fusion", default="min", choices=["min", "sum", "both"],
                    help="'min' is the operating rule; 'sum' is the "
                         "naive-Bayes baseline the paper compares against")
    ap.add_argument("--out", default=str(ROOT / "results" / "ablation.csv"))
    a = ap.parse_args()

    audio = load_audio_scores(Path(a.audio), Path(a.split))
    cmds = load_command_scores(Path(a.commands))
    live = load_liveness_scores(Path(a.liveness))
    cals = load_calibrators(Path(a.calibrators))
    policy = RiskPolicy(CostModel())

    missing = [l for l in ("L1", "L2", "L3", "L4") if l not in cals]
    if missing:
        sys.exit(f"missing calibrators for {missing} -- fit them first")

    print("Evaluation pools (held-out halves only):")
    for k, v in sorted(audio.items()):
        print(f"  audio/{k:10s} n={len(v)}")
    for k, v in sorted(cmds.items()):
        print(f"  cmd/{k:20s} n={len(v)}")
    for k, v in sorted(live.items()):
        print(f"  liveness/{k:15s} n={len(v)}")

    rules = ["min", "sum"] if a.fusion == "both" else [a.fusion]
    rng = random.Random(a.seed)
    rows: list[dict] = []

    for rule in rules:
      for label, layers in SUBSETS:
        engine = DecisionEngine({k: v for k, v in cals.items() if k in layers},
                                policy, fusion_rule=rule)
        for threat, (acond, ccond, lcond) in THREATS.items():
            pool_a, pool_c, pool_l = audio[acond], cmds[ccond], live[lcond]
            outcomes = {"ALLOW": 0, "CHALLENGE": 0, "DENY": 0}
            trial_rng = random.Random(a.seed)   # same trials for every subset
            for _ in range(a.n_trials):
                sa, sc = trial_rng.choice(pool_a), trial_rng.choice(pool_c)
                sl = trial_rng.choice(pool_l)
                raw = {"L1": sa["L1"], "L2": sl, "L3": sa["L3"], "L4": sc["L4"]}
                raw = {k: (v if k in layers else None) for k, v in raw.items()}
                amt = a.amount if a.amount is not None else sc["amount"]
                outcomes[engine.decide(raw, amt).outcome] += 1

            n = a.n_trials
            allowed = outcomes["ALLOW"] / n
            genuine_row = threat == "genuine user"
            rows.append({
                "fusion_rule": rule,
                "subset": label,
                "n_layers": len(layers),
                "threat": threat,
                "n": n,
                "allow_pct": round(allowed * 100, 2),
                "challenge_pct": round(outcomes["CHALLENGE"] / n * 100, 2),
                "deny_pct": round(outcomes["DENY"] / n * 100, 2),
                # For attacks the headline is FAR = fraction wrongly allowed.
                # For the genuine row it is FRR = fraction not allowed.
                "error_pct": round((1 - allowed) * 100 if genuine_row
                                   else allowed * 100, 2),
                "error_kind": "FRR" if genuine_row else "FAR",
            })

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- printed table ----------------------------------------------------
    threats = list(THREATS)
    w0 = max(len(s) for s, _ in SUBSETS) + 2
    for rule in rules:
        print(f"\n=== fusion rule: {rule} "
              f"({'conjunctive / weakest link' if rule == 'min' else 'naive-Bayes sum (baseline)'}) ===")
        print("FAR by threat (% of attacks wrongly ALLOWED); "
              "last column is FRR on genuine users\n")
        hdr = "subset".ljust(w0) + "".join(t[:17].rjust(19) for t in threats[1:]) \
            + "genuine FRR".rjust(19)
        print(hdr)
        print("-" * len(hdr))
        for label, _ in SUBSETS:
            line = label.ljust(w0)
            for t in threats[1:]:
                r = next(x for x in rows if x["subset"] == label
                         and x["threat"] == t and x["fusion_rule"] == rule)
                line += f"{r['error_pct']:.1f}%".rjust(19)
            g = next(x for x in rows if x["subset"] == label
                     and x["threat"] == "genuine user"
                     and x["fusion_rule"] == rule)
            line += f"{g['error_pct']:.1f}%".rjust(19)
            print(line)

    print("\nL2 scores come from 15 live/playback pairs captured on one machine "
          "(one speaker, one room, the laptop's own speakers). The replay row "
          "is real but narrow -- a high-quality loudspeaker is untested.")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
