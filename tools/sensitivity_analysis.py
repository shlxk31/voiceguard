"""
tools/sensitivity_analysis.py

How much do the decisions depend on the numbers we chose by hand?

THE CRITICISM THIS ANSWERS
"You derived tau from a cost formula, but you invented the costs. So the
threshold is still arbitrary, just with more steps."

That criticism is partly right and must be met with a measurement, not a
rebuttal. Two things are true at once:

  DERIVED   The FORM of the threshold -- that it rises logarithmically with
            the stake, that allow and deny come from different costs, that
            the prior enters as log-odds -- follows from Bayes risk
            minimisation. It is not a choice. (Kinnunen et al., Odyssey 2018.)

  ASSUMED   The four constants ARE our assumptions:
              handling_cost  = Rs.500    fixed cost of a fraud incident
              challenge_cost = Rs.20     cost of asking a genuine user again
              reject_cost    = Rs.400    cost of refusing a genuine user
              attack_prior   = 0.01      how often a request is an attack
            We have no fraud logs to estimate them from. Anyone is entitled
            to disagree with any of them.

So the question that actually matters is not "are the constants right" but
"HOW WRONG CAN THEY BE BEFORE THE SYSTEM BEHAVES DIFFERENTLY". A policy whose
verdicts are stable across every plausible setting is not arbitrary in any
sense that matters; one that flips on a factor of two is, and we would need
to say so loudly.

This script sweeps each constant over a wide plausible range, re-runs the
whole decision set, and reports how many verdicts move.
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
from tools.run_ablation import (THREATS, load_audio_scores,  # noqa: E402
                                load_command_scores, load_liveness_scores)

#: Ranges swept. Each spans at least an order of magnitude around the default,
#: which is far wider than any reasonable disagreement about the true value.
SWEEPS = {
    "handling_cost":  [100, 250, 500, 1_000, 2_500, 5_000],
    "challenge_cost": [5, 10, 20, 50, 100, 200],
    "reject_cost":    [100, 200, 400, 800, 1_500, 3_000],
    "attack_prior":   [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
}


def build_trials(audio, cmds, live, n_per_threat, seed):
    """One fixed set of composite trials, reused for every cost setting."""
    trials = []
    for threat, (acond, ccond, lcond) in THREATS.items():
        rng = random.Random(seed)
        for _ in range(n_per_threat):
            sa = rng.choice(audio[acond])
            sc = rng.choice(cmds[ccond])
            sl = rng.choice(live[lcond])
            trials.append((threat,
                           {"L1": sa["L1"], "L2": sl, "L3": sa["L3"], "L4": sc["L4"]},
                           sc["amount"]))
    return trials


def run(trials, cals, costs) -> list[str]:
    engine = DecisionEngine(cals, RiskPolicy(costs))
    return [engine.decide(raw, amt).outcome for _, raw, amt in trials]


def rates(trials, outcomes) -> dict[str, float]:
    out = {}
    for threat in THREATS:
        idx = [i for i, (t, _, _) in enumerate(trials) if t == threat]
        allowed = sum(1 for i in idx if outcomes[i] == "ALLOW") / len(idx)
        out[threat] = (1 - allowed) if threat == "genuine user" else allowed
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-threat", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results" / "sensitivity.csv"))
    a = ap.parse_args()

    audio = load_audio_scores(ROOT / "results" / "spoof_vs_speaker_verification.csv",
                              ROOT / "results" / "calibration_split.json")
    cmds = load_command_scores(ROOT / "results" / "l4_scores.json")
    live = load_liveness_scores(ROOT / "results" / "l2_scores.json")
    cals = load_calibrators(ROOT / "models" / "calibrators.json")

    trials = build_trials(audio, cmds, live, a.n_per_threat, a.seed)
    base_costs = CostModel()
    base = run(trials, cals, base_costs)
    base_rates = rates(trials, base)

    print(f"{len(trials)} composite trials, {a.n_per_threat} per threat")
    print(f"\nBaseline (handling {base_costs.handling_cost:g}, challenge "
          f"{base_costs.challenge_cost:g}, reject {base_costs.reject_cost:g}, "
          f"prior {base_costs.attack_prior:g}):")
    for k, v in base_rates.items():
        kind = "FRR" if k == "genuine user" else "FAR"
        print(f"  {k:26s} {kind} {v * 100:5.1f}%")

    rows = []
    print("\nSweeping each constant over an order of magnitude or more.")
    print("'verdicts changed' counts trials whose ALLOW/CHALLENGE/DENY differs "
          "from baseline.\n")

    for param, values in SWEEPS.items():
        print(f"{param}:")
        for v in values:
            kwargs = {param: v}
            try:
                costs = CostModel(**{**base_costs.to_dict(), **kwargs})
            except ValueError as exc:
                print(f"  {v:<10g} skipped ({exc})")
                continue
            outs = run(trials, cals, costs)
            changed = sum(1 for x, y in zip(outs, base) if x != y) / len(outs)
            r = rates(trials, outs)
            worst_far = max(val for k, val in r.items() if k != "genuine user")
            mark = " <- default" if v == getattr(base_costs, param) else ""
            print(f"  {v:<10g} verdicts changed {changed * 100:5.1f}%   "
                  f"worst FAR {worst_far * 100:5.1f}%   "
                  f"genuine FRR {r['genuine user'] * 100:5.1f}%{mark}")
            rows.append({"param": param, "value": v,
                         "verdicts_changed_pct": round(changed * 100, 2),
                         "worst_far_pct": round(worst_far * 100, 2),
                         "genuine_frr_pct": round(r["genuine user"] * 100, 2)})
        print()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ---- the headline: how stable are the ATTACK verdicts overall? ---------
    fars = [r["worst_far_pct"] for r in rows]
    frrs = [r["genuine_frr_pct"] for r in rows]
    print("Across every setting swept:")
    print(f"  worst-case FAR ranged {min(fars):.1f}% - {max(fars):.1f}%")
    print(f"  genuine FRR    ranged {min(frrs):.1f}% - {max(frrs):.1f}%")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
