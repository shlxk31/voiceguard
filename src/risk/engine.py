"""
risk/engine.py

Orchestrates the layers and produces a fully explainable decision.

DESIGN GOAL
Every number the system shows must be traceable to either (a) a measured
model output, or (b) a formula with a citation. The engine therefore never
returns a bare verdict -- it returns a `Decision` carrying the complete audit
trail: each layer's raw score, its calibrated evidence, how much that
evidence moved the outcome, the derived thresholds, and the margin.

ARCHITECTURE (the reframe that makes the design defensible)
This is not "five layers of security". It is:

    four independent EVIDENCE SOURCES  ->  calibration  ->  one DECISION POLICY

Layers 1-4 are detectors that answer different questions and cover different
threats. Layer 5 is not a detector at all -- it is the policy that fuses the
evidence and chooses allow / challenge / deny.

THREAT COVERAGE (why more than one layer is needed at all)
The argument for layering is not "more is safer", it is that each layer
catches something the others structurally cannot:

    Threat                     L1 voice  L2 liveness  L3 anti-spoof  L4 context
    Different human            CATCHES   -            -              partial
    Replay of a recording      -         CATCHES      -              partial
    TTS / voice-conversion     fails     partial      CATCHES        -
    Stolen device, real voice  -         -            -              CATCHES

Two of those cells are measured in this project: L1 rejects 100% of impostors
but lets through 54% of synthetic clones; L3 catches 87% of those clones.

FUSION -- AND WHY THE OBVIOUS RULE IS WRONG
Calibrated LLRs are log-odds, so it is tempting to combine them by addition:
that is the standard naive-Bayes fusion of independent likelihood ratios, and
it is what this engine originally did. The ablation study showed it is
invalid here, and the failure is worth stating precisely because it is easy
to reproduce and easy to miss.

Addition assumes every layer scores the SAME hypothesis. Ours do not:

    L1 "genuine" means  the enrolled speaker
    L2 "genuine" means  spoken into a live microphone
    L3 "genuine" means  a real human vocal tract, not a synthesiser
    L4 "genuine" means  a normal request from this user

Consider an impostor -- a different human, speaking live. L3 is entirely
correct to report "real human" at LLR +2.72. But summed, that correct
statement cancels L1's correct -2.63 and the impostor is scored +0.09,
i.e. genuine. Adding a working layer made the system worse, and the measured
ablation confirmed it: impostor FAR went from 0.0% with L1 alone to 47.5%
with L1+L3.

The error is in the fusion, not the layers. Defence in depth is a
CONJUNCTION -- an attacker must defeat every layer, so a request is genuine
only if EVERY applicable layer finds it genuine. The corresponding rule on
log-odds is the weakest link:

    fused = min over available layers of ( weight * LLR )

Read as a bound, this is the log-odds of the single most incriminating piece
of evidence, and it has the property the security argument requires:
adding a layer can only ever lower the fused score, so a new layer can never
open a hole -- it can only cost false rejections. That is the real trade the
ablation measures.

`fusion_rule="sum"` is retained ONLY so the paper can report the naive-Bayes
baseline alongside the conjunctive rule. It is not a supported operating mode.

Weights exist so the ablation study can switch layers off without changing
any other code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .calibration import ScoreCalibrator
from .policy import RiskPolicy, ThresholdDerivation


# Static description of each layer: what it is and which threat it exists for.
LAYER_INFO = {
    "L1": {
        "name": "Voiceprint",
        "model": "ECAPA-TDNN (192-d embedding, cosine similarity)",
        "question": "Is this the enrolled user's voice?",
        "threat": "Impostor - a different human",
        "score_label": "cosine similarity",
        "higher_is_genuine": True,
    },
    "L2": {
        "name": "Liveness",
        "model": "Air-conduction replay detector",
        "question": "Did this come from a mouth, or a loudspeaker?",
        "threat": "Replay of a recording",
        "score_label": "P(playback)",
        "higher_is_genuine": False,
    },
    "L3": {
        "name": "Anti-spoofing",
        "model": "AASIST-L (graph attention, raw waveform)",
        "question": "Is this a real human, or AI-generated?",
        "threat": "TTS / voice-conversion clone",
        "score_label": "P(synthetic)",
        "higher_is_genuine": False,
    },
    "L4": {
        "name": "Context",
        "model": "Contextual risk scorer",
        "question": "Is this request normal for this user?",
        "threat": "Stolen device with a genuine voice",
        "score_label": "normality score",
        "higher_is_genuine": True,
    },
}


@dataclass
class LayerEvidence:
    """One layer's contribution, with everything needed to explain it."""

    layer_id: str
    name: str
    model: str
    question: str
    threat: str
    score_label: str
    available: bool
    raw_score: float | None = None
    llr: float | None = None
    weight: float = 1.0
    contribution: float = 0.0
    reading: str = ""
    unavailable_reason: str = ""

    @property
    def favours(self) -> str:
        if not self.available or self.llr is None:
            return "n/a"
        if abs(self.llr) < 0.2:
            return "neutral"
        return "genuine" if self.llr > 0 else "attack"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["favours"] = self.favours
        return d


@dataclass
class Decision:
    """The complete, auditable outcome of one authorization request."""

    outcome: str
    amount: float
    fused_llr: float
    thresholds: ThresholdDerivation
    evidence: list[LayerEvidence] = field(default_factory=list)
    fusion_rule: str = "min"

    @property
    def margin(self) -> float:
        """How far the evidence sits from the boundary that decided it."""
        if self.outcome == RiskPolicy.ALLOW:
            return self.fused_llr - self.thresholds.tau_allow
        if self.outcome == RiskPolicy.DENY:
            return self.thresholds.tau_deny - self.fused_llr
        return min(self.thresholds.tau_allow - self.fused_llr,
                   self.fused_llr - self.thresholds.tau_deny)

    def summary(self) -> str:
        active = [e for e in self.evidence if e.available]
        drivers = sorted(active, key=lambda e: abs(e.contribution), reverse=True)
        if not drivers:
            return "No evidence available."
        top = drivers[0]
        word = "weakest layer" if self.fusion_rule == "min" else "total evidence"
        return (f"{self.outcome} at Rs.{self.amount:,.0f}: {word} "
                f"{self.fused_llr:+.2f} vs required {self.thresholds.tau_allow:.2f} "
                f"to allow. Strongest signal: {top.name} ({top.contribution:+.2f}).")

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "fusion_rule": self.fusion_rule,
            "amount": self.amount,
            "fused_llr": self.fused_llr,
            "margin": self.margin,
            "summary": self.summary(),
            "thresholds": self.thresholds.to_dict(),
            "derivation": self.thresholds.explain(),
            "evidence": [e.to_dict() for e in self.evidence],
        }


class DecisionEngine:
    """
    Fuses calibrated layer evidence and applies the amount-dependent policy.

    Layers absent from `calibrators` are reported as unavailable rather than
    silently skipped -- an honest UI should show which defences were actually
    in play for a given decision.
    """

    #: "min" -- conjunctive / weakest-link, the correct rule (see module doc).
    #: "sum" -- naive-Bayes addition, retained only as the paper's baseline.
    FUSION_RULES = ("min", "sum")

    def __init__(self, calibrators: dict[str, ScoreCalibrator],
                 policy: RiskPolicy | None = None,
                 weights: dict[str, float] | None = None,
                 fusion_rule: str = "min"):
        if fusion_rule not in self.FUSION_RULES:
            raise ValueError(f"fusion_rule must be one of {self.FUSION_RULES}")
        self.calibrators = calibrators
        self.policy = policy or RiskPolicy()
        self.weights = weights or {}
        self.fusion_rule = fusion_rule

    def decide(self, raw_scores: dict[str, float | None], amount: float) -> Decision:
        """
        Parameters
        ----------
        raw_scores : each layer's native output, e.g.
            {"L1": 0.72, "L3": 0.0004, "L2": None, "L4": None}
            None (or a missing key) marks that layer as unavailable.
        amount : transaction amount, which sets the thresholds.
        """
        evidence: list[LayerEvidence] = []
        contributions: list[float] = []

        for layer_id, info in LAYER_INFO.items():
            score = raw_scores.get(layer_id)
            cal = self.calibrators.get(layer_id)

            if score is None or cal is None:
                reason = ("not implemented / no calibration fitted" if cal is None
                          else "no score produced for this request")
                evidence.append(LayerEvidence(
                    layer_id=layer_id, name=info["name"], model=info["model"],
                    question=info["question"], threat=info["threat"],
                    score_label=info["score_label"], available=False,
                    unavailable_reason=reason,
                ))
                continue

            llr = float(cal.llr(float(score)))
            weight = float(self.weights.get(layer_id, 1.0))
            contribution = weight * llr
            contributions.append(contribution)

            evidence.append(LayerEvidence(
                layer_id=layer_id, name=info["name"], model=info["model"],
                question=info["question"], threat=info["threat"],
                score_label=info["score_label"], available=True,
                raw_score=float(score), llr=llr, weight=weight,
                contribution=contribution,
                reading=cal.odds_statement(float(score)),
            ))

        # No layer available means no evidence either way -- 0.0 is the
        # uninformative log-odds, and the amount-dependent threshold then
        # decides, which for any non-trivial amount means "do not allow".
        if not contributions:
            fused = 0.0
        elif self.fusion_rule == "min":
            fused = min(contributions)
        else:
            fused = sum(contributions)

        outcome, thresholds = self.policy.decide(fused, amount)
        return Decision(outcome=outcome, amount=float(amount), fused_llr=fused,
                        thresholds=thresholds, evidence=evidence,
                        fusion_rule=self.fusion_rule)
