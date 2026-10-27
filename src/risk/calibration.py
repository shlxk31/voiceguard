"""
risk/calibration.py

Turns each layer's raw score into a calibrated log-likelihood ratio (LLR).

WHY THIS EXISTS
Every detector in this system outputs a number on its own arbitrary scale:
ECAPA-TDNN gives a cosine similarity in [-1, 1], AASIST-L gives a softmax
probability in [0, 1]. These are NOT comparable and NOT interpretable --
"0.66 similarity" and "0.99 spoof probability" cannot be meaningfully added,
weighted, or compared against a cost-derived threshold.

Calibration fixes this. After calibration every layer speaks the same
language:

    LLR = log [ p(score | genuine) / p(score | attack) ]

    LLR > 0  ->  this evidence favours "genuine", by exp(LLR) : 1 odds
    LLR = 0  ->  this evidence is uninformative
    LLR < 0  ->  this evidence favours "attack"

Because LLRs are log-odds, they ADD. Combining independent evidence is then
just a sum, and the result can be compared directly against the Bayes
threshold derived in policy.py. Without this step, any fusion weight or
threshold would be an arbitrary hand-tuned number -- exactly the criticism
this module exists to remove.

METHOD
Platt scaling / logistic calibration: fit a 1-D logistic regression mapping
raw score -> P(genuine | score) on a held-out development set, then convert
the posterior back to a likelihood ratio by removing the training prior:

    LLR = logit( P(genuine|score) ) - logit( P_train(genuine) )

Subtracting the training prior is what makes the output a likelihood ratio
(evidence) rather than a posterior (evidence + whatever class balance the
training set happened to have). This matters here because our dev sets are
deliberately class-imbalanced.

Reference for cost/LLR-based assessment in this domain: Kinnunen et al.,
"t-DCF: a Detection Cost Function for the Tandem Assessment of Spoofing
Countermeasures and Automatic Speaker Verification", Odyssey 2018.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


def _logit(p: np.ndarray | float, eps: float = 1e-6) -> np.ndarray | float:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


@dataclass
class ScoreCalibrator:
    """
    Maps one layer's raw score to a log-likelihood ratio.

    Attributes
    ----------
    slope, intercept : the fitted logistic parameters (P(genuine|s) =
        sigmoid(slope * s + intercept)).
    train_prior : fraction of genuine samples in the calibration set; removed
        from the posterior so the output is an LLR, not a posterior.
    higher_is_genuine : True if a larger raw score means "more genuine"
        (e.g. ECAPA similarity); False if larger means "more attack"
        (e.g. AASIST P(synthetic)). Recorded so the sign convention of every
        layer is explicit rather than implicit.
    """

    slope: float
    intercept: float
    train_prior: float
    higher_is_genuine: bool
    layer_id: str = ""
    n_train: int = 0

    # ---------------------------------------------------------------- fit --
    @classmethod
    def fit(cls, scores: np.ndarray, is_genuine: np.ndarray, *,
            higher_is_genuine: bool, layer_id: str = "",
            max_iter: int = 1000) -> "ScoreCalibrator":
        """
        Fit on a DEVELOPMENT set only. `is_genuine` is a boolean array:
        True for genuine/bona fide trials, False for any attack type.
        """
        from sklearn.linear_model import LogisticRegression

        scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(is_genuine).astype(int).reshape(-1)
        if len(np.unique(y)) < 2:
            raise ValueError(f"[{layer_id}] calibration needs both classes present")

        lr = LogisticRegression(max_iter=max_iter)
        lr.fit(scores, y)
        return cls(
            slope=float(lr.coef_[0][0]),
            intercept=float(lr.intercept_[0]),
            train_prior=float(np.mean(y)),
            higher_is_genuine=higher_is_genuine,
            layer_id=layer_id,
            n_train=len(y),
        )

    # ------------------------------------------------------------- apply --
    def llr(self, raw_score: float | np.ndarray) -> float | np.ndarray:
        """Raw score -> calibrated log-likelihood ratio (positive = genuine)."""
        s = np.asarray(raw_score, dtype=np.float64)
        posterior = 1.0 / (1.0 + np.exp(-(self.slope * s + self.intercept)))
        out = _logit(posterior) - _logit(self.train_prior)
        return float(out) if np.isscalar(raw_score) or out.ndim == 0 else out

    def odds_statement(self, raw_score: float) -> str:
        """Plain-English reading of the evidence, for the UI and audit log."""
        v = self.llr(raw_score)
        odds = float(np.exp(abs(v)))
        if abs(v) < 0.2:
            return "uninformative (no meaningful evidence either way)"
        side = "genuine" if v > 0 else "attack"
        return f"{odds:,.0f}:1 in favour of {side}"

    # -------------------------------------------------------------- io ----
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScoreCalibrator":
        return cls(**d)


def save_calibrators(calibrators: dict[str, ScoreCalibrator], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: v.to_dict() for k, v in calibrators.items()}, f, indent=2)


def load_calibrators(path: Path) -> dict[str, ScoreCalibrator]:
    with open(path) as f:
        return {k: ScoreCalibrator.from_dict(v) for k, v in json.load(f).items()}
