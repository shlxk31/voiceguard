"""
detector/liveness_l2.py

Layer 2 -- liveness. "Did this come from a mouth, or a loudspeaker?"

Loads the logistic model fitted by tools/train_l2_liveness.py and scores one
utterance. The raw score is P(playback), so HIGHER means more attack, the
same convention as L3's P(synthetic).

WHAT THIS DETECTS, PHYSICALLY
A replayed utterance has been through the air twice and through a transducer
chain the live path never touches: mouth -> air -> mic, versus mouth -> air
-> mic -> speaker -> air -> mic. The extra loudspeaker and room convolution
reshapes the spectral envelope, which is why the strongest discriminating
features are the low-order MFCC means.

SCOPE, WHICH IS NARROW AND MUST BE QUOTED WITH THE NUMBER
Fitted on 15 live/playback pairs from one speaker, one room, and one
playback device -- a laptop's built-in speakers, which are poor reproducers
and therefore leave an obvious signature. It reached 100% under
leave-one-pair-out cross-validation and classified 20 held-out genuine clips
correctly, so it is not merely memorising its training pairs. It has NOT been
tested against a good loudspeaker, another room, or another voice, and the
honest expectation is that a high-fidelity replay is considerably harder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "l2_liveness.json"
SAMPLE_RATE = 16000


@dataclass
class LivenessResult:
    score: float          # P(playback) -- higher is more likely a replay
    available: bool = True
    note: str = ""


class LivenessDetector:
    def __init__(self, model_path: Path | str = MODEL_PATH):
        self.model_path = Path(model_path)
        self._m = None

    @property
    def available(self) -> bool:
        return self.model_path.exists()

    def _load(self):
        if self._m is None:
            d = json.loads(self.model_path.read_text())
            self._m = {
                "mean": np.array(d["mean"]), "scale": np.array(d["scale"]),
                "coef": np.array(d["coef"]), "intercept": float(d["intercept"]),
                "crop": float(d.get("crop_seconds", 5.5)),
            }
        return self._m

    def score(self, audio: np.ndarray) -> LivenessResult:
        """Never raises; an unavailable layer is reported, not guessed."""
        if not self.available:
            return LivenessResult(0.0, available=False,
                                  note="no liveness model fitted")
        try:
            from liveness_detector import extract_features
        except Exception as exc:
            return LivenessResult(0.0, available=False,
                                  note=f"feature extractor unavailable: {exc}")
        try:
            m = self._load()
            # Crop exactly as training did, so the feature distribution matches.
            n = int(m["crop"] * SAMPLE_RATE)
            a = np.pad(audio, (0, max(0, n - len(audio))))[:n]
            x = (extract_features(a, SAMPLE_RATE) - m["mean"]) / m["scale"]
            p_live = 1.0 / (1.0 + np.exp(-(float(x @ m["coef"]) + m["intercept"])))
            return LivenessResult(float(1.0 - p_live))
        except Exception as exc:
            return LivenessResult(0.0, available=False,
                                  note=f"{type(exc).__name__}: {exc}")
