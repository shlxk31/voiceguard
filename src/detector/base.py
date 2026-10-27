"""
detector/base.py

Defines the common interface every detection backend must implement.
This is the "swappable front-end" contract — the streaming layer and the
FastAPI layer only ever talk to this interface, never to a specific model.
That way you can swap LFCC-GMM <-> AASIST-L <-> a future SSL-based model
without touching any other file.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class DetectionResult:
    """Result of scoring one chunk (or one full utterance) of audio."""
    score: float          # higher = more likely SPOOF/synthetic. Raw model score.
    is_spoof: bool         # thresholded decision
    confidence: float      # 0-1, calibrated if possible, else abs(score - threshold)
    latency_ms: float      # wall-clock inference time for THIS call, for the paper's
                            # latency/EER tradeoff plots — always measure this.


class SpoofDetector(ABC):
    """
    Abstract base for a bonafide-vs-spoof classifier.

    Implementations:
      - LFCCGMMDetector   (src/detector/lfcc_gmm.py)  -- classical baseline, CPU-only
      - AasistLDetector   (src/detector/aasist_l.py)  -- primary model, pretrained checkpoint

    Contract:
      - `score()` must accept a 1D float32 numpy array at 16kHz mono
        (resample upstream if needed -- keep resampling OUT of the detector
        so the streaming layer controls chunk boundaries precisely).
      - Must be safe to call repeatedly on short (e.g. 250ms-1s) chunks;
        if the underlying model expects a minimum length, pad/handle it
        internally and document the minimum in the subclass docstring.
    """

    #: Decision threshold in score-space. Tune this on a DEV split only
    #: (never on the eval/test split -- see README threshold-selection note).
    threshold: float = 0.5

    @abstractmethod
    def score(self, audio: np.ndarray, sample_rate: int = 16000) -> DetectionResult:
        """Score a single chunk/utterance. Must measure and return latency_ms."""
        raise NotImplementedError

    def score_batch(self, chunks: list[np.ndarray], sample_rate: int = 16000) -> list[DetectionResult]:
        """Default: score sequentially. Override if the backend supports true batching."""
        return [self.score(c, sample_rate) for c in chunks]
