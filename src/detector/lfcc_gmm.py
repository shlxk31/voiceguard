"""
detector/lfcc_gmm.py

Classical baseline: LFCC features + two GMMs (one for bonafide, one for spoof),
score = log-likelihood ratio. This is the ASVspoof-2017/2019-baseline-style
approach (Todisco/Delgado/Evans lineage).

WHY THIS FIRST (before AASIST-L):
  - Zero GPU dependency, trains/runs in seconds -> lets you validate the whole
    pipeline (data loading -> VoIP degradation -> streaming -> FastAPI) before
    touching a heavier model.
  - Legitimate citable baseline for your results table / ablation.

TODO (Week 1, Day 1-2):
  1. pip install -r requirements.txt
  2. Implement `extract_lfcc()` using librosa or torchaudio.compliance.kaldi
     (20 static + delta + delta-delta LFCCs is the standard ASVspoof recipe).
  3. Fit `bonafide_gmm` and `spoof_gmm` (sklearn.mixture.GaussianMixture,
     n_components=512 is the ASVspoof2019 baseline default, but start with
     64-128 components for speed while iterating) on ASVspoof19 LA train split.
  4. Pick `threshold` on the DEV split by sweeping and taking the EER point.
     Freeze it. Never touch it again after looking at eval/test results.
"""

import time
import numpy as np
import torch
import torchaudio
from sklearn.mixture import GaussianMixture

from .base import SpoofDetector, DetectionResult

# ASVspoof2019-baseline-style framing: 25ms window / 10ms hop at 16kHz.
_N_FFT = 512
_WIN_LENGTH = 400  # 25ms @ 16kHz
_HOP_LENGTH = 160  # 10ms @ 16kHz


def extract_lfcc(audio: np.ndarray, sample_rate: int = 16000, n_lfcc: int = 20) -> np.ndarray:
    """
    Extract LFCC + delta + delta-delta features, mean-pooled over time.

    Returns a (3 * n_lfcc,) float32 vector: [static | delta | delta-delta],
    each mean-pooled across frames.

    NOTE: mean-pooling loses temporal structure -- fine for a GMM baseline,
    but this is exactly why the CNN/AASIST-L step later matters: it can use
    the full sequence, not just its average.
    """
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)  # (1, T)

    lfcc_transform = torchaudio.transforms.LFCC(
        sample_rate=sample_rate,
        n_filter=128,
        n_lfcc=n_lfcc,
        dct_type=2,
        norm="ortho",
        log_lf=False,
        speckwargs={
            "n_fft": _N_FFT,
            "win_length": _WIN_LENGTH,
            "hop_length": _HOP_LENGTH,
            "center": True,
        },
    )
    feats = lfcc_transform(waveform)  # (1, n_lfcc, n_frames)
    deltas = torchaudio.functional.compute_deltas(feats)
    delta2 = torchaudio.functional.compute_deltas(deltas)

    stacked = torch.cat([feats, deltas, delta2], dim=1)  # (1, 3*n_lfcc, n_frames)
    pooled = stacked.mean(dim=-1).squeeze(0)  # (3*n_lfcc,)
    return pooled.numpy().astype(np.float32)


class LFCCGMMDetector(SpoofDetector):
    def __init__(self, n_components: int = 128):
        self.bonafide_gmm = GaussianMixture(n_components=n_components, covariance_type="diag")
        self.spoof_gmm = GaussianMixture(n_components=n_components, covariance_type="diag")
        self._fitted = False
        self.threshold = 0.0  # log-likelihood-ratio threshold, set via dev-set sweep

    def fit(self, bonafide_feats: np.ndarray, spoof_feats: np.ndarray) -> None:
        self.bonafide_gmm.fit(bonafide_feats)
        self.spoof_gmm.fit(spoof_feats)
        self._fitted = True

    def spoof_scores(self, feats: np.ndarray) -> np.ndarray:
        """
        Batch LLR-based spoof score for already-extracted features, shape
        (n_samples, n_features) -> (n_samples,). Higher = more spoof-like.
        Used directly by evaluation scripts to avoid re-extracting features
        and to vectorize scoring across a whole split.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before scoring, or load pretrained GMMs.")
        llr = self.bonafide_gmm.score_samples(feats) - self.spoof_gmm.score_samples(feats)
        # Higher llr = more bonafide-like, so flip sign to match "higher = spoofier"
        return -llr

    def score(self, audio: np.ndarray, sample_rate: int = 16000) -> DetectionResult:
        t0 = time.perf_counter()
        feats = extract_lfcc(audio, sample_rate).reshape(1, -1)
        spoof_score = float(self.spoof_scores(feats)[0])
        latency_ms = (time.perf_counter() - t0) * 1000
        is_spoof = spoof_score > self.threshold
        confidence = min(abs(spoof_score - self.threshold) / (abs(self.threshold) + 1e-6), 1.0)
        return DetectionResult(score=spoof_score, is_spoof=is_spoof,
                                confidence=confidence, latency_ms=latency_ms)
