"""
detector/metrics.py

Equal Error Rate (EER) is THE standard metric in this field — every paper
you've read in the research phase reports it. Implement it once, correctly,
here, and reuse everywhere (baseline reproduction, degradation experiments,
streaming experiments) so your numbers are internally consistent.
"""

import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> tuple[float, float]:
    """
    Standard EER computation.

    Args:
        bonafide_scores: model scores for genuine audio (higher = more spoof-like,
                          matching DetectionResult.score's convention)
        spoof_scores: model scores for spoofed/synthetic audio

    Returns:
        (eer, threshold) — eer as a fraction (0.05 = 5%), threshold is the
        score cutoff at which EER occurs (useful as a starting point for
        DetectionResult.threshold, but re-validate on YOUR dev split, not
        this one, before freezing it).
    """
    labels = np.concatenate([np.zeros(len(bonafide_scores)), np.ones(len(spoof_scores))])
    scores = np.concatenate([bonafide_scores, spoof_scores])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    # EER is where FPR and FNR cross
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(thresholds[idx])


def eer_vs_latency_table(results_by_config: dict) -> None:
    """
    Pretty-prints the core results table for your paper:
    window_size x degradation_config -> (EER, mean_latency_ms)

    TODO (Week 5-8): call this after each experiment sweep. Args should be
    { (window_ms, degradation_label): (eer, mean_latency_ms) }.
    """
    print(f"{'Window':<10}{'Degradation':<20}{'EER (%)':<10}{'Latency (ms)':<12}")
    for (window_ms, degradation), (eer, latency) in results_by_config.items():
        print(f"{window_ms:<10}{degradation:<20}{eer*100:<10.2f}{latency:<12.1f}")
