"""
tests/eval_common.py

Shared helpers for the ASVspoof2019 LA evaluation scripts (run_lfcc_baseline.py,
run_aasist_baseline.py, build_degraded_eval_set.py, eval_degraded.py). Keeping
these in one place means every script subsamples/loads data identically, which
matters for fair clean-vs-degraded and detector-vs-detector comparisons.
"""

import random
from pathlib import Path

import numpy as np
import soundfile as sf

PROTOCOL_NAMES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}


def load_protocol(data_root: Path, split: str) -> list[tuple[str, str]]:
    """Returns [(utt_id, label), ...] with label in {"bonafide", "spoof"}."""
    proto_path = data_root / "ASVspoof2019_LA_cm_protocols" / PROTOCOL_NAMES[split]
    entries = []
    with open(proto_path) as f:
        for line in f:
            fields = line.strip().split()
            # SPEAKER_ID AUDIO_FILE_NAME SYSTEM_ID- KEY  (columns vary slightly
            # across LA protocol files, but utt id is field[1] and label is field[-1])
            utt_id, label = fields[1], fields[-1]
            entries.append((utt_id, label))
    return entries


def load_audio(data_root: Path, split: str, utt_id: str) -> np.ndarray:
    flac_path = data_root / f"ASVspoof2019_LA_{split}" / "flac" / f"{utt_id}.flac"
    audio, sr = sf.read(flac_path, dtype="float32")
    assert sr == 16000, f"expected 16kHz, got {sr}Hz for {flac_path}"
    return audio


def split_by_label(items: np.ndarray, labels: np.ndarray):
    bonafide = items[labels == "bonafide"]
    spoof = items[labels == "spoof"]
    return bonafide, spoof


def stratified_subsample(
    entries: list[tuple[str, str]],
    n_bonafide: int | None,
    n_spoof: int | None,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """
    Deterministically subsample a protocol's entries, class-balanced.

    Using a fixed seed everywhere means two different scripts (e.g. clean
    vs. degraded eval) reference the exact same utt_ids, which is required
    for a fair comparison -- especially since apply_packet_loss() uses
    unseeded randomness internally, so the *files on disk* must already be
    identical across conditions before that randomness is applied.

    n_bonafide/n_spoof of None means "take all of that class".
    """
    bonafide = [e for e in entries if e[1] == "bonafide"]
    spoof = [e for e in entries if e[1] == "spoof"]
    rng = random.Random(seed)
    rng.shuffle(bonafide)
    rng.shuffle(spoof)
    if n_bonafide is not None:
        bonafide = bonafide[:n_bonafide]
    if n_spoof is not None:
        spoof = spoof[:n_spoof]
    return bonafide + spoof


def save_scores_npz(path: Path, utt_ids: list[str], labels: list[str], scores: np.ndarray, **extra_arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        utt_ids=np.array(utt_ids),
        labels=np.array(labels),
        scores=np.asarray(scores),
        **extra_arrays,
    )


def load_scores_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}
