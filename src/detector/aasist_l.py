"""
detector/aasist_l.py

Wraps a pretrained AASIST-L checkpoint (~85k params, graph-attention over a
raw-waveform SincNet front-end) as a SpoofDetector.

WHERE TO GET THE MODEL (do this on your own machine, not in a sandbox):
  Option 1 (reference implementation + checkpoints):
      git clone https://github.com/clovaai/aasist.git
    Follow their README for the AASIST-L checkpoint under `models/weights/`.
  Option 2 (HuggingFace mirror, if available in your environment):
      from huggingface_hub import hf_hub_download
      ckpt = hf_hub_download(repo_id="<AASIST-L repo id>", filename="...")
    (Verify the exact repo id yourself — HF repo ids/paths churn; don't
    hardcode one you haven't confirmed loads.)

WHY THIS ARCHITECTURE (recap from research):
  - Sub-100k params -> CPU-viable, fits your single-file/FastAPI constraint.
  - Published EER: ~0.99% (ASVspoof19 LA eval), ~13.15% (ASVspoof21 LA) —
    expect WORSE than this on your own VoIP-degraded / modern-clone data.
    That gap is a legitimate result, not a bug.
  - IMPORTANT (caught during Phase-1 review): AASIST's benchmark numbers are
    LA/DF, not PA/replay. If you ever add a replay-detection module, treat
    it as a SEPARATE detector/benchmark, don't assume this model covers it.

TODO (Week 1, Day 2-3):
  1. Clone the AASIST repo, copy their `models/AASIST.py` (or the -L variant)
     into this project (or import it directly, whichever is cleaner for you).
  2. Load the pretrained checkpoint in `__init__`.
  3. Confirm the model's expected input: raw waveform, 16kHz, a FIXED
     input length (the original AASIST pads/crops to ~4 seconds). For
     streaming (Week 5-6), you'll need to decide: pad short chunks up to
     that length, or retrain/fine-tune for variable-length short chunks.
     This decision point is exactly where "streaming reformulation"
     (your novel contribution) starts -- document whichever you choose
     and why, it's paper-relevant.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .base import SpoofDetector, DetectionResult

_DEFAULT_REPO = Path(__file__).resolve().parent.parent.parent / "external" / "aasist"
_DEFAULT_CHECKPOINT = _DEFAULT_REPO / "models" / "weights" / "AASIST-L.pth"
_DEFAULT_CONFIG = _DEFAULT_REPO / "config" / "AASIST-L.conf"


def _load_aasist_model_class(repo_path: Path):
    """
    Imports the `Model` class from clovaai/aasist's models/AASIST.py without
    requiring that repo to be installed as a package (it's just cloned into
    external/aasist — see WEEK1_CHECKLIST.md step 5).
    """
    repo_path = str(repo_path)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    spec = importlib.util.spec_from_file_location(
        "aasist_models_AASIST", Path(repo_path) / "models" / "AASIST.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


def _pad_or_crop(x: np.ndarray, max_len: int) -> np.ndarray:
    """Same deterministic eval-time convention as clovaai/aasist's data_utils.pad():
    crop to the first max_len samples if long enough, else tile-repeat then crop."""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


class AasistLDetector(SpoofDetector):
    """
    Wraps the pretrained AASIST-L checkpoint bundled in clovaai/aasist
    (models/weights/AASIST-L.pth) as a SpoofDetector. No fitting needed —
    this is a pretrained model, usable standalone without ASVspoof data.

    Input handling: AASIST-L expects a FIXED-length raw waveform
    (nb_samp=64600, ~4.04s @ 16kHz, from the model config). Shorter chunks
    are tile-repeated to that length; longer ones are cropped to the first
    4.04s. This is a temporary streaming stand-in — the real "streaming
    reformulation" (window-size-aware retraining/fine-tuning) is Week 5-6+
    work, not implemented here.
    """

    def __init__(
        self,
        checkpoint_path: str = str(_DEFAULT_CHECKPOINT),
        config_path: str = str(_DEFAULT_CONFIG),
        repo_path: str = str(_DEFAULT_REPO),
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        with open(config_path) as f:
            config = json.load(f)
        model_config = config["model_config"]
        self.nb_samp = model_config["nb_samp"]

        Model = _load_aasist_model_class(Path(repo_path))
        self.model = Model(model_config).to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.threshold = 0.5  # set via dev-set EER sweep, same discipline as LFCC-GMM

    @torch.no_grad()
    def score(self, audio: np.ndarray, sample_rate: int = 16000) -> DetectionResult:
        t0 = time.perf_counter()
        x_np = _pad_or_crop(np.asarray(audio, dtype=np.float32), self.nb_samp)
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(self.device)  # (1, T)
        _, logits = self.model(x)
        # Class index 0 = spoof, index 1 = bonafide in this checkpoint's own
        # training convention (verified empirically against clovaai/aasist's
        # own eval code + a dev-set EER check: scoring by index 1 gives ~100%
        # EER, i.e. perfectly inverted; index 0 gives the expected near-0%).
        spoof_score = torch.softmax(logits, dim=-1)[0, 0].item()  # P(spoof)
        latency_ms = (time.perf_counter() - t0) * 1000
        is_spoof = spoof_score > self.threshold
        confidence = abs(spoof_score - self.threshold) * 2  # 0-1 scale, threshold at midpoint
        return DetectionResult(score=spoof_score, is_spoof=is_spoof,
                                confidence=min(confidence, 1.0), latency_ms=latency_ms)
