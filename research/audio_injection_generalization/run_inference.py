"""
run_inference.py — VoiceSHIELD-Small audio inference (zero-shot smoke test)

PURPOSE
-------
Loads VoiceSHIELD-Small (Whisper-small encoder + classifier head) from the
Hugging Face Hub and runs inference on audio files. This is a PRELIMINARY
SMOKE TEST only — it confirms the model loads and produces predictions. It is
NOT a valid research evaluation, and no results should be cited as findings.

NO MODEL TRAINING. NO DATASET DOWNLOAD. NO CLAIMED RESULTS.

USAGE
-----
    .venv/bin/python research/audio_injection_generalization/run_inference.py <audio_file> [audio_file ...]

If no audio file is provided, runs on a synthetic sine-wave clip (execution
test only — any label is expected and not meaningful).

Outputs to stdout:
  - model identity (VoiceSHIELD-Small, Emvo-ai/voiceSHIELD-small)
  - per-file: predicted label, confidence, P(malicious), transcript
  - execution details: device, load time, inference time

Results are saved to:
  research/audio_injection_generalization/inference_results.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# 1. Load Whisper-small encoder from local HF cache
# ---------------------------------------------------------------------------
HF_CACHE = Path(os.path.expanduser("~/.cache/huggingface/hub"))
WHISPER_SNAPSHOT = HF_CACHE / "models--openai--whisper-small/snapshots"
VOICESHIELD_SNAPSHOT = HF_CACHE / "models--Emvo-ai--voiceSHIELD-small/snapshots"


def _find_snapshot(cache_dir: Path) -> Path:
    """Return the first snapshot subdirectory in a model cache dir."""
    snaps = sorted(cache_dir.glob("*"))
    if not snaps:
        raise FileNotFoundError(f"No snapshot found in {cache_dir}")
    return snaps[0]


def load_whisper_encoder() -> tuple:
    """Load Whisper-small encoder weights from the HF cache."""
    import glob

    snap_dirs = list(WHISPER_SNAPSHOT.glob("*"))
    if not snap_dirs:
        raise FileNotFoundError(
            f"Whisper-small snapshot not found in {WHISPER_SNAPSHOT}. "
            "Run: .venv/bin/python -c \"from huggingface_hub import hf_hub_download; "
            "hf_hub_download(repo_id='openai/whisper-small', filename='pytorch_model.bin')\""
        )
    whisper_snap = snap_dirs[0]

    # Find the model weights file
    weight_files = list(whisper_snap.glob("*.bin")) + list(whisper_snap.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No model weights found in {whisper_snap}")
    weight_file = weight_files[0]
    print(f"  Whisper weights: {weight_file} ({weight_file.stat().st_size / 1e6:.1f} MB)")

    from transformers import WhisperModel
    whisper = WhisperModel.from_pretrained(
        str(whisper_snap),
        device_map=None,
        torch_dtype=torch.float32,
    )
    encoder = whisper.encoder
    encoder.eval()
    return encoder, encoder.config.d_model


def load_voiceshield_classifier(encoder: torch.nn.Module, d_model: int) -> torch.nn.Module:
    """Build VoiceSHIELD classifier head and load pretrained weights."""
    import glob

    snap_dirs = list(VOICESHIELD_SNAPSHOT.glob("*"))
    if not snap_dirs:
        raise FileNotFoundError(
            f"VoiceSHIELD-Small snapshot not found in {VOICESHIELD_SNAPSHOT}"
        )
    vs_snap = snap_dirs[0]

    weight_files = list(vs_snap.glob("*.safetensors")) + list(vs_snap.glob("*.bin"))
    if not weight_files:
        raise FileNotFoundError(f"No VoiceSHIELD weights found in {vs_snap}")
    weight_file = weight_files[0]
    print(f"  VoiceSHIELD weights: {weight_file} ({weight_file.stat().st_size / 1e6:.1f} MB)")

    from safetensors.torch import load_file
    state_dict = load_file(str(weight_file))

    # Build classifier head matching the VoiceSHIELD architecture
    classifier = torch.nn.Sequential(
        torch.nn.Linear(d_model, 512),
        torch.nn.GELU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 128),
        torch.nn.GELU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(128, 2),
    )

    # Load only classifier.* weights
    classifier_weights = {
        k.replace("classifier.", ""): v
        for k, v in state_dict.items()
        if k.startswith("classifier.")
    }
    missing, unexpected = classifier.load_state_dict(classifier_weights, strict=True)
    if missing:
        print(f"  Warning: missing keys: {missing}")
    if unexpected:
        print(f"  Warning: unexpected keys: {unexpected}")

    classifier.eval()
    return classifier


def predict(audio: np.ndarray, sample_rate: int,
            encoder: torch.nn.Module,
            classifier: torch.nn.Module,
            feature_extractor) -> dict:
    """Run inference on one audio clip. Returns label, confidence, P(malicious)."""
    # Use Whisper feature extractor for proper mel + padding (30s, 3000 frames)
    inputs = feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    input_features = inputs.input_features  # (1, 80, 3000)

    # Whisper encoder expects (batch, n_mels, time)
    with torch.no_grad():
        encoder_out = encoder(input_features)
        hidden = encoder_out.last_hidden_state  # (batch, seq, d_model)
        pooled = hidden.mean(dim=1)  # (batch, d_model)
        logits = classifier(pooled)  # (batch, 2)
        probs = torch.softmax(logits, dim=-1)[0]

    p_safe = float(probs[0])
    p_malicious = float(probs[1])
    label = "malicious" if p_malicious > p_safe else "safe"
    confidence = float(max(p_safe, p_malicious))

    return {
        "label": label,
        "confidence": confidence,
        "p_safe": p_safe,
        "p_malicious": p_malicious,
    }


def load_audio_file(path: Path) -> tuple:
    """Load audio file, return (numpy_array, sample_rate)."""
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def make_synthetic_clip() -> tuple:
    """Create a short synthetic sine wave (16 kHz mono)."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio, sr


def main() -> int:
    audio_files = [Path(f) for f in sys.argv[1:] if Path(f).exists()]
    use_synthetic = len(audio_files) == 0

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model_id = "Emvo-ai/voiceSHIELD-small"
    print(f"Model: {model_id}")
    print("Loading VoiceSHIELD-Small...")

    t0 = time.time()
    encoder, d_model = load_whisper_encoder()
    print(f"  Whisper-small encoder loaded (d_model={d_model})")

    from transformers import WhisperFeatureExtractor
    whisper_snap = sorted(WHISPER_SNAPSHOT.glob("*"))[0]
    feature_extractor = WhisperFeatureExtractor.from_pretrained(str(whisper_snap))
    print(f"  Whisper feature extractor loaded")

    classifier = load_voiceshield_classifier(encoder, d_model)
    print(f"  VoiceSHIELD classifier head loaded")
    load_time = time.time() - t0
    print(f"  Total load time: {load_time:.1f}s")

    device = next(encoder.parameters()).device
    print(f"  Device: {device}")

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    results = []

    if use_synthetic:
        print("\nNo audio files provided — running on synthetic sine-wave clip.")
        print("(Execution test only — any label is expected and NOT meaningful.)")
        audio, sr = make_synthetic_clip()
        files_to_test = [("synthetic_sine_wave.wav", audio, sr)]
    else:
        files_to_test = []
        for f in audio_files:
            audio, sr = load_audio_file(f)
            files_to_test.append((f.name, audio, sr))

    for name, audio, sr in files_to_test:
        print(f"\n  File: {name} ({len(audio) / sr:.2f}s, {sr} Hz)")
        t0 = time.time()
        result = predict(audio, sr, encoder, classifier, feature_extractor)
        inf_time = time.time() - t0
        print(f"    label={result['label']}")
        print(f"    confidence={result['confidence']:.4f}")
        print(f"    P(safe)={result['p_safe']:.4f}  P(malicious)={result['p_malicious']:.4f}")
        print(f"    inference_time={inf_time:.3f}s")

        results.append({
            "file": name,
            "duration_s": len(audio) / sr,
            "sample_rate": sr,
            **result,
            "inference_time_s": inf_time,
        })

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = PROJECT_ROOT / "research" / "audio_injection_generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "inference_results.json"

    output = {
        "model_id": model_id,
        "model_type": "VoiceSHIELD-Small",
        "architecture": "Whisper-small encoder + mean-pool + MLP classifier head",
        "license": "MIT",
        "device": str(device),
        "load_time_s": load_time,
        "synthetic_only": use_synthetic,
        "note": "PRELIMINARY SMOKE TEST — NOT a valid research evaluation. No results claimed.",
        "results": results,
    }
    out_file.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to: {out_file}")
    print("\nSMOKE TEST COMPLETE — model loads and runs. No research results claimed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
