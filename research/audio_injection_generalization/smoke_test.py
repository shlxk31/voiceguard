"""
smoke_test.py — VoiceSHIELD-Small load + inference smoke test

PURPOSE
-------
Verify that VoiceSHIELD-Small can be loaded from the Hugging Face Hub and run
inference on actual audio. This is a SMOKE TEST only: it confirms the model
loads and produces a label/confidence/transcript. It is NOT a valid research
evaluation, and no results from it should be cited as research findings.

REQUIREMENTS
------------
- torch, torchaudio (already in this project's venv)
- transformers, safetensors, accelerate (NOT yet installed — see FEASIBILITY_REPORT.md §6)
- AudioAgentSecurity audio downloaded locally (requires manual HF agreement — see FEASIBILITY_REPORT.md §2.1)

USAGE
-----
    .venv/bin/python research/audio_injection_generalization/smoke_test.py

The script tries three audio sources in order:
1. Any WAV files found under research/audio_injection_generalization/audio/ (place downloaded AudioAgentSecurity samples here).
2. A short synthetic sine-wave clip (to confirm the model runs at all).
3. Exits with instructions if neither is available and transformers is missing.

NO MODEL TRAINING, NO DATASET DOWNLOAD, NO CLAIMED RESULTS.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

AUDIO_DIR = PROJECT_ROOT / "research" / "audio_injection_generalization" / "audio"


def check_dependencies() -> list[str]:
    """Return list of missing packages."""
    missing = []
    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")
    try:
        import safetensors  # noqa: F401
    except ImportError:
        missing.append("safetensors")
    try:
        import accelerate  # noqa: F401
    except ImportError:
        missing.append("accelerate")
    return missing


def find_audio_files() -> list[Path]:
    """Look for user-placed AudioAgentSecurity WAV files."""
    if not AUDIO_DIR.exists():
        return []
    return sorted(AUDIO_DIR.glob("*.wav")) + sorted(AUDIO_DIR.glob("*.WAV"))


def make_synthetic_clip() -> tuple:
    """Create a short synthetic sine wave (16 kHz mono) so the model can be exercised."""
    import numpy as np
    import torch

    sr = 16000
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # 440 Hz tone (A4) — not speech, but exercises the pipeline
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return torch.from_numpy(audio), sr


def main() -> int:
    missing = check_dependencies()
    if missing:
        print("SMOKE TEST BLOCKED — missing Python packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print()
        print("Install them with:")
        print("    .venv/bin/pip install transformers safetensors accelerate")
        print()
        print("Then re-run this smoke test.")
        return 1

    import torch
    from transformers import pipeline

    model_id = "Emvo-ai/voiceSHIELD-small"
    print(f"Loading VoiceSHIELD-Small from {model_id} ...")
    print("(requires trust_remote_code=True — executes code from Hugging Face Hub)")
    try:
        pipe = pipeline(
            "audio-classification",
            model=model_id,
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"FAILED to load model: {exc}")
        print()
        print("Possible causes:")
        print("  - No network access to Hugging Face Hub")
        print("  - transformers/safetensors/accelerate not installed correctly")
        print("  - Model requires GPU and none is available")
        return 1
    print("Model loaded successfully.")

    audio_files = find_audio_files()
    if audio_files:
        print(f"\nFound {len(audio_files)} audio file(s) under {AUDIO_DIR}:")
        for f in audio_files[:5]:
            print(f"  - {f.name}")
        print("(Testing on first 3 files — smoke test only, not research evaluation.)")
        for f in audio_files[:3]:
            try:
                result = pipe(str(f))
                print(f"\n  {f.name}:")
                print(f"    label={result.get('label')} "
                      f"confidence={result.get('score', result.get('confidence'))}")
                if "transcript" in result:
                    print(f"    transcript={result['transcript']!r}")
            except Exception as exc:
                print(f"\n  {f.name}: inference failed — {exc}")
    else:
        print(f"\nNo audio files found under {AUDIO_DIR}.")
        print("To test on real AudioAgentSecurity samples:")
        print(f"  1. Manually accept access at https://huggingface.co/datasets/Limax11/AudioAgentSecurity")
        print(f"  2. Download benchmark_dataset_mixed/ to {AUDIO_DIR}")
        print("  3. Re-run this smoke test.")
        print()
        print("Running on a synthetic sine-wave clip instead (pipeline smoke test only):")
        audio, sr = make_synthetic_clip()
        try:
            result = pipe({"array": audio.numpy(), "sampling_rate": sr})
            print(f"  label={result.get('label')} "
                  f"confidence={result.get('score', result.get('confidence'))}")
            if "transcript" in result:
                print(f"  transcript={result['transcript']!r}")
            print("(Sine wave is not speech — any label is expected and not meaningful.)")
        except Exception as exc:
            print(f"  Inference on synthetic clip failed: {exc}")
            return 1

    print("\nSMOKE TEST COMPLETE — model loads and runs. No research results claimed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
