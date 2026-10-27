"""
liveness_detector.py

Standalone research prototype: is a short speech clip LIVE (spoken directly
into a microphone) or PLAYBACK (played through a speaker and re-captured)?

WHY THIS IS A DIFFERENT PROBLEM FROM THE REST OF THIS REPO
The detectors elsewhere in this project (src/detector/) tell "real human
speech" apart from "AI-synthesized speech." This script asks a different
question: did the sound come directly out of a mouth, or did it bounce
through a speaker first? That's exactly why a smart speaker sometimes
falsely triggers on its wake word from a TV ad or a YouTube video -- there
is currently no fully reliable software-only way to tell the two apart.
This is an honest, small research prototype to explore the idea, not a
production-grade fix -- expect it to be imperfect, and treat a poor result
as a real finding to report, not a bug to hide.

WHAT SIGNAL IT LOOKS FOR
Playback (speaker -> air -> mic) adds real, if imperfect, acoustic
artifacts vs. a direct human-to-mic path: consumer speakers roll off
high-frequency content, the extra room round-trip raises the background
noise floor and adds echo/reverb coloration, and the signal has already
been through at least one prior recording+playback chain. The features
below (spectral rolloff, high-frequency energy ratio, spectral flatness,
noise floor, plus generic MFCC/ZCR/energy stats) are standard descriptors
from the replay-attack detection literature, used here as a research probe
rather than an established production defense.

USAGE

  1. List input devices (find your microphone's index if the default is wrong):
       python liveness_detector.py list-devices

  2. Record training samples (repeat several times per label -- 8-10+ each
     is a reasonable minimum for a first pass):
       python liveness_detector.py record --label live     --out recordings/live/sample1.wav
       python liveness_detector.py record --label playback --out recordings/playback/sample1.wav
     For "live" samples: speak a short phrase directly at the mic.
     For "playback" samples: play one of your recorded live clips out loud
     through a speaker (phone, laptop, external speaker -- any real
     playback path) with the mic recording nearby, so it re-captures the
     speaker's output rather than your direct voice.

  3. Train + evaluate a classifier on what you recorded:
       python liveness_detector.py train --data-dir recordings --model model.pkl
     Uses leave-one-out cross-validation (appropriate for the small
     datasets this workflow produces) and prints per-file results so you
     can see exactly which recordings the classifier got wrong.

  4. Classify a new clip:
       python liveness_detector.py classify --model model.pkl --file new_clip.wav

This file is deliberately self-contained (no imports from src/) so it can
be handed to someone else as a single script.
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
FRAME_LENGTH = 1024
HOP_LENGTH = 256


def list_devices():
    import sounddevice as sd
    print(sd.query_devices())


def record(out_path: Path, seconds: float, device: int | None):
    import sounddevice as sd

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Recording in 3...", end=" ", flush=True)
    time.sleep(1)
    print("2...", end=" ", flush=True)
    time.sleep(1)
    print("1...", flush=True)
    time.sleep(1)
    print(f"RECORDING for {seconds:.1f}s -- speak/play now")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1,
                    dtype="float32", device=device)
    sd.wait()
    print("Done.")
    audio = audio.reshape(-1)
    sf.write(out_path, audio, SAMPLE_RATE)
    print(f"Saved to {out_path} ({len(audio) / SAMPLE_RATE:.2f}s)")


def _frame_signal(audio: np.ndarray) -> np.ndarray:
    """Split into overlapping frames, shape (n_frames, FRAME_LENGTH)."""
    n_frames = 1 + max(0, (len(audio) - FRAME_LENGTH) // HOP_LENGTH)
    if n_frames <= 0:
        audio = np.pad(audio, (0, FRAME_LENGTH - len(audio)))
        n_frames = 1
    frames = np.stack([
        audio[i * HOP_LENGTH: i * HOP_LENGTH + FRAME_LENGTH]
        if i * HOP_LENGTH + FRAME_LENGTH <= len(audio)
        else np.pad(audio[i * HOP_LENGTH:], (0, FRAME_LENGTH - (len(audio) - i * HOP_LENGTH)))
        for i in range(n_frames)
    ])
    return frames


def _mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    import librosa
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                 n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)


def extract_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Self-contained feature extractor (no cross-file imports, per the
    "single file" design goal). Returns a fixed-length vector combining
    replay/liveness-relevant descriptors with generic spectral stats.
    """
    import librosa

    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 1e-8:
        audio = audio / np.max(np.abs(audio))  # normalize level -- we care about
        # spectral SHAPE and noise floor RELATIVE to signal, not absolute loudness,
        # which otherwise just reflects how close you held the mic.

    mfcc = _mfcc(audio, sr)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=FRAME_LENGTH,
                                                  hop_length=HOP_LENGTH)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=FRAME_LENGTH,
                                                hop_length=HOP_LENGTH, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=FRAME_LENGTH,
                                                  hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=FRAME_LENGTH,
                                              hop_length=HOP_LENGTH)[0]
    rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]

    # High-frequency energy ratio: energy above 4kHz / total energy. Consumer
    # speakers commonly roll off well before a direct-to-mic human voice would.
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio)) if len(audio) > 1 else audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    total_energy = np.sum(spec ** 2) + 1e-12
    hf_energy = np.sum(spec[freqs > 4000] ** 2)
    hf_ratio = hf_energy / total_energy

    # Noise floor: 10th-percentile frame RMS, a proxy for background hiss/room
    # noise -- typically higher for a re-captured (speaker-to-mic) signal.
    noise_floor = np.percentile(rms, 10)

    feats = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        [centroid.mean(), centroid.std()],
        [rolloff.mean(), rolloff.std()],
        [flatness.mean(), flatness.std()],
        [zcr.mean(), zcr.std()],
        [rms.mean(), rms.std()],
        [hf_ratio, noise_floor],
    ])
    return feats.astype(np.float32)


FEATURE_NAMES = (
    [f"mfcc{i}_mean" for i in range(13)] + [f"mfcc{i}_std" for i in range(13)] +
    ["centroid_mean", "centroid_std", "rolloff_mean", "rolloff_std",
     "flatness_mean", "flatness_std", "zcr_mean", "zcr_std", "rms_mean", "rms_std",
     "hf_energy_ratio", "noise_floor"]
)


AUDIO_EXTENSIONS = (".wav", ".m4a", ".mp3", ".caf", ".aiff", ".flac", ".ogg")


def load_audio_any(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Decodes ANY common audio file (wav, and also iPhone Voice Memos' .m4a,
    plus mp3/caf/aiff/etc.) via PyAV, resampled to mono target_sr. This
    means you can drop iPhone recordings straight into recordings/live/ or
    recordings/playback/ with no manual conversion step.
    """
    import av

    container = av.open(str(path))
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
    chunks = []
    for frame in container.decode(audio=0):
        for rframe in resampler.resample(frame):
            chunks.append(rframe.to_ndarray().reshape(-1))
    container.close()
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)


def _load_dataset(data_dir: Path):
    files, labels = [], []
    for label in ("live", "playback"):
        label_dir = data_dir / label
        if not label_dir.exists():
            continue
        for audio_path in sorted(label_dir.iterdir()):
            if audio_path.suffix.lower() in AUDIO_EXTENSIONS:
                files.append(audio_path)
                labels.append(label)
    return files, labels


def train(data_dir: Path, model_path: Path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler

    files, labels = _load_dataset(data_dir)
    n_live = labels.count("live")
    n_playback = labels.count("playback")
    print(f"Found {n_live} 'live' and {n_playback} 'playback' recordings under {data_dir}")
    if n_live < 3 or n_playback < 3:
        print("WARNING: fewer than 3 samples in one class -- results below are not "
              "statistically meaningful, just a smoke test. Record more before trusting them.")
    if len(files) < 2:
        raise SystemExit("Need at least 2 recordings total (both classes) to train anything.")

    X = np.stack([extract_features(load_audio_any(f), SAMPLE_RATE) for f in files])
    y = np.array([1 if l == "playback" else 0 for l in labels])  # 1 = playback

    loo = LeaveOneOut()
    preds = np.zeros(len(y), dtype=int)
    probs = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X[train_idx]), y[train_idx])
        probs[test_idx] = clf.predict_proba(scaler.transform(X[test_idx]))[:, 1]
        preds[test_idx] = clf.predict(scaler.transform(X[test_idx]))

    accuracy = float(np.mean(preds == y))
    print(f"\nLeave-one-out cross-validated accuracy: {accuracy * 100:.1f}% "
          f"({len(y)} recordings)")
    print("\nPer-file results (label, predicted, P(playback)):")
    for f, true_label, pred, prob in zip(files, labels, preds, probs):
        pred_label = "playback" if pred == 1 else "live"
        flag = "  <-- MISCLASSIFIED" if pred_label != true_label else ""
        print(f"  {f.name:30s} true={true_label:8s} pred={pred_label:8s} "
              f"P(playback)={prob:.3f}{flag}")

    # Fit the final model on ALL data for actual use via `classify`.
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X), y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as fh:
        pickle.dump({"scaler": scaler, "clf": clf, "loo_accuracy": accuracy,
                     "n_live": n_live, "n_playback": n_playback}, fh)
    print(f"\nFinal model (fit on all {len(y)} recordings) saved to {model_path}")


def classify(model_path: Path, file_path: Path):
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    audio = load_audio_any(file_path)
    feats = extract_features(audio, SAMPLE_RATE).reshape(1, -1)
    prob_playback = model["clf"].predict_proba(model["scaler"].transform(feats))[0, 1]
    label = "playback" if prob_playback > 0.5 else "live"
    print(f"{file_path.name}: {label}  (P(playback)={prob_playback:.3f})")
    print(f"[model's own leave-one-out accuracy on its training set: "
          f"{model['loo_accuracy'] * 100:.1f}% over {model['n_live']} live + "
          f"{model['n_playback']} playback samples -- treat this prediction's "
          f"reliability accordingly]")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-devices")

    p_record = sub.add_parser("record")
    p_record.add_argument("--label", choices=["live", "playback"], required=True)
    p_record.add_argument("--out", type=Path, required=True)
    p_record.add_argument("--seconds", type=float, default=4.0)
    p_record.add_argument("--device", type=int, default=None)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data-dir", type=Path, default=Path("recordings"))
    p_train.add_argument("--model", type=Path, default=Path("liveness_model.pkl"))

    p_classify = sub.add_parser("classify")
    p_classify.add_argument("--model", type=Path, default=Path("liveness_model.pkl"))
    p_classify.add_argument("--file", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "list-devices":
        list_devices()
    elif args.command == "record":
        record(args.out, args.seconds, args.device)
    elif args.command == "train":
        train(args.data_dir, args.model)
    elif args.command == "classify":
        classify(args.model, args.file)


if __name__ == "__main__":
    main()
