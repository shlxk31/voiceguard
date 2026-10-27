"""
voip_sim/degrade.py

Simulates what happens to audio when it travels over a real VoIP call:
  1. Lossy codec encode/decode (Opus — what WhatsApp/Zoom/Discord/Teams use)
  2. Packet loss using a Gilbert-Elliott two-state Markov model (captures
     the "losses come in bursts" reality of real networks, not just
     independent random drops)
  3. (optional, Week 3-4 stretch) jitter buffer simulation

WHY THIS FILE MATTERS FOR YOUR PAPER:
  This IS your experimental control knob. Every result in your latency/EER
  tables will be reported AS A FUNCTION OF the settings here (bitrate,
  loss rate). Keep every run's exact parameters logged (see `DegradationConfig`)
  so your results are reproducible.

TODO (Week 3-4):
  1. Implement `encode_decode_opus()` using PyAV (`av` package) or by
     shelling out to ffmpeg (ffmpeg -c:a libopus -b:a <bitrate> ...).
  2. Implement `apply_packet_loss()`: chop audio into ~20ms frames (Opus's
     native frame size), drop frames per the Gilbert-Elliott state machine,
     then either zero-fill or apply simple packet-loss-concealment (PLC)
     interpolation for the dropped frames -- report BOTH, since real VoIP
     stacks use PLC and "zero-fill" is an overly pessimistic lower bound.
  3. Validate: run bonafide-only audio through this at increasing loss rates
     and confirm your baseline detector's EER degrades in the expected
     direction (sanity check the harness before trusting any result on it).
"""

import io
from dataclasses import dataclass

import av
import numpy as np


@dataclass
class DegradationConfig:
    codec: str = "opus"
    bitrate_kbps: int = 16          # try 8/16/32/64 kbps to span the realistic range
    packet_loss_rate: float = 0.0   # fraction of 20ms frames lost, e.g. 0.05 = 5%
    burst_factor: float = 0.5       # Gilbert-Elliott burstiness (0=random, 1=very bursty)
    frame_ms: int = 20              # Opus's native frame size
    use_plc: bool = True            # simple concealment vs. hard zero-fill on loss


def encode_decode_opus(audio: np.ndarray, sample_rate: int, config: DegradationConfig) -> np.ndarray:
    """
    Round-trip audio through Opus at the configured bitrate via PyAV.

    Opus's encoder introduces a small algorithmic pre-skip/delay, so the
    decoded signal won't be exactly `len(audio)` samples (empirically ~1ms
    short at 16kHz). We crop/zero-pad the tail to match the input length
    exactly so downstream code (packet loss framing, detector scoring) can
    assume a fixed length -- a documented simplification, not lossless
    alignment of the actual encoder delay.
    """
    assert sample_rate in (8000, 12000, 16000, 24000, 48000), (
        f"Opus requires one of its native rates (8/12/16/24/48kHz), got {sample_rate}Hz"
    )

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="ogg")
    stream = container.add_stream("libopus", rate=sample_rate)
    stream.bit_rate = config.bitrate_kbps * 1000

    frame = av.AudioFrame.from_ndarray(
        audio.reshape(1, -1).astype(np.float32), format="fltp", layout="mono"
    )
    frame.sample_rate = sample_rate
    frame.pts = 0
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):  # flush
        container.mux(packet)
    container.close()

    buf.seek(0)
    container = av.open(buf, mode="r")
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
    out_chunks = []
    for frame in container.decode(audio=0):
        for rframe in resampler.resample(frame):
            out_chunks.append(rframe.to_ndarray().reshape(-1))
    container.close()

    out = np.concatenate(out_chunks) if out_chunks else np.zeros(0, dtype=np.float32)
    out = out.astype(np.float32)

    # Align length to the input exactly (see docstring above).
    if len(out) < len(audio):
        out = np.pad(out, (0, len(audio) - len(out)))
    else:
        out = out[:len(audio)]
    return out


def apply_packet_loss(audio: np.ndarray, sample_rate: int, config: DegradationConfig) -> np.ndarray:
    """Drop/interpolate frames according to a 2-state (good/bad) Markov chain."""
    frame_len = int(sample_rate * config.frame_ms / 1000)
    n_frames = len(audio) // frame_len
    # Gilbert-Elliott: P(stay in bad state) grows with burst_factor
    p_good_to_bad = config.packet_loss_rate * (1 - config.burst_factor) + 1e-6
    p_bad_to_good = 1 - config.burst_factor + 1e-6
    state = "good"
    out = audio.copy()
    for i in range(n_frames):
        if state == "good" and np.random.rand() < p_good_to_bad:
            state = "bad"
        elif state == "bad" and np.random.rand() < p_bad_to_good:
            state = "good"
        if state == "bad":
            start, end = i * frame_len, (i + 1) * frame_len
            if config.use_plc:
                # crude PLC: repeat the previous frame, attenuated (placeholder —
                # a real PLC uses waveform extrapolation; document this simplification
                # explicitly in the paper's limitations section)
                if i > 0:
                    out[start:end] = out[(i - 1) * frame_len:i * frame_len] * 0.7
            else:
                out[start:end] = 0.0
    return out


def degrade(audio: np.ndarray, sample_rate: int, config: DegradationConfig) -> np.ndarray:
    """Full pipeline: codec round-trip, then packet loss."""
    x = encode_decode_opus(audio, sample_rate, config)
    x = apply_packet_loss(x, sample_rate, config)
    return x
