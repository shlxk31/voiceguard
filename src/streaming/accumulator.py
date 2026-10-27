"""
streaming/accumulator.py

THE NOVEL PART OF THIS PROJECT.

Turns a stream of per-chunk DetectionResults into a single "raise the alarm
now" decision, and — critically — measures HOW LONG into the call it took
to reach that decision (time-to-detection), traded off against latency and
false-alarm rate.

Two things every result in your paper will reference:
  1. Per-chunk EER at each window size (250ms / 500ms / 1s) under each
     VoIP degradation setting.
  2. Time-to-detection using this accumulator, at a fixed false-alarm rate.

TODO (Week 7-8):
  1. Tune `window_size` (N) and `required_votes` (M) on a DEV split by
     sweeping and plotting time-to-detection vs. false-alarm rate (an
     ROC-like curve, but on "call-level" outcomes, not chunk-level).
  2. Add an EWMA (exponentially-weighted moving average) variant as a
     comparison ablation against the simple N-of-M vote — reviewers will
     ask "why this rule and not a smoothed score," so having both in your
     results table pre-empts that.
"""

from collections import deque
from dataclasses import dataclass, field

from ..detector.base import DetectionResult


@dataclass
class AccumulatorConfig:
    window_size: int = 10          # look at the last N chunk decisions
    required_votes: int = 6         # ...need at least M of them flagged spoof
    min_confidence: float = 0.3     # ignore low-confidence chunks entirely


@dataclass
class StreamState:
    chunk_index: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    alarm_raised: bool = False
    alarm_at_chunk: int | None = None


class CallAccumulator:
    """
    Feed it one DetectionResult per chunk (in order) as a call progresses.
    It tells you whether/when to raise a mid-call fraud alarm.
    """

    def __init__(self, config: AccumulatorConfig, chunk_duration_ms: float):
        self.config = config
        self.chunk_duration_ms = chunk_duration_ms  # needed to convert chunk index -> time-to-detection
        self.state = StreamState(history=deque(maxlen=config.window_size))

    def update(self, result: DetectionResult) -> bool:
        """Returns True the FIRST time the alarm condition is met (call once per chunk)."""
        self.state.chunk_index += 1
        if result.confidence >= self.config.min_confidence:
            self.state.history.append(result.is_spoof)

        if self.state.alarm_raised:
            return False  # already raised, don't re-trigger

        votes = sum(self.state.history)
        if len(self.state.history) == self.config.window_size and votes >= self.config.required_votes:
            self.state.alarm_raised = True
            self.state.alarm_at_chunk = self.state.chunk_index
            return True
        return False

    @property
    def time_to_detection_ms(self) -> float | None:
        if self.state.alarm_at_chunk is None:
            return None
        return self.state.alarm_at_chunk * self.chunk_duration_ms
