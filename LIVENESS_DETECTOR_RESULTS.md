# Liveness Detector — First Real-World Result

## What this is
A side experiment answering a different question from the rest of this
project: not "is this voice AI-generated" but "did this audio come
directly from a mouth, or did it bounce through a speaker first" — the
mechanism behind smart speakers (Alexa/Siri/Google Home) falsely
triggering from a TV ad or video. Standalone tool: `liveness_detector.py`.

## Setup
- 8 "live" recordings: real phrases spoken into an iPhone's mic (Voice
  Memos), covering distinct assistant-style commands ("Hey Siri, place an
  order for...", "Alexa, unlock the front door...", etc.) plus 3 unrelated
  phrases from an earlier batch.
- 11 "playback" recordings: each live clip replayed out loud through the
  Mac's built-in speakers and re-captured live through the Mac's built-in
  mic (`afplay` + simultaneous recording, real acoustic round-trip, not
  simulated).
- Features: MFCC stats, spectral centroid/rolloff/flatness, zero-crossing
  rate, RMS energy, high-frequency energy ratio (>4kHz), and a noise-floor
  estimate — see `liveness_detector.py`'s `extract_features()` docstring
  for why each one is a plausible playback-vs-live cue.
- Classifier: logistic regression on standardized features, evaluated with
  leave-one-out cross-validation (appropriate given the small N).

## Result
**100% leave-one-out accuracy, 19/19 recordings.**

| | P(playback) range |
|---|---|
| Live samples (n=8) | 0.004 - 0.073 |
| Playback samples (n=11) | 0.967 - 0.993 |

Every held-out prediction landed far from the 0.5 decision boundary —
clean separation, not a lucky borderline call.

## How to read this honestly
- This is real data (an actual speaker-to-mic round trip on one Mac, one
  room), not a toy/simulated result -- a genuinely encouraging first pass.
- **But n=19 from a single room, single device pair, single speaker's
  voice is a small, narrow sample.** It does NOT yet show this
  generalizes to: other rooms (different echo/reverb), other
  speaker/mic hardware (a TV vs. a phone vs. a smart speaker's own mic),
  other voices, or background noise. The features chosen (speaker
  frequency rolloff, noise floor, etc.) are real, literature-grounded
  replay-detection cues, but 100% accuracy on 19 same-room samples is
  exactly the situation where a classifier can be right for a narrower
  reason than intended (e.g. it may partly be picking up on this specific
  room's echo signature, not "playback" in general).
- Treat this as: "the core idea has real signal in it, worth pursuing
  further" -- not "solved." That distinction matters if this goes in
  front of a mentor.

## Natural next steps (not yet done)
- Record in a different room / with a different speaker (phone vs. laptop
  vs. an actual smart-speaker-style driver) to test whether accuracy holds
  up outside this one setup.
- Get a second person's voice into the "live" set.
- Try genuinely different playback sources (TV, YouTube video, a phone
  call) rather than just the same Mac replaying its own recording.

## Reproduce it
```
python liveness_detector.py train --data-dir recordings
```
Data lives in `recordings/live/*.m4a` and `recordings/playback/*.wav`.
