"""
assistant/passphrase.py

Verifies the enrolled challenge phrase spoken during a CHALLENGE step.

WHY THIS EXISTS
The assistant was telling users "please repeat your passphrase" while the
system had no passphrase at all -- the challenge endpoint simply re-scored
whatever audio arrived. The words promised a capability the code did not
have, which is worse than not offering it: it invites the user to believe a
check happened.

WHAT IT ADDS, AND WHAT IT DOES NOT
A fixed enrolled phrase makes the challenge TEXT-DEPENDENT. The second
utterance must be the right words in the right voice, so an attacker who can
mimic or synthesise the voice must also know the phrase.

Its limitation must be stated plainly in the paper: because the phrase is
fixed, ANY recording of the user saying it satisfies the text check. It
raises the bar against an attacker who lacks the phrase; it does nothing
against one who has a recording of it. A system-chosen random phrase would
close that gap, since no pre-recorded clip can answer an unpredictable
prompt, and that is the natural next step.

MATCHING IS FUZZY BY NECESSITY
ASR does not return the same string twice ("open sesame" / "open sesame."
/ "open, sesame"). We normalise aggressively and compare with a token-level
similarity, accepting above a threshold. The threshold is a usability choice,
not a security parameter -- the security comes from the voice layers; the
phrase is a second factor whose job is to be hard to guess, not hard to
pronounce.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher

#: Token-similarity above which the spoken phrase counts as a match.
MATCH_THRESHOLD = 0.75

_FILLER = {"um", "uh", "er", "the", "a", "an", "please", "okay", "ok"}


def normalise(text: str) -> str:
    """Lowercase, strip punctuation and filler, collapse whitespace."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    kept = [w for w in words if w not in _FILLER]
    return " ".join(kept or words)


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalise(a), normalise(b)).ratio()


@dataclass
class PassphraseCheck:
    enrolled: bool
    matched: bool
    score: float
    heard: str
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def verify(enrolled_phrase: str, heard_text: str) -> PassphraseCheck:
    """Compare what was said against the enrolled phrase."""
    if not enrolled_phrase:
        return PassphraseCheck(
            enrolled=False, matched=False, score=0.0, heard=heard_text,
            note="no passphrase enrolled for this user; "
                 "the challenge falls back to voice re-verification only")
    if not heard_text.strip():
        return PassphraseCheck(enrolled=True, matched=False, score=0.0,
                               heard="", note="nothing was transcribed")
    score = similarity(enrolled_phrase, heard_text)
    ok = score >= MATCH_THRESHOLD
    return PassphraseCheck(
        enrolled=True, matched=ok, score=score, heard=heard_text,
        note="phrase matched" if ok else
             f"phrase did not match (similarity {score:.2f} < {MATCH_THRESHOLD})")
