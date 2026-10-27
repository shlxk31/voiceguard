"""
nlu/understand.py

Transcribes the spoken command and extracts the transaction amount from it.

WHY THIS EXISTS
The transaction amount must come from what the user actually SAID, not from a
slider the demo operator moves by hand. In a real voice-ordering system there
is no slider: the user says "order the headphones for four thousand rupees",
and the system has to work out that this is a Rs.4,000 transaction before it
can decide how much verification that transaction deserves.

This closes the loop in the architecture:

    speech --> transcript --> amount --> risk threshold tau(A) --> decision
                     |
                     +--> (also shown to the user, so the system's
                           understanding of the command is auditable)

SECURITY NOTE WORTH STATING IN THE PAPER
Deriving the amount from speech does not hand the attacker a lever. The amount
sets the *strictness* of verification: a higher amount raises the evidence bar.
An attacker who understates the amount to obtain a lower threshold thereby
also limits themselves to a small transaction, which is exactly the incentive
structure we want. The amount is used for risk scaling only -- it is never
treated as authentication evidence.

ASR: faster-whisper (CTranslate2 Whisper) running locally. No API key, no
network call, so the demo works offline and no audio leaves the machine --
which also matters for the voice-privacy concerns raised in the PVA
literature.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field

import numpy as np

SAMPLE_RATE = 16000

# --------------------------------------------------------------------------
# Number-word parsing, including Indian numbering (lakh / crore).
# --------------------------------------------------------------------------
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {
    "hundred": 100, "thousand": 1_000, "lakh": 100_000, "lac": 100_000,
    "lakhs": 100_000, "million": 1_000_000, "crore": 10_000_000,
    "crores": 10_000_000, "billion": 1_000_000_000,
}
_CURRENCY_HINTS = ("rupee", "rupees", "rs", "inr", "dollar", "dollars", "usd", "₹", "$")

# A number followed by one of these is a SPECIFICATION, not a price:
# "Galaxy Fold 8 Ultra", "MacBook M3 16 inch", "512 GB iPhone 15 Pro Max".
# This was not hypothetical -- all three of these were logged as real
# transactions, and "Samsung Galaxy Fold 8 Ultra" was authorised as Rs.8.
_UNIT_BLOCK = {
    "inch", "inches", "gb", "tb", "mb", "kb", "ghz", "mhz", "hz", "mp",
    "megapixel", "watt", "watts", "w", "mah", "kg", "kgs", "gram", "grams",
    "ml", "litre", "litres", "liter", "liters", "percent", "pct", "degree",
    "degrees", "am", "pm", "oclock", "kmph", "km", "cm", "mm", "px", "bit",
    # product-tier words that follow a model number
    "pro", "ultra", "max", "plus", "mini", "air", "lite", "series", "gen",
    "generation", "version", "edition", "model",
}

# A number preceded (within two tokens) by one of these is being priced.
_PRICE_CONTEXT = {
    "for", "of", "at", "worth", "costs", "cost", "costing", "price", "priced",
    "around", "about", "approximately", "roughly", "under", "over",
    "pay", "paying", "transfer", "transferring", "send", "sending",
    "spend", "spending", "settle", "recharge", "add", "charge", "bill",
}


def _words_to_number(tokens: list[str]) -> float | None:
    """Convert a run of number words ('fifty thousand') into a value."""
    if not tokens:
        return None
    total, current, seen = 0.0, 0.0, False
    for tok in tokens:
        if tok in _UNITS:
            current += _UNITS[tok]; seen = True
        elif tok in _TENS:
            current += _TENS[tok]; seen = True
        elif tok in _SCALES:
            scale = _SCALES[tok]
            if current == 0:
                current = 1.0
            if scale >= 1000:
                total += current * scale
                current = 0.0
            else:
                current *= scale
            seen = True
        elif tok in ("and", "-"):
            continue
        else:
            break
    return (total + current) if seen else None


@dataclass
class AmountCandidate:
    value: float
    source_text: str
    method: str          # "digits" | "words"
    has_currency_cue: bool
    #: Empty when the candidate was accepted; otherwise why it was discarded.
    rejected_because: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _token_window(tokens: list[str], i: int, j: int) -> tuple[list[str], str | None]:
    """Tokens in the two positions before span [i, j), and the token after."""
    before = tokens[max(0, i - 2): i]
    after = tokens[j] if j < len(tokens) else None
    return before, after


def extract_amount(transcript: str) -> tuple[float | None, list[AmountCandidate]]:
    """
    Find the transaction amount in a transcript.

    Returns (chosen_amount, all_candidates).

    THE RULE: A BARE NUMBER IS NEVER AN AMOUNT.
    An earlier version took the largest number in the sentence whenever no
    currency word appeared. That is unsafe in both directions, and it failed
    in practice: "buy me a Samsung Galaxy Fold 8 Ultra" was authorised as an
    Rs.8 transaction, which set the LOWEST possible verification bar on a
    request whose real value was unknown. Silently under-estimating the stake
    is the worst failure mode this module has, because tau(A) is monotonic in
    the amount.

    A candidate is now accepted only with positive evidence that it is a
    price -- an adjacent currency cue, a monetary scale word (lakh/crore), or
    a pricing/payment word within two tokens before it -- and is rejected
    outright if a unit or product-tier word follows it. When nothing
    qualifies the amount is None, and the caller falls back rather than
    inventing a number.
    """
    text = transcript.lower().replace(",", "")
    candidates: list[AmountCandidate] = []
    rejected: list[AmountCandidate] = []
    all_tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", text)

    # --- digit forms: "4000", "Rs 4000", "4000 rupees", "5k", "2.5 lakh" ---
    for m in re.finditer(
        r"(?:(₹|\$|rs\.?|inr|usd)\s*)?"
        r"(\d+(?:\.\d+)?)"
        r"\s*(k|thousand|lakh|lakhs|lac|crore|crores|million|billion)?"
        r"\s*(rupees|rupee|dollars|dollar|rs\.?|inr|usd)?",
        text,
    ):
        pre_cur, num, scale_word, post_cur = m.groups()
        if not num:
            continue
        value = float(num)
        if scale_word:
            value *= 1_000 if scale_word == "k" else _SCALES.get(scale_word, 1)
        cue = bool(pre_cur or post_cur or scale_word in ("lakh", "lakhs", "lac", "crore", "crores"))
        span = m.group(0).strip()
        if value <= 0:
            continue

        # Locate this number in the token stream to inspect its neighbours.
        num_tok = num
        try:
            pos = next(k for k, t in enumerate(all_tokens) if t == num_tok)
        except StopIteration:
            pos = -1
        before, after = _token_window(all_tokens, pos, pos + 1) if pos >= 0 else ([], None)
        # A scale word consumed by the regex is not the blocking "after".
        if after in ("k", "thousand", "lakh", "lakhs", "lac", "crore",
                     "crores", "million", "billion"):
            _, after = _token_window(all_tokens, pos, pos + 2)

        blocked_by_unit = after in _UNIT_BLOCK
        priced = cue or any(b in _PRICE_CONTEXT for b in before)
        cand = AmountCandidate(value, span, "digits", cue)
        if blocked_by_unit or not priced:
            cand.rejected_because = ("followed by a unit or model word "
                                     f"({after!r})" if blocked_by_unit
                                     else "no currency or pricing word nearby")
            rejected.append(cand)
        else:
            candidates.append(cand)

    # --- word forms: "fifty thousand rupees" ---
    tokens = re.findall(r"[a-z]+", text)
    i = 0
    while i < len(tokens):
        if tokens[i] in _UNITS or tokens[i] in _TENS or tokens[i] in _SCALES:
            j = i
            run: list[str] = []
            while j < len(tokens) and (
                tokens[j] in _UNITS or tokens[j] in _TENS
                or tokens[j] in _SCALES or tokens[j] in ("and",)
            ):
                run.append(tokens[j]); j += 1
            value = _words_to_number(run)
            if value and value > 0:
                nearby = " ".join(tokens[max(0, i - 3): min(len(tokens), j + 3)])
                cue = any(c in nearby for c in _CURRENCY_HINTS)
                after = tokens[j] if j < len(tokens) else None
                before = tokens[max(0, i - 2): i]
                priced = cue or any(b in _PRICE_CONTEXT for b in before)
                cand = AmountCandidate(value, " ".join(run), "words", cue)
                if after in _UNIT_BLOCK or not priced:
                    cand.rejected_because = ("followed by a unit or model word"
                                             if after in _UNIT_BLOCK
                                             else "no currency or pricing word nearby")
                    rejected.append(cand)
                else:
                    candidates.append(cand)
            i = j
        else:
            i += 1

    if not candidates:
        # Rejected candidates are still returned so the UI can show WHY a
        # number in the transcript was not treated as money.
        return None, rejected

    with_cue = [c for c in candidates if c.has_currency_cue]
    pool = with_cue or candidates
    chosen = max(pool, key=lambda c: c.value)
    return chosen.value, candidates + rejected


@dataclass
class Understanding:
    """What the system believes was said and requested."""

    transcript: str = ""
    language: str = ""
    amount: float | None = None
    amount_source: str = ""
    candidates: list[AmountCandidate] = field(default_factory=list)
    asr_available: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "transcript": self.transcript,
            "language": self.language,
            "amount": self.amount,
            "amount_source": self.amount_source,
            "candidates": [c.to_dict() for c in self.candidates],
            "asr_available": self.asr_available,
            "note": self.note,
        }


class CommandUnderstanding:
    """Local Whisper transcription plus amount extraction."""

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device=self.device,
                                        compute_type="int8")
        return self._model

    def understand(self, audio: np.ndarray) -> Understanding:
        try:
            model = self._load()
            segments, info = model.transcribe(
                audio.astype(np.float32), beam_size=1, language="en",
                vad_filter=False,
            )
            transcript = " ".join(s.text for s in segments).strip()
        except Exception as exc:  # ASR must never break authorization
            return Understanding(
                asr_available=False,
                note=f"transcription unavailable ({type(exc).__name__}); "
                     f"falling back to the manually entered amount",
            )

        amount, candidates = extract_amount(transcript)
        chosen = next((c for c in candidates if c.value == amount), None)
        return Understanding(
            transcript=transcript,
            language=getattr(info, "language", "") or "",
            amount=amount,
            amount_source=(chosen.source_text if chosen else ""),
            candidates=candidates,
            note="" if amount is not None
                 else "no transaction amount detected in the command",
        )
