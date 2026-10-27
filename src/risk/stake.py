"""
risk/stake.py

Generalises the risk threshold from "how many rupees" to "how much is at
stake".

THE PROBLEM THIS FIXES
policy.py derives tau(A) = log((A + C_h)/C) + logit(pi), where A is the
transaction amount. That is sound whenever the loss IS an amount. But a voice
assistant is asked for many damaging things that carry no rupee figure:

    "unlock the front door"
    "forward the OTP that just arrived"
    "delete all the backups"
    "add this person to my account"
    "borrow my friend's card and buy it"

Under an amount-only policy every one of these is A = 0, which produces the
LOWEST threshold the system can issue -- the cheapest possible verification
for some of the most consequential requests it will ever receive. The
monotonicity of tau(A) turns silence about money into permissiveness.

THE FIX, WHICH DOES NOT DISTURB THE DERIVATION
The Bayes rule needs C_fa: the cost of wrongly approving. A rupee amount is
one way to estimate C_fa, not the definition of it. So we keep the derivation
exactly as it is and broaden the estimate:

    stake = max( monetary amount , consequence of the action itself )

Both terms are in rupees, so tau(stake) is the same formula with a better
input. Nothing in policy.py changes and the Odyssey 2018 citation still
covers it.

WHERE THE NUMBERS COME FROM, AND WHAT THEY ARE NOT
The anchors below are DECLARED POLICY CONSTANTS, in the same sense as
CostModel.handling_cost: they are business judgements about how bad an
outcome is, stated openly so a reviewer can disagree with a specific number
rather than with an opaque score. They are not measured or learned, and the
paper must present them as assumptions. What IS defensible is the structure:
that consequence is monotonic in severity and irreversibility, and that an
unrecognised action must not default to zero.

THE LLM'S ROLE IS CLASSIFICATION ONLY
The category is chosen by the language model from a fixed enum and validated
against it. The model never emits a rupee figure and never sees the anchor
table -- if it could name its own stake, an attacker who controls the
transcript would control the threshold. It selects a bucket; policy prices
the bucket.

ATTACKER INCENTIVES (the same argument as for the spoken amount)
Understating the stake to obtain a lower bar also limits the attacker to a
low-stake action, because the action actually performed is the one that was
described. The estimate is used for risk scaling only; it is never treated
as authentication evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

#: Consequence anchors in rupees: the assumed cost of wrongly approving an
#: action of this kind, EXCLUDING any money it moves (that is counted
#: separately and the larger of the two wins).
#:
#: UNKNOWN is deliberately not the cheapest bucket. A request the classifier
#: could not place is a request we do not understand, and the safe reading of
#: "I don't know what this does" is not "it must be harmless".
#: NOTE on device_control vs physical_access. These were one category, and
#: it priced "turn off the lights" at Rs.50,000 -- the same as unlocking the
#: front door -- so a light switch demanded tau = 3.24, more evidence than a
#: Rs.10,000 transfer. The classifier was not wrong; the enum was too coarse.
#: Toggling an appliance is trivially reversible and harms nothing; defeating
#: a lock or an alarm is a security event. They are now separate.
SEVERITY_ANCHORS: dict[str, tuple[float, str]] = {
    "informational":  (0.0,      "answering a question; no state changes"),
    "device_control": (100.0,    "toggles an appliance (lights, fan, TV); trivially reversible"),
    "reversible_low": (200.0,    "trivially undone (play music, add to cart)"),
    "purchase":       (1_000.0,  "buys something; refundable with effort"),
    "account_change": (10_000.0, "alters the account (address, linked phone, new payee)"),
    "data_loss":      (25_000.0, "deletes or overwrites data"),
    "credential":     (50_000.0, "discloses an OTP, password, or card detail"),
    "physical_access":(50_000.0, "defeats a lock or alarm (doors, safes, security systems)"),
    "unknown":        (5_000.0,  "could not be classified; treated as non-trivial"),
}

#: Irreversibility multiplier. An action that cannot be undone costs more when
#: it is wrong, because the mitigation ("just reverse it") is unavailable.
IRREVERSIBLE_MULTIPLIER = 2.0

#: An explicitly unlawful or unethical request (theft, coercion, impersonating
#: the account holder) is not merely expensive -- it is one the system should
#: essentially never approve on voice evidence alone. Rather than special-case
#: it in the policy, we price it high enough that tau(stake) demands evidence
#: no single utterance can supply.
ILLEGITIMATE_STAKE = 500_000.0

VALID_CATEGORIES = tuple(SEVERITY_ANCHORS)


#: Bands used to describe the stake to a user. The rupee figure is what the
#: threshold is actually computed from, but quoting it for a non-monetary
#: action reads as invented precision -- "turning off the lights costs Rs.100"
#: is not a claim anyone should make. The band is the honest summary; the
#: figure stays available for the audit trail and for anyone checking the
#: arithmetic.
RISK_BANDS = ((500.0, "Low"), (5_000.0, "Moderate"),
              (50_000.0, "High"), (float("inf"), "Critical"))


@dataclass
class StakeEstimate:
    """The stake used for tau(), with every term that produced it."""

    stake: float
    monetary: float
    consequence: float
    category: str
    irreversible: bool
    illegitimate: bool
    basis: str
    estimated_price: float = 0.0
    price_item: str = ""

    @property
    def level(self) -> str:
        """Plain-language risk band, for display instead of a bare number."""
        if self.stake <= 0:
            return "Minimal"
        for ceiling, name in RISK_BANDS:
            if self.stake < ceiling:
                return name
        return "Critical"

    @property
    def is_monetary(self) -> bool:
        """True when a rupee figure is meaningful to show the user."""
        return self.monetary > 0 or self.estimated_price > 0

    @property
    def headline(self) -> str:
        """One line describing what is at stake, without invented precision."""
        if self.monetary > 0:
            return f"Rs.{self.monetary:,.0f}"
        if self.estimated_price > 0:
            return f"~Rs.{self.estimated_price:,.0f} ({self.price_item})"
        return SEVERITY_ANCHORS.get(self.category, (0, "unclassified"))[1]

    def explain(self) -> list[str]:
        lines = [
            f"Spoken amount              = Rs.{self.monetary:,.0f}",
            f"Action category            = {self.category} "
            f"({SEVERITY_ANCHORS.get(self.category, (0, 'n/a'))[1]})",
            f"Consequence of that action = Rs.{self.consequence:,.0f}"
            + (f"  (x{IRREVERSIBLE_MULTIPLIER:g}, irreversible)"
               if self.irreversible else ""),
        ]
        if self.estimated_price > 0:
            lines.append(
                f"Estimated price of '{self.price_item}' = "
                f"Rs.{self.estimated_price:,.0f}  (model knowledge, not a live lookup)")
        if self.illegitimate:
            lines.append(
                f"Request flagged unlawful/unethical -> stake floored at "
                f"Rs.{ILLEGITIMATE_STAKE:,.0f}")
        lines.append(
            f"Stake A = max(amount, consequence, price) = Rs.{self.stake:,.0f}  "
            f"-> this is what tau(A) is computed from")
        return lines

    def to_dict(self) -> dict:
        d = asdict(self)
        d["explain"] = self.explain()
        d["level"] = self.level
        d["is_monetary"] = self.is_monetary
        d["headline"] = self.headline
        return d


def estimate_stake(amount: float | None, category: str = "unknown",
                   *, irreversible: bool = False,
                   illegitimate: bool = False,
                   estimated_price: float = 0.0,
                   price_item: str = "") -> StakeEstimate:
    """
    Combine the spoken amount, the consequence of the action, and (for
    purchases) the estimated price of the item named.

    `category` must be one of VALID_CATEGORIES; anything else is coerced to
    "unknown", which is priced as non-trivial rather than free.

    `estimated_price` comes from risk/pricing.py and enters through a MAXIMUM,
    so an underestimate is absorbed by the category anchor and cannot lower
    the bar. See that module for why that property matters.
    """
    if category not in SEVERITY_ANCHORS:
        category = "unknown"

    consequence = SEVERITY_ANCHORS[category][0]
    if irreversible:
        consequence *= IRREVERSIBLE_MULTIPLIER

    monetary = float(amount or 0.0)
    priced = max(0.0, float(estimated_price or 0.0))
    stake = max(monetary, consequence, priced)

    if illegitimate:
        stake = max(stake, ILLEGITIMATE_STAKE)

    if illegitimate:
        basis = "request judged unlawful or unethical"
    elif stake == monetary and monetary > 0:
        basis = "spoken transaction amount"
    elif stake == priced and priced > 0:
        basis = f"estimated price of {price_item or 'the item'}"
    elif consequence > 0:
        basis = f"consequence of a '{category}' action"
    else:
        basis = "no money and no state change"

    return StakeEstimate(stake=stake, monetary=monetary, consequence=consequence,
                         category=category, irreversible=irreversible,
                         illegitimate=illegitimate, basis=basis,
                         estimated_price=priced, price_item=price_item)
