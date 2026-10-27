"""
risk/policy.py

Derives the decision threshold from cost, instead of picking it by hand.

THE QUESTION THIS ANSWERS
"Why is the threshold 0.75? Why does a Rs.10,000 order need trust >= 85?"
If the answer is "it felt about right", the system is not defensible. This
module replaces those hand-picked numbers with a derivation.

THE PRINCIPLE (Bayes risk minimisation)
A detector should minimise EXPECTED COST, not maximise accuracy. The standard
formulation in speaker verification is the Detection Cost Function:

    E[cost] = C_miss * P_miss * P_target  +  C_fa * P_fa * (1 - P_target)

Minimising this yields the Bayes-optimal rule: accept when the calibrated
log-likelihood ratio exceeds

    tau = log[ C_fa / C_miss ] + log[ P(attack) / P(genuine) ]

Reference: T. Kinnunen, K. A. Lee, H. Delgado, N. Evans, M. Todisco,
M. Sahidullah, J. Yamagishi, D. A. Reynolds, "t-DCF: a Detection Cost
Function for the Tandem Assessment of Spoofing Countermeasures and Automatic
Speaker Verification", Odyssey 2018 (arXiv:1804.09618). ASVspoof adopted this
as its primary metric precisely because EER alone ignores operational cost.

WHY THE THRESHOLD MUST MOVE WITH THE TRANSACTION AMOUNT
The two error costs in a payment system are asymmetric and only one of them
scales:

  * C_fa  -- cost of approving a fraudulent transaction. This SCALES WITH THE
    AMOUNT: approving a fraudulent Rs.50,000 order costs 100x approving a
    fraudulent Rs.500 order. We model C_fa(A) = A + C_handling.

  * C_miss -- cost of wrongly rejecting a genuine user. This is roughly
    CONSTANT: a support call, some friction, some goodwill. It does not
    depend on the amount.

Substituting C_fa(A) into the Bayes rule gives an amount-dependent threshold:

    tau(A) = log( (A + C_handling) / C_reject ) + log( pi / (1 - pi) )

The threshold therefore rises LOGARITHMICALLY with the amount. That is a
derived curve, not three hand-picked tiers -- and tiers, if you still want
them for the UI, can be produced by quantising this curve rather than by
intuition.

THREE OUTCOMES, NOT TWO
Binary allow/deny throws away the most useful response to uncertainty: ask
again. We therefore use two boundaries derived from two different costs:

  * Wrongly CHALLENGING a genuine user is cheap (mild friction) -> C_challenge
  * Wrongly DENYING a genuine user is expensive (lost customer)  -> C_reject

    tau_allow(A)  = log( (A + C_h) / C_challenge ) + logit(pi)
    tau_deny(A)   = log( (A + C_h) / C_reject )    + logit(pi)

Because C_reject > C_challenge, tau_deny < tau_allow, and the band between
them is where the system asks for re-verification. Both boundaries come from
the same formula with a different cost -- nothing is invented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict


@dataclass
class CostModel:
    """
    The operational costs that determine the thresholds. These are the ONLY
    tunable numbers in the decision policy, and each is a business quantity
    with a real-world meaning -- not an opaque score cutoff.

    All monetary values in the same currency unit (INR here).
    """

    #: Fixed cost of handling a fraud incident beyond the amount lost
    #: (investigation, chargeback processing, reputational cost).
    handling_cost: float = 500.0

    #: Cost of wrongly challenging a genuine user: a few seconds of friction.
    challenge_cost: float = 20.0

    #: Cost of wrongly denying a genuine user outright: support contact,
    #: abandoned transaction, goodwill. Must exceed challenge_cost.
    reject_cost: float = 400.0

    #: Prior probability that any given request is an attack. Deliberately
    #: conservative; a real deployment would estimate this from fraud logs.
    attack_prior: float = 0.01

    def __post_init__(self):
        if not (0.0 < self.attack_prior < 1.0):
            raise ValueError("attack_prior must lie strictly between 0 and 1")
        if self.challenge_cost >= self.reject_cost:
            raise ValueError(
                "challenge_cost must be less than reject_cost -- challenging a "
                "genuine user is by definition cheaper than denying them"
            )

    @property
    def prior_log_odds(self) -> float:
        """log[ P(attack) / P(genuine) ] -- the prior term of the Bayes rule."""
        return math.log(self.attack_prior / (1.0 - self.attack_prior))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ThresholdDerivation:
    """Every term of tau(A), kept so the UI and audit log can show the working."""

    amount: float
    handling_cost: float
    fraud_cost: float          # A + C_handling
    challenge_cost: float
    reject_cost: float
    prior_log_odds: float
    tau_allow: float
    tau_deny: float

    def explain(self) -> list[str]:
        """Human-readable derivation, line by line, for display."""
        return [
            # "Stake", not "transaction amount": A is the cost of wrongly
            # approving, which for a non-monetary action has no rupee figure
            # in the request at all. Calling it a transaction amount implied
            # the user had said one.
            f"Stake A = Rs.{self.amount:,.0f}",
            f"Cost of approving it wrongly  C_fa = A + Rs.{self.handling_cost:,.0f} "
            f"= Rs.{self.fraud_cost:,.0f}",
            f"Cost of wrongly challenging a genuine user = Rs.{self.challenge_cost:,.0f}",
            f"Cost of wrongly denying a genuine user     = Rs.{self.reject_cost:,.0f}",
            f"Prior log-odds of attack = {self.prior_log_odds:.2f}",
            f"tau_allow = log({self.fraud_cost:,.0f} / {self.challenge_cost:,.0f}) "
            f"+ ({self.prior_log_odds:.2f}) = {self.tau_allow:.2f}",
            f"tau_deny  = log({self.fraud_cost:,.0f} / {self.reject_cost:,.0f}) "
            f"+ ({self.prior_log_odds:.2f}) = {self.tau_deny:.2f}",
        ]

    def to_dict(self) -> dict:
        return asdict(self)


class RiskPolicy:
    """Amount-dependent Bayes thresholds and the three-way decision."""

    ALLOW = "ALLOW"
    CHALLENGE = "CHALLENGE"
    DENY = "DENY"

    def __init__(self, costs: CostModel | None = None):
        self.costs = costs or CostModel()

    def thresholds(self, amount: float) -> ThresholdDerivation:
        c = self.costs
        fraud_cost = amount + c.handling_cost
        return ThresholdDerivation(
            amount=float(amount),
            handling_cost=c.handling_cost,
            fraud_cost=fraud_cost,
            challenge_cost=c.challenge_cost,
            reject_cost=c.reject_cost,
            prior_log_odds=c.prior_log_odds,
            tau_allow=math.log(fraud_cost / c.challenge_cost) + c.prior_log_odds,
            tau_deny=math.log(fraud_cost / c.reject_cost) + c.prior_log_odds,
        )

    def decide(self, fused_llr: float, amount: float) -> tuple[str, ThresholdDerivation]:
        t = self.thresholds(amount)
        if fused_llr >= t.tau_allow:
            return self.ALLOW, t
        if fused_llr >= t.tau_deny:
            return self.CHALLENGE, t
        return self.DENY, t

    def curve(self, amounts: list[float]) -> list[dict]:
        """tau(A) sampled over a range -- used to plot the threshold curve."""
        out = []
        for a in amounts:
            t = self.thresholds(a)
            out.append({"amount": a, "tau_allow": t.tau_allow, "tau_deny": t.tau_deny})
        return out
