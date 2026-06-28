"""Logistic death-probability model.

Extracted from ``Main.death_probability`` so both the cost evaluator and the
queueing shortage model share a single, tested implementation (DRY + SRP).
"""

from __future__ import annotations

import math

from ..domain.problem_data import DeathModelParameters


class DeathProbabilityModel:
    """Probability that an injured person dies after waiting time ``t``.

    Implements ``phi_min + (phi_max - phi_min) / (1 + exp(ks * (tm - t)))``.
    """

    def probability(self, params: DeathModelParameters, t: float) -> float:
        return params.phi_min + (params.phi_max - params.phi_min) / (
            1 + math.exp(params.ks * (params.tm - t))
        )
