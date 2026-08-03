"""Result value objects shared across services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class AllocationResult:
    """Output of allocating displaced population to shelters."""

    demand: Dict[int, float]                       # shelter_id -> relief demand
    total_shortage: float                          # unmet shelter capacity
    weighted_distance: float                       # sum(alloc * distance)
    allocations: Dict[int, Dict[int, float]]       # da_id -> {shelter_id: people}
    unsettled_population: Dict[int, float]                 # shelter_id -> potential shortage


@dataclass
class TransportPlan:
    """Ground/air transport split for one origin-destination injured flow."""

    ground_flow: int
    air_flow: int
    ground_vehicles: float
    air_vehicles: float
    ground_cost: float
    air_cost: float

    @property
    def total_cost(self) -> float:
        return self.ground_cost + self.air_cost


@dataclass
class EvaluationResult:
    """Objective values for a whole population of candidate solutions."""

    objectives: List[List[float]]            # raw [F1, F2, F3] per chromosome
    constraint_violations: List[float]       # per chromosome
    normalized_objectives: List[List[float]] # min-max normalised [F1, F2, F3]

    def as_legacy_tuple(self):
        """Match the original ``cost_function`` return signature."""
        return self.objectives, self.constraint_violations, self.normalized_objectives
