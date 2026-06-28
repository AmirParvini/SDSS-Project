"""Immutable description of a humanitarian-logistics problem instance.

This replaces the dozens of loose ``self.*`` attributes that the original
``Main`` class carried.  Grouping them into a single, explicit value object
gives every service one well-typed dependency to depend on (Dependency
Inversion) instead of reaching into a god-object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class FacilityCapacities:
    """Capacity look-ups keyed by facility id."""

    ambulance: Dict[str, int]  # e.g. {"injured_type1": 2, "injured_type2": 4}
    helicopter: Dict[str, int]  # e.g. {"injured_type1": 4, "injured_type2": 12}
    hospital: Dict[int, float]  # hospital_id -> capacity
    tmc: Dict[int, float]  # temporary medical centre id -> capacity
    shelter: Dict[int, float]  # shelter id -> capacity


@dataclass(frozen=True)
class Distances:
    """Pre-computed distance dictionaries keyed by ``"source,target"``."""

    dc_to_shelter: Dict[str, float]
    da_to_ec: Dict[str, float]
    da_to_h: Dict[str, float]
    da_to_h_helicopter: Dict[str, float]
    da_to_tmc: Dict[str, float]
    da_to_tmc_helicopter: Dict[str, float]


@dataclass(frozen=True)
class DeathModelParameters:
    """Parameters of the logistic death-probability model per injury class."""

    phi_min: float
    phi_max: float
    ks: float
    tm: float


@dataclass(frozen=True)
class ProblemData:
    """All static information describing one optimisation scenario."""

    # Entity id lists
    dc_id: List[int]
    ec_id: List[int]
    da_id: List[int]
    h_id: List[int]
    tmc_id: List[int]

    # Demographics (keyed by damage-area id)
    affected_pop: Dict[int, float]
    severe_injured: Dict[int, float]
    minor_injured: Dict[int, float]
    homeless: Dict[int, float]

    # Costs, distances, capacities
    cost: Dict[str, float]
    distance: Distances
    capacity: FacilityCapacities

    # Vehicle speeds
    ambulance_speed: float
    helicopter_speed: float

    # Death-probability models
    severe_death_model: DeathModelParameters
    moderate_death_model: DeathModelParameters

    # Queue / treatment parameters
    injured_treated_same_time: int
    waiting_time: float  # `wt` in the original code

    # Budget constraint
    budget: float  # `B` in the original code

    # Convenience dimensions -------------------------------------------------
    @property
    def n_shelters(self) -> int:
        return len(self.ec_id)

    @property
    def n_distribution(self) -> int:
        return len(self.dc_id)

    @property
    def n_damage_points(self) -> int:
        return len(self.da_id)

    @property
    def n_hospitals(self) -> int:
        return len(self.h_id)

    @property
    def n_temp_medical(self) -> int:
        return len(self.tmc_id)
