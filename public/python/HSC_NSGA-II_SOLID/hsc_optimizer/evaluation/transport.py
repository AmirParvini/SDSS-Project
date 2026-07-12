"""Casualty transport planning.

The original cost function and decoder each repeated the same ground/air
ambulance vs. helicopter arithmetic four times.  This service computes it once,
removing a large amount of duplication (DRY).
"""

from __future__ import annotations

import math

from ..domain.problem_data import ProblemData
from ..domain.results import TransportPlan


class TransportPlanner:
    """Compute ground/air flows, vehicle counts and costs for an injured flow."""

    #: capacity keys for the two injury classes
    SEVERE = "injured_type1"
    MODERATE = "injured_type2"

    def __init__(self, problem: ProblemData) -> None:
        self._problem = problem

    def plan(self, injured: float, ground_ratio: float, injured_type: str) -> TransportPlan:
        """Split ``injured`` people between ground and air transport.

        ``injured`` is the (already destination-weighted) number of casualties;
        ``ground_ratio`` is the share carried by ground ambulance.
        """
        cost = self._problem.cost
        capacity = self._problem.capacity

        ground_flow = math.ceil(injured * ground_ratio)
        air_flow = math.ceil(injured * (1 - ground_ratio))

        ground_vehicles = math.ceil(ground_flow / capacity.ambulance[injured_type])
        air_vehicles = math.ceil(air_flow / capacity.helicopter[injured_type])

        return TransportPlan(
            ground_flow=ground_flow,
            air_flow=air_flow,
            ground_vehicles=ground_vehicles,
            air_vehicles=air_vehicles,
            ground_cost=ground_vehicles * cost["ambulance"],
            air_cost=air_vehicles * cost["helicopter"],
        )

    # -- travel times --------------------------------------------------------
    # NOTE: the original code divides *both* the ground and the helicopter
    # distance by ``ambulance_speed`` (the helicopter speed is never used in the
    # travel-time computation).  That quirk is preserved deliberately so results
    # match the original model exactly.
    def ground_minutes(self, distance: float) -> float:
        return (distance / self._problem.ambulance_speed) * 60

    def air_minutes(self, helicopter_distance: float) -> float:
        return (helicopter_distance / self._problem.ambulance_speed) * 60
