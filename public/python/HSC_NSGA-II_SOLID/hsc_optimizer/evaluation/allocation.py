"""Shelter allocation service.

Pure relocation of ``Main.allocate_population_under_capacity`` into a focused
service that depends only on :class:`ProblemData`.  Behaviour (including the
proportional split, last-bucket remainder and shortage accounting) is preserved
exactly.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List

from ..domain.problem_data import ProblemData
from ..domain.results import AllocationResult


class ShelterAllocationService:
    """Allocate displaced (homeless) population to the selected shelters."""

    def __init__(self, problem: ProblemData) -> None:
        self._problem = problem

    def allocate(self, da_ec: Dict[int, List[int]]) -> AllocationResult:
        problem = self._problem

        demand = {ec_id: 0 for ec_list in da_ec.values() for ec_id in ec_list}
        allocations: Dict[int, Dict[int, float]] = {da_id: {} for da_id in da_ec}
        ec_shortages = {ec_id: 0 for ec_list in da_ec.values() for ec_id in ec_list}

        total_shortage = 0
        total_weighted_dist = 0
        remaining_capacity = copy.deepcopy(problem.capacity.shelter)

        # Raw potential demand (ignoring capacity), used only for shortage stats.
        total_potential_demand = {
            ec_id: 0 for ec_list in da_ec.values() for ec_id in ec_list
        }
        for da_id, ec_id_list in da_ec.items():
            share = problem.homeless[f'{da_id}'] / len(ec_id_list)
            for ec_id in ec_id_list:
                total_potential_demand[ec_id] += share

        # Capacity-aware allocation.
        for da_id, ec_id_list in da_ec.items():
            homeless_count = problem.homeless[f'{da_id}']
            available_caps = [remaining_capacity[f'{v}'] for v in ec_id_list]
            total_available = sum(available_caps)

            if total_available >= homeless_count:
                remaining_pop = homeless_count
                for i, ec_id in enumerate(ec_id_list):
                    is_last = i == len(ec_id_list) - 1
                    alloc = (
                        remaining_pop
                        if is_last
                        else math.floor(homeless_count * (available_caps[i] / total_available))
                    )
                    allocations[da_id][ec_id] = alloc
                    dist = problem.distance.da_to_ec[f"{da_id},{ec_id}"]
                    remaining_capacity[f'{ec_id}'] -= alloc
                    demand[ec_id] += math.ceil(alloc / 5)
                    total_weighted_dist += alloc * dist
                    remaining_pop -= alloc
            else:
                for ec_id in ec_id_list:
                    alloc = remaining_capacity[f'{ec_id}']
                    allocations[da_id][ec_id] = alloc
                    dist = problem.distance.da_to_ec[f"{da_id},{ec_id}"]
                    demand[ec_id] += math.ceil(alloc / 5)
                    total_weighted_dist += alloc * dist
                    remaining_capacity[f'{ec_id}'] = 0
                total_shortage += homeless_count - total_available

        # Final per-shelter shortage relative to its initial capacity.
        for ec_id in ec_shortages:
            initial_cap = problem.capacity.shelter[f'{ec_id}']
            if total_potential_demand[ec_id] > initial_cap:
                ec_shortages[ec_id] = total_potential_demand[ec_id] - initial_cap

        return AllocationResult(
            demand=demand,
            total_shortage=total_shortage,
            weighted_distance=total_weighted_dist,
            allocations=allocations,
            ec_shortages=ec_shortages,
        )
