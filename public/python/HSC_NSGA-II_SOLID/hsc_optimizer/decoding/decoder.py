"""Decode Pareto-front chromosomes into human-readable solution records.

Replaces the ``Decoder`` class, which previously borrowed all of ``Main``'s
state through ``__getattr__`` (hidden, fragile coupling) and re-implemented the
cost function's traversal.  The decoder now receives an explicit
:class:`ProblemData` and reuses the *same* assignment / allocation / transport
services as the evaluator (DRY), so the two can never drift apart.

The detailed output schema and historical quirks are preserved exactly; quirks
are flagged with ``LEGACY`` comments.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from ..optimization.individual import Individual
from ..domain.chromosome import Chromosome
from ..domain.problem_data import ProblemData
from ..domain.results import TransportPlan
from ..evaluation.allocation import ShelterAllocationService
from ..evaluation.assignment import DamageShelterAssigner
from ..evaluation.transport import TransportPlanner


class SolutionDecoder:
    """Turn raw chromosomes into structured solution dictionaries."""

    def __init__(
        self,
        problem: ProblemData,
        assigner: DamageShelterAssigner,
        allocator: ShelterAllocationService,
        transport: TransportPlanner,
    ) -> None:
        self._problem = problem
        self._assigner = assigner
        self._allocator = allocator
        self._transport = transport

        self.package_flow_cost = 0
        self.package_cost = 0
        self.ground_vehicle_cost = 0
        self.air_vehicle_cost = 0
        self.shelter_establish_cost = 0
        self.tmc_establish_cost = 0
        self.total_cost = 0

    def decode(self, chromosomes: List[Chromosome], pareto_pops: List[Individual]) -> List[dict]:
        return [self._decode_one(idx+1, chrom, pareto_pops[idx]) for idx, chrom in enumerate(chromosomes)]

    # -- per-solution -------------------------------------------------------
    def _decode_one(self, solution_id: int, chromosome: Chromosome, pareto_pop: Individual) -> dict:
        from copy import deepcopy

        problem = self._problem
        hospital_cap = deepcopy(problem.capacity.hospital)
        tmc_cap = deepcopy(problem.capacity.tmc)

        da_ec_alloc, num_selected_ec = self._assigner.assign(chromosome)
        self.shelter_establish_cost = num_selected_ec * problem.cost['ec_cost']
        allocation = self._allocator.allocate(da_ec_alloc)
        demand = allocation.demand

        shelter_allocations = self._shelter_allocations(allocation.allocations)
        package_flows = self._package_flows(chromosome, demand)

        hospital_allocations: List[dict] = []
        shortage_severe: Dict[int, float] = {}
        shortage_moderate: Dict[int, float] = {}
        tmc_shortage: Dict[int, float] = {}

        self._severe_to_hospital(
            chromosome, hospital_cap, hospital_allocations, shortage_severe
        )
        self._moderate_to_hospital(
            chromosome,
            hospital_cap,
            hospital_allocations,
            shortage_severe,
            shortage_moderate,
        )
        tmc_allocations, num_tmc = self._moderate_to_tmc(chromosome, tmc_cap, tmc_shortage)
        self.tmc_establish_cost = num_tmc * problem.cost['tmc_cost']
        self.total_cost = self.package_flow_cost + self.package_cost + self.ground_vehicle_cost +\
              self.air_vehicle_cost + self.shelter_establish_cost + self.tmc_establish_cost +\
                self.total_cost
        return {
            "solution_id": solution_id,
            "F1": pareto_pop.normal_cost[0],
            "F2": pareto_pop.cost[1],
            "F3": pareto_pop.normal_cost[2],
            "package_flows": package_flows,
            "shelter_allocations": shelter_allocations,
            "hospital_allocations": hospital_allocations,
            "tmc_allocations": tmc_allocations,
            "solution_shelter_shortage": allocation.ec_shortages,
            "solution_hospital_shortage_severe": shortage_severe,
            "solution_hospital_shortage_moderate": shortage_moderate,
            "solution_tmc_shortage": tmc_shortage,
            "costs": {
                "package_flow_cost": self.package_flow_cost,
                "package_cost": self.package_cost,
                "ground_vehicle_cost": self.ground_vehicle_cost,
                "air_vehicle_cost": self.air_vehicle_cost,
                "shelter_establish_cost": self.shelter_establish_cost,
                "tmc_establish_cost": self.tmc_establish_cost,
                "total_cost": self.total_cost
            }
        }

    # -- sections -----------------------------------------------------------
    def _shelter_allocations(
        self, allocations: Dict[int, Dict[int, float]]
    ) -> List[dict]:
        records = []
        for da_id, shelter_flow in allocations.items():
            for shelter_id, flow in shelter_flow.items():
                records.append(
                    {"source_id": da_id, "target_id": shelter_id, "flow": flow}
                )
        return records

    def _package_flows(self, chromosome: Chromosome, demand) -> List[dict]:
        problem = self._problem
        records = []
        for idx, dc in enumerate(chromosome.dc_to_shelter):
            if dc <= 0:
                continue
            shelter_id = problem.ec_id[idx]
            dist = problem.distance.dc_to_shelter[f"{dc},{shelter_id}"]
            flow = math.ceil(chromosome.shelter_flow_ratio[idx] * demand[shelter_id])
            self.package_cost += flow * problem.cost['reliefpackage_cost']
            flow_cost = dist * problem.cost["reliefpackage_transportation_cost"] * flow
            self.package_flow_cost += flow_cost
            records.append(
                {
                    "source_id": dc,
                    "target_id": shelter_id,
                    "flow": flow,
                    "flow_cost": flow_cost,                }
            )
        return records

    def _severe_to_hospital(
        self, chromosome, hospital_cap, records, shortage_severe
    ) -> None:
        problem = self._problem
        for idx, raw_row in enumerate(chromosome.severe_split):
            row = np.array(raw_row) / sum(raw_row)
            severe = problem.severe_injured[f'{problem.da_id[idx]}']
            for h_idx, j in enumerate(row):
                if j <= 0:
                    continue
                hospital_id = problem.h_id[h_idx]
                self._record_shortage(
                    shortage_severe,
                    hospital_id,
                    severe * j,
                    hospital_cap,
                    key_table=shortage_severe,
                )
                plan = self._transport.plan(
                    severe * j,
                    chromosome.severe_ground_ratio[idx][h_idx],
                    TransportPlanner.SEVERE,
                )
                self.ground_vehicle_cost += plan.ground_cost
                self.air_vehicle_cost += plan.air_cost
                records.append(self._severe_record(problem.da_id[idx], hospital_id, plan))

    def _moderate_to_hospital(
        self, chromosome, hospital_cap, records, shortage_severe, shortage_moderate
    ) -> None:
        problem = self._problem
        n_hosp = problem.n_hospitals
        for idx, raw_row in enumerate(chromosome.moderate_split):
            row = np.array(raw_row) / sum(raw_row)
            minor = problem.minor_injured[f'{problem.da_id[idx]}']
            for h_idx, j in enumerate(row[:n_hosp]):
                if j <= 0:
                    continue
                hospital_id = problem.h_id[h_idx]
                # LEGACY: presence is tested against the *severe* shortage dict
                # (a bug in the original) while writing into the moderate dict.
                self._record_shortage(
                    shortage_moderate,
                    hospital_id,
                    minor * j,
                    hospital_cap,
                    key_table=shortage_severe,
                )
                plan = self._transport.plan(
                    minor * j,
                    chromosome.moderate_ground_ratio[idx][h_idx],
                    TransportPlanner.MODERATE,
                )
                self.ground_vehicle_cost += plan.ground_cost
                self.air_vehicle_cost += plan.air_cost
                records.append(self._moderate_record(problem.da_id[idx], hospital_id, plan))

    def _moderate_to_tmc(self, chromosome, tmc_cap, tmc_shortage) -> List[dict]:
        problem = self._problem
        n_hosp = problem.n_hospitals
        legacy_idx = n_hosp - 1  # LEGACY: leftover ``h_idx`` from the hospital loop
        records: List[dict] = []
        n_tmc = []
        for idx, raw_row in enumerate(chromosome.moderate_split):
            row = np.array(raw_row) / sum(raw_row)
            minor = problem.minor_injured[f'{problem.da_id[idx]}']
            for tmc_idx, j in enumerate(row[n_hosp:]):
                if j <= 0:
                    continue
                tmc_id = problem.tmc_id[tmc_idx]
                n_tmc.append(tmc_id)
                if minor * j > tmc_cap[f'{tmc_id}']:
                    self._accumulate(
                        tmc_shortage, tmc_id, round(minor * j) - tmc_cap[f'{tmc_id}']
                    )
                    tmc_cap[f'{tmc_id}'] = 0
                else:
                    # LEGACY: reads capacity at ``legacy_idx`` (preserved bug).
                    tmc_cap[f'{tmc_id}'] = tmc_cap[f'{problem.tmc_id[legacy_idx]}'] - round(
                        minor * j
                    )
                # LEGACY: uses the leftover ``legacy_idx`` ground ratio column.
                plan = self._transport.plan(
                    minor * j,
                    chromosome.moderate_ground_ratio[idx][legacy_idx],
                    TransportPlanner.MODERATE,
                )
                self.ground_vehicle_cost += plan.ground_cost
                self.air_vehicle_cost += plan.air_cost
                records.append(self._moderate_record(problem.da_id[idx], tmc_id, plan))
        num_tmc = len(set(n_tmc))
        return records, num_tmc

    # -- helpers -----------------------------------------------------------
    def _record_shortage(
        self, target, facility_id, injured, cap_table, key_table
    ) -> None:
        """Replicate the original capacity-consumption + shortage bookkeeping."""
        if injured > cap_table[f'{facility_id}']:
            delta = round(injured) - cap_table[f'{facility_id}']
            # LEGACY: the "already seen?" test uses ``key_table`` which, for the
            # moderate hospital pass, is the *severe* shortage dict.
            if facility_id in key_table:
                target[facility_id] = target.get(facility_id, 0) + delta
            else:
                target[facility_id] = delta
            cap_table[f'{facility_id}'] = 0
        else:
            cap_table[f'{facility_id}'] -= round(injured)

    @staticmethod
    def _accumulate(table: Dict[int, float], key: int, delta: float) -> None:
        table[key] = table.get(key, 0) + delta if key in table else delta

    @staticmethod
    def _severe_record(source_id: int, target_id: int, plan: TransportPlan) -> dict:
        return {
            "source_id": source_id,
            "target_id": target_id,
            "g_flow_severe": plan.ground_flow,
            "num_gv_severe": plan.ground_vehicles,
            "g_flow_cost_severe": plan.ground_cost,
            "a_flow_severe": plan.air_flow,
            "num_av_severe": plan.air_vehicles,
            "a_flow_cost_severe": plan.air_cost,
        }

    @staticmethod
    def _moderate_record(source_id: int, target_id: int, plan: TransportPlan) -> dict:
        return {
            "source_id": source_id,
            "target_id": target_id,
            "g_flow_moderate": plan.ground_flow,
            "num_gv_moderate": plan.ground_vehicles,
            "g_flow_cost_moderate": plan.ground_cost,
            "a_flow_moderate": plan.air_flow,
            "num_av_moderate": plan.air_vehicles,
            "a_flow_cost_moderate": plan.air_cost,
        }
