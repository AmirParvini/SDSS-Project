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

    def decode(
        self, chromosomes: List[Chromosome], pareto_pops: List[Individual]
    ) -> List[dict]:
        return [
            self._decode_one(idx + 1, chrom, pareto_pops[idx])
            for idx, chrom in enumerate(chromosomes)
        ]

    # -- per-solution -------------------------------------------------------
    def _decode_one(
        self, solution_id: int, chromosome: Chromosome, pareto_pop: Individual
    ) -> dict:
        from copy import deepcopy

        problem = self._problem
        hospital_cap = deepcopy(problem.capacity.hospital)
        tmc_cap = deepcopy(problem.capacity.tmc)

        self.package_flow_cost = 0
        self.package_cost = 0
        self.ground_vehicle_cost = 0
        self.air_vehicle_cost = 0

        da_ec_alloc, num_selected_ec = self._assigner.assign(chromosome)
        shelter_establish_cost = num_selected_ec * problem.cost["ec_cost"]
        allocation = self._allocator.allocate(da_ec_alloc)
        demand = allocation.demand

        shelter_allocations = self._shelter_allocations(allocation.allocations)
        package_flows, dc_id_list, ec_id_list = self._package_flows(chromosome, demand)

        hospital_allocations: List[dict] = []
        h_id_list: List = []
        shortage_severe: Dict[int, float] = {}
        shortage_moderate: Dict[int, float] = {}
        tmc_id_list: List = []
        tmc_shortage: Dict[int, float] = {}

        # ابتدا تخصیص بیماران بدحال (Severe) به بیمارستان‌ها انجام می‌شود تا ظرفیت‌ها کسر شود.
        self._injured_to_hospital(
            chromosome, hospital_cap, hospital_allocations, shortage_severe, h_id_list
        )
        # سپس محاسبه‌ی کمبود ظرفیت بیماران سرپایی (Moderate) با ظرفیت باقی‌مانده بیمارستان‌ها انجام می‌گیرد.
        self._process_moderate_hospital_shortage(
            chromosome, hospital_cap, shortage_moderate, shortage_severe
        )
        tmc_allocations, num_tmc = self._injured_to_tmc(
            chromosome, tmc_cap, tmc_shortage, tmc_id_list
        )
        tmc_establish_cost = num_tmc * problem.cost["tmc_cost"]
        total_cost = (
            self.package_flow_cost
            + self.package_cost
            + self.ground_vehicle_cost
            + self.air_vehicle_cost
            + shelter_establish_cost
            + tmc_establish_cost
        )
        return {
            "solution_id": solution_id,
            "dc_id": dc_id_list,
            "ec_id": ec_id_list,
            "h_id": h_id_list,
            "tmc_id": tmc_id_list,
            "F1": pareto_pop.cost[0],
            "F2": pareto_pop.cost[1],
            "F3": pareto_pop.cost[2],
            "package_flows": package_flows,
            "shelter_allocations": shelter_allocations,
            "hospital_allocations": hospital_allocations,
            "tmc_allocations": tmc_allocations,
            "solution_unsettled_population": allocation.unsettled_population,
            "solution_hospital_shortage_severe": shortage_severe,
            "solution_hospital_shortage_moderate": shortage_moderate,
            "solution_tmc_shortage": tmc_shortage,
            "costs": {
                "package_flow_cost": self.package_flow_cost,
                "package_cost": self.package_cost,
                "ground_vehicle_cost": self.ground_vehicle_cost,
                "air_vehicle_cost": self.air_vehicle_cost,
                "shelter_establish_cost": shelter_establish_cost,
                "tmc_establish_cost": tmc_establish_cost,
                "total_cost": total_cost,
            },
        }

    # -- sections -----------------------------------------------------------
    def _shelter_allocations(
        self, allocations: Dict[int, Dict[int, float]]
    ) -> List[dict]:
        problem = self._problem
        records = []
        for da_id, shelter_flow in allocations.items():
            for shelter_id, flow in shelter_flow.items():
                records.append(
                    {
                        "source_id": da_id,
                        "target_id": shelter_id,
                        "geometry": problem.path.get(f"{da_id},{shelter_id}", {}).get(
                            "air", None
                        ),
                        "flow": flow,
                    }
                )
        return records

    def _package_flows(self, chromosome: Chromosome, demand) -> List[dict]:
        problem = self._problem
        records = []
        dc_id_list = []
        ec_id_list = []
        for idx, dc in enumerate(chromosome.dc_to_shelter):
            if dc <= 0:
                continue
            dc_id_list.append(dc)
            shelter_id = problem.ec_id[idx]
            ec_id_list.append(shelter_id)
            dist = problem.distance.dc_to_shelter[f"{dc},{shelter_id}"]
            flow = math.ceil(chromosome.shelter_flow_ratio[idx] * demand[shelter_id])
            unmet_demand = math.ceil(
                (1 - chromosome.shelter_flow_ratio[idx]) * demand[shelter_id]
            )
            self.package_cost += flow * problem.cost["reliefpackage_cost"]
            flow_cost = dist * problem.cost["reliefpackage_transportation_cost"] * flow
            self.package_flow_cost += flow_cost
            records.append(
                {
                    "source_id": dc,
                    "target_id": shelter_id,
                    "geometry": problem.path.get(f"{dc},{shelter_id}", {}).get(
                        "ground", None
                    ),
                    "flow": flow,
                    "flow_cost": flow_cost,
                    "unmet_demand": unmet_demand,
                }
            )
        return records, dc_id_list, ec_id_list

    def _injured_to_hospital(
        self,
        chromosome: Chromosome,
        hospital_cap,
        records,
        shortage_severe,
        h_id_list: List,
    ) -> None:
        problem = self._problem
        for idx, severe_raw_row in enumerate(chromosome.severe_split):
            moderate_raw_row = chromosome.moderate_split[idx]
            severe_row = np.array(severe_raw_row) / sum(severe_raw_row)
            moderate_row = np.array(moderate_raw_row) / sum(moderate_raw_row)
            severe = problem.severe_injured[f"{problem.da_id[idx]}"]
            moderate = problem.minor_injured[f"{problem.da_id[idx]}"]
            for h_idx, j in enumerate(severe_row):
                if j <= 0 and moderate_row[h_idx] <= 0:
                    continue
                hospital_id = problem.h_id[h_idx]
                if hospital_id not in h_id_list:
                    h_id_list.append(hospital_id)

                # ثبت کمبود ظرفیت فقط برای مجروحین severe
                self._record_shortage(
                    shortage_severe,
                    hospital_id,
                    severe * j,
                    hospital_cap,
                    key_table=shortage_severe,
                )

                severe_plan = self._transport.plan(
                    severe * j,
                    chromosome.severe_ground_ratio[idx][h_idx],
                    TransportPlanner.SEVERE,
                )
                self.ground_vehicle_cost += severe_plan.ground_cost
                self.air_vehicle_cost += severe_plan.air_cost
                moderate_plan = self._transport.plan(
                    moderate * moderate_row[h_idx],
                    chromosome.moderate_ground_ratio[idx][h_idx],
                    TransportPlanner.MODERATE,
                )
                self.ground_vehicle_cost += moderate_plan.ground_cost
                self.air_vehicle_cost += moderate_plan.air_cost
                records.append(
                    self._hospital_record(
                        problem,
                        problem.da_id[idx],
                        hospital_id,
                        severe_plan,
                        moderate_plan,
                    )
                )

    def _process_moderate_hospital_shortage(
        self, chromosome: Chromosome, hospital_cap, shortage_moderate, shortage_severe
    ) -> None:
        """محاسبه کمبود ظرفیت بیمارستان‌ها برای مجروحین moderate پس از تخصیص کامل ظرفیت به severe."""
        problem = self._problem
        for idx, moderate_raw_row in enumerate(chromosome.moderate_split):
            moderate_row = np.array(moderate_raw_row) / sum(moderate_raw_row)
            moderate = problem.minor_injured[f"{problem.da_id[idx]}"]
            for h_idx, j in enumerate(
                moderate_row[: problem.n_hospitals]
            ):  # فقط ظرفیت‌های مربوط به بیمارستان‌ها بررسی شود
                if j <= 0:
                    continue
                hospital_id = problem.h_id[h_idx]

                # محاسبه کمبود ظرفیت برای بیماران moderate با ظرفیت باقی‌مانده (پس از کسر severe)
                self._record_shortage(
                    shortage_moderate,
                    hospital_id,
                    moderate * j,
                    hospital_cap,
                    key_table=shortage_severe,
                )

    def _injured_to_tmc(
        self, chromosome, tmc_cap, tmc_shortage, tmc_id_list: List
    ) -> List[dict]:
        problem = self._problem
        n_hosp = problem.n_hospitals
        legacy_idx = n_hosp - 1  # LEGACY: leftover ``h_idx`` from the hospital loop
        records: List[dict] = []
        n_tmc = []
        for idx, raw_row in enumerate(chromosome.moderate_split):
            row = np.array(raw_row) / sum(raw_row)
            minor = problem.minor_injured[f"{problem.da_id[idx]}"]
            for tmc_idx, j in enumerate(row[n_hosp:]):
                if j <= 0:
                    continue
                tmc_id = problem.tmc_id[tmc_idx]
                if tmc_id not in tmc_id_list:
                    tmc_id_list.append(tmc_id)
                n_tmc.append(tmc_id)
                if minor * j > tmc_cap[f"{tmc_id}"]:
                    self._accumulate(
                        tmc_shortage, tmc_id, round(minor * j) - tmc_cap[f"{tmc_id}"]
                    )
                    tmc_cap[f"{tmc_id}"] = 0
                else:
                    # LEGACY: reads capacity at ``legacy_idx`` (preserved bug).
                    tmc_cap[f"{tmc_id}"] = tmc_cap[
                        f"{problem.tmc_id[legacy_idx]}"
                    ] - round(minor * j)
                # LEGACY: uses the leftover ``legacy_idx`` ground ratio column.
                plan = self._transport.plan(
                    minor * j,
                    chromosome.moderate_ground_ratio[idx][legacy_idx],
                    TransportPlanner.MODERATE,
                )
                self.ground_vehicle_cost += plan.ground_cost
                self.air_vehicle_cost += plan.air_cost
                records.append(
                    self._tmc_record(problem, problem.da_id[idx], tmc_id, plan)
                )
        num_tmc = len(set(n_tmc))
        return records, num_tmc

    # -- helpers -----------------------------------------------------------
    def _record_shortage(
        self, target, facility_id, injured, cap_table, key_table
    ) -> None:
        """Replicate the original capacity-consumption + shortage bookkeeping."""
        if round(injured) > cap_table[f"{facility_id}"]:
            delta = round(injured) - cap_table[f"{facility_id}"]
            # LEGACY: the "already seen?" test uses ``key_table`` which, for the
            # moderate hospital pass, is the *severe* shortage dict.
            if facility_id in key_table:
                target[facility_id] = target.get(facility_id, 0) + delta
            else:
                target[facility_id] = delta
            cap_table[f"{facility_id}"] = 0
        else:
            cap_table[f"{facility_id}"] -= round(injured)

    @staticmethod
    def _accumulate(table: Dict[int, float], key: int, delta: float) -> None:
        table[key] = table.get(key, 0) + delta if key in table else delta

    @staticmethod
    def _hospital_record(
        problem: ProblemData,
        source_id: int,
        target_id: int,
        severe_plan: TransportPlan,
        moderate_plan: TransportPlan,
    ) -> dict:
        if severe_plan.ground_flow == 0 and moderate_plan.ground_flow == 0:
            geometry = problem.path.get(f"{source_id},{target_id}", {}).get("air", None)
        else:
            geometry = problem.path.get(f"{source_id},{target_id}", {}).get(
                "ground", None
            )
        return {
            "source_id": source_id,
            "target_id": target_id,
            "geometry": geometry,
            "g_flow_severe": severe_plan.ground_flow,
            "num_gv_severe": severe_plan.ground_vehicles,
            "g_flow_cost_severe": severe_plan.ground_cost,
            "a_flow_severe": severe_plan.air_flow,
            "num_av_severe": severe_plan.air_vehicles,
            "a_flow_cost_severe": severe_plan.air_cost,
            "g_flow_moderate": moderate_plan.ground_flow,
            "num_gv_moderate": moderate_plan.ground_vehicles,
            "g_flow_cost_moderate": moderate_plan.ground_cost,
            "a_flow_moderate": moderate_plan.air_flow,
            "num_av_moderate": moderate_plan.air_vehicles,
            "a_flow_cost_moderate": moderate_plan.air_cost,
        }

    @staticmethod
    def _tmc_record(
        problem: ProblemData, source_id: int, target_id: int, plan: TransportPlan
    ) -> dict:
        if plan.ground_flow == 0 and plan.ground_flow == 0:
            geometry = problem.path.get(f"{source_id},{target_id}", {}).get("air", None)
        else:
            geometry = problem.path.get(f"{source_id},{target_id}", {}).get(
                "ground", None
            )
        return {
            "source_id": source_id,
            "target_id": target_id,
            "geometry": geometry,
            "g_flow_moderate": plan.ground_flow,
            "num_gv_moderate": plan.ground_vehicles,
            "g_flow_cost_moderate": plan.ground_cost,
            "a_flow_moderate": plan.air_flow,
            "num_av_moderate": plan.air_vehicles,
            "a_flow_cost_moderate": plan.air_cost,
        }
