"""Multi-objective evaluation of candidate solutions.

This is the clean-architecture replacement for
``Main.complex_humanitarian_cost``.  The monolithic method is decomposed into:

* an :class:`ObjectiveEvaluator` port (so the NSGA-II engine depends on an
  abstraction, not on ``Main``), and
* :class:`HumanitarianCostEvaluator`, which orchestrates the focused services
  (assignment, allocation, transport, death probability).

The three objectives are preserved exactly:

* **F1** - weighted distance from damaged areas to shelters,
* **F2** - unmet relief demand + a fraction of shelter capacity shortage,
* **F3** - expected deaths (in-transit + treatment-queue shortages).

Several historical quirks are intentionally retained for numerical parity with
the original model; each is flagged with a ``LEGACY`` comment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Sequence

import numpy as np

from ..domain.chromosome import Chromosome
from ..domain.problem_data import DeathModelParameters, ProblemData
from ..domain.results import EvaluationResult
from .allocation import ShelterAllocationService
from .assignment import DamageShelterAssigner
from .death_probability import DeathProbabilityModel
from .transport import TransportPlanner


class ObjectiveEvaluator(ABC):
    """Port: evaluate a population of chromosomes into objective values."""

    @abstractmethod
    def evaluate(
        self, chromosomes: Sequence[Chromosome]
    ) -> EvaluationResult:  # pragma: no cover
        raise NotImplementedError


class _SolutionMetrics:
    """Mutable accumulators for a single chromosome's evaluation."""

    __slots__ = (
        "weighted_distance",
        "ec_shortage",
        "unmet_demand",
        "ecs_cost",
        "tmcs_cost",
        "last_relief_package_cost",
        "transport_cost",
        "death_prob",
        "hospital_shortage_severe",
        "hospital_shortage_minor",
        "tmc_shortage",
    )

    def __init__(self) -> None:
        self.weighted_distance = 0.0
        self.ec_shortage = 0.0
        self.unmet_demand: dict = {}
        self.ecs_cost = 0.0
        self.tmcs_cost = 0.0
        self.last_relief_package_cost = 0.0
        self.transport_cost = 0.0
        self.death_prob = 0.0
        self.hospital_shortage_severe = 0
        self.hospital_shortage_minor = 0
        self.tmc_shortage = 0


class HumanitarianCostEvaluator(ObjectiveEvaluator):
    """Evaluate chromosomes for the humanitarian-logistics problem."""

    def __init__(
        self,
        problem: ProblemData,
        assigner: DamageShelterAssigner,
        allocator: ShelterAllocationService,
        transport: TransportPlanner,
        death_model: DeathProbabilityModel,
    ) -> None:
        self._problem = problem
        self._assigner = assigner
        self._allocator = allocator
        self._transport = transport
        self._death = death_model

    # -- public API ---------------------------------------------------------
    def evaluate(self, chromosomes: Sequence[Chromosome]) -> EvaluationResult:
        f1, f2, f3, violations = [], [], [], []
        for chromosome in chromosomes:
            metrics = self._evaluate_one(chromosome)
            f1.append(metrics.weighted_distance)
            f2.append(sum(metrics.unmet_demand.values()) + metrics.ec_shortage / 5)
            f3.append(self._total_expected_deaths(metrics))
            violations.append(self._constraint_violation(metrics))

        objectives = [list(t) for t in zip(f1, f2, f3)]
        return EvaluationResult(
            objectives=objectives,
            constraint_violations=violations,
            normalized_objectives=objectives,
        )

    # -- per-chromosome evaluation -----------------------------------------
    def _evaluate_one(self, chromosome: Chromosome) -> _SolutionMetrics:
        problem = self._problem
        metrics = _SolutionMetrics()

        # Capacities mutate as we consume them; start from a fresh copy.
        hospital_cap = deepcopy(problem.capacity.hospital)
        tmc_cap = deepcopy(problem.capacity.tmc)

        # Part 3 + shelter allocation.
        da_ec_alloc, num_selected_ec = self._assigner.assign(chromosome)
        allocation = self._allocator.allocate(da_ec_alloc)
        metrics.weighted_distance = allocation.weighted_distance
        metrics.ec_shortage = allocation.total_shortage

        self._relief_packages(chromosome, allocation.demand, metrics)
        self._severe_to_hospital(chromosome, hospital_cap, metrics)
        self._moderate_to_hospital(chromosome, hospital_cap, metrics)
        self._moderate_to_tmc(chromosome, tmc_cap, metrics)
        return metrics

    def _relief_packages(self, chromosome, demand, metrics) -> None:
        import math

        problem = self._problem
        for idx, dc in enumerate(chromosome.dc_to_shelter):
            if dc <= 0:
                continue
            shelter_id = problem.ec_id[idx]
            metrics.ecs_cost += problem.cost["ec_cost"]
            dist = problem.distance.dc_to_shelter[f"{dc},{shelter_id}"]
            flow = math.ceil(chromosome.shelter_flow_ratio[idx] * demand[shelter_id])
            # LEGACY: only the *last* active shelter's package cost feeds the
            # budget constraint (the original kept a non-accumulated variable).
            metrics.last_relief_package_cost += (
                dist * problem.cost["reliefpackage_transportation_cost"] * flow) +\
            (problem.cost["reliefpackage_cost"] * round( demand[shelter_id] * (1 - chromosome.shelter_flow_ratio[idx])))
            metrics.unmet_demand[shelter_id] = round(
                demand[shelter_id] * (1 - chromosome.shelter_flow_ratio[idx])
            )

    def _severe_to_hospital(self, chromosome, hospital_cap, metrics) -> None:
        problem = self._problem
        model = problem.severe_death_model
        for idx, raw_row in enumerate(chromosome.severe_split):
            row = np.array(raw_row) / sum(raw_row)
            severe = problem.severe_injured[f'{problem.da_id[idx]}']
            for h_idx, j in enumerate(row):
                if j <= 0:
                    continue
                hospital_id = problem.h_id[h_idx]
                self._consume_capacity(
                    hospital_cap,
                    hospital_id,
                    severe * j,
                    metrics,
                    "hospital_shortage_severe",
                )
                ground_ratio = chromosome.severe_ground_ratio[idx][h_idx]
                self._add_transport(
                    severe * j, ground_ratio, TransportPlanner.SEVERE, metrics
                )
                self._add_in_transit_deaths(
                    model,
                    problem.distance.da_to_h[f"{problem.da_id[idx]},{hospital_id}"],
                    problem.distance.da_to_h_helicopter[
                        f"{problem.da_id[idx]},{hospital_id}"
                    ],
                    severe,
                    j,
                    metrics,
                )

    def _moderate_to_hospital(self, chromosome, hospital_cap, metrics) -> None:
        problem = self._problem
        # LEGACY: moderate-to-hospital in-transit deaths use the *severe* death
        # model parameters in the original code.
        model = problem.severe_death_model
        n_hosp = problem.n_hospitals
        for idx, raw_row in enumerate(chromosome.moderate_split):
            row = np.array(raw_row) / sum(raw_row)
            minor = problem.minor_injured[f'{problem.da_id[idx]}']
            for h_idx, j in enumerate(row[:n_hosp]):
                if j <= 0:
                    continue
                metrics.tmcs_cost += problem.cost["tmc_cost"]
                hospital_id = problem.h_id[h_idx]
                self._consume_capacity(
                    hospital_cap,
                    hospital_id,
                    minor * j,
                    metrics,
                    "hospital_shortage_minor",
                )
                ground_ratio = chromosome.moderate_ground_ratio[idx][h_idx]
                self._add_transport(
                    minor * j, ground_ratio, TransportPlanner.MODERATE, metrics
                )
                self._add_in_transit_deaths(
                    model,
                    problem.distance.da_to_h[f"{problem.da_id[idx]},{hospital_id}"],
                    problem.distance.da_to_h_helicopter[
                        f"{problem.da_id[idx]},{hospital_id}"
                    ],
                    minor,
                    j,
                    metrics,
                )

    def _moderate_to_tmc(self, chromosome, tmc_cap, metrics) -> None:
        problem = self._problem
        model = problem.moderate_death_model
        n_hosp = problem.n_hospitals
        # LEGACY: the original reused ``h_idx`` (left over from the hospital
        # loop, always == n_hosp - 1) when reading the ground ratio and the
        # capacity look-up below.
        legacy_idx = n_hosp - 1
        for idx, raw_row in enumerate(chromosome.moderate_split):
            row = np.array(raw_row) / sum(raw_row)
            minor = problem.minor_injured[f'{problem.da_id[idx]}']
            for tmc_idx, j in enumerate(row[n_hosp:]):
                if j <= 0:
                    continue
                tmc_id = problem.tmc_id[tmc_idx]
                if minor * j > tmc_cap[f'{tmc_id}']:
                    metrics.tmc_shortage += round(minor * j) - tmc_cap[f'{tmc_id}']
                    tmc_cap[f'{tmc_id}'] = 0
                else:
                    # LEGACY: reads capacity at ``legacy_idx`` (a bug preserved
                    # for numerical parity with the original implementation).
                    tmc_cap[f'{tmc_id}'] = tmc_cap[f'{problem.tmc_id[legacy_idx]}'] - round(
                        minor * j
                    )
                ground_ratio = chromosome.moderate_ground_ratio[idx][legacy_idx]
                self._add_transport(
                    minor * j, ground_ratio, TransportPlanner.MODERATE, metrics
                )
                self._add_in_transit_deaths(
                    model,
                    problem.distance.da_to_tmc[f"{problem.da_id[idx]},{tmc_id}"],
                    problem.distance.da_to_tmc_helicopter[
                        f"{problem.da_id[idx]},{tmc_id}"
                    ],
                    minor,
                    j,
                    metrics,
                )

    # -- shared building blocks --------------------------------------------
    def _consume_capacity(
        self, cap_table, facility_id, injured, metrics, shortage_attr
    ) -> None:
        if injured > cap_table[f'{facility_id}']:
            setattr(
                metrics,
                shortage_attr,
                getattr(metrics, shortage_attr)
                + round(injured)
                - cap_table[f'{facility_id}'],
            )
            cap_table[f'{facility_id}'] = 0
        else:
            cap_table[f'{facility_id}'] -= round(injured)

    def _add_transport(self, injured, ground_ratio, injured_type, metrics) -> None:
        plan = self._transport.plan(injured, ground_ratio, injured_type)
        metrics.transport_cost += plan.total_cost

    def _add_in_transit_deaths(
        self,
        model: DeathModelParameters,
        ground_dist,
        air_dist,
        injured,
        fraction,
        metrics,
    ) -> None:
        t_ground = self._transport.ground_minutes(ground_dist)
        t_air = self._transport.air_minutes(air_dist)
        metrics.death_prob += self._death.probability(model, t_ground) * round(
            injured * fraction
        ) + self._death.probability(model, t_air) * round(injured * (1 - fraction))

    # -- objective / constraint aggregation --------------------------------
    def _total_expected_deaths(self, metrics: _SolutionMetrics) -> float:
        problem = self._problem
        
        # ۱. محاسبه صف یکپارچه و مشترک برای بیمارستان‌ها (با اولویت مجروحین شدید)
        hospital_queue_deaths = self._combined_hospital_queue_deaths(
            metrics.hospital_shortage_severe,
            metrics.hospital_shortage_minor,
            problem.severe_death_model,
            problem.moderate_death_model
        )
        
        # ۲. محاسبه صف مستقل برای TMCها (TMC فقط مجروحین متوسط دارد و صف آن جداست)
        tmc_queue_deaths = self._queue_deaths(
            metrics.tmc_shortage, 
            problem.moderate_death_model
        )
        
        return metrics.death_prob + hospital_queue_deaths + tmc_queue_deaths

    def _combined_hospital_queue_deaths(
        self, severe_shortage, minor_shortage, severe_model: DeathModelParameters, minor_model: DeathModelParameters
    ) -> float:
        """
        محاسبه احتمال مرگ در صف بیمارستان با رعایت اولویت.
        ابتدا مجروحین شدید وارد دسته‌های درمان می‌شوند. مجروحین متوسط
        از ظرفیت باقی‌مانده دسته‌ها استفاده می‌کنند و زمان انتظارشان به درستی محاسبه می‌شود.
        """
        problem = self._problem
        total = 0.0
        
        # مقداردهی اولیه با صفر انجام می‌شود تا در دسته اول (i=0) زمان انتظار دقیقاً برابر با یک waiting_time شود.
        t = 0.0 
        
        total_shortage = int(severe_shortage) + int(minor_shortage)
        severe_count = int(severe_shortage)
        
        for i in range(total_shortage):
            # هر گاه دسته جدیدی شروع شود (مثلاً هر 50 نفر)، زمان انتظار یک پله بالا می‌رود
            if i % problem.injured_treated_same_time == 0:
                t += problem.waiting_time
            
            # اگر در بازه مجروحین شدید هستیم (اولویت بالا)
            if i < severe_count:
                total += self._death.probability(severe_model, t)
            # اگر ظرفیت مجروحین شدید تمام شد، ادامه ظرفیت دسته به مجروحین متوسط می‌رسد
            else:
                total += self._death.probability(minor_model, t)
                
        return total

    def _queue_deaths(self, shortage, model: DeathModelParameters) -> float:
        """محاسبه صف برای مراکزی که فقط یک نوع مجروح دارند (مثل TMC)"""
        if shortage <= 0:
            return 0.0
        problem = self._problem
        total = 0.0
        
        # اصلاح باگ قبلی: مقدار دهی با 0
        t = 0.0 
        
        for i in range(int(shortage)):
            if i % problem.injured_treated_same_time == 0:
                t += problem.waiting_time
            total += self._death.probability(model, t)
        return total

    def _constraint_violation(self, metrics: _SolutionMetrics) -> float:
        sum_costs = (
            metrics.last_relief_package_cost
            + metrics.transport_cost
            + metrics.ecs_cost
            + metrics.tmcs_cost
        )
        budget = self._problem.budget
        return (sum_costs - budget) / budget if sum_costs > budget else 0