"""Creation and validation of chromosomes.

Replaces ``NSGA2_Humanitarian.create_random_chromosome`` and
``_validate_and_convert_chromosome``.  Both concern *building valid genotypes*,
so they belong together and separate from the search algorithm (SRP).
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Optional

import numpy as np

from ..domain.chromosome import Chromosome
from ..domain.problem_data import ProblemData
from .repair import ChromosomeRepair


class ChromosomeFactory:
    """Produce random and externally-supplied chromosomes."""

    def __init__(self, problem: ProblemData, repair: ChromosomeRepair) -> None:
        self._problem = problem
        self._repair = repair

    # -- random generation --------------------------------------------------
    def create_random(self) -> Chromosome:
        problem = self._problem
        n_shelters = problem.n_shelters
        n_distribution = problem.n_distribution
        n_damage = problem.n_damage_points
        n_hosp = problem.n_hospitals
        n_tmc = problem.n_temp_medical

        # Part 1: activate a random subset of shelters.
        n_active = random.randint(1, n_shelters)
        active_shelters = np.random.choice(n_shelters, n_active, replace=False)
        dc_to_shelter = [0] * n_shelters
        for idx in active_shelters:
            dc_to_shelter[idx] = random.randint(1, n_distribution)

        # Part 2: random flow ratios.
        shelter_flow_ratio = [np.random.random() for _ in range(n_shelters)]

        # Part 3: damage-area permutation, padded with random damage areas.
        damage_areas = deepcopy(problem.da_id)
        np.random.shuffle(damage_areas)
        damage_to_shelter = list(damage_areas)
        for _ in range(n_damage, n_shelters):
            damage_to_shelter.append(np.random.choice(problem.da_id))

        # Part 4: severe split (each row has at least one active hospital).
        severe_split = np.zeros((n_damage, n_hosp))
        for i in range(n_damage):
            self._fill_random_row(severe_split[i], n_hosp)

        # Part 5: ground-ambulance share where part 4 is active.
        severe_ground_ratio = np.zeros((n_damage, n_hosp))
        for i in range(n_damage):
            for j in range(n_hosp):
                if severe_split[i][j] > 0:
                    severe_ground_ratio[i][j] = np.random.random()

        # Part 6: moderate split across hospitals + TMCs.
        moderate_split = np.zeros((n_damage, n_hosp + n_tmc))
        for i in range(n_damage):
            num_active = np.random.randint(1, n_hosp + n_tmc)
            self._fill_random_row(moderate_split[i], n_hosp + n_tmc, num_active)

        # Part 7: ground-ambulance share where part 6 is active.
        moderate_ground_ratio = np.zeros((n_damage, n_hosp + n_tmc))
        for i in range(n_damage):
            for j in range(n_hosp + n_tmc):
                if moderate_split[i][j] > 0:
                    moderate_ground_ratio[i][j] = np.random.random()

        return Chromosome(
            dc_to_shelter=dc_to_shelter,
            shelter_flow_ratio=shelter_flow_ratio,
            damage_to_shelter=damage_to_shelter,
            severe_split=severe_split,
            severe_ground_ratio=severe_ground_ratio,
            moderate_split=moderate_split,
            moderate_ground_ratio=moderate_ground_ratio,
        )

    @staticmethod
    def _fill_random_row(row: np.ndarray, n_cols: int, num_active: Optional[int] = None) -> None:
        """Place a normalised random weight vector into ``num_active`` columns."""
        if num_active is None:
            num_active = random.randint(1, n_cols)
        active_indices = random.sample(range(n_cols), num_active)
        weights = [np.random.random() for _ in range(num_active)]
        total = sum(weights)
        for k, dest in enumerate(active_indices):
            row[dest] = weights[k] / total

    # -- external (e.g. LLM-generated) chromosomes -------------------------
    def from_external_dict(self, item: dict) -> Optional[Chromosome]:
        """Validate and convert a ``part1..part7`` dict, or return ``None``."""
        problem = self._problem
        try:
            p1 = self._as_list(item.get("part1"))
            if p1 is None or len(p1) != problem.n_shelters:
                return None
            if any((not isinstance(x, (int, float))) or x < 0 or x > problem.n_distribution for x in p1):
                return None
            p1 = [int(round(x)) for x in p1]

            p2 = self._as_list(item.get("part2"))
            if p2 is None or len(p2) != problem.n_shelters:
                return None
            p2 = [float(min(1.0, max(0.0, x))) for x in p2]

            p3 = self._as_list(item.get("part3"))
            if p3 is None or len(p3) != problem.n_shelters:
                return None
            p3 = [int(x) for x in p3]
            if sorted(p3[: problem.n_damage_points]) != sorted(problem.da_id):
                return None

            p4 = self._as_matrix(item.get("part4"), problem.n_damage_points, problem.n_hospitals)
            if p4 is None or not all(np.any(r > 0) for r in p4):
                return None

            p5 = self._as_matrix(item.get("part5"), problem.n_damage_points, problem.n_hospitals)
            if p5 is None:
                return None
            p5 = np.clip(p5, 0.0, 1.0)
            p5[p4 == 0] = 0.0

            width = problem.n_hospitals + problem.n_temp_medical
            p6 = self._as_matrix(item.get("part6"), problem.n_damage_points, width)
            if p6 is None or not all(np.any(r > 0) for r in p6):
                return None

            p7 = self._as_matrix(item.get("part7"), problem.n_damage_points, width)
            if p7 is None:
                return None
            p7 = np.clip(p7, 0.0, 1.0)
            p7[p6 == 0] = 0.0

            chromosome = Chromosome.from_parts([p1, p2, p3, p4, p5, p6, p7])
            return self._repair.repair(chromosome)
        except Exception:
            return None

    @staticmethod
    def _as_list(value: Any):
        return list(value) if isinstance(value, list) else None

    @staticmethod
    def _as_matrix(value: Any, rows: int, cols: int):
        if not (isinstance(value, list) and len(value) == rows):
            return None
        arr = np.array(value, dtype=float)
        return arr if arr.shape == (rows, cols) else None
