"""Genetic operators: crossover and mutation.

Faithful refactor of the crossover / order_crossover / mutate methods from
``NSGA2_Humanitarian``.  Behaviour (including the per-part swap scheme) is
preserved; the operators now depend on the explicit :class:`Chromosome` fields
instead of magic list indices, and on an injected :class:`ChromosomeRepair`.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

from ..domain.chromosome import Chromosome
from ..domain.problem_data import ProblemData
from .repair import ChromosomeRepair


class CrossoverOperator:
    """Produce two children from two parents."""

    def __init__(self, problem: ProblemData, repair: ChromosomeRepair) -> None:
        self._problem = problem
        self._repair = repair

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        problem = self._problem
        n_shelters = problem.n_shelters
        n_damage = problem.n_damage_points
        n_hosp = problem.n_hospitals
        n_total = n_hosp + problem.n_temp_medical

        child1 = parent1.clone()
        child2 = parent2.clone()

        # Parts 1 & 2: single-point swap of the shelter/flow prefixes.
        point = random.choice(range(1, n_shelters))
        child1.dc_to_shelter[:point], child2.dc_to_shelter[:point] = (
            child2.dc_to_shelter[:point], child1.dc_to_shelter[:point])
        child1.shelter_flow_ratio[:point], child2.shelter_flow_ratio[:point] = (
            child2.shelter_flow_ratio[:point], child1.shelter_flow_ratio[:point])
        if sum(bool(x) for x in child1.dc_to_shelter) == 0:
            child1.dc_to_shelter = list(parent1.dc_to_shelter)
            child1.shelter_flow_ratio = list(parent2.shelter_flow_ratio)
        if sum(bool(x) for x in child2.dc_to_shelter) == 0:
            child2.dc_to_shelter = list(parent2.dc_to_shelter)
            child2.shelter_flow_ratio = list(parent2.shelter_flow_ratio)

        # Part 3: order crossover on the damage-area permutation prefix, random
        # swap on the padded tail.
        if n_damage > 0:
            c1_prefix, c2_prefix = self._order_crossover(
                parent1.damage_to_shelter[:n_damage], parent2.damage_to_shelter[:n_damage])
            child1.damage_to_shelter[:n_damage] = c1_prefix
            child2.damage_to_shelter[:n_damage] = c2_prefix
        for i in range(n_damage, n_shelters):
            if np.random.rand() < 0.5:
                child1.damage_to_shelter[i], child2.damage_to_shelter[i] = (
                    child2.damage_to_shelter[i], child1.damage_to_shelter[i])

        # Part 4: uniform (masked) crossover.
        mask = np.random.rand(n_damage, n_hosp) < 0.5
        child1.severe_split[mask] = parent2.severe_split[mask]
        child2.severe_split[mask] = parent1.severe_split[mask]

        # Part 5: block swap.
        p1 = np.random.choice(range(1, n_damage))
        p2 = np.random.choice(range(1, n_hosp))
        child1.severe_ground_ratio[:p1, :p2], child2.severe_ground_ratio[:p1, :p2] = (
            child2.severe_ground_ratio[:p1, :p2], child1.severe_ground_ratio[:p1, :p2])
        child1.severe_ground_ratio[p1:, p2:], child2.severe_ground_ratio[p1:, p2:] = (
            child2.severe_ground_ratio[p1:, p2:], child1.severe_ground_ratio[p1:, p2:])

        # Part 6: uniform (masked) crossover.
        mask = np.random.rand(n_damage, n_total) < 0.5
        child1.moderate_split[mask] = parent2.moderate_split[mask]
        child2.moderate_split[mask] = parent1.moderate_split[mask]

        # Part 7: multi-block swap.
        p1 = np.random.choice(range(1, n_damage))
        p2, p3 = sorted(random.sample(range(1, n_total), k=2))
        child1.moderate_ground_ratio[:p1, p2:p3], child2.moderate_ground_ratio[:p1, p2:p3] = (
            child2.moderate_ground_ratio[:p1, p2:p3], child1.moderate_ground_ratio[:p1, p2:p3])
        child1.moderate_ground_ratio[p1:, :p2], child2.moderate_ground_ratio[p1:, :p2] = (
            child2.moderate_ground_ratio[p1:, :p2], child1.moderate_ground_ratio[p1:, :p2])
        child1.moderate_ground_ratio[p1:, p3:], child2.moderate_ground_ratio[p1:, p3:] = (
            child2.moderate_ground_ratio[p1:, p3:], child1.moderate_ground_ratio[p1:, p3:])

        return self._repair.repair(child1), self._repair.repair(child2)

    @staticmethod
    def _order_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Order crossover (OX) that preserves permutations."""
        size = len(parent1)
        child1: List = [None] * size
        child2: List = [None] * size

        start, end = sorted(random.sample(range(size), 2))
        child1[start:end + 1] = parent1[start:end + 1]
        child2[start:end + 1] = parent2[start:end + 1]

        CrossoverOperator._fill_remaining(child1, parent2, start, end)
        CrossoverOperator._fill_remaining(child2, parent1, start, end)
        return child1, child2

    @staticmethod
    def _fill_remaining(child: List, donor: List[int], start: int, end: int) -> None:
        segment = child[start:end + 1]
        donor_idx = 0
        for i in range(len(child)):
            if child[i] is not None:
                continue
            while donor[donor_idx] in segment:
                donor_idx += 1
            child[i] = donor[donor_idx]
            donor_idx += 1


class MutationOperator:
    """Mutate a single chromosome in place on a clone."""

    def __init__(self, problem: ProblemData, repair: ChromosomeRepair) -> None:
        self._problem = problem
        self._repair = repair

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        problem = self._problem
        n_shelters = problem.n_shelters
        n_damage = problem.n_damage_points
        n_hosp = problem.n_hospitals
        n_tmc = problem.n_temp_medical

        mutated = chromosome.clone()

        # Part 1: reassign a random shelter's distribution centre.
        i = np.random.randint(0, n_shelters)
        mutated.dc_to_shelter[i] = random.randint(1, problem.n_distribution)

        # Part 2: perturb a non-zero flow ratio.
        i = np.random.randint(0, n_shelters)
        while mutated.shelter_flow_ratio[i] == 0:
            i = np.random.randint(0, n_shelters)
        mutated.shelter_flow_ratio[i] = max(
            0, np.clip(mutated.shelter_flow_ratio[i] + np.random.normal(0, 0.1), 0, 1))

        # Part 3: mutate the damage-area assignment.
        i = np.random.randint(0, n_shelters)
        if mutated.damage_to_shelter[i] == 0:
            mutated.damage_to_shelter[i] = random.choice(problem.da_id)
        elif i > n_damage:
            mutated.damage_to_shelter[i] = 0
        elif i < n_damage:
            j = random.choice([k for k in range(n_damage) if k != i])
            mutated.damage_to_shelter[i], mutated.damage_to_shelter[j] = (
                mutated.damage_to_shelter[j], mutated.damage_to_shelter[i])

        # Parts 4 & 5.
        for row in range(n_damage):
            col = random.randint(0, n_hosp - 1)
            if mutated.severe_split[row][col] == 0:
                mutated.severe_split[row][col] = random.uniform(0.1, 0.5)
            else:
                mutated.severe_split[row][col] = max(
                    0, np.clip(mutated.severe_split[row][col] + np.random.normal(0, 0.1), 0, 1))
            mutated.severe_ground_ratio[row][col] = np.clip(
                mutated.severe_ground_ratio[row][col] + np.random.normal(0, 0.1), 0, 1)

        # Parts 6 & 7.
        # LEGACY: the original loops over ``range(n_hospitals)`` here even though
        # the matrices are indexed by damage area; preserved for parity.
        for row in range(n_hosp):
            col = random.randint(0, n_hosp - 1)
            self._mutate_moderate_cell(mutated, row, col)
            tmc_col = random.randint(n_hosp, n_hosp + n_tmc - 1)
            self._mutate_moderate_cell(mutated, row, tmc_col)

        return self._repair.repair(mutated)

    @staticmethod
    def _mutate_moderate_cell(chromosome: Chromosome, row: int, col: int) -> None:
        if chromosome.moderate_split[row][col] == 0:
            chromosome.moderate_split[row][col] = random.uniform(0, 0.2)
        else:
            chromosome.moderate_split[row][col] = max(
                0, np.clip(chromosome.moderate_split[row][col] + np.random.normal(0, 0.1), 0, 1))
        chromosome.moderate_ground_ratio[row][col] = max(
            0, np.clip(chromosome.moderate_ground_ratio[row][col] + np.random.normal(0, 0.1), 0, 1))
