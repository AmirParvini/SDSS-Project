"""Chromosome repair (feasibility restoration).

Extracted from ``NSGA2_Humanitarian.chromosome_repair``.  Repair is a distinct
responsibility from crossover/mutation, so it gets its own class that can be
reused by the factory, crossover and mutation operators alike.
"""

from __future__ import annotations

import random

import numpy as np

from ..domain.chromosome import Chromosome
from ..domain.problem_data import ProblemData


class ChromosomeRepair:
    """Ensure a chromosome satisfies the basic structural constraints."""

    def __init__(self, problem: ProblemData) -> None:
        self._problem = problem

    def repair(self, chromosome: Chromosome) -> Chromosome:
        repaired = chromosome.clone()
        self._ensure_active_shelter(repaired)
        self._ensure_nonzero_rows(repaired.severe_split)
        self._ensure_nonzero_rows(repaired.moderate_split)
        return repaired

    def _ensure_active_shelter(self, chromosome: Chromosome) -> None:
        if all(gene == 0 for gene in chromosome.dc_to_shelter):
            n_active = random.randint(1, self._problem.n_shelters)
            active = np.random.choice(self._problem.n_shelters, n_active, replace=False)
            for idx in active:
                chromosome.dc_to_shelter[idx] = random.randint(1, self._problem.n_distribution)

    @staticmethod
    def _ensure_nonzero_rows(matrix: np.ndarray) -> None:
        rows, cols = matrix.shape
        for i in range(rows):
            if np.sum(matrix[i, :]) != 0:
                continue
            num_active = random.randint(1, cols)
            active_indices = random.sample(range(cols), num_active)
            weights = [np.random.random() for _ in range(num_active)]
            total = sum(weights)
            for k, dest in enumerate(active_indices):
                matrix[i][dest] = weights[k] / total
