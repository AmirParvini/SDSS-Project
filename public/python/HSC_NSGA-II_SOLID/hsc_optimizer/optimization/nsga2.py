"""NSGA-II evolutionary engine.

The algorithm now depends only on small, injected collaborators (evaluator,
initializer, operators, selection, ranking, checkpoint).  This is the heart of
the Dependency-Inversion improvement: the engine knows *nothing* about the
humanitarian problem, file formats or cost details - it just evolves a
population using the ports it is given.
"""

from __future__ import annotations

import logging
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List

import numpy as np

from ..domain.chromosome import Chromosome
from ..evaluation.objectives import ObjectiveEvaluator
from .checkpoint import CheckpointRepository, CheckpointState, NullCheckpointRepository
from .individual import Individual
from .initialization import PopulationInitializer
from .operators import CrossoverOperator, MutationOperator
from .ranking import Ranking
from .selection import SelectionStrategy

logger = logging.getLogger(__name__)


@dataclass
class NSGA2Config:
    """Tunable hyper-parameters for a run."""

    max_iter: int = 100
    pop_size: int = 100
    p_crossover: float = 0.7
    p_mutation: float = 0.3
    resume: bool = True


class NSGA2:
    """Multi-objective evolutionary optimiser (constraint-aware NSGA-II)."""

    def __init__(
        self,
        evaluator: ObjectiveEvaluator,
        initializer: PopulationInitializer,
        crossover: CrossoverOperator,
        mutation: MutationOperator,
        selection: SelectionStrategy,
        ranking: Ranking,
        config: NSGA2Config,
        checkpoint: CheckpointRepository | None = None,
    ) -> None:
        self._evaluator = evaluator
        self._initializer = initializer
        self._crossover = crossover
        self._mutation = mutation
        self._selection = selection
        self._ranking = ranking
        self._config = config
        self._checkpoint = checkpoint or NullCheckpointRepository()

    # -- public API ---------------------------------------------------------
    def run(self) -> List[Chromosome]:
        """Run the optimisation and return the final Pareto-front chromosomes."""
        pop, fronts, pareto_history, start_it = self._initialise_state()

        pareto_pop: List[Individual] = []
        for iteration in range(start_it, self._config.max_iter):
            offspring = self._reproduce(pop)
            pop = pop + offspring
            self._evaluate_pending(pop)

            pop, fronts = self._ranking.non_dominated_sort(pop)
            pop = self._ranking.assign_crowding_distance(pop, fronts)
            pop, fronts = self._ranking.sort(pop)
            pop, fronts = self._ranking.truncate(pop, fronts, self._config.pop_size)

            pareto_pop = self._extract_pareto(pop, fronts)
            pareto_history.append(deepcopy(pareto_pop))

            logger.info("Iteration %d: Pareto members = %d", iteration + 1, len(pareto_pop))
            self._save_checkpoint(iteration, pop, fronts, pareto_history)

        return [ind.chromosome for ind in pareto_pop]

    # -- initialisation / resume -------------------------------------------
    def _initialise_state(self):
        if self._config.resume:
            state = self._safe_load_checkpoint()
            if state is not None:
                logger.info("Resumed from checkpoint at iteration %d.", state.iteration + 1)
                return state.pop, state.fronts, state.pareto_history, state.iteration + 1

        pop = self._initializer.initialize(self._config.pop_size)
        self._evaluate_pending(pop)
        pop, fronts = self._ranking.non_dominated_sort(pop)
        pop = self._ranking.assign_crowding_distance(pop, fronts)
        pop, fronts = self._ranking.sort(pop)
        return pop, fronts, [], 0

    def _safe_load_checkpoint(self):
        try:
            return self._checkpoint.load()
        except Exception as exc:  # noqa: BLE001 - corrupt checkpoint should not crash
            logger.warning("Failed to load checkpoint, starting fresh: %s", exc)
            return None

    # -- evolutionary operators --------------------------------------------
    def _reproduce(self, pop: List[Individual]) -> List[Individual]:
        offspring = self._crossover_phase(pop)
        return self._mutation_phase(pop, offspring)

    def _crossover_phase(self, pop: List[Individual]) -> List[Individual]:
        offspring: List[Individual] = []
        for _ in range(self._config.pop_size // 2):
            p1 = self._selection.select(pop)
            p2 = self._selection.select(pop)
            if random.uniform(0, 1) < self._config.p_crossover:
                c1, c2 = self._crossover.crossover(p1.chromosome, p2.chromosome)
                offspring.append(Individual(chromosome=c1))
                offspring.append(Individual(chromosome=c2))
            else:
                offspring.append(deepcopy(p1))
                offspring.append(deepcopy(p2))

        if len(offspring) < self._config.pop_size:
            offspring.append(deepcopy(self._selection.select(pop)))
        return offspring

    def _mutation_phase(self, pop, offspring: List[Individual]) -> List[Individual]:
        mutated: List[Individual] = []
        for child in offspring:
            if random.uniform(0, 1) < self._config.p_mutation:
                mutated.append(Individual(chromosome=self._mutation.mutate(child.chromosome)))
            else:
                mutated.append(child)
        return mutated

    # -- evaluation --------------------------------------------------------
    def _evaluate_pending(self, pop: List[Individual]) -> None:
        pending = [ind for ind in pop if not ind.is_evaluated]
        if not pending:
            return
        result = self._evaluator.evaluate([ind.chromosome for ind in pending])
        for ind, objective, violation, normalized in zip(
            pending, result.objectives, result.constraint_violations, result.normalized_objectives
        ):
            ind.cost = np.array(objective)
            ind.normal_cost = np.array(normalized)
            ind.constraint_violation = violation

    # -- helpers -----------------------------------------------------------
    def _extract_pareto(self, pop: List[Individual], fronts) -> List[Individual]:
        if fronts and fronts[0]:
            return [pop[i] for i in fronts[0]]
        sorted_pop, sorted_fronts = self._ranking.sort(pop)
        first = sorted_fronts[0] if sorted_fronts else []
        return [sorted_pop[i] for i in first]

    def _save_checkpoint(self, iteration, pop, fronts, pareto_history) -> None:
        try:
            self._checkpoint.save(CheckpointState(
                iteration=iteration, pop=pop, fronts=fronts, pareto_history=pareto_history))
        except Exception as exc:  # noqa: BLE001 - persistence must not break the run
            logger.warning("Failed to save checkpoint at iteration %d: %s", iteration, exc)
