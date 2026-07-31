"""NSGA-II ranking primitives.

Constraint-aware dominance, fast non-dominated sorting, crowding distance,
sorting and truncation - lifted from ``NSGA2_Humanitarian`` into a cohesive,
testable unit separate from the evolutionary loop.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .individual import Individual

Front = List[int]
Fronts = List[Front]


class DominanceComparator:
    """Constraint-dominance: feasible beats infeasible, then Pareto order."""

    def dominates(self, p: Individual, q: Individual) -> bool:
        pc, qc = p.constraint_violation, q.constraint_violation
        if pc > 0 and qc > 0:
            return pc < qc
        if pc > 0 and qc == 0:
            return False
        if pc == 0 and qc > 0:
            return True
        return bool(np.all(p.normal_cost <= q.normal_cost) and np.any(p.normal_cost < q.normal_cost))


class Ranking:
    """Non-dominated sorting + crowding distance + ordering helpers."""

    def __init__(self, comparator: DominanceComparator) -> None:
        self._comparator = comparator

    def non_dominated_sort(self, pop: List[Individual]) -> Tuple[List[Individual], Fronts]:
        n = len(pop)
        dominated_by = [[] for _ in range(n)]
        domination_count = [0] * n
        fronts: Fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._comparator.dominates(pop[i], pop[j]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif self._comparator.dominates(pop[j], pop[i]):
                    dominated_by[j].append(i)
                    domination_count[i] += 1
            if domination_count[i] == 0:
                pop[i].rank = 0
                fronts[0].append(i)

        k = 0
        while True:
            nxt: Front = []
            for i in fronts[k]:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        pop[j].rank = k + 1
                        nxt.append(j)
            if not nxt:
                break
            fronts.append(nxt)
            k += 1
        return pop, fronts

    def assign_crowding_distance(self, pop: List[Individual], fronts: Fronts) -> List[Individual]:
        n_obj = len(pop[0].normal_cost)
        for front in fronts:
            costs = np.array([pop[i].normal_cost for i in front])
            n = len(front)
            distances = np.zeros((n, n_obj))
            for j in range(n_obj):
                order = np.argsort(costs[:, j])
                distances[order[0], j] = np.inf
                distances[order[-1], j] = np.inf
                span = costs[order[-1], j] - costs[order[0], j]
                for i in range(1, n - 1):
                    distances[order[i], j] = costs[order[i + 1], j] - costs[order[i - 1], j]
                    if span != 0:
                        distances[order[i], j] /= span
            for i in range(n):
                pop[front[i]].crowding_distance = float(np.sum(distances[i, :]))
        return pop

    def sort(self, pop: List[Individual]) -> Tuple[List[Individual], Fronts]:
        pop = sorted(pop, key=lambda x: (x.rank, -x.crowding_distance))
        max_rank = pop[-1].rank
        fronts = [[i for i in range(len(pop)) if pop[i].rank == r] for r in range(max_rank + 1)]
        return pop, fronts

    @staticmethod
    def truncate(pop: List[Individual], fronts: Fronts, pop_size: int) -> Tuple[List[Individual], Fronts]:
        if len(pop) <= pop_size:
            return pop, fronts
        pop = pop[:pop_size]
        fronts = [[i for i in front if i < pop_size] for front in fronts]
        return pop, fronts
