"""Parent selection strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .individual import Individual


class SelectionStrategy(ABC):
    """Port: pick one parent from the population."""

    @abstractmethod
    def select(self, pop: List[Individual]) -> Individual:  # pragma: no cover
        raise NotImplementedError


class CrowdingTournamentSelection(SelectionStrategy):
    """Binary tournament: lower rank wins, ties broken by crowding distance."""

    def __init__(self, tournament_size: int = 2) -> None:
        self._tournament_size = tournament_size

    def select(self, pop: List[Individual]) -> Individual:
        idx = np.random.choice(range(len(pop)), size=self._tournament_size, replace=False)
        p1, p2 = pop[idx[0]], pop[idx[1]]
        if p1.rank < p2.rank:
            return p1
        if p2.rank < p1.rank:
            return p2
        return p1 if p1.crowding_distance > p2.crowding_distance else p2
