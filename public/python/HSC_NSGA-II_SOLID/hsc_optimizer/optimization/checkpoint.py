"""Checkpoint persistence for resumable runs.

The original engine hard-coded pickle paths and ``open(...)`` calls inside the
main loop.  Persistence is now a port (:class:`CheckpointRepository`) so the
engine can be run with pickle files, in memory, or with no persistence at all.
"""

from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from .individual import Individual
from .ranking import Fronts


@dataclass
class CheckpointState:
    """Everything needed to resume a run."""

    iteration: int
    pop: List[Individual]
    fronts: Fronts
    pareto_history: List[List[Individual]] = field(default_factory=list)


class CheckpointRepository(ABC):
    """Port: load/save optimisation state."""

    @abstractmethod
    def load(self) -> Optional[CheckpointState]:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def save(self, state: CheckpointState) -> None:  # pragma: no cover
        raise NotImplementedError


class NullCheckpointRepository(CheckpointRepository):
    """No-op repository (runs always start fresh, nothing is persisted)."""

    def load(self) -> Optional[CheckpointState]:
        return None

    def save(self, state: CheckpointState) -> None:
        return None


class PickleCheckpointRepository(CheckpointRepository):
    """Persist state to a pickle file."""

    def __init__(self, path: str) -> None:
        self._path = path

    def load(self) -> Optional[CheckpointState]:
        if not os.path.exists(self._path):
            return None
        with open(self._path, "rb") as handle:
            return pickle.load(handle)

    def save(self, state: CheckpointState) -> None:
        directory = os.path.dirname(self._path)
        if directory and directory != ".":
            os.makedirs(directory, exist_ok=True)
        with open(self._path, "wb") as handle:
            pickle.dump(state, handle)
