"""Domain model for a humanitarian-logistics solution candidate.

The original code represented a chromosome as a bare ``list`` indexed by magic
numbers (``chromosome[0] .. chromosome[6]``).  That made every operator, the
cost function and the decoder hard to read and easy to break.  Here the seven
"parts" become explicitly named fields with a single source of truth for their
meaning, while still offering loss-less conversion to/from the legacy list form
so the model can interoperate with any code that still expects it.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class Chromosome:
    """A single candidate solution.

    Attributes
    ----------
    dc_to_shelter:
        Part 1 - for each shelter, the distribution centre assigned to it
        (``0`` means the shelter is not activated).
    shelter_flow_ratio:
        Part 2 - relief-package flow ratio (0..1) for each shelter.
    damage_to_shelter:
        Part 3 - damage-area assignment per shelter slot (the first
        ``n_damage_points`` entries are a permutation of the damage-area ids).
    severe_split:
        Part 4 - (n_damage_points x n_hospitals) probability split of severe
        injured from each damage area to each hospital.
    severe_ground_ratio:
        Part 5 - (n_damage_points x n_hospitals) share of severe injured moved
        by ground ambulance (the remainder goes by helicopter).
    moderate_split:
        Part 6 - (n_damage_points x (n_hospitals + n_temp_medical)) split of
        moderately injured to hospitals and temporary medical centres.
    moderate_ground_ratio:
        Part 7 - matching ground-ambulance share for the moderate split.
    """

    dc_to_shelter: List[int]
    shelter_flow_ratio: List[float]
    damage_to_shelter: List[int]
    severe_split: np.ndarray
    severe_ground_ratio: np.ndarray
    moderate_split: np.ndarray
    moderate_ground_ratio: np.ndarray

    # -- legacy interop ----------------------------------------------------
    def to_parts(self) -> list:
        """Return the legacy ``list``-of-seven representation."""
        return [
            self.dc_to_shelter,
            self.shelter_flow_ratio,
            self.damage_to_shelter,
            self.severe_split,
            self.severe_ground_ratio,
            self.moderate_split,
            self.moderate_ground_ratio,
        ]

    @classmethod
    def from_parts(cls, parts: Sequence) -> "Chromosome":
        """Build a :class:`Chromosome` from the legacy list representation."""
        return cls(
            dc_to_shelter=list(parts[0]),
            shelter_flow_ratio=list(parts[1]),
            damage_to_shelter=list(parts[2]),
            severe_split=np.asarray(parts[3], dtype=float),
            severe_ground_ratio=np.asarray(parts[4], dtype=float),
            moderate_split=np.asarray(parts[5], dtype=float),
            moderate_ground_ratio=np.asarray(parts[6], dtype=float),
        )

    def clone(self) -> "Chromosome":
        """Return a deep copy (operators must never mutate their inputs)."""
        return deepcopy(self)
