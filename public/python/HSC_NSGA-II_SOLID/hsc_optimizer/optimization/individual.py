"""Population member used by the NSGA-II engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..domain.chromosome import Chromosome


@dataclass
class Individual:
    """A chromosome plus the bookkeeping NSGA-II attaches to it."""

    chromosome: Optional[Chromosome] = None
    cost: Optional[np.ndarray] = None              # raw objective vector
    normal_cost: Optional[np.ndarray] = None       # normalised objective vector
    rank: Optional[int] = None
    crowding_distance: float = 0.0
    constraint_violation: float = 0.0

    @property
    def is_evaluated(self) -> bool:
        return self.cost is not None
