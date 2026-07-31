"""Damage-area -> shelter assignment.

The "part 3" decoding step (matching active shelters to damaged areas, cycling
the shorter list to cover the longer one) was duplicated verbatim in both the
cost function and the decoder.  It now lives in one place.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Dict, List

from ..domain.chromosome import Chromosome
from ..domain.problem_data import ProblemData


class DamageShelterAssigner:
    """Map each damaged area to the shelters that serve it."""

    def __init__(self, problem: ProblemData) -> None:
        self._problem = problem

    def assign(self, chromosome: Chromosome) -> Dict[int, List[int]]:
        ec_id = self._problem.ec_id

        # پیدا کردن پناهگاه‌های انتخاب شده بدون تکرار (با فرض اینکه در کروموزوم یکتا هستند)
        selected_shelters = [
            ec_id[idx] for idx, x in enumerate(chromosome.dc_to_shelter) if x != 0
        ]
        n_shelters = len(selected_shelters)
        damaged_areas = [x for x in chromosome.damage_to_shelter if x != 0]

        # هم‌طول کردن دو لیست با متد قبلی شما
        if len(selected_shelters) > len(damaged_areas):
            damaged_areas = list(
                itertools.islice(itertools.cycle(damaged_areas), len(selected_shelters))
            )
        elif len(selected_shelters) < len(damaged_areas):
            selected_shelters = list(
                itertools.islice(itertools.cycle(selected_shelters), len(damaged_areas))
            )

        da_ec_alloc: Dict[int, List[int]] = defaultdict(list)
        
        for idx, area in enumerate(damaged_areas):
            shelter = selected_shelters[idx]
            # اصلاح اصلی: فقط اگر پناهگاه قبلاً برای این منطقه ثبت نشده باشد، اضافه شود
            if shelter not in da_ec_alloc[area]:
                da_ec_alloc[area].append(shelter)
                
        return da_ec_alloc, n_shelters
