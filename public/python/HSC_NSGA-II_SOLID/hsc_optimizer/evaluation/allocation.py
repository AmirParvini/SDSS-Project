"""Shelter allocation service.

Pure relocation of ``Main.allocate_population_under_capacity`` into a focused
service that depends only on :class:`ProblemData`.  Behaviour (including the
proportional split, last-bucket remainder and shortage accounting) is preserved
exactly.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List

from ..domain.problem_data import ProblemData
from ..domain.results import AllocationResult


class ShelterAllocationService:
    """Allocate displaced (homeless) population to the selected shelters."""

    def __init__(self, problem: ProblemData) -> None:
        self._problem = problem

    def allocate(self, da_ec: Dict[int, List[int]]) -> AllocationResult:
        problem = self._problem

        demand = {ec_id: 0 for ec_list in da_ec.values() for ec_id in ec_list}
        allocations: Dict[int, Dict[int, float]] = {da_id: {} for da_id in da_ec}
        # مقدار اولیه کمبودها را اعشاری (0.0) در نظر می‌گیریم تا خطای گرد کردن در طول حلقه رخ ندهد
        ec_shortages = {ec_id: 0.0 for ec_list in da_ec.values() for ec_id in ec_list}

        total_shortage = 0
        total_weighted_dist = 0
        remaining_capacity = copy.deepcopy(problem.capacity.shelter)

        # Capacity-aware allocation.
        for da_id, ec_id_list in da_ec.items():
            homeless_count = problem.homeless[f'{da_id}']
            
            # استفاده از max(0, ...) برای اطمینان از اینکه ظرفیت‌های منفی احتمالی ناشی از محاسبات دیگر وارد محاسبات نشوند
            available_caps = [max(0, remaining_capacity[f'{v}']) for v in ec_id_list]
            total_available = sum(available_caps)

            remaining_pop = homeless_count

            if total_available >= homeless_count and total_available > 0:
                for i, ec_id in enumerate(ec_id_list):
                    is_last = i == len(ec_id_list) - 1
                    
                    if is_last:
                        # در مرکز آخر، بیشترین مقدار ممکن که ظرفیتش اجازه می‌دهد را تخصیص می‌دهیم
                        alloc = min(remaining_pop, max(0, remaining_capacity[f'{ec_id}']))
                    else:
                        # برای سایر مراکز، سهم متناسب را محاسبه کرده و مطمئن می‌شویم از ظرفیت واقعی بیشتر نشود
                        calculated_alloc = round(homeless_count * (available_caps[i] / total_available))
                        alloc = min(calculated_alloc, max(0, remaining_capacity[f'{ec_id}']))
                    
                    allocations[da_id][ec_id] = alloc
                    dist = problem.distance.da_to_ec[f"{da_id},{ec_id}"]
                    remaining_capacity[f'{ec_id}'] -= alloc
                    demand[ec_id] += math.ceil(alloc / 5)
                    total_weighted_dist += alloc * dist
                    remaining_pop -= alloc
            else:
                # اگر ظرفیت کل کمتر از جمعیت بود، تمام ظرفیت باقی‌مانده را تخصیص می‌دهیم
                for ec_id in ec_id_list:
                    alloc = max(0, remaining_capacity[f'{ec_id}'])
                    allocations[da_id][ec_id] = alloc
                    dist = problem.distance.da_to_ec[f"{da_id},{ec_id}"]
                    remaining_capacity[f'{ec_id}'] -= alloc
                    demand[ec_id] += math.ceil(alloc / 5)
                    total_weighted_dist += alloc * dist
                    remaining_pop -= alloc
                
            # -- بخش اصلاح شده: ثبت دقیق و پویای کمبود هر مرکز --
                # اگر پس از تخصیص، هنوز جمعیتی از این منطقه باقی مانده باشد:
                if remaining_pop > 0:
                    total_shortage += remaining_pop
                    
                    # توزیع کسریِ این منطقه بین مراکز متصل به آن، بر اساس نسبت ظرفیت اولیه‌شان
                    initial_caps = [max(0, problem.capacity.shelter[f'{v}']) for v in ec_id_list]
                    total_init = sum(initial_caps)
                    
                    for i, ec_id in enumerate(ec_id_list):
                        if total_init > 0:
                            ec_shortages[ec_id] += remaining_pop * (initial_caps[i] / total_init)
                        else:
                            # حالت استثنایی که ظرفیت اولیه همه مراکز صفر باشد
                            ec_shortages[ec_id] += remaining_pop / len(ec_id_list)

            # تبدیل مقادیر نهایی کمبود به عدد صحیح (رفع خطاهای جزئی اعشاری)
            ec_shortages = {k: round(v) for k, v in ec_shortages.items()}

        return AllocationResult(
            demand=demand,
            total_shortage=total_shortage,
            weighted_distance=total_weighted_dist,
            allocations=allocations,
            ec_shortages=ec_shortages,
        )
