import numpy as np
from typing import List, Tuple, Optional
import warnings

class Hypervolume3D:
    """
    محاسبه دقیق Hypervolume برای مسائل سه‌هدفه
    
    الگوریتم‌های پیاده‌سازی شده:
    1. WFG Algorithm (While et al., 2012) - استاندارد طلایی
    2. Inclusion-Exclusion Method - برای تعداد کم جواب‌ها
    3. Monte Carlo Approximation - برای سرعت بالا
    
    References:
    - While, L., Bradstreet, L., & Barone, L. (2012). 
      "A fast way of calculating exact hypervolumes"
    """
    
    def __init__(self, reference_point: Optional[np.ndarray] = None):
        """
        Parameters:
        -----------
        reference_point: نقطه مرجع برای محاسبه hypervolume
                        اگر None باشد، به صورت خودکار محاسبه می‌شود
        """
        self.reference_point = reference_point
        
    def compute(self, pareto_fronts_list: np.ndarray, method: str = 'wfg') -> float:
        """
        محاسبه Hypervolume
        
        Parameters:
        -----------
        pareto_front: آرایه با shape (n_solutions, 3)
        method: 'wfg', 'inclusion_exclusion', یا 'monte_carlo'
        
        Returns:
        --------
        hypervolume: مقدار hypervolume
        """
        
        # نرمال‌سازی
        # normalized_pf = self._normalize(pareto_front, self.reference_point)
        
        # انتخاب روش محاسبه
        if method == 'wfg':
            hv = self._wfg_algorithm(pareto_fronts_list, ref_point=self.reference_point)
        elif method == 'inclusion_exclusion':
            hv = self._inclusion_exclusion(pareto_fronts_list)
        elif method == 'monte_carlo':
            hv = self._monte_carlo_normal(pareto_fronts_list=pareto_fronts_list, ref_point=self.reference_point, n_samples=100000)
        else:
            raise ValueError(f"روش نامعتبر: {method}")
        
        return hv
    
    
    def _normalize(self, pareto_front: np.ndarray, reference_point: np.ndarray) -> np.ndarray:
        """
        نرمال‌سازی نسبت به reference point
        همه مقادیر باید بین 0 و 1 باشند
        """
        # بررسی اینکه همه نقاط dominated به reference point هستند
        # if np.any(pareto_front > reference_point):
        #     warnings.warn("برخی نقاط از reference point بدترند! Reference point تنظیم می‌شود.")
        #     reference_point = np.max(pareto_front, axis=0) * 1.1
        #     self.reference_point = reference_point
        normalized = pareto_front / reference_point
        return normalized
    
    # ================== WFG Algorithm ==================
    
    def _wfg_algorithm(self, pareto_fronts_list: np.ndarray, ref_point) -> List:
        """
        الگوریتم WFG برای محاسبه دقیق Hypervolume در 3D
        
        این الگوریتم بر اساس تجزیه فضا به hyperboxes است
        پیچیدگی زمانی: O(n^2) برای 3 هدف
        
        Algorithm:
        1. مرتب‌سازی بر اساس objective اول
        2. محاسبه exclusive hypervolume هر نقطه
        3. جمع کردن تمام exclusive hypervolumes
        """
        ref = np.asarray(ref_point, dtype=float)
        hv: list[float] = []

        for pareto_front in pareto_fronts_list:

            # اگر پرتو خالی است
            if pareto_front is None or len(pareto_front) == 0:
                hv.append(0.0)
                continue

            pf = np.asarray(pareto_front, dtype=float)

            # اگر فقط یک نقطه دارد → حجم همان یک جعبه
            if pf.shape[0] == 1:
                hv.append(self._calculate_box_volume(pf[0], ref))
                continue
            
            # مرتب‌سازی بر اساس اولین هدف (ascending)
            pf = np.array(pareto_front)
            sorted_indices = np.argsort(pf[:, 0])
            sorted_pf = pf[sorted_indices]
            
            # محاسبه سهم اکسکلوزیو هر نقطه نسبت به بقیه‌ی نقاط همان پرتو
            hypervolume = 0.0
            for i in range(sorted_pf.shape[0]):
                point = sorted_pf[i]

                # مجموعه‌ی سایر نقاط (F \ {point})
                other_points = np.delete(sorted_pf, i, axis=0)

                # HV_excl(point | other_points)
                exclusive_hv = self._exclusive_hypervolume_3d(point, other_points)

                hypervolume += exclusive_hv
                
            hv.append(hypervolume)
            
        return hv
    
    def _exclusive_hypervolume_3d(self, point: np.ndarray, dominated_points: np.ndarray) -> float:
        """
        محاسبه دقیق exclusive hypervolume یک نقطه در فضای 3D
        
        تعریف:
            HV_excl(p | S) = HV({p} ∪ S) - HV(S)
        
        این پیاده‌سازی از یک الگوریتم grid-based برای محاسبه‌ی 
        union حجم جعبه‌های [point, reference] استفاده می‌کند.
        از نظر ریاضی دقیق است، ولی پیچیدگی زمانی آن از WFG کلاسیک بیشتر است
        (برای تعداد نقاط کم/متوسط در 3D معمولاً قابل قبول است).
        """
        import numpy as np

        # نقطه‌ی مرجع را از کلاس بگیر؛ اگر نداشتی، فرض کن [1,1,1]
        if hasattr(self, "reference_point"):
            ref = np.asarray(self.reference_point, dtype=float)
        else:
            ref = np.ones(3, dtype=float)

        point = np.asarray(point, dtype=float)
        dominated_points = np.asarray(dominated_points, dtype=float)

        # اگر هیچ نقطه‌ی دیگری نیست، سهم اکسکلوزیو همان حجم جعبه‌ی خودش است
        if dominated_points.size == 0:
            return self._calculate_box_volume(point, ref)

        # --- تابع کمکی: محاسبه hypervolume دقیق 3D برای مجموعه‌ای از نقاط ---
        def _hv3d(points: np.ndarray) -> float:
            """
            محاسبه‌ی دقیق hypervolume در 3D برای مجموعه‌ای از نقاط (minimization)
            هر نقطه p، جعبه [p, ref] را تعریف می‌کند.
            HV = حجم union همه‌ی این جعبه‌ها.
            """
            pts = np.asarray(points, dtype=float)

            # نقاطی که بیرون از ref هستند کمکی نمی‌کنند
            mask = np.all(pts < ref, axis=1)
            pts = pts[mask]

            if pts.shape[0] == 0:
                return 0.0

            # مختصات متمایز در هر بعد (به‌علاوه‌ی ref)
            xs = sorted(set(pts[:, 0].tolist() + [ref[0]]))
            ys = sorted(set(pts[:, 1].tolist() + [ref[1]]))
            zs = sorted(set(pts[:, 2].tolist() + [ref[2]]))

            hv = 0.0

            # روی هر سلول کوچک شبکه حلقه می‌زنیم و اگر این سلول داخل حداقل یک جعبه بود، حجمش را اضافه می‌کنیم
            for i in range(len(xs) - 1):
                xL, xR = xs[i], xs[i + 1]
                xmid = 0.5 * (xL + xR)

                for j in range(len(ys) - 1):
                    yL, yR = ys[j], ys[j + 1]
                    ymid = 0.5 * (yL + yR)

                    for k in range(len(zs) - 1):
                        zL, zR = zs[k], zs[k + 1]
                        zmid = 0.5 * (zL + zR)

                        center = np.array([xmid, ymid, zmid], dtype=float)

                        # اگر این نقطه‌ی مرکزی توسط حداقل یک نقطه p پوشش داده شود
                        # (minimization → center >= p در همه‌ی ابعاد)
                        covered = False
                        for p in pts:
                            if np.all(center >= p):
                                covered = True
                                break

                        if covered:
                            hv += (xR - xL) * (yR - yL) * (zR - zL)

            return hv

        # HV({p} ∪ S)
        all_points = np.vstack([point, dominated_points])
        hv_with = _hv3d(all_points)

        # HV(S)
        hv_without = _hv3d(dominated_points)

        exclusive = hv_with - hv_without

        # به خاطر خطاهای عددی ممکن است مقدار خیلی کوچک منفی بشود → 0 کن
        if exclusive < 0.0 and exclusive > -1e-12:
            exclusive = 0.0

        return exclusive
    
    def _calculate_2d_overlap(self, point: np.ndarray, dominated_points: np.ndarray) -> float:
        """
        محاسبه حجم overlapping در projection به صفحات 2D
        
        استراتژی: محاسبه hypervolume 2D در صفحه (obj2, obj3)
        برای هر slice در obj1
        """
        if len(dominated_points) == 0:
            return 0.0
        
        # مرتب‌سازی بر اساس obj1
        sorted_points = dominated_points[np.argsort(dominated_points[:, 0])]
        
        total_overlap = 0.0
        prev_x = point[0]
        
        for i, dom_point in enumerate(sorted_points):
            # عرض slice در dimension اول
            slice_width = dom_point[0] - prev_x
            
            if slice_width <= 0:
                continue
            
            # نقاط فعال در این slice (تمام نقاط تا i)
            active_points = sorted_points[:i+1, 1:]  # فقط obj2 و obj3
            point_2d = point[1:]  # obj2 و obj3 از نقطه اصلی
            
            # محاسبه 2D hypervolume
            hv_2d = self._hypervolume_2d(active_points, point_2d)
            
            total_overlap += slice_width * hv_2d
            prev_x = dom_point[0]
        
        return total_overlap
    
    def _hypervolume_2d(self, points_2d: np.ndarray, reference_2d: np.ndarray) -> float:
        """
        محاسبه دقیق Hypervolume در 2D
        
        در 2D، hypervolume به سادگی مجموع مساحت‌های مستطیل‌های non-overlapping است
        """
        if len(points_2d) == 0:
            return (1.0 - reference_2d[0]) * (1.0 - reference_2d[1])
        
        # فیلتر نقاط dominated
        points_2d = self._filter_dominated_2d(points_2d, reference_2d)
        
        if len(points_2d) == 0:
            return (1.0 - reference_2d[0]) * (1.0 - reference_2d[1])
        
        # مرتب‌سازی بر اساس اولین بعد
        sorted_points = points_2d[np.argsort(points_2d[:, 0])]
        
        hv = 0.0
        prev_y = reference_2d[1]
        
        for point in sorted_points:
            if point[1] < prev_y:
                width = 1.0 - point[0]
                height = prev_y - point[1]
                hv += width * height
                prev_y = point[1]
        
        return hv
    
    def _filter_dominated_2d(self, points_2d: np.ndarray, reference_2d: np.ndarray) -> np.ndarray:
        """حذف نقاط dominated در 2D و نقاطی که از reference بدترند"""
        # حذف نقاط بدتر از reference
        valid_mask = (points_2d[:, 0] <= reference_2d[0]) & (points_2d[:, 1] <= reference_2d[1])
        points_2d = points_2d[valid_mask]
        
        if len(points_2d) <= 1:
            return points_2d
        
        # حذف dominated
        n = len(points_2d)
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(i + 1, n):
                if is_dominated[j]:
                    continue
                if np.all(points_2d[i] <= points_2d[j]) and np.any(points_2d[i] < points_2d[j]):
                    is_dominated[j] = True
                elif np.all(points_2d[j] <= points_2d[i]) and np.any(points_2d[j] < points_2d[i]):
                    is_dominated[i] = True
                    break
        
        return points_2d[~is_dominated]
    
    def _calculate_box_volume(self, point: np.ndarray, reference: np.ndarray) -> float:
        """محاسبه حجم box از point تا reference"""
        diff = reference - point
        if np.any(diff < 0):
            return 0.0
        return np.prod(diff)
    
    def _single_point_hv(self, point: np.ndarray) -> float:
        """Hypervolume برای یک نقطه منفرد"""
        return np.prod(1.0 - point)
    
    # ================== Inclusion-Exclusion Method ==================
    
    def _inclusion_exclusion(self, pareto_front: np.ndarray) -> float:
        """
        روش Inclusion-Exclusion برای محاسبه Hypervolume
        
        این روش برای تعداد کم جواب‌ها (< 10) کارآمد است
        فرمول: HV = Σ V(i) - Σ V(i,j) + Σ V(i,j,k) - ...
        
        پیچیدگی: O(2^n) - برای n کوچک قابل استفاده
        """
        n = len(pareto_front)
        
        if n == 0:
            return 0.0
        
        if n > 15:
            warnings.warn("Inclusion-Exclusion برای تعداد زیاد جواب کند است. از WFG استفاده کنید.")
        
        hypervolume = 0.0
        
        # تولید تمام subsets غیرخالی
        from itertools import combinations
        
        for size in range(1, n + 1):
            sign = (-1) ** (size + 1)
            
            for subset_indices in combinations(range(n), size):
                subset = pareto_front[list(subset_indices)]
                
                # پیدا کردن intersection box
                worst_point = np.max(subset, axis=0)
                
                # حجم box
                volume = np.prod(1.0 - worst_point)
                hypervolume += sign * volume
        
        return hypervolume
    
    # ================== Monte Carlo Approximation ==================
    
    def _monte_carlo(self, pareto_fronts_list: np.ndarray,ref_point: np.ndarray,
                   ideal_point: np.ndarray = None, n_samples: int = 100000) -> list:
        """
        تقریب Monte Carlo برای Hypervolume
        
        مناسب برای:
        - تعداد زیاد جواب‌ها
        - نیاز به سرعت بالا
        - دقت کامل لازم نیست
        
        Parameters:
        -----------
        n_samples: تعداد نمونه‌های تصادفی
        
        Returns:
        --------
        تقریب hypervolume
        """
        # sample in the same cube
        rng = np.random.default_rng(seed=42)
        rnd = rng.uniform(0.0, ref_point, size=(n_samples, np.array(pareto_fronts_list[0]).shape[1]))
        
        # 1) ideal point
        # if ideal_point is None:
        #     all_costs = [cost for pareto_front in pareto_fronts_list for cost in pareto_front]
        #     ideal_point = np.min(all_costs, axis=0)

        # 2) guard: ref must dominate (be worse than) all points
        # if any ref <= ideal in a dim, expand it a bit
        # span = ref_point - ideal_point
        hv = []
        pareto_fronts_list = [pareto_fronts_list[-1]]
        for pareto_front in pareto_fronts_list:
            if len(pareto_front) == 0:
                hv.append(0.0)
                continue
            # 3) normalize PF into [0,1]^3
            # pf_norm = (pareto_front - ideal_point) / span
            # pf_norm = np.clip(pf_norm, 0.0, 1.0)

            # 4) dominance test (vectorized):
            # a rnd point x is dominated if exists s in PF with s <= x (componentwise)
            # shape tricks: compare all rnd to all pf
            # pf_norm[None, :, :] -> (1, N, M); rnd[:, None, :] -> (S, 1, M)
            pareto_front = np.array(pareto_front)
            dominated_by_any = np.all(pareto_front[None, :, :] <= rnd[:, None, :], axis=2).any(axis=1)
            count = sum(d for d in dominated_by_any if d == True)
            volume_box = np.prod(ref_point)
            hv_estimate = volume_box * (count / n_samples)
            hv.append(hv_estimate)  # in [0,1]
            
        return hv
    
    def _monte_carlo_normal(self, pareto_fronts_list: np.ndarray, ref_point: np.ndarray,
                 ideal_point: np.ndarray = None, n_samples: int = 100000) -> list:
        """
        تقریب Monte Carlo برای Hypervolume (نرمال‌شده)

        مناسب برای:
        - تعداد زیاد جواب‌ها
        - نیاز به سرعت بالا
        - دقت کامل لازم نیست

        Parameters:
        -----------
        n_samples: تعداد نمونه‌های تصادفی

        Returns:
        --------
        تقریب hypervolume نرمال‌شده در بازه [0, 1]
        """
        rng = np.random.default_rng(seed=42)
        rnd = rng.uniform(
            0.0,
            ref_point,
            size=(n_samples, np.array(pareto_fronts_list[0]).shape[1])
        )

        # حجم کل جعبه مرجع (مخرج نرمال‌سازی)
        volume_box = np.prod(ref_point)

        hv = []
        
        for pareto_front in pareto_fronts_list:
            if len(pareto_front) == 0:
                hv.append(0.0)
                continue

            pareto_front = np.array(pareto_front)

            # بررسی دومینانس: آیا هر نقطه تصادفی توسط حداقل یک نقطه PF دومینیت می‌شود؟
            # pareto_front[None, :, :] -> (1, N, M)
            # rnd[:, None, :]          -> (S, 1, M)
            dominated_by_any = np.all(
                pareto_front[None, :, :] <= rnd[:, None, :],
                axis=2
            ).any(axis=1)

            count = np.sum(dominated_by_any)

            # نرمال‌سازی: تقسیم بر حجم کل جعبه
            hv_normalized = count / n_samples  # معادل: (volume_box * count/n_samples) / volume_box

            hv.append(hv_normalized)

        return hv


# ================== تابع کمکی برای استفاده آسان ==================

def calculate_hypervolume_3d(pareto_fronts_list: np.ndarray, 
                             reference_point: Optional[np.ndarray] = None,
                             method: str = 'wfg') -> float:
    """
    تابع wrapper ساده برای محاسبه Hypervolume
    
    Parameters:
    -----------
    pareto_front: آرایه numpy با shape (n_solutions, 3)
    reference_point: نقطه مرجع (optional)
    method: 'wfg' (دقیق), 'inclusion_exclusion' (کوچک), 'monte_carlo' (سریع)
    
    Returns:
    --------
    hypervolume: مقدار hypervolume
    
    Examples:
    ---------
    >>> pf = np.array([[1, 2, 3], [2, 1, 3], [3, 3, 1]])
    >>> hv = calculate_hypervolume_3d(pf)
    >>> print(f"Hypervolume: {hv}")
    """
    hv_calculator = Hypervolume3D(reference_point)
    return hv_calculator.compute(pareto_fronts_list, method=method)
