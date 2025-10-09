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
        
    def compute(self, pareto_front: np.ndarray, method: str = 'wfg') -> float:
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
        if len(pareto_front) == 0:
            return 0.0
            
        if pareto_front.shape[1] != 3:
            raise ValueError("این کلاس فقط برای مسائل 3-هدفه است")
        
        # تعیین reference point
        if self.reference_point is None:
            self.reference_point = self._auto_reference_point(pareto_front)
        
        # نرمال‌سازی
        normalized_pf = self._normalize(pareto_front, self.reference_point)
        
        # بررسی dominated solutions
        normalized_pf = self._filter_dominated(normalized_pf)
        
        # انتخاب روش محاسبه
        if method == 'wfg':
            hv = self._wfg_algorithm(normalized_pf)
        elif method == 'inclusion_exclusion':
            hv = self._inclusion_exclusion(normalized_pf)
        elif method == 'monte_carlo':
            hv = self._monte_carlo(normalized_pf, n_samples=100000)
        else:
            raise ValueError(f"روش نامعتبر: {method}")
        
        # de-normalize
        volume_factor = np.prod(self.reference_point)
        return hv * volume_factor
    
    def _auto_reference_point(self, pareto_front: np.ndarray, margin: float = 0.1) -> np.ndarray:
        """
        محاسبه خودکار reference point
        
        Reference point باید بدترین از همه جواب‌ها باشد
        margin: حاشیه اضافی (معمولاً 10%)
        """
        worst_point = np.max(pareto_front, axis=0)
        reference = worst_point * (1.0 + margin)
        return reference
    
    def _normalize(self, pareto_front: np.ndarray, reference_point: np.ndarray) -> np.ndarray:
        """
        نرمال‌سازی نسبت به reference point
        همه مقادیر باید بین 0 و 1 باشند
        """
        # بررسی اینکه همه نقاط dominated به reference point هستند
        if np.any(pareto_front >= reference_point):
            warnings.warn("برخی نقاط از reference point بدترند! Reference point تنظیم می‌شود.")
            reference_point = np.max(pareto_front, axis=0) * 1.1
            self.reference_point = reference_point
        
        normalized = pareto_front / reference_point
        return normalized
    
    def _filter_dominated(self, pareto_front: np.ndarray) -> np.ndarray:
        """
        حذف جواب‌های dominated از Pareto front
        """
        n = len(pareto_front)
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(i + 1, n):
                if is_dominated[j]:
                    continue
                
                # بررسی domination
                if self._dominates(pareto_front[i], pareto_front[j]):
                    is_dominated[j] = True
                elif self._dominates(pareto_front[j], pareto_front[i]):
                    is_dominated[i] = True
                    break
        
        return pareto_front[~is_dominated]
    
    def _dominates(self, point1: np.ndarray, point2: np.ndarray) -> bool:
        """بررسی اینکه point1 بر point2 dominate می‌کند"""
        return np.all(point1 <= point2) and np.any(point1 < point2)
    
    # ================== WFG Algorithm ==================
    
    def _wfg_algorithm(self, pareto_front: np.ndarray) -> float:
        """
        الگوریتم WFG برای محاسبه دقیق Hypervolume در 3D
        
        این الگوریتم بر اساس تجزیه فضا به hyperboxes است
        پیچیدگی زمانی: O(n^2) برای 3 هدف
        
        Algorithm:
        1. مرتب‌سازی بر اساس objective اول
        2. محاسبه exclusive hypervolume هر نقطه
        3. جمع کردن تمام exclusive hypervolumes
        """
        if len(pareto_front) == 0:
            return 0.0
        
        if len(pareto_front) == 1:
            return self._single_point_hv(pareto_front[0])
        
        # مرتب‌سازی بر اساس اولین هدف (ascending)
        sorted_indices = np.argsort(pareto_front[:, 0])
        sorted_pf = pareto_front[sorted_indices]
        
        hypervolume = 0.0
        
        # محاسبه exclusive hypervolume برای هر نقطه
        for i in range(len(sorted_pf)):
            point = sorted_pf[i]
            
            # پیدا کردن dominated space توسط نقاط بعدی
            # این نقاط می‌توانند exclusive space این نقطه را کاهش دهند
            dominated_points = sorted_pf[i+1:]
            
            if len(dominated_points) == 0:
                # آخرین نقطه - کل فضای باقی‌مانده
                exclusive_hv = self._calculate_box_volume(point, np.ones(3))
            else:
                # محاسبه exclusive hypervolume با استفاده از 2D slicing
                exclusive_hv = self._exclusive_hypervolume_3d(point, dominated_points)
            
            hypervolume += exclusive_hv
        
        return hypervolume
    
    def _exclusive_hypervolume_3d(self, point: np.ndarray, dominated_points: np.ndarray) -> float:
        """
        محاسبه exclusive hypervolume یک نقطه در فضای 3D
        با در نظر گرفتن نقاطی که ممکن است بخشی از فضا را بپوشانند
        
        این روش از 2D slicing استفاده می‌کند
        """
        # فضای کامل این نقطه تا reference point
        full_volume = self._calculate_box_volume(point, np.ones(3))
        
        # فیلتر نقاطی که می‌توانند exclusive space را کاهش دهند
        # فقط نقاطی که در بعد اول بزرگ‌تر هستند ولی در بعدهای دیگر کوچک‌ترند
        relevant_points = dominated_points[
            (dominated_points[:, 0] > point[0]) &
            (dominated_points[:, 1] <= point[1]) &
            (dominated_points[:, 2] <= point[2])
        ]
        
        if len(relevant_points) == 0:
            return full_volume
        
        # محاسبه حجم overlapping با استفاده از 2D hypervolume
        overlapping_volume = self._calculate_2d_overlap(point, relevant_points)
        
        return full_volume - overlapping_volume
    
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
    
    def _monte_carlo(self, pareto_front: np.ndarray, n_samples: int = 100000) -> float:
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
        if len(pareto_front) == 0:
            return 0.0
        
        # تولید نقاط تصادفی در فضای [0, 1]^3
        random_points = np.random.uniform(0, 1, size=(n_samples, 3))
        
        # شمارش نقاطی که توسط حداقل یک جواب dominate می‌شوند
        dominated_count = 0
        
        for point in random_points:
            # آیا این نقطه توسط حداقل یکی از جواب‌های pareto dominated می‌شود؟
            is_dominated = np.any(np.all(pareto_front <= point, axis=1))
            
            if is_dominated:
                dominated_count += 1
        
        # تخمین hypervolume
        hypervolume = dominated_count / n_samples
        
        return hypervolume


# ================== تابع کمکی برای استفاده آسان ==================

def calculate_hypervolume_3d(pareto_front: np.ndarray, 
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
    return hv_calculator.compute(pareto_front, method=method)
