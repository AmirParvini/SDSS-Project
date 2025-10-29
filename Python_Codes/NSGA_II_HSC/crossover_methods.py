import numpy as np
import random
from copy import deepcopy
from typing import List, Tuple

class CrossoverMethods:
    """
    کلاس جامع برای متدهای مختلف Crossover (تقاطع/ترکیب)
    
    این کلاس شامل تمام روش‌های رایج crossover برای انواع مختلف داده‌ها است:
    - List-based crossover
    - Matrix-based crossover
    - Permutation-based crossover
    - Continuous value crossover
    
    هر متد دو parent می‌گیرد و دو child برمی‌گرداند.
    """
    
    def __init__(self):
        """Constructor"""
        pass
    
    # ==================== LIST-BASED CROSSOVER METHODS ====================
    
    @staticmethod
    def one_point_crossover_list(parent1: List, parent2: List) -> Tuple[List, List]:
        """
        One-Point Crossover برای لیست‌ها
        
        یک نقطه تقاطع تصادفی انتخاب می‌شود و عناصر بعد از آن نقطه بین والدین جابجا می‌شوند.
        
        Parameters:
        -----------
        parent1: لیست اول
        parent2: لیست دوم
        
        Returns:
        --------
        tuple: (child1, child2)
        
        Example:
        --------
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        point = 2
        child1 = [1, 2, 8, 9, 10]
        child2 = [6, 7, 3, 4, 5]
        """
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same length")
        
        if len(parent1) <= 1:
            return deepcopy(parent1), deepcopy(parent2)
        
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)
        
        # انتخاب نقطه تقاطع (بین 1 و len-1)
        point = random.randint(1, len(parent1) - 1)
        
        # جابجایی قسمت‌های بعد از نقطه تقاطع
        child1[point:] = parent2[point:]
        child2[point:] = parent1[point:]
        
        return child1, child2
    
    @staticmethod
    def two_point_crossover_list(parent1: List, parent2: List) -> Tuple[List, List]:
        """
        Two-Point Crossover برای لیست‌ها
        
        دو نقطه تقاطع تصادفی انتخاب می‌شود و قسمت بین این دو نقطه جابجا می‌شود.
        
        Parameters:
        -----------
        parent1: لیست اول
        parent2: لیست دوم
        
        Returns:
        --------
        tuple: (child1, child2)
        
        Example:
        --------
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        points = (1, 3)
        child1 = [1, 7, 8, 4, 5]
        child2 = [6, 2, 3, 9, 10]
        """
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same length")
        
        if len(parent1) <= 2:
            return deepcopy(parent1), deepcopy(parent2)
        
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)
        
        # انتخاب دو نقطه تقاطع
        point1, point2 = sorted(random.sample(range(1, len(parent1)), 2))
        
        # جابجایی قسمت بین دو نقطه
        child1[point1:point2] = parent2[point1:point2]
        child2[point1:point2] = parent1[point1:point2]
        
        return child1, child2
    
    @staticmethod
    def uniform_crossover_list(parent1: List, parent2: List, swap_prob: float = 0.5) -> Tuple[List, List]:
        """
        Uniform Crossover برای لیست‌ها
        
        برای هر موقعیت، با احتمال swap_prob عناصر بین والدین جابجا می‌شوند.
        
        Parameters:
        -----------
        parent1: لیست اول
        parent2: لیست دوم
        swap_prob: احتمال جابجایی هر عنصر (پیش‌فرض: 0.5)
        
        Returns:
        --------
        tuple: (child1, child2)
        """
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same length")
        
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)
        
        for i in range(len(parent1)):
            if random.random() < swap_prob:
                child1[i], child2[i] = child2[i], child1[i]
        
        return child1, child2
    
    @staticmethod
    def order_crossover_list(parent1: List, parent2: List) -> Tuple[List, List]:
        """
        Order Crossover (OX) برای permutation lists
        
        این روش برای لیست‌هایی که ترتیب مهم است و نباید عنصر تکراری داشته باشند.
        معمولاً برای مسائل TSP و مسیریابی استفاده می‌شود.
        
        Parameters:
        -----------
        parent1: لیست permutation اول
        parent2: لیست permutation دوم
        
        Returns:
        --------
        tuple: (child1, child2)
        """
        size = len(parent1)
        if size <= 2:
            return deepcopy(parent1), deepcopy(parent2)
        
        child1 = [None] * size
        child2 = [None] * size
        
        # انتخاب دو نقطه تقاطع
        start, end = sorted(random.sample(range(size), 2))
        
        # کپی قسمت بین دو نقطه
        child1[start:end+1] = parent1[start:end+1]
        child2[start:end+1] = parent2[start:end+1]
        
        # پر کردن بقیه موقعیت‌ها با حفظ ترتیب
        # برای child1 از parent2 استفاده می‌کنیم
        p2_idx = 0
        for i in range(size):
            if child1[i] is None:
                while parent2[p2_idx] in child1[start:end+1]:
                    p2_idx += 1
                child1[i] = parent2[p2_idx]
                p2_idx += 1
        
        # برای child2 از parent1 استفاده می‌کنیم
        p1_idx = 0
        for i in range(size):
            if child2[i] is None:
                while parent1[p1_idx] in child2[start:end+1]:
                    p1_idx += 1
                child2[i] = parent1[p1_idx]
                p1_idx += 1
        
        return child1, child2
    
    @staticmethod
    def arithmetic_crossover_list(parent1: List[float], parent2: List[float], 
                                  alpha: float = None) -> Tuple[List[float], List[float]]:
        """
        Arithmetic Crossover برای مقادیر پیوسته
        
        ترکیب خطی از دو والد:
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        
        Parameters:
        -----------
        parent1: لیست مقادیر پیوسته اول
        parent2: لیست مقادیر پیوسته دوم
        alpha: ضریب ترکیب (اگر None باشد، تصادفی انتخاب می‌شود)
        
        Returns:
        --------
        tuple: (child1, child2)
        """
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same length")
        
        if alpha is None:
            alpha = random.random()
        
        alpha = np.clip(alpha, 0, 1)
        
        child1 = [alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2)]
        child2 = [(1 - alpha) * p1 + alpha * p2 for p1, p2 in zip(parent1, parent2)]
        
        return child1, child2
    
    @staticmethod
    def blend_crossover_list(parent1: List[float], parent2: List[float], 
                            alpha: float = 0.5) -> Tuple[List[float], List[float]]:
        """
        Blend Crossover (BLX-α) برای مقادیر پیوسته
        
        هر عنصر child از بازه [min - α*d, max + α*d] انتخاب می‌شود
        که d = |parent1[i] - parent2[i]|
        
        Parameters:
        -----------
        parent1: لیست مقادیر پیوسته اول
        parent2: لیست مقادیر پیوسته دوم
        alpha: پارامتر blend (پیش‌فرض: 0.5)
        
        Returns:
        --------
        tuple: (child1, child2)
        """
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same length")
        
        child1 = []
        child2 = []
        
        for p1, p2 in zip(parent1, parent2):
            min_val = min(p1, p2)
            max_val = max(p1, p2)
            d = max_val - min_val
            
            # محدوده برای child1
            low = min_val - alpha * d
            high = max_val + alpha * d
            
            child1.append(random.uniform(low, high))
            child2.append(random.uniform(low, high))
        
        return child1, child2
    
    # ==================== MATRIX-BASED CROSSOVER METHODS ====================
    
    @staticmethod
    def one_point_crossover_matrix(parent1: np.ndarray, parent2: np.ndarray, 
                                   axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        One-Point Crossover برای ماتریس‌ها
        
        Parameters:
        -----------
        parent1: ماتریس اول
        parent2: ماتریس دوم
        axis: محور تقاطع (0 برای سطر، 1 برای ستون)
        
        Returns:
        --------
        tuple: (child1, child2)
        """
        if parent1.shape != parent2.shape:
            raise ValueError("Parents must have the same shape")
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        if axis == 0:  # تقاطع بر اساس سطر
            if parent1.shape[0] <= 1:
                return child1, child2
            point = random.randint(1, parent1.shape[0] - 1)
            child1[point:, :] = parent2[point:, :]
            child2[point:, :] = parent1[point:, :]
        else:  # تقاطع بر اساس ستون
            if parent1.shape[1] <= 1:
                return child1, child2
            point = random.randint(1, parent1.shape[1] - 1)
            child1[:, point:] = parent2[:, point:]
            child2[:, point:] = parent1[:, point:]
        
        return child1, child2
    
    @staticmethod
    def two_point_crossover_matrix(parent1: np.ndarray, parent2: np.ndarray,
                                   axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Two-Point Crossover برای ماتریس‌ها
        
        Parameters:
        -----------
        parent1: ماتریس اول
        parent2: ماتریس دوم
        axis: محور تقاطع (0 برای سطر، 1 برای ستون)
        
        Returns:
        --------
        tuple: (child1, child2)
        """
        if parent1.shape != parent2.shape:
            raise ValueError("Parents must have the same shape")
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        if axis == 0:  # تقاطع بر اساس سطر
            if parent1.shape[0] <= 2:
                return child1, child2
            point1, point2 = sorted(random.sample(range(1, parent1.shape[0]), 2))
            child1[point1:point2, :] = parent2[point1:point2, :]
            child2[point1:point2, :] = parent1[point1:point2, :]
        else:  # تقاطع بر اساس ستون
            if parent1.shape[1] <= 2:
                return child1, child2
            point1, point2 = sorted(random.sample(range(1, parent1.shape[1]), 2))
            child1[:, point1:point2] = parent2[:, point1:point2]
            child2[:, point1:point2] = parent1[:, point1:point2]
        
        return child1, child2
    
    @staticmethod
    def uniform_crossover_matrix(parent1: np.ndarray, parent2: np.ndarray,
                                swap_prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uniform Crossover برای ماتریس‌ها
        
        برای هر عنصر ماتریس، با احتمال swap_prob جابجا می‌شود.
        
        Parameters:
        -----------
        parent1: ماتریس اول
        parent2: ماتریس دوم
        swap_prob: احتمال جابجایی هر عنصر
        
        Returns:
        --------
        tuple: (child1, child2)
        """
        if parent1.shape != parent2.shape:
            raise ValueError("Parents must have the same shape")
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # ایجاد ماسک تصادفی
        mask = np.random.rand(*parent1.shape) < swap_prob
        
        # جابجایی عناصر بر اساس ماسک
        child1[mask] = parent2[mask]
        child2[mask] = parent1[mask]
        
        return child1, child2
    
    @staticmethod
    def block_crossover_matrix(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Block Crossover برای ماتریس‌ها
        
        یک بلوک مستطیلی تصادفی انتخاب و بین والدین جابجا می‌شود.
        
        Parameters:
        -----------
        parent1: ماتریس اول
        parent2: ماتریس دوم
        
        Returns:
        --------
        tuple: (child1, child2)
        """
        if parent1.shape != parent2.shape:
            raise ValueError("Parents must have the same shape")
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        rows, cols = parent1.shape
        
        if rows <= 1 or cols <= 1:
            return child1, child2
        
        # انتخاب نقاط گوشه بلوک
        r1, r2 = sorted(random.sample(range(rows), 2))
        c1, c2 = sorted(random.sample(range(cols), 2))
        
        # جابجایی بلوک
        child1[r1:r2+1, c1:c2+1] = parent2[r1:r2+1, c1:c2+1]
        child2[r1:r2+1, c1:c2+1] = parent1[r1:r2+1, c1:c2+1]
        
        return child1, child2
    
    @staticmethod
    def row_wise_crossover_matrix(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Row-wise Crossover برای ماتریس‌ها
        
        برای هر سطر، تصمیم می‌گیرد که از کدام والد کپی شود.
        
        Parameters:
        -----------
        parent1: ماتریس اول
        parent2: ماتریس دوم
        
        Returns:
        --------
        tuple: (child1, child2)
        """
        if parent1.shape != parent2.shape:
            raise ValueError("Parents must have the same shape")
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(parent1.shape[0]):
            if random.random() < 0.5:
                child1[i, :] = parent2[i, :]
                child2[i, :] = parent1[i, :]
        
        return child1, child2
    
    @staticmethod
    def arithmetic_crossover_matrix(parent1: np.ndarray, parent2: np.ndarray,
                                   alpha: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Arithmetic Crossover برای ماتریس‌ها با مقادیر پیوسته
        
        Parameters:
        -----------
        parent1: ماتریس اول
        parent2: ماتریس دوم
        alpha: ضریب ترکیب (اگر None باشد، تصادفی انتخاب می‌شود)
        
        Returns:
        --------
        tuple: (child1, child2)
        """
        if parent1.shape != parent2.shape:
            raise ValueError("Parents must have the same shape")
        
        if alpha is None:
            alpha = random.random()
        
        alpha = np.clip(alpha, 0, 1)
        
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        
        return child1, child2
    
    # ==================== MIXED/HYBRID CROSSOVER METHODS ====================
    
    @staticmethod
    def multi_point_crossover_list(parent1: List, parent2: List, 
                                   n_points: int = 3) -> Tuple[List, List]:
        """
        Multi-Point Crossover برای لیست‌ها
        
        چندین نقطه تقاطع انتخاب و قسمت‌های بین آن‌ها به صورت متناوب جابجا می‌شوند.
        
        Parameters:
        -----------
        parent1: لیست اول
        parent2: لیست دوم
        n_points: تعداد نقاط تقاطع
        
        Returns:
        --------
        tuple: (child1, child2)
        """
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same length")
        
        if len(parent1) <= n_points:
            return CrossoverMethods.one_point_crossover_list(parent1, parent2)
        
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)
        
        # انتخاب نقاط تقاطع
        points = sorted(random.sample(range(1, len(parent1)), min(n_points, len(parent1) - 1)))
        points = [0] + points + [len(parent1)]
        
        # جابجایی قسمت‌های متناوب
        for i in range(len(points) - 1):
            if i % 2 == 1:  # قسمت‌های فرد را جابجا کن
                child1[points[i]:points[i+1]] = parent2[points[i]:points[i+1]]
                child2[points[i]:points[i+1]] = parent1[points[i]:points[i+1]]
        
        return child1, child2
    
    @staticmethod
    def partially_mapped_crossover(parent1: List, parent2: List) -> Tuple[List, List]:
        """
        Partially Mapped Crossover (PMX) برای permutations
        
        مناسب برای مسائلی که ترتیب عناصر مهم است و نباید تکرار داشته باشند.
        
        Parameters:
        -----------
        parent1: لیست permutation اول
        parent2: لیست permutation دوم
        
        Returns:
        --------
        tuple: (child1, child2)
        """
        size = len(parent1)
        if size <= 2:
            return deepcopy(parent1), deepcopy(parent2)
        
        # انتخاب دو نقطه
        point1, point2 = sorted(random.sample(range(size), 2))
        
        # ایجاد children
        child1 = [None] * size
        child2 = [None] * size
        
        # کپی بخش میانی
        child1[point1:point2+1] = parent1[point1:point2+1]
        child2[point1:point2+1] = parent2[point1:point2+1]
        
        # ایجاد mapping
        def fill_child(child, parent_main, parent_other, p1, p2):
            # Create the mapping between the crossover segments: parent_other[i] -> parent_main[i]
            # This mapping is used to resolve conflicts in the outside sections.
            mapping_dict = {}
            for i in range(p1, p2 + 1):
                mapping_dict[parent_other[i]] = parent_main[i]
                
            for i in range(size):
                # Only process elements outside the crossover segment
                if i < p1 or i > p2:
                    # Get the gene from the corresponding position in the other parent
                    value = parent_other[i]
                    
                    # Resolve conflicts using the mapping chain
                    # If 'value' is present in the main parent's middle segment (which is copied to the child), 
                    # we must replace it using the mapping rule.
                    while value in mapping_dict:
                        value = mapping_dict[value] # Follow the chain: A -> B, if B is the conflict, use A's partner
                        
                    # Place the final resolved value into the child
                    child[i] = value
        
        fill_child(child1, parent2, parent1, point1, point2)
        fill_child(child2, parent1, parent2, point1, point2)
        
        return child1, child2
    
    @staticmethod
    def simulated_binary_crossover(parent1: List[float], parent2: List[float],
                                   eta: float = 20.0) -> Tuple[List[float], List[float]]:
        """
        Simulated Binary Crossover (SBX) برای مقادیر پیوسته
        
        این روش رفتار single-point crossover در binary را برای مقادیر real شبیه‌سازی می‌کند.
        
        Parameters:
        -----------
        parent1: لیست مقادیر پیوسته اول
        parent2: لیست مقادیر پیوسته دوم
        eta: پارامتر توزیع (مقادیر بزرگتر -> children نزدیک‌تر به والدین)
        
        Returns:
        --------
        tuple: (child1, child2)
        """
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same length")
        
        child1 = []
        child2 = []
        
        for p1, p2 in zip(parent1, parent2):
            u = random.random()
            
            if u <= 0.5:
                beta = (2 * u) ** (1.0 / (eta + 1))
            else:
                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
            
            c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
            c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
            
            child1.append(c1)
            child2.append(c2)
        
        return child1, child2


# ==================== تابع Helper برای انتخاب خودکار متد ====================

def select_crossover_method(data_type: str, **kwargs):
    """
    تابع کمکی برای انتخاب خودکار متد crossover مناسب
    
    Parameters:
    -----------
    data_type: نوع داده ('list', 'matrix', 'permutation', 'continuous')
    **kwargs: پارامترهای اضافی
    
    Returns:
    --------
    function: تابع crossover مناسب
    """
    crossover = CrossoverMethods()
    
    if data_type == 'list_one_point':
        return crossover.one_point_crossover_list
    elif data_type == 'list_two_point':
        return crossover.two_point_crossover_list
    elif data_type == 'list_uniform':
        return crossover.uniform_crossover_list
    elif data_type == 'permutation_order':
        return crossover.order_crossover_list
    elif data_type == 'permutation_pmx':
        return crossover.partially_mapped_crossover
    elif data_type == 'continuous_arithmetic':
        return crossover.arithmetic_crossover_list
    elif data_type == 'continuous_blend':
        return crossover.blend_crossover_list
    elif data_type == 'continuous_sbx':
        return crossover.simulated_binary_crossover
    elif data_type == 'matrix_one_point':
        return crossover.one_point_crossover_matrix
    elif data_type == 'matrix_two_point':
        return crossover.two_point_crossover_matrix
    elif data_type == 'matrix_uniform':
        return crossover.uniform_crossover_matrix
    elif data_type == 'matrix_block':
        return crossover.block_crossover_matrix
    elif data_type == 'matrix_row_wise':
        return crossover.row_wise_crossover_matrix
    elif data_type == 'matrix_arithmetic':
        return crossover.arithmetic_crossover_matrix
    else:
        raise ValueError(f"Unknown data type: {data_type}")