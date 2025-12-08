import numpy as np
import random
from copy import deepcopy
from typing import List

class MutationMethods:
    """
    کلاس جامع برای متدهای مختلف Mutation (جهش)
    
    این کلاس شامل تمام روش‌های رایج mutation برای انواع مختلف داده‌ها است:
    - List-based mutation
    - Matrix-based mutation
    - Permutation-based mutation
    - Continuous value mutation
    
    هر متد یک individual می‌گیرد و نسخه جهش‌یافته آن را برمی‌گرداند.
    """
    
    def __init__(self):
        """Constructor"""
        pass
    
    # ==================== LIST-BASED MUTATION METHODS ====================
    
    @staticmethod
    def bit_flip_mutation(individual: List, mutation_rate: float = 0.1,
                               value_range: tuple = None, **mutation_args) -> List:
        """
        Bit-Flip Mutation برای لیست‌ها
        
        هر عنصر با احتمال mutation_rate تغییر می‌کند.
        برای مقادیر integer، مقدار تصادفی جدید انتخاب می‌شود.
        
        Parameters:
        -----------
        individual: لیست ورودی
        mutation_rate: احتمال جهش هر عنصر
        value_range: محدوده مقادیر ممکن (min, max)
        
        Returns:
        --------
        mutated_individual: لیست جهش‌یافته
        
        Example:
        --------
        individual = [1, 2, 0, 3, 0]
        mutated = bit_flip_mutation(individual, 0.2, (0, 5))
        """
        mutated = deepcopy(individual)
        # override from mutation_args if provided
        mutation_rate = float(mutation_args.get("mutation_rate", mutation_rate))
        value_range = mutation_args.get("value_range", value_range)
        
        if value_range is None:
            # تخمین محدوده از داده‌های موجود
            non_zero = [x for x in individual if x != 0]
            if non_zero:
                value_range = (0, max(non_zero) + 1)
            else:
                value_range = (0, 1)
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = random.randint(value_range[0], value_range[1] - 1)
        
        return mutated
    
    @staticmethod
    def swap_mutation(individual: List, n_swaps: int = 1, **mutation_args) -> List:
        """
        Swap Mutation برای لیست‌ها
        
        دو یا چند عنصر تصادفی با هم جابجا می‌شوند.
        مناسب برای permutations و مسائل مرتب‌سازی.
        
        Parameters:
        -----------
        individual: لیست ورودی
        n_swaps: تعداد جابجایی‌ها
        
        Returns:
        --------
        mutated_individual: لیست جهش‌یافته
        """
        if len(individual) < 2:
            return deepcopy(individual)
        
        mutated = deepcopy(individual)
        n_swaps = int(mutation_args.get("n_swaps", n_swaps))
        
        for _ in range(n_swaps):
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        return mutated
    
    @staticmethod
    def inversion_mutation(individual: List, **mutation_args) -> List:
        """
        Inversion Mutation برای لیست‌ها
        
        یک بخش تصادفی از لیست انتخاب و معکوس می‌شود.
        مناسب برای permutations و TSP.
        
        Parameters:
        -----------
        individual: لیست ورودی
        
        Returns:
        --------
        mutated_individual: لیست جهش‌یافته
        
        Example:
        --------
        individual = [1, 2, 3, 4, 5]
        points = (1, 3)
        mutated = [1, 4, 3, 2, 5]  # بخش [2,3,4] معکوس شد
        """
        if len(individual) < 2:
            return deepcopy(individual)
        
        mutated = deepcopy(individual)
        
        # انتخاب دو نقطه
        point1, point2 = sorted(random.sample(range(len(mutated)), 2))
        
        # معکوس کردن بخش بین دو نقطه
        mutated[point1:point2+1] = reversed(mutated[point1:point2+1])
        
        return mutated
    
    @staticmethod
    def scramble_mutation(individual: List, **mutation_args) -> List:
        """
        Scramble Mutation برای لیست‌ها
        
        یک بخش تصادفی انتخاب و عناصر آن به صورت تصادفی مرتب می‌شوند.
        
        Parameters:
        -----------
        individual: لیست ورودی
        
        Returns:
        --------
        mutated_individual: لیست جهش‌یافته
        """
        if len(individual) < 2:
            return deepcopy(individual)
        
        mutated = deepcopy(individual)
        
        # انتخاب دو نقطه
        point1, point2 = sorted(random.sample(range(len(mutated)), 2))
        
        # شافل کردن بخش بین دو نقطه
        subset = mutated[point1:point2+1]
        random.shuffle(subset)
        mutated[point1:point2+1] = subset
        
        return mutated
    
    @staticmethod
    def insertion_mutation(individual: List, **mutation_args) -> List:
        """
        Insertion Mutation برای لیست‌ها
        
        یک عنصر تصادفی انتخاب و در موقعیت تصادفی دیگری قرار می‌گیرد.
        
        Parameters:
        -----------
        individual: لیست ورودی
        
        Returns:
        --------
        mutated_individual: لیست جهش‌یافته
        """
        if len(individual) < 2:
            return deepcopy(individual)
        
        mutated = deepcopy(individual)
        
        # انتخاب عنصر برای جابجایی
        from_idx = random.randint(0, len(mutated) - 1)
        to_idx = random.randint(0, len(mutated) - 1)
        
        # حذف و درج در موقعیت جدید
        element = mutated.pop(from_idx)
        mutated.insert(to_idx, element)
        
        return mutated
    
    @staticmethod
    def displacement_mutation(individual: List, **mutation_args) -> List:
        """
        Displacement Mutation برای لیست‌ها
        
        یک بخش تصادفی انتخاب و در موقعیت تصادفی دیگری قرار می‌گیرد.
        
        Parameters:
        -----------
        individual: لیست ورودی
        
        Returns:
        --------
        mutated_individual: لیست جهش‌یافته
        """
        if len(individual) < 3:
            return deepcopy(individual)
        
        mutated = deepcopy(individual)
        
        # انتخاب بخش برای جابجایی
        size = random.randint(1, len(mutated) // 2)
        start = random.randint(0, len(mutated) - size)
        
        # حذف بخش
        segment = mutated[start:start+size]
        del mutated[start:start+size]
        
        # درج در موقعیت تصادفی
        insert_pos = random.randint(0, len(mutated))
        mutated[insert_pos:insert_pos] = segment
        
        return mutated
    
    @staticmethod
    def gaussian_mutation(individual: List[float], mutation_rate: float = 0.1,
                               sigma: float = 0.1, bounds: tuple = (0.0, 1.0), **mutation_args) -> List[float]:
        """
        Gaussian Mutation برای مقادیر پیوسته
        
        به هر عنصر نویز گاوسی اضافه می‌شود.
        
        Parameters:
        -----------
        individual: لیست مقادیر پیوسته
        mutation_rate: احتمال جهش هر عنصر
        sigma: انحراف معیار توزیع گاوسی
        bounds: محدوده مقادیر (min, max)
        
        Returns:
        --------
        mutated_individual: لیست جهش‌یافته
        """
        mutated = deepcopy(individual)
        mutation_rate = float(mutation_args.get("mutation_rate", mutation_rate))
        sigma = float(mutation_args.get("sigma", sigma))
        bounds = mutation_args.get("bounds", bounds)
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                # اضافه کردن نویز گاوسی
                mutated[i] += np.random.normal(0, sigma)
                # اعمال محدودیت
                mutated[i] = np.clip(mutated[i], bounds[0], bounds[1])
        
        return mutated
    
    @staticmethod
    def uniform_mutation(individual: List[float], mutation_rate: float = 0.1,
                             bounds: tuple = (0.0, 1.0), **mutation_args) -> List[float]:
        """
        Uniform Mutation برای مقادیر پیوسته
        
        هر عنصر با مقدار تصادفی یکنواخت جایگزین می‌شود.
        
        Parameters:
        -----------
        individual: لیست مقادیر پیوسته
        mutation_rate: احتمال جهش هر عنصر
        bounds: محدوده مقادیر (min, max)
        
        Returns:
        --------
        mutated_individual: لیست جهش‌یافته
        """
        mutated = deepcopy(individual)
        mutation_rate = float(mutation_args.get("mutation_rate", mutation_rate))
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = random.uniform(bounds[0], bounds[1])
        
        return mutated
    
    @staticmethod
    def polynomial_mutation(individual: List[float], mutation_rate: float = 0.1,
                                 eta: float = 20.0, bounds: tuple = (0.0, 1.0), **mutation_args) -> List[float]:
        """
        Polynomial Mutation برای مقادیر پیوسته
        
        جهش با استفاده از توزیع چندجمله‌ای.
        مناسب برای NSGA-II و الگوریتم‌های تکاملی مدرن.
        
        Parameters:
        -----------
        individual: لیست مقادیر پیوسته
        mutation_rate: احتمال جهش هر عنصر
        eta: پارامتر توزیع (مقادیر بزرگتر -> تغییرات کوچکتر)
        bounds: محدوده مقادیر (min, max)
        
        Returns:
        --------
        mutated_individual: لیست جهش‌یافته
        """
        mutated = deepcopy(individual)
        mutation_rate = float(mutation_args.get("mutation_rate", mutation_rate))
        eta = float(mutation_args.get("eta", eta))
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                y = mutated[i]
                yl, yu = bounds
                
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                
                rand = random.random()
                mut_pow = 1.0 / (eta + 1.0)
                
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - val ** mut_pow
                
                y = y + deltaq * (yu - yl)
                mutated[i] = np.clip(y, yl, yu)
        
        return mutated
    
    @staticmethod
    def boundary_mutation(individual: List[float], mutation_rate: float = 0.1,
                               bounds: tuple = (0.0, 1.0), **mutation_args) -> List[float]:
        """
        Boundary Mutation برای مقادیر پیوسته
        
        عنصر انتخابی به یکی از مرزهای محدوده می‌رود.
        
        Parameters:
        -----------
        individual: لیست مقادیر پیوسته
        mutation_rate: احتمال جهش هر عنصر
        bounds: محدوده مقادیر (min, max)
        
        Returns:
        --------
        mutated_individual: لیست جهش‌یافته
        """
        mutated = deepcopy(individual)
        mutation_rate = float(mutation_args.get("mutation_rate", mutation_rate))
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = random.choice([bounds[0], bounds[1]])
        
        return mutated
    
    # ==================== MATRIX-BASED MUTATION METHODS ====================
    
    @staticmethod
    def random_element_mutation_matrix(matrix: np.ndarray, mutation_rate: float = 0.1,
                                      value_range: tuple = None, **mutation_args) -> np.ndarray:
        """
        Random Element Mutation برای ماتریس‌ها
        
        عناصر تصادفی ماتریس با مقادیر جدید جایگزین می‌شوند.
        
        Parameters:
        -----------
        matrix: ماتریس ورودی
        mutation_rate: احتمال جهش هر عنصر
        value_range: محدوده مقادیر (min, max)
        
        Returns:
        --------
        mutated_matrix: ماتریس جهش‌یافته
        """
        mutated = matrix.copy()
        mutation_rate = float(mutation_args.get("mutation_rate", mutation_rate))
        value_range = mutation_args.get("value_range", value_range)
        
        if value_range is None:
            value_range = (0, 1)
        
        # ایجاد ماسک برای عناصری که جهش می‌یابند
        mask = np.random.rand(*matrix.shape) < mutation_rate
        
        # تولید مقادیر تصادفی جدید
        random_values = np.random.uniform(value_range[0], value_range[1], matrix.shape)
        
        # جایگزینی عناصر
        mutated[mask] = random_values[mask]
        
        return mutated
    
    @staticmethod
    def gaussian_mutation_matrix(matrix: np.ndarray, mutation_rate: float = 0.1,
                                sigma: float = 0.1, bounds: tuple = (0,1), **mutation_args) -> np.ndarray:
        """
        Gaussian Mutation برای ماتریس‌ها
        
        نویز گاوسی به عناصر تصادفی اضافه می‌شود.
        
        Parameters:
        -----------
        matrix: ماتریس ورودی
        mutation_rate: احتمال جهش هر عنصر
        sigma: انحراف معیار نویز
        bounds: محدوده مقادیر (min, max)
        
        Returns:
        --------
        mutated_matrix: ماتریس جهش‌یافته
        """
        mutated = matrix.copy()
        mutation_rate = float(mutation_args.get("mutation_rate", mutation_rate))
        sigma = float(mutation_args.get("sigma", sigma))
        
        # ماسک برای عناصری که جهش می‌یابند
        mask = np.random.rand(*matrix.shape) < mutation_rate
        
        # اضافه کردن نویز گاوسی
        noise = np.random.normal(0, sigma, matrix.shape)
        mutated[mask] += noise[mask]
        
        # اعمال محدودیت
        if bounds is not None:
            mutated = np.clip(mutated, bounds[0], bounds[1])
        
        return mutated
    
    @staticmethod
    def row_mutation_matrix(matrix: np.ndarray, mutation_rate: float = 0.1, **mutation_args) -> np.ndarray:
        """
        Row Mutation برای ماتریس‌ها
        
        سطرهای کامل با احتمال معین جهش می‌یابند.
        
        Parameters:
        -----------
        matrix: ماتریس ورودی
        mutation_rate: احتمال جهش هر سطر
        
        Returns:
        --------
        mutated_matrix: ماتریس جهش‌یافته
        """
        mutated = matrix.copy()
        mutation_rate = float(mutation_args.get("mutation_rate", mutation_rate))
        
        for i in range(matrix.shape[0]):
            if random.random() < mutation_rate:
                # جهش سطر (مثلاً شافل کردن یا تغییر مقادیر)
                np.random.shuffle(mutated[i, :])
        
        return mutated
    
    @staticmethod
    def column_mutation_matrix(matrix: np.ndarray, mutation_rate: float = 0.1, **mutation_args) -> np.ndarray:
        """
        Column Mutation برای ماتریس‌ها
        
        ستون‌های کامل با احتمال معین جهش می‌یابند.
        
        Parameters:
        -----------
        matrix: ماتریس ورودی
        mutation_rate: احتمال جهش هر ستون
        
        Returns:
        --------
        mutated_matrix: ماتریس جهش‌یافته
        """
        mutated = matrix.copy()
        mutation_rate = float(mutation_args.get("mutation_rate", mutation_rate))
        
        for j in range(matrix.shape[1]):
            if random.random() < mutation_rate:
                # جهش ستون
                np.random.shuffle(mutated[:, j])
        
        return mutated
    
    @staticmethod
    def block_mutation_matrix(matrix: np.ndarray, block_size: tuple = None, **mutation_args) -> np.ndarray:
        """
        Block Mutation برای ماتریس‌ها
        
        یک بلوک تصادفی از ماتریس انتخاب و جهش می‌یابد.
        
        Parameters:
        -----------
        matrix: ماتریس ورودی
        block_size: اندازه بلوک (rows, cols). اگر None باشد، تصادفی انتخاب می‌شود
        
        Returns:
        --------
        mutated_matrix: ماتریس جهش‌یافته
        """
        mutated = matrix.copy()
        rows, cols = matrix.shape
        
        block_size = mutation_args.get("block_size", block_size)
        if block_size is None:
            block_rows = random.randint(1, max(1, rows // 2))
            block_cols = random.randint(1, max(1, cols // 2))
        else:
            block_rows, block_cols = block_size
        
        # انتخاب موقعیت شروع بلوک
        start_row = random.randint(0, rows - block_rows)
        start_col = random.randint(0, cols - block_cols)
        
        # جهش بلوک (مثلاً با مقادیر تصادفی)
        mutated[start_row:start_row+block_rows, start_col:start_col+block_cols] = \
            np.random.rand(block_rows, block_cols)
        
        return mutated
    
    @staticmethod
    def swap_rows_mutation_matrix(matrix: np.ndarray, n_swaps: int = 1, **mutation_args) -> np.ndarray:
        """
        Swap Rows Mutation برای ماتریس‌ها
        
        دو سطر تصادفی با هم جابجا می‌شوند.
        
        Parameters:
        -----------
        matrix: ماتریس ورودی
        n_swaps: تعداد جابجایی‌ها
        
        Returns:
        --------
        mutated_matrix: ماتریس جهش‌یافته
        """
        if matrix.shape[0] < 2:
            return matrix.copy()
        
        mutated = matrix.copy()
        n_swaps = int(mutation_args.get("n_swaps", n_swaps))
        
        for _ in range(n_swaps):
            idx1, idx2 = random.sample(range(matrix.shape[0]), 2)
            mutated[[idx1, idx2], :] = mutated[[idx2, idx1], :]
        
        return mutated
    
    @staticmethod
    def swap_columns_mutation_matrix(matrix: np.ndarray, n_swaps: int = 1, **mutation_args) -> np.ndarray:
        """
        Swap Columns Mutation برای ماتریس‌ها
        
        دو ستون تصادفی با هم جابجا می‌شوند.
        
        Parameters:
        -----------
        matrix: ماتریس ورودی
        n_swaps: تعداد جابجایی‌ها
        
        Returns:
        --------
        mutated_matrix: ماتریس جهش‌یافته
        """
        if matrix.shape[1] < 2:
            return matrix.copy()
        
        mutated = matrix.copy()
        n_swaps = int(mutation_args.get("n_swaps", n_swaps))
        
        for _ in range(n_swaps):
            idx1, idx2 = random.sample(range(matrix.shape[1]), 2)
            mutated[:, [idx1, idx2]] = mutated[:, [idx2, idx1]]
        
        return mutated
    
    @staticmethod
    def creep_mutation_matrix(matrix: np.ndarray, mutation_rate: float = 0.1,
                             step_size: float = 0.01, bounds: tuple = None, **mutation_args) -> np.ndarray:
        """
        Creep Mutation برای ماتریس‌ها
        
        تغییرات کوچک و تدریجی در عناصر.
        
        Parameters:
        -----------
        matrix: ماتریس ورودی
        mutation_rate: احتمال جهش هر عنصر
        step_size: اندازه گام تغییر
        bounds: محدوده مقادیر (min, max)
        
        Returns:
        --------
        mutated_matrix: ماتریس جهش‌یافته
        """
        mutated = matrix.copy()
        mutation_rate = float(mutation_args.get("mutation_rate", mutation_rate))
        step_size = float(mutation_args.get("step_size", step_size))
        
        mask = np.random.rand(*matrix.shape) < mutation_rate
        
        # تغییرات کوچک تصادفی
        changes = np.random.uniform(-step_size, step_size, matrix.shape)
        mutated[mask] += changes[mask]
        
        if bounds is not None:
            mutated = np.clip(mutated, bounds[0], bounds[1])
        
        return mutated
    
    # ==================== SPECIALIZED MUTATION METHODS ====================
    
    @staticmethod
    def adaptive_mutation_list(individual: List[float], mutation_rate: float = 0.1,
                              generation: int = 0, max_generations: int = 100,
                              bounds: tuple = (0.0, 1.0), **mutation_args) -> List[float]:
        """
        Adaptive Mutation با نرخ متغیر
        
        نرخ جهش در طول نسل‌ها کاهش می‌یابد.
        
        Parameters:
        -----------
        individual: لیست ورودی
        mutation_rate: نرخ جهش اولیه
        generation: نسل فعلی
        max_generations: تعداد کل نسل‌ها
        bounds: محدوده مقادیر
        
        Returns:
        --------
        mutated_individual: لیست جهش‌یافته
        """
        # کاهش نرخ جهش با پیشرفت نسل‌ها
        mutation_rate = float(mutation_args.get("mutation_rate", mutation_rate))
        generation = int(mutation_args.get("generation", generation))
        max_generations = int(mutation_args.get("max_generations", max_generations))
        adaptive_rate = mutation_rate * (1 - generation / max_generations)
        
        return MutationMethods.gaussian_mutation(
            individual, adaptive_rate, sigma=0.1, bounds=bounds
        )
    
    @staticmethod
    def non_uniform_mutation(individual: List[float], mutation_rate: float = 0.1,
                                  generation: int = 0, max_generations: int = 100,
                                  b: float = 5.0, bounds: tuple = (0.0, 1.0), **mutation_args) -> List[float]:
        """
        Non-Uniform Mutation
        
        جهش با توزیع غیریکنواخت که با گذشت زمان دقیق‌تر می‌شود.
        
        Parameters:
        -----------
        individual: لیست ورودی
        mutation_rate: احتمال جهش
        generation: نسل فعلی
        max_generations: تعداد کل نسل‌ها
        b: پارامتر شکل توزیع
        bounds: محدوده مقادیر
        
        Returns:
        --------
        mutated_individual: لیست جهش‌یافته
        """
        mutated = deepcopy(individual)
        mutation_rate = float(mutation_args.get("mutation_rate", mutation_rate))
        generation = int(mutation_args.get("generation", generation))
        max_generations = int(mutation_args.get("max_generations", max_generations))
        b = float(mutation_args.get("b", b))
        
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                y = mutated[i]
                yl, yu = bounds
                
                if random.random() < 0.5:
                    delta = yu - y
                else:
                    delta = y - yl
                
                r = random.random()
                t = generation / max_generations
                delta_t = delta * (1 - r ** ((1 - t) ** b))
                
                if random.random() < 0.5:
                    mutated[i] = y + delta_t
                else:
                    mutated[i] = y - delta_t
                
                mutated[i] = np.clip(mutated[i], yl, yu)
        
        return mutated


# ==================== تابع Helper برای انتخاب خودکار متد ====================

def select_mutation_method(data_type: str, **kwargs):
    """
    تابع کمکی برای انتخاب خودکار متد mutation مناسب
    
    Parameters:
    -----------
    data_type: نوع داده
    **kwargs: پارامترهای اضافی
    
    Returns:
    --------
    function: تابع mutation مناسب
    """
    mutation = MutationMethods()
    
    if data_type == 'list_bit_flip':
        return mutation.bit_flip_mutation
    elif data_type == 'list_swap':
        return mutation.swap_mutation
    elif data_type == 'list_inversion':
        return mutation.inversion_mutation
    elif data_type == 'list_scramble':
        return mutation.scramble_mutation
    elif data_type == 'list_insertion':
        return mutation.insertion_mutation
    elif data_type == 'list_displacement':
        return mutation.displacement_mutation
    elif data_type == 'continuous_gaussian':
        return mutation.gaussian_mutation
    elif data_type == 'continuous_uniform':
        return mutation.uniform_mutation
    elif data_type == 'continuous_polynomial':
        return mutation.polynomial_mutation
    elif data_type == 'continuous_boundary':
        return mutation.boundary_mutation
    elif data_type == 'matrix_random':
        return mutation.random_element_mutation_matrix
    elif data_type == 'matrix_gaussian':
        return mutation.gaussian_mutation_matrix
    elif data_type == 'matrix_row':
        return mutation.row_mutation_matrix
    elif data_type == 'matrix_column':
        return mutation.column_mutation_matrix
    elif data_type == 'matrix_block':
        return mutation.block_mutation_matrix
    elif data_type == 'matrix_creep':
        return mutation.creep_mutation_matrix
    else:
        raise ValueError(f"Unknown data type: {data_type}")