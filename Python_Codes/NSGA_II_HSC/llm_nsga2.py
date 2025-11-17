import numpy as np
from copy import deepcopy
import random
import time
from convergence_metrics import ConvergenceMetrics
from diagnostic_metrics import DiagnosticMetrics
from crossover_methods import CrossoverMethods
from mutation_methods import MutationMethods
from selection_methods import SelectionMethods
from openrouter_client import OpenRouterClient
from hypervolume_manager import HypervolumeManager
from ai_config import AI_MODEL
import traceback
import os
import pickle


class LLM_NSGA2_Humanitarian:
    """
    NSGA-II for humanitarian logistics optimization with custom chromosome structure

    Features:
    - Multi-objective optimization for humanitarian logistics
    - Custom chromosome structure for shelter, distribution, and medical facility allocation
    - Elitism: preserves best solutions across generations
    - Crowding distance for diversity maintenance
    - Constraint handling for feasible solutions
    """

    def __init__(
        self,
        shelter_id,
        distribution_center_id,
        damage_points_id,
        hospital_id,
        temporary_medical_id,
        max_iter=300,
        pop_size=150,
        p_crossover=0.9,
        p_mutation=0.1,
        elitism_rate=0.1,
        verbose=True,
        resume = True,
        openrouter_api_key=None,
        base_url=None,
        use_ai_optimization=True,
        llm_iter = 5,
        use_llm_init_pop=True,
        distances=None,
        homeless=None,
        severe_injured=None,
        minor_injured=None,
        cost=None,
        capacity=None,
    ):
        """
        Constructor for humanitarian logistics NSGA-II

        Parameters:
        -----------
        shelter_id: List of shelter IDs
        distribution_center_id: List of distribution center IDs
        damage_points_id: List of damage point IDs
        hospital_id: List of hospital IDs
        temporary_medical_id: List of temporary medical center IDs
        max_iter: Maximum number of iterations (default: 100)
        pop_size: Population size (default: 100)
        p_crossover: Crossover probability (default: 0.7)
        p_mutation: Mutation probability (default: 0.3)
        elitism_rate: Percentage of elite individuals to preserve (default: 0.1)
        verbose: Print iteration information (default: True)
        openrouter_api_key: API key for OpenRouter (optional)
        use_ai_optimization: Enable AI-based method optimization (default: False)
        distances: dict of distance matrices to inject into AI prompt
        homeless: dict of homeless counts per damage point id to inject into AI prompt
        """
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.elitism_rate = elitism_rate
        self.n_elite = max(1, int(elitism_rate * pop_size))  # حداقل یک فرد نخبه
        self.verbose = verbose
        self.resume = resume
        self.use_ai_optimization = use_ai_optimization
        self.use_llm_init_pop = bool(use_llm_init_pop)
        self.llm_iter = llm_iter

        # تنظیم OpenRouter Client
        self.openrouter_client = None
        if use_ai_optimization and openrouter_api_key:
            self.openrouter_client = OpenRouterClient(openrouter_api_key, AI_MODEL, base_url)

        # ذخیره متدهای فعلی (هر بخش به صورت جداگانه)
        self.current_methods = {
            "crossover_methods": {
                "part1": "default",
                "part2": "default",
                "part3_p1": "default",
                "part3_p2": "default",
                "part4": "default",
                "part5": "default",
                "part6": "default",
                "part7": "default",
            },
            "mutation_methods": {
                "part1": "default",
                "part2": "default",
                "part3_p1": "default",
                "part3_p2": "default",
                "part4": "default",
                "part5": "default",
                "part6": "default",
                "part7": "default",
            },
            "selection_method": "default",
            "selection_args": {
                "k": None,
                "power": None,
                "refs": None,
                "tau": None,
                "prefer_younger": None,
                "eps": None,
            },
            "mutation_args_per_part": {
                "part1": {},
                "part2": {},
                "part3_p1": {},
                "part3_p2": {},
                "part4": {},
                "part5": {},
                "part6": {},
                "part7": {},
            },
            "global_crossover_probability": p_crossover,
            "global_mutation_probability": p_mutation,
            "crossover_part_probability": {
                "part1": 1.0,
                "part2": 1.0,
                "part3_p1": 1.0,
                "part3_p2": 1.0,
                "part4": 1.0,
                "part5": 1.0,
                "part6": 1.0,
                "part7": 1.0,
            },
            "mutation_part_probability": {
                "part1": 1.0,
                "part2": 1.0,
                "part3_p1": 1.0,
                "part3_p2": 1.0,
                "part4": 1.0,
                "part5": 1.0,
                "part6": 1.0,
                "part7": 1.0,
            },
            "history": {
                "crossover_methods_history":{
                    "part1": [],
                    "part2": [],
                    "part3_p1": [],
                    "part3_p2": [],
                    "part4": [],
                    "part5": [],
                    "part6": [],
                    "part7": [],
                    },
                "mutation_methods_history":{
                    "part1": [],
                    "part2": [],
                    "part3_p1": [],
                    "part3_p2": [],
                    "part4": [],
                    "part5": [],
                    "part6": [],
                    "part7": [],
                },
                "selection_method_history": [],
                "global_crossover_probability_history": [],
                "global_mutation_probability_history": [],
                "crossover_part_probability_history": {
                    "part1": [],
                    "part2": [],
                    "part3_p1": [],
                    "part3_p2": [],
                    "part4": [],
                    "part5": [],
                    "part6": [],
                    "part7": [],
                },
                "mutation_part_probability_history": {
                    "part1": [],
                    "part2": [],
                    "part3_p1": [],
                    "part3_p2": [],
                    "part4": [],
                    "part5": [],
                    "part6": [],
                    "part7": [],
                },
                }
        }

        # Problem dimensions
        self.da_id = damage_points_id
        self.idc_id = distribution_center_id
        # Store full ID lists for prompt injection
        self.shelter_id_list = shelter_id
        self.hospital_id_list = hospital_id
        self.tmc_id_list = temporary_medical_id
        # Dimensions
        self.n_shelters = len(shelter_id)
        self.n_distribution = len(distribution_center_id)
        self.n_damage_points = len(damage_points_id)
        self.n_hospitals = len(hospital_id)
        self.n_temp_medical = len(temporary_medical_id)
        # Optional context for AI prompt
        self.distances = distances or {}
        self.homeless = homeless or {}
        self.severe_injured = severe_injured or {}
        self.minor_injured = minor_injured or {}
        self.cost = cost or {}
        self.capacity = capacity or {}

        self.crossover_methods = {
            "1": CrossoverMethods.one_point_crossover_list,
            "2": CrossoverMethods.two_point_crossover_list,
            "3": CrossoverMethods.uniform_crossover_list,
            "4": CrossoverMethods.order_crossover_list,
            "5": CrossoverMethods.arithmetic_crossover_list,
            "6": CrossoverMethods.blend_crossover_list,
            "7": CrossoverMethods.one_point_crossover_matrix,
            "8": CrossoverMethods.two_point_crossover_matrix,
            "9": CrossoverMethods.uniform_crossover_matrix,
            "10": CrossoverMethods.block_crossover_matrix,
            "11": CrossoverMethods.row_wise_crossover_matrix,
            "12": CrossoverMethods.arithmetic_crossover_matrix,
            "13": CrossoverMethods.multi_point_crossover_list,
            "14": CrossoverMethods.partially_mapped_crossover,
            "15": CrossoverMethods.simulated_binary_crossover,
        }
        self.mutation_methods = {
            "1": MutationMethods.bit_flip_mutation_list,
            "2": MutationMethods.swap_mutation_list,
            "3": MutationMethods.inversion_mutation_list,
            "4": MutationMethods.scramble_mutation_list,
            "5": MutationMethods.insertion_mutation_list,
            "6": MutationMethods.displacement_mutation_list,
            "7": MutationMethods.gaussian_mutation_list,
            "8": MutationMethods.uniform_mutation_list,
            "9": MutationMethods.polynomial_mutation_list,
            "10": MutationMethods.boundary_mutation_list,
            "11": MutationMethods.random_element_mutation_matrix,
            "12": MutationMethods.gaussian_mutation_matrix,
            "13": MutationMethods.row_mutation_matrix,
            "14": MutationMethods.column_mutation_matrix,
            "15": MutationMethods.block_mutation_matrix,
            "16": MutationMethods.creep_mutation_matrix,
        }

        # Instantiate SelectionMethods to bind methods
        self.selection_methods_instance = SelectionMethods()

        self.selection_methods = {
            "1": self.selection_methods_instance.crowded_binary_tournament,
            "2": self.selection_methods_instance.adaptive_k_tournament,
            "3": self.selection_methods_instance.rank_based_roulette,
            "4": self.selection_methods_instance.reference_biased_tournament,
            "5": self.selection_methods_instance.age_diversity_tournament,
            "6": self.selection_methods_instance.epsilon_dominance_tournament,
        }

        self.metrics = ConvergenceMetrics()
        self.diagnostics = DiagnosticMetrics()

        # مدیریت هایپرولیوم با نقطه مرجع ثابت
        self.hypervolume_manager = HypervolumeManager(estimation_method="conservative")

        # برای ذخیره تاریخچه شاخص‌ها
        # فرمت: دیکشنری که کلیدهایش همان کلیدهای current_metrics هستند
        # و مقدار هر کلید لیستی از مقادیر تاریخی آن شاخص است
        self.metrics_history = {}
        # آمار بخشی نسل به نسل
        # فرمت: دیکشنری که کلیدهایش همان کلیدهای section_stats هستند
        # و مقدار هر کلید یک دیکشنری است که کلیدهایش (entropy/variance) هستند
        # و مقدار هر کدام لیستی از مقادیر تاریخی است
        self.section_stats_history = {}
        self.offspring_survival_history = []
        
    def _validate_and_convert_chromosome(self, item):
        """
        Validate a chromosome dict from AI and convert to the internal format:
        [list, list, list, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Returns None if invalid.
        """
        try:
            # part1
            p1 = list(item["part1"]) if isinstance(item.get("part1"), list) else None
            if p1 is None or len(p1) != self.n_shelters:
                return None
            if any((not isinstance(x, (int, float))) or (x < 0) or (x > self.n_distribution) for x in p1):
                return None
            p1 = [int(round(x)) for x in p1]

            # part2
            p2 = list(item["part2"]) if isinstance(item.get("part2"), list) else None
            if p2 is None or len(p2) != self.n_shelters:
                return None
            p2 = [float(min(1.0, max(0.0, x))) for x in p2]

            # part3
            p3 = list(item["part3"]) if isinstance(item.get("part3"), list) else None
            if p3 is None or len(p3) != self.n_shelters:
                return None
            p3 = [int(x) for x in p3]
            # basic sanity: first section should be permutation of DA ids
            prefix = p3[: self.n_damage_points]
            if sorted(prefix) != sorted(self.da_id):
                return None

            # matrices
            import numpy as np
            # part4
            p4 = item.get("part4")
            if not (isinstance(p4, list) and len(p4) == self.n_damage_points):
                return None
            p4_arr = np.array(p4, dtype=float)
            if p4_arr.shape != (self.n_damage_points, self.n_hospitals):
                return None
            # each row has at least one positive
            if not all(np.any(row > 0) for row in p4_arr):
                return None

            # part5
            p5 = item.get("part5")
            if not (isinstance(p5, list) and len(p5) == self.n_damage_points):
                return None
            p5_arr = np.array(p5, dtype=float)
            if p5_arr.shape != (self.n_damage_points, self.n_hospitals):
                return None
            p5_arr = np.clip(p5_arr, 0.0, 1.0)
            # enforce mask where p4>0 => p5 can be >0, else 0
            p5_arr[p4_arr == 0] = 0.0

            # part6
            p6 = item.get("part6")
            if not (isinstance(p6, list) and len(p6) == self.n_damage_points):
                return None
            p6_arr = np.array(p6, dtype=float)
            if p6_arr.shape != (self.n_damage_points, self.n_hospitals + self.n_temp_medical):
                return None
            if not all(np.any(row > 0) for row in p6_arr):
                return None

            # part7
            p7 = item.get("part7")
            if not (isinstance(p7, list) and len(p7) == self.n_damage_points):
                return None
            p7_arr = np.array(p7, dtype=float)
            if p7_arr.shape != (self.n_damage_points, self.n_hospitals + self.n_temp_medical):
                return None
            p7_arr = np.clip(p7_arr, 0.0, 1.0)
            p7_arr[p6_arr == 0] = 0.0

            chrom = [p1, p2, p3, p4_arr, p5_arr, p6_arr, p7_arr]
            # final light repair
            chrom = self.chromosome_repair(chrom)
            return chrom
        except Exception:
            return None

    def _entropy(self, values):
        """
        محاسبه آنتروپی ژنی بخش اول جمعیت
        Input:
            part1_list: list of lists (بخش اول همه کروموزوم‌ها)
        Output:
            entropy_per_gene: آرایه‌ای از آنتروپی هر موقعیت
            avg_entropy: میانگین آنتروپی کل
        """
        arr = np.array(values, dtype=int)  # شکل (n_individuals, n_genes)
        n_genes = arr.shape[1]
        entropy_per_gene = np.zeros(n_genes)

        for j in range(n_genes):
            gene_values = arr[:, j]
            unique, counts = np.unique(gene_values, return_counts=True)
            p = counts / counts.sum()
            p = p[p > 0]  # حذف احتمال صفر
            entropy_per_gene[j] = round(-np.sum(p * np.log(p + 1e-12)), 4)  # آنتروپی شانون

        avg_entropy = round(np.mean(entropy_per_gene), 4)
        return entropy_per_gene, avg_entropy

    def _variance(self, values):
        """
        محاسبه واریانس ژن‌ها یا سلول‌ها در جمعیت و میانگین کل آن‌ها.
        
        ورودی:
            values: لیست افراد جمعیت
                    هر عضو می‌تواند:
                    - لیست یک‌بعدی از اعداد (مثلاً بخش‌های پیوسته مثل part2)
                    - ماتریس دوبعدی (مثلاً بخش‌های 4 تا 7)
        
        خروجی:
            variance_per_gene : آرایه واریانس هر ژن یا سلول
            avg_variance      : میانگین واریانس‌ها (شاخص کلی تنوع)
        """
        arr = np.round(np.array(values, dtype=float), decimals=4)
        arr[~np.isfinite(arr)] = np.nan  # حذف NaN یا inf

        if arr.ndim == 2 or arr.ndim == 3:
            variance_per_gene = np.nanvar(arr, axis=0)
        else:
            # اگر فقط لیست ساده باشد (مثل تک ژن‌ها)
            arr = arr.flatten()
            if arr.size == 0:
                return np.array([0.0]), 0.0
            variance_per_gene = np.array([np.var(arr)])
        
        avg_variance = np.round(float(np.nanmean(variance_per_gene)), decimals=4)
        return variance_per_gene, avg_variance

    def compute_section_stats(self, pop):
        """
        محاسبه آمار بخشی (entropy/variance) فقط برای هر بخش/سکشن (بدون نرخ تعمیر)
        """
        if not pop:
            return {}
        # جمع‌آوری مقادیر
        # برای entropy: باید لیست‌ها را به صورت جداگانه نگه داریم (2D array)
        # برای variance: می‌توانیم همه را flatten کنیم
        part1_vals_list, part3_vals_list = [], []  # برای entropy (2D)
        part2_vals, part4_vals, part5_vals, part6_vals, part7_vals = [], [], [], [], []  # برای variance (flat)
        for ind in pop:
            chrom = ind["chromosome"]
            if chrom is None:
                continue
            part1_vals_list.append(chrom[0])  # لیست جداگانه برای هر کروموزوم
            part2_vals.extend(chrom[1])
            part3_vals_list.append(chrom[2])  # لیست جداگانه برای هر کروموزوم
            part4_vals.extend(chrom[3])
            part5_vals.extend(chrom[4])
            part6_vals.extend(chrom[5])
            part7_vals.extend(chrom[6])
        section_stats = {
            "part1": {
                "entropy": self._entropy(part1_vals_list),
            },
            "part2": {"variance": self._variance(part2_vals)},
            "part3": {"entropy": self._entropy(part3_vals_list)},
            "part4": {
                "variance": self._variance(part4_vals),
            },
            "part5": {"variance": self._variance(part5_vals)},
            "part6": {
                "variance": self._variance(part6_vals),
            },
            "part7": {"variance": self._variance(part7_vals)},
        }
        return section_stats

    def create_random_chromosome(self):
        """
        Create a random valid chromosome

        Chromosome structure:
        [distribution_assignment, flow_values, damage_to_shelter, damage_to_hospital,
         temp_medical_binary, severe_to_hospital, moderate_to_hospital, moderate_to_temp]
        """
        chromosome = []

        # Part 1: Distribution center assignment to shelters (0 means shelter not selected)
        # Randomly select which shelters to activate (at least 1)
        n_active = random.randint(1, self.n_shelters)
        active_shelters = np.random.choice(self.n_shelters, n_active, replace=False)
        dist_assignment = [0] * self.n_shelters
        for idx in active_shelters:
            dist_assignment[idx] = random.randint(1, self.n_distribution)
        chromosome.append(dist_assignment)

        # Part 2: Flow values from distribution centers to shelters
        flow_values = []
        for i in range(self.n_shelters):
            # Use normal distribution with mean 0.6 and std 0.15
            ratio = np.clip(np.random.normal(0.6, 0.15), 0, 1)
            flow_values.append(ratio)
        chromosome.append(flow_values)

        # Part 3: Damage point to shelter assignment
        np.random.shuffle(self.da_id)
        damage_to_shelter: list = deepcopy(self.da_id)
        for _ in range(self.n_damage_points, self.n_shelters):
            damage_to_shelter.append(np.random.choice(self.da_id))
        chromosome.append(damage_to_shelter)

        # Part 4: damaged_points to hospital assignment for transfer of severely injured
        part4 = np.zeros((self.n_damage_points, self.n_hospitals))
        for i in range(self.n_damage_points):
            num_active = random.randint(1, self.n_hospitals)
            active_indices = random.sample(range(self.n_hospitals), num_active)
            weights = [np.random.random() for _ in range(num_active)]
            total = sum(weights)
            for idx, dest in enumerate(active_indices):
                part4[i][dest] = weights[idx] / total
        chromosome.append(part4)

        # Part 5: Percentage of severely injured people transported by ambulance
        part5 = np.zeros((self.n_damage_points, self.n_hospitals))
        for i in range(self.n_damage_points):
            for j in range(self.n_hospitals):
                if part4[i][j] > 0:
                    part5[i][j] = np.random.random()
        chromosome.append(part5)

        # Part 6: damaged_points to TMC & hospital assignment for transfer of minor injured
        part6 = np.zeros((self.n_damage_points, self.n_hospitals + self.n_temp_medical))
        for i in range(self.n_damage_points):
            num_active = np.random.randint(1, self.n_hospitals + self.n_temp_medical)
            active_indices = random.sample(
                range(self.n_hospitals + self.n_temp_medical), num_active
            )
            weights = [np.random.random() for _ in range(num_active)]
            for idx, dest in enumerate(active_indices):
                part6[i][dest] = weights[idx]
        chromosome.append(part6)

        # Part 7: Percentage of minor injured people transported by ambulance
        part7 = np.zeros((self.n_damage_points, self.n_hospitals + self.n_temp_medical))
        for i in range(self.n_damage_points):
            for j in range(self.n_hospitals + self.n_temp_medical):
                if part6[i][j] > 0:
                    part7[i][j] = np.random.random()
        chromosome.append(part7)

        return chromosome

    def chromosome_repair(self, chromosome):
        """
        Repair chromosome parts 1, 4, and 6

        Parameters:
        -----------
        chromosome: list
            Complete chromosome to repair

        Returns:
        --------
        list: repaired chromosome
        """
        repaired_chromosome = deepcopy(chromosome)

        # تعمیر بخش 1: اگر همه ژن‌ها صفر باشند، حداقل یکی را غیر صفر کن
        if all(gene == 0 for gene in repaired_chromosome[0]):
            # انتخاب تصادفی یک یا چند پناهگاه برای فعال‌سازی
            n_active = random.randint(1, self.n_shelters)
            active_shelters = np.random.choice(self.n_shelters, n_active, replace=False)

            for idx in active_shelters:
                repaired_chromosome[0][idx] = random.randint(1, self.n_distribution)

        # تعمیر بخش 4: ماتریس احتمال برای مجروحان شدید
        if len(repaired_chromosome) > 3:
            matrix = repaired_chromosome[3]
            rows, cols = matrix.shape

            for i in range(rows):
                # بررسی مجموع سطر
                if np.sum(matrix[i, :]) == 0:
                    num_active = random.randint(1, cols)
                    active_indices = random.sample(range(cols), num_active)
                    weights = [np.random.random() for _ in range(num_active)]
                    total = sum(weights)

                    for idx, dest in enumerate(active_indices):
                        matrix[i][dest] = weights[idx] / total

        # تعمیر بخش 6: ماتریس احتمال برای مجروحان خفیف
        if len(repaired_chromosome) > 5:
            matrix = repaired_chromosome[5]
            rows, cols = matrix.shape

            for i in range(rows):
                # بررسی مجموع سطر
                if np.sum(matrix[i, :]) == 0:
                    num_active = random.randint(1, cols)
                    active_indices = random.sample(range(cols), num_active)
                    weights = [np.random.random() for _ in range(num_active)]
                    total = sum(weights)

                    for idx, dest in enumerate(active_indices):
                        matrix[i][dest] = weights[idx] / total

        return repaired_chromosome

    def order_crossover(self, parent1, parent2):
        """
        Order crossover for permutations to ensure no duplicates
        """
        size = len(parent1)
        child1 = [None] * size
        child2 = [None] * size

        # Select two random crossover points
        start, end = sorted(random.sample(range(size), 2))

        # Copy the segment from parent1 to child1 and from parent2 to child2
        child1[start : end + 1] = parent1[start : end + 1]
        child2[start : end + 1] = parent2[start : end + 1]

        # Fill the remaining positions in child1 with elements from parent2 not in the copied segment
        p2_idx = 0
        for i in range(size):
            if child1[i] is None:
                while parent2[p2_idx] in child1[start : end + 1]:
                    p2_idx += 1
                child1[i] = parent2[p2_idx]
                p2_idx += 1

        # Fill the remaining positions in child2 with elements from parent1 not in the copied segment
        p1_idx = 0
        for i in range(size):
            if child2[i] is None:
                while parent1[p1_idx] in child2[start : end + 1]:
                    p1_idx += 1
                child2[i] = parent1[p1_idx]
                p1_idx += 1

        return child1, child2

    def crossover(self, parent1, parent2):
        """
        Custom crossover for humanitarian logistics chromosome
        """
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)

        # Part 1 & 2
        point = random.choice(range(1, self.n_shelters))
        child1[0][:point], child2[0][:point] = child2[0][:point], child1[0][:point]
        child1[1][:point], child2[1][:point] = child2[1][:point], child1[1][:point]

        # Part 3: Use order crossover for damage points assignment (0 to n_damage_points)
        if self.n_damage_points > 0:
            part3_p1 = parent1[2][: self.n_damage_points]
            part3_p2 = parent2[2][: self.n_damage_points]
            child1_part3, child2_part3 = self.order_crossover(part3_p1, part3_p2)
            child1[2][: self.n_damage_points] = child1_part3
            child2[2][: self.n_damage_points] = child2_part3
        # For the rest (n_damage_points to n_shelters), swap as before
        for i in range(self.n_damage_points, self.n_shelters):
            if np.random.rand() < 0.5:
                child1[2][i], child2[2][i] = child2[2][i], child1[2][i]

        # Part 4
        mask_matrix = np.random.rand(self.n_damage_points, self.n_hospitals) < 0.5
        child1[3][mask_matrix] = parent2[3][mask_matrix]
        child2[3][mask_matrix] = parent1[3][mask_matrix]

        # Part 5
        point1 = np.random.choice(range(1, self.n_damage_points))
        point2 = np.random.choice(range(1, self.n_hospitals))
        child1[4][:point1, :point2], child2[4][:point1, :point2] = (
            child2[4][:point1, :point2],
            child1[4][:point1, :point2],
        )
        child1[4][point1:, point2:], child2[4][point1:, point2:] = (
            child2[4][point1:, point2:],
            child1[4][point1:, point2:],
        )

        # Part 6
        mask_matrix = (
            np.random.rand(self.n_damage_points, self.n_hospitals + self.n_temp_medical)
            < 0.5
        )
        child1[5][mask_matrix] = parent2[5][mask_matrix]
        child2[5][mask_matrix] = parent1[5][mask_matrix]

        # Part 7
        point1 = np.random.choice(range(1, self.n_damage_points))
        point2, point3 = sorted(
            random.sample(range(1, self.n_hospitals + self.n_temp_medical), k=2)
        )
        child1[6][:point1, point2:point3], child2[6][:point1, point2:point3] = (
            child2[6][:point1, point2:point3],
            child1[6][:point1, point2:point3],
        )
        child1[6][point1:, :point2], child2[6][point1:, :point2] = (
            child2[6][point1:, :point2],
            child1[6][point1:, :point2],
        )
        child1[6][point1:, point3:], child2[6][point1:, point3:] = (
            child2[6][point1:, point3:],
            child1[6][point1:, point3:],
        )

        # تعمیر کروموزوم‌ها (بخش‌های 1، 4، و 6)
        child1 = self.chromosome_repair(child1)
        child2 = self.chromosome_repair(child2)

        return child1, child2

    def mutate(self, chromosome):
        """
        Custom mutation for humanitarian logistics chromosome
        """
        mutated = deepcopy(chromosome)
        # Part 1
        i = np.random.randint(0, self.n_shelters)
        if mutated[0][i] == 0:
            mutated[0][i] = random.randint(1, self.n_distribution)
        else:
            mutated[0][i] = random.randint(1, self.n_distribution)

        # Part 2
        i = np.random.randint(0, self.n_shelters)
        while mutated[1][i] == 0:
            i = np.random.randint(0, self.n_shelters)
        mutated[1][i] = max(0, np.clip(mutated[1][i] + np.random.normal(0, 0.1), 0, 1))

        # Part 3
        i = np.random.randint(0, self.n_shelters)
        if mutated[2][i] == 0:
            mutated[2][i] = random.choice(self.da_id)
        elif mutated[2][i] != 0 and i > self.n_damage_points:
            mutated[2][i] = 0
        elif mutated[2][i] != 0 and i < self.n_damage_points:
            j = [k for k in range(self.n_damage_points) if k != i]
            j = random.choice(j)
            mutated[2][i], mutated[2][j] = mutated[2][j], mutated[2][i]

        # Part 4 & 5
        for i in range(self.n_damage_points):
            j = random.randint(0, self.n_hospitals - 1)
            if mutated[3][i][j] == 0:
                # if random.random() < 0.1:
                mutated[3][i][j] = random.uniform(0.1, 0.5)
                mutated[4][i][j] = np.clip(
                    mutated[4][i][j] + np.random.normal(0, 0.1), 0, 1
                )
            else:
                # if random.random() < 0.1:
                #     mutated[3][i][j] = 0
                # else:
                mutated[3][i][j] = max(
                    0, np.clip(mutated[3][i][j] + np.random.normal(0, 0.1), 0, 1)
                )
                mutated[4][i][j] = np.clip(
                    mutated[4][i][j] + np.random.normal(0, 0.1), 0, 1
                )

        # Part 6 & 7
        for i in range(self.n_damage_points):
            j = random.randint(0, self.n_hospitals - 1)
            if mutated[5][i][j] == 0:
                # if random.random() < 0.1:
                mutated[5][i][j] = random.uniform(0, 0.2)
                mutated[6][i][j] = max(
                    0, np.clip(mutated[6][i][j] + np.random.normal(0, 0.1), 0, 1)
                )
            else:
                # if random.random() < 0.1:
                #     mutated[5][i][j] = 0
                # else:
                mutated[5][i][j] = max(
                    0, np.clip(mutated[5][i][j] + np.random.normal(0, 0.1), 0, 1)
                )
                mutated[6][i][j] = max(
                    0, np.clip(mutated[6][i][j] + np.random.normal(0, 0.1), 0, 1)
                )
            k = random.randint(
                self.n_hospitals, self.n_hospitals + self.n_temp_medical - 1
            )
            if mutated[5][i][k] == 0:
                # if random.random() < 0.1:
                mutated[5][i][k] = random.uniform(0, 0.2)
                mutated[6][i][k] = max(
                    0, np.clip(mutated[6][i][k] + np.random.normal(0, 0.1), 0, 1)
                )
            else:
                # if random.random() < 0.1:
                #     mutated[5][i][k] = 0
                # else:
                mutated[5][i][k] = max(
                    0, np.clip(mutated[5][i][k] + np.random.normal(0, 0.1), 0, 1)
                )
                mutated[6][i][k] = max(
                    0, np.clip(mutated[6][i][k] + np.random.normal(0, 0.1), 0, 1)
                )
        mutated = self.chromosome_repair(mutated)
        return mutated

    def crossover_per_part(self, parent1, parent2):
        """
        Crossover per-part with AI-selected methods for each part, considering per-part probability
        """
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)
        cpp = self.current_methods.get("crossover_part_probability", {})

        def get_prob(part):
            p = cpp.get(part)
            return float(p) if p is not None else 1.0

        # Part 1
        if np.random.rand() < get_prob("part1"):
            part1_method = self.current_methods["crossover_methods"]["part1"]
            if part1_method != "default" and part1_method in self.crossover_methods:
                c1, c2 = self.crossover_methods[part1_method](parent1[0], parent2[0])
                child1[0] = c1
                child2[0] = c2
            else:
                point = random.choice(range(1, self.n_shelters))
                child1[0][:point], child2[0][:point] = (
                    child2[0][:point],
                    child1[0][:point],
                )
        else:
            child1[0] = deepcopy(parent1[0])
            child2[0] = deepcopy(parent2[0])

        # Part 2
        if np.random.rand() < get_prob("part2"):
            part2_method = self.current_methods["crossover_methods"]["part2"]
            if part2_method != "default" and part2_method in self.crossover_methods:
                c1, c2 = self.crossover_methods[part2_method](parent1[1], parent2[1])
                child1[1] = c1
                child2[1] = c2
            else:
                point = random.choice(range(1, self.n_shelters))
                child1[1][:point], child2[1][:point] = (
                    child2[1][:point],
                    child1[1][:point],
                )
        else:
            child1[1] = deepcopy(parent1[1])
            child2[1] = deepcopy(parent2[1])

        # Part 3 Section 1
        if np.random.rand() < get_prob("part3_p1"):
            part3_p1_method = self.current_methods["crossover_methods"]["part3_p1"]
            if (
                part3_p1_method != "default"
                and part3_p1_method in self.crossover_methods
            ):
                p3_p1 = parent1[2][: self.n_damage_points]
                p3_p2 = parent2[2][: self.n_damage_points]
                c1, c2 = self.crossover_methods[part3_p1_method](p3_p1, p3_p2)
                child1[2][: self.n_damage_points] = c1
                child2[2][: self.n_damage_points] = c2
            else:
                if self.n_damage_points > 0:
                    part3_p1 = parent1[2][: self.n_damage_points]
                    part3_p2 = parent2[2][: self.n_damage_points]
                    child1_part3, child2_part3 = self.order_crossover(
                        part3_p1, part3_p2
                    )
                    child1[2][: self.n_damage_points] = child1_part3
                    child2[2][: self.n_damage_points] = child2_part3
        else:
            child1[2][: self.n_damage_points] = deepcopy(
                parent1[2][: self.n_damage_points]
            )
            child2[2][: self.n_damage_points] = deepcopy(
                parent2[2][: self.n_damage_points]
            )

        # Part 3 Section 2
        if np.random.rand() < get_prob("part3_p2"):
            part3_p2_method = self.current_methods["crossover_methods"]["part3_p2"]
            if (
                part3_p2_method != "default"
                and part3_p2_method in self.crossover_methods
            ):
                p3_p1_sec2 = parent1[2][self.n_damage_points :]
                p3_p2_sec2 = parent2[2][self.n_damage_points :]
                c1_sec2, c2_sec2 = self.crossover_methods[part3_p2_method](
                    p3_p1_sec2, p3_p2_sec2
                )
                child1[2][self.n_damage_points :] = c1_sec2
                child2[2][self.n_damage_points :] = c2_sec2
            else:
                for i in range(self.n_damage_points, self.n_shelters):
                    if np.random.rand() < 0.5:
                        child1[2][i], child2[2][i] = child2[2][i], child1[2][i]
        else:
            child1[2][self.n_damage_points :] = deepcopy(
                parent1[2][self.n_damage_points :]
            )
            child2[2][self.n_damage_points :] = deepcopy(
                parent2[2][self.n_damage_points :]
            )

        # Part 4
        if np.random.rand() < get_prob("part4"):
            part4_method = self.current_methods["crossover_methods"]["part4"]
            if part4_method != "default" and part4_method in self.crossover_methods:
                c1, c2 = self.crossover_methods[part4_method](parent1[3], parent2[3])
                child1[3] = np.array(c1).reshape(parent1[3].shape)
                child2[3] = np.array(c2).reshape(parent2[3].shape)
            else:
                mask_matrix = (
                    np.random.rand(self.n_damage_points, self.n_hospitals) < 0.5
                )
                child1[3][mask_matrix] = parent2[3][mask_matrix]
                child2[3][mask_matrix] = parent1[3][mask_matrix]
        else:
            child1[3] = deepcopy(parent1[3])
            child2[3] = deepcopy(parent2[3])

        # Part 5
        if np.random.rand() < get_prob("part5"):
            part5_method = self.current_methods["crossover_methods"]["part5"]
            if part5_method != "default" and part5_method in self.crossover_methods:
                c1, c2 = self.crossover_methods[part5_method](parent1[4], parent2[4])
                child1[4] = np.array(c1).reshape(parent1[4].shape)
                child2[4] = np.array(c2).reshape(parent2[4].shape)
            else:
                point1 = np.random.choice(range(1, self.n_damage_points))
                point2 = np.random.choice(range(1, self.n_hospitals))
                child1[4][:point1, :point2], child2[4][:point1, :point2] = (
                    child2[4][:point1, :point2],
                    child1[4][:point1, :point2],
                )
                child1[4][point1:, point2:], child2[4][point1:, point2:] = (
                    child2[4][point1:, point2:],
                    child1[4][point1:, point2:],
                )
        else:
            child1[4] = deepcopy(parent1[4])
            child2[4] = deepcopy(parent2[4])

        # Part 6
        if np.random.rand() < get_prob("part6"):
            part6_method = self.current_methods["crossover_methods"]["part6"]
            if part6_method != "default" and part6_method in self.crossover_methods:
                c1, c2 = self.crossover_methods[part6_method](parent1[5], parent2[5])
                child1[5] = np.array(c1).reshape(parent1[5].shape)
                child2[5] = np.array(c2).reshape(parent2[5].shape)
            else:
                mask_matrix = (
                    np.random.rand(
                        self.n_damage_points, self.n_hospitals + self.n_temp_medical
                    )
                    < 0.5
                )
                child1[5][mask_matrix] = parent2[5][mask_matrix]
                child2[5][mask_matrix] = parent1[5][mask_matrix]
        else:
            child1[5] = deepcopy(parent1[5])
            child2[5] = deepcopy(parent2[5])

        # Part 7
        if np.random.rand() < get_prob("part7"):
            part7_method = self.current_methods["crossover_methods"]["part7"]
            if part7_method != "default" and part7_method in self.crossover_methods:
                c1, c2 = self.crossover_methods[part7_method](parent1[6], parent2[6])
                child1[6] = np.array(c1).reshape(parent1[6].shape)
                child2[6] = np.array(c2).reshape(parent2[6].shape)
            else:
                point1 = np.random.choice(range(1, self.n_damage_points))
                point2, point3 = sorted(
                    random.sample(range(1, self.n_hospitals + self.n_temp_medical), k=2)
                )
                child1[6][:point1, point2:point3], child2[6][:point1, point2:point3] = (
                    child2[6][:point1, point2:point3],
                    child1[6][:point1, point2:point3],
                )
                child1[6][point1:, :point2], child2[6][point1:, :point2] = (
                    child2[6][point1:, :point2],
                    child1[6][point1:, :point2],
                )
                child1[6][point1:, point3:], child2[6][point1:, point3:] = (
                    child2[6][point1:, point3:],
                    child1[6][point1:, point3:],
                )
        else:
            child1[6] = deepcopy(parent1[6])
            child2[6] = deepcopy(parent2[6])

        # تعمیر کروموزوم‌ها
        child1 = self.chromosome_repair(child1)
        child2 = self.chromosome_repair(child2)
        return child1, child2

    def mutate_per_part(self, chromosome):
        """
        Mutation per-part with AI-selected methods for each part, considering per-part probability
        """
        mutated = deepcopy(chromosome)
        mpp = self.current_methods.get("mutation_part_probability", {})
        mutation_args_per_part = self.current_methods.get("mutation_args_per_part", {})

        def get_prob(part):
            p = mpp.get(part)
            return float(p) if p is not None else 1.0
        
        def get_mut_args(part_key: str, include_inner: bool = False) -> dict:
            base = {}
            # per-part overrides
            part_overrides = mutation_args_per_part.get(part_key, {}) if isinstance(mutation_args_per_part, dict) else {}
            if isinstance(part_overrides, dict):
                base.update(part_overrides)
            return base

        # Part 1
        if np.random.rand() < get_prob("part1"):
            part1_method = self.current_methods["mutation_methods"]["part1"]
            if part1_method != "default" and part1_method in self.mutation_methods:
                mutated[0] = self.mutation_methods[part1_method](chromosome[0], **get_mut_args("part1"))
            else:
                i = np.random.randint(0, self.n_shelters)
                if mutated[0][i] == 0:
                    mutated[0][i] = random.randint(1, self.n_distribution)
                else:
                    mutated[0][i] = random.randint(1, self.n_distribution)
        # else: keep unchanged

        # Part 2
        if np.random.rand() < get_prob("part2"):
            part2_method = self.current_methods["mutation_methods"]["part2"]
            if part2_method != "default" and part2_method in self.mutation_methods:
                mutated[1] = self.mutation_methods[part2_method](chromosome[1], **get_mut_args("part2", include_inner=True))
            else:
                i = np.random.randint(0, self.n_shelters)
                while mutated[1][i] == 0:
                    i = np.random.randint(0, self.n_shelters)
                mutated[1][i] = max(
                    0, np.clip(mutated[1][i] + np.random.normal(0, 0.1), 0, 1)
                )
        # else: keep unchanged

        # Part 3 Section 1
        if np.random.rand() < get_prob("part3_p1"):
            part3_p1_method = self.current_methods["mutation_methods"]["part3_p1"]
            if (
                part3_p1_method != "default"
                and part3_p1_method in self.mutation_methods
            ):
                mutated_p1 = self.mutation_methods[part3_p1_method](chromosome[2][: self.n_damage_points], **get_mut_args("part3_p1"))
                mutated[2][: self.n_damage_points] = mutated_p1
            else:
                if self.n_damage_points > 1:
                    i = np.random.randint(0, self.n_damage_points)
                    j = [k for k in range(self.n_damage_points) if k != i]
                    j = random.choice(j)
                    mutated[2][i], mutated[2][j] = mutated[2][j], mutated[2][i]
        # else: keep unchanged

        # Part 3 Section 2
        if np.random.rand() < get_prob("part3_p2"):
            part3_p2_method = self.current_methods["mutation_methods"]["part3_p2"]
            if (
                part3_p2_method != "default"
                and part3_p2_method in self.mutation_methods
            ):
                mutated_p2 = self.mutation_methods[part3_p2_method](chromosome[2][self.n_damage_points :], **get_mut_args("part3_p2"))
                mutated[2][self.n_damage_points :] = mutated_p2
            else:
                if self.n_shelters > self.n_damage_points:
                    i = np.random.randint(self.n_damage_points, self.n_shelters)
                    if mutated[2][i] == 0:
                        mutated[2][i] = random.choice(self.da_id)
                    else:
                        mutated[2][i] = 0
        # else: keep unchanged

        # The below (matrix) parts are left unchanged if skip, not forcibly mutated.
        # Parts 4,5,6,7:
        for part_idx, part_key in zip(
            range(3, 7), ["part4", "part5", "part6", "part7"]
        ):
            if np.random.rand() < get_prob(part_key):
                method = self.current_methods["mutation_methods"][part_key]
                if method != "default" and method in self.mutation_methods:
                    mutated_flat = self.mutation_methods[method](chromosome[part_idx], **get_mut_args(part_key, include_inner=True))
                    mutated[part_idx] = np.array(mutated_flat).reshape(
                        chromosome[part_idx].shape
                    )
                else:
                    # Use vanilla default(s) as in old code
                    if part_key == "part4":
                        for i in range(self.n_damage_points):
                            j = random.randint(0, self.n_hospitals - 1)
                            if mutated[3][i][j] == 0:
                                mutated[3][i][j] = random.uniform(0.1, 0.5)
                            else:
                                mutated[3][i][j] = max(
                                    0,
                                    np.clip(
                                        mutated[3][i][j] + np.random.normal(0, 0.1),
                                        0,
                                        1,
                                    ),
                                )
                    elif part_key == "part5":
                        for i in range(self.n_damage_points):
                            j = random.randint(0, self.n_hospitals - 1)
                            if mutated[4][i][j] > 0:
                                mutated[4][i][j] = np.clip(
                                    mutated[4][i][j] + np.random.normal(0, 0.1), 0, 1
                                )
                    elif part_key == "part6":
                        for i in range(self.n_damage_points):
                            j = random.randint(
                                0, self.n_hospitals + self.n_temp_medical - 1
                            )
                            if mutated[5][i][j] == 0:
                                mutated[5][i][j] = random.uniform(0, 0.2)
                            else:
                                mutated[5][i][j] = max(
                                    0,
                                    np.clip(
                                        mutated[5][i][j] + np.random.normal(0, 0.1),
                                        0,
                                        1,
                                    ),
                                )
                            if mutated[6][i][j] > 0:
                                mutated[6][i][j] = max(
                                    0,
                                    np.clip(
                                        mutated[6][i][j] + np.random.normal(0, 0.1),
                                        0,
                                        1,
                                    ),
                                )
                    elif part_key == "part7":
                        for i in range(self.n_damage_points):
                            j = random.randint(
                                0, self.n_hospitals + self.n_temp_medical - 1
                            )
                            if mutated[6][i][j] > 0:
                                mutated[6][i][j] = max(
                                    0,
                                    np.clip(
                                        mutated[6][i][j] + np.random.normal(0, 0.1),
                                        0,
                                        1,
                                    ),
                                )
                            if mutated[5][i][j] > 0:
                                mutated[5][i][j] = max(
                                    0,
                                    np.clip(
                                        mutated[5][i][j] + np.random.normal(0, 0.1),
                                        0,
                                        1,
                                    ),
                                )
        mutated = self.chromosome_repair(mutated)
        return mutated

    def run(self, problem):
        """
        Run NSGA-II for humanitarian logistics problem
        """
        # Extract problem info
        cost_function = problem["cost_function"]
        checkpoint_path = problem.get("checkpoint_path", "exports/llm_nsga2_checkpoint.pkl")
        resume = self.resume

        # Ensure the checkpoint directory exists
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir and checkpoint_dir != ".":
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Empty individual
        empty_individual = {
            "chromosome": None,
            "cost": None,
            "normal_cost": None,
            "rank": None,
            "crowding_distance": None,
            "constriant_violation": 0,
        }

        # Initialize or resume population/state
        pop = None
        F = None
        pareto_pop_list = []
        all_pop_list = []  # Store entire population history
        start_it = 0
        if resume and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "rb") as f:
                    state = pickle.load(f)
                start_it = int(state.get("iteration", -1)) + 1
                pop = state.get("pop")
                F = state.get("F")
                pareto_pop_list = state.get("pareto_history", [])
                all_pop_list = state.get("all_pop_history", [])
                # Restore histories and operators if present
                loaded_history = state.get("metrics_history", {})
                # تبدیل فرمت قدیم (list of dicts) به فرمت جدید (dict with lists) برای backward compatibility
                if isinstance(loaded_history, list):
                    # فرمت قدیم: list of dicts
                    self.metrics_history = {}
                    for metrics_dict in loaded_history:
                        for key, value in metrics_dict.items():
                            if key not in self.metrics_history:
                                self.metrics_history[key] = []
                            self.metrics_history[key].append(round(value, 4))
                else:
                    # فرمت جدید: dict with lists
                    self.metrics_history = loaded_history
                
                # Restore section_stats_history
                loaded_section_stats = state.get("section_stats_history", {})
                # تبدیل فرمت قدیم (list of dicts) به فرمت جدید (dict with nested dicts and lists)
                if isinstance(loaded_section_stats, list):
                    # فرمت قدیم: list of dicts
                    self.section_stats_history = {}
                    for stats_dict in loaded_section_stats:
                        for part_key, part_stats in stats_dict.items():
                            if part_key not in self.section_stats_history:
                                self.section_stats_history[part_key] = {}
                            if isinstance(part_stats, dict):
                                for stat_key, stat_value in part_stats.items():
                                    if stat_key not in self.section_stats_history[part_key]:
                                        self.section_stats_history[part_key][stat_key] = []
                                    self.section_stats_history[part_key][stat_key].append(stat_value)
                else:
                    # فرمت جدید: dict with nested dicts and lists
                    self.section_stats_history = loaded_section_stats
                
                if "current_methods" in state:
                    self.current_methods = state["current_methods"]
                if "hypervolume_manager" in state:
                    self.hypervolume_manager = state["hypervolume_manager"]
                if self.verbose:
                    print(f"Resumed from checkpoint at iteration {start_it}")
            except Exception:
                if self.verbose:
                    print("Failed to load checkpoint. Starting fresh.")
                pop = None
        if pop is None:
            pop = [deepcopy(empty_individual) for _ in range(self.pop_size)]
            # Try AI-generated initial population first if enabled
            ai_init_ok = False
            if self.use_llm_init_pop and self.openrouter_client:
                dims = {
                    "n_shelters": self.n_shelters,
                    "n_distribution": self.n_distribution,
                    "n_damage_points": self.n_damage_points,
                    "n_hospitals": self.n_hospitals,
                    "n_temp_medical": self.n_temp_medical,
                }
                try:
                    ids_payload = {
                        "idc_id": self.idc_id,
                        "ec_id": self.shelter_id_list,
                        "da_id": self.da_id,
                        "h_id": self.hospital_id_list,
                        "tmc_id": self.tmc_id_list,
                    }
                    ai_pop = self.openrouter_client.get_initial_population(
                        self.pop_size,
                        dims,
                        ids_payload,
                        self.distances,
                        self.homeless,
                        self.severe_injured,
                        self.minor_injured,
                        self.cost,
                        self.capacity,
                    )
                except Exception:
                    ai_pop = None
                if isinstance(ai_pop, list) and len(ai_pop) > 0:
                    filled = 0
                    for item in ai_pop:
                        chrom = self._validate_and_convert_chromosome(item)
                        if chrom is not None and filled < self.pop_size:
                            pop[filled]["chromosome"] = chrom
                            filled += 1
                    # fill remaining with random if needed
                    for i in range(filled, self.pop_size):
                        pop[i]["chromosome"] = self.create_random_chromosome()
                    ai_init_ok = filled > 0
            else:
                # If LLM-based init is disabled, try loading from initial_population.json
                try:
                    import json
                    init_path = os.path.join(os.path.dirname(__file__), "initial_population.json")
                    if os.path.exists(init_path):
                        if self.verbose:
                            print(f"✓ Found initial_population.json at: {init_path}")
                        with open(init_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if "population" in data and len(data["population"]) > 0:
                            filled = 0
                            for chromosome_dict in data["population"]:
                                chrom = self._validate_and_convert_chromosome(chromosome_dict)
                                if chrom is not None and filled < self.pop_size:
                                    pop[filled]["chromosome"] = chrom
                                    filled += 1
                            # fill remaining with random if needed
                            for i in range(filled, self.pop_size):
                                pop[i]["chromosome"] = self.create_random_chromosome()
                            ai_init_ok = filled > 0
                            if self.verbose:
                                print(f"✓ Loaded {filled} chromosomes from initial_population.json")
                        else:
                            if self.verbose:
                                print(f"⚠ initial_population.json exists but is empty or invalid format")
                    else:
                        if self.verbose:
                            print(f"ℹ initial_population.json not found at: {init_path}")
                            print(f"  Using random initialization instead")
                except Exception as e:
                    # Any error -> ignore and fallback later
                    if self.verbose:
                        print(f"⚠ Error loading initial_population.json: {e}")
                    ai_init_ok = False
            # fallback if AI/file init disabled or failed
            if not ai_init_ok:
                for i in range(self.pop_size):
                    pop[i]["chromosome"] = self.create_random_chromosome()
            chromosom_list = []
            for p in pop:
                chromosom_list.append(p["chromosome"])
            costs, constriants_violation, normal_cost = cost_function(chromosom_list)
            for idx, i in enumerate(costs):
                pop[idx]["cost"] = np.array(i)
                pop[idx]["normal_cost"] = np.array(normal_cost[idx])
                pop[idx]["constriant_violation"] = constriants_violation[idx]
            # Non-dominated sorting
            pop, F = self.non_dominated_sorting(pop)
            # Calculate crowding distance
            pop = self.calc_crowding_distance(pop, F)
            # Sort population
            pop, F = self.sort_population(pop)

        # ذخیره مقادیر اولیه در history (فقط اگر history خالی باشد)
        # این برای حالت اولیه (start_it == 0) یا برای backward compatibility با checkpoint های قدیمی
        if start_it == 0 or not self._is_history_initialized():
            self._save_initial_methods_to_history()
        
        # Main loop
        pareto_pop = []  # safe default to avoid UnboundLocalError on early failures
        for it in range(start_it, self.max_iter):
            # انتخاب افراد نخبه از نسل فعلی
            elite_individuals = self.select_elite(pop, F)

            # محاسبه تعداد افراد جدید مورد نیاز (کل جمعیت منهای نخبگان)
            n_new_individuals = self.pop_size - len(elite_individuals)

            # جمع‌آوری شاخص‌های عملکرد
            current_metrics = self.collect_metrics(pop, F, it)
            # اضافه کردن مقادیر به تاریخچه (به فرمت dict با لیست مقادیر)
            for key, value in current_metrics.items():
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                if isinstance(value, list):
                    self.metrics_history[key].append([round(v, 4) for v in value])
                else:
                    self.metrics_history[key].append(round(value, 4))

            # درخواست پیشنهادات AI هر 5 نسل
            if (
                self.use_ai_optimization
                and self.openrouter_client
                and it > 0
                and it % self.llm_iter == 0
            ):
                print(f"\n🤖 Requesting AI recommendations for generation {it}...")
                # اگر خطا رخ دهد، exception پرتاب می‌شود و الگوریتم متوقف می‌شود
                prompt_args = {
                    'llm_iter': self.llm_iter,
                    'it': it,
                    'current_methods': self.current_methods,
                    'metrics_history': self.metrics_history,
                    'offspring_survival_history': self.offspring_survival_history,
                    'section_stats_history': self.section_stats_history
                }
                start_time = time.time()
                recommendations = self.get_ai_recommendations(**prompt_args)
                end_time = time.time()
                exe_time = end_time - start_time
                print(f"AI recomendation time: {exe_time} seconds")
                self.update_genetic_operators(recommendations)

            # Crossover
            popc = []
            n_crossover_pairs = n_new_individuals // 2
            for _ in range(n_crossover_pairs):
                p1 = self.crowding_tournament_selection(pop)
                p2 = self.crowding_tournament_selection(pop)
                if random.uniform(0, 1) < self.p_crossover:
                    # چک کردن آیا روش‌های per-part تعریف شده‌اند
                    has_custom_methods = any(
                        [
                            self.current_methods["crossover_methods"]["part1"]
                            != "default",
                            self.current_methods["crossover_methods"]["part2"]
                            != "default",
                            self.current_methods["crossover_methods"]["part3_p1"]
                            != "default",
                            self.current_methods["crossover_methods"]["part3_p2"]
                            != "default",
                            self.current_methods["crossover_methods"]["part4"]
                            != "default",
                            self.current_methods["crossover_methods"]["part5"]
                            != "default",
                            self.current_methods["crossover_methods"]["part6"]
                            != "default",
                            self.current_methods["crossover_methods"]["part7"]
                            != "default",
                        ]
                    )

                    if has_custom_methods:
                        # استفاده از متدهای per-part
                        c1_chrom, c2_chrom = self.crossover_per_part(
                            p1["chromosome"], p2["chromosome"]
                        )
                    else:
                        # استفاده از روش پیش‌فرض
                        c1_chrom, c2_chrom = self.crossover(
                            p1["chromosome"], p2["chromosome"]
                        )

                    c1 = deepcopy(empty_individual)
                    c2 = deepcopy(empty_individual)
                    c1["chromosome"] = c1_chrom
                    c2["chromosome"] = c2_chrom
                    popc.append(c1)
                    popc.append(c2)
                else:
                    popc.append(deepcopy(p1))
                    popc.append(deepcopy(p2))

            # اگر تعداد فرد باقی‌مانده فرد باشد، یک فرد اضافی تولید کن
            if len(popc) < n_new_individuals:
                p = self.crowding_tournament_selection(pop)
                popc.append(deepcopy(p))

            # Mutation
            popm = []
            for i in range(len(popc)):
                if random.uniform(0, 1) < self.p_mutation:
                    m = deepcopy(empty_individual)
                    # چک کردن آیا روش‌های per-part تعریف شده‌اند
                    has_custom_methods = any(
                        [
                            self.current_methods["mutation_methods"]["part1"]
                            != "default",
                            self.current_methods["mutation_methods"]["part2"]
                            != "default",
                            self.current_methods["mutation_methods"]["part3_p1"]
                            != "default",
                            self.current_methods["mutation_methods"]["part3_p2"]
                            != "default",
                            self.current_methods["mutation_methods"]["part4"]
                            != "default",
                            self.current_methods["mutation_methods"]["part5"]
                            != "default",
                            self.current_methods["mutation_methods"]["part6"]
                            != "default",
                            self.current_methods["mutation_methods"]["part7"]
                            != "default",
                        ]
                    )

                    if has_custom_methods:
                        # استفاده از متدهای per-part
                        m["chromosome"] = self.mutate_per_part(popc[i]["chromosome"])
                    else:
                        # استفاده از روش پیش‌فرض
                        m["chromosome"] = self.mutate(popc[i]["chromosome"])
                    popm.append(m)
                else:
                    popm.append(popc[i])

            # ترکیب نخبگان با افراد جدید
            pop = elite_individuals + popm

            # محاسبه cost برای افراد جدید (نخبگان قبلاً محاسبه شده‌اند)
            new_chromosomes = []
            for i in range(len(elite_individuals), len(pop)):
                if pop[i]["cost"] is None:
                    new_chromosomes.append(pop[i]["chromosome"])

            if new_chromosomes:
                costs, constraints_violation, normal_costs = cost_function(
                    new_chromosomes
                )
                new_idx = 0
                for i in range(len(elite_individuals), len(pop)):
                    if pop[i]["cost"] is None:
                        pop[i]["cost"] = np.array(costs[new_idx])
                        pop[i]["normal_cost"] = np.array(normal_costs[new_idx])
                        pop[i]["constriant_violation"] = constraints_violation[new_idx]
                        new_idx += 1

            # Non-dominated sorting
            pop, F = self.non_dominated_sorting(pop)

            # Calculate crowding distance
            pop = self.calc_crowding_distance(pop, F)

            # Sort population
            pop, F = self.sort_population(pop)

            # Truncate (اگر لازم باشد)
            pop, F = self.truncate_population(pop, F)

            # تعداد فرزندان زنده‌مانده بعد از انتخاب محیطی (بدون نخبگان)
            try:
                survived_offspring = self.count_survived_offspring(elite_individuals, popm, pop)
                self.offspring_survival_history.append(survived_offspring)
            except Exception:
                survived_offspring = None

            # اضافه کردن محاسبه metrics
            if len(F) > 0 and len(F[0]) > 0:
                pareto_pop = [pop[i] for i in F[0]]
            else:
                # fallback: take top-k by rank if fronts empty
                sorted_pop, sorted_F = self.sort_population(pop)
                pareto_pop = [sorted_pop[i] for i in (sorted_F[0] if len(sorted_F) > 0 else [])]
            pareto_pop_list.append(deepcopy(pareto_pop))
            all_pop_list.append(deepcopy(pop))  # Store entire population
            # self.diagnostics.update_all_metrics(
            # population=pop,
            # pareto_pop=pareto_pop,
            # offspring_pop=popc,
            # mutation_pop=popm
            # )

            # Display iteration info
            if self.verbose:
                print(f"\nIteration {it + 1}:")
                print(f"    Number of Pareto Members = {len(F[0])}")
                print(f"    Elite individuals preserved: {len(elite_individuals)}")
                print(f"    New individuals generated: {len(popm)}")
                if survived_offspring is not None:
                    print(f"    Offspring that survived selection: {survived_offspring}")
                # if it > 0:
                # print(f'   Hypervolume: {self.metrics.hypervolume_history[-1]:.6f}')
                # print(f'   Spacing: {self.metrics.spacing_history[-1]:.6f}')
                # if it > 0 and len(self.diagnostics.diversity_history) > 0:
                #     print(f'   Diversity: {self.diagnostics.diversity_history[-1]:.4f}')
                #     print(f'   Selection Pressure: {self.diagnostics.selection_pressure[-1]:.4f}')

            # Save checkpoint at the end of each successful iteration
            try:
                state = {
                    "iteration": it,
                    "pop": pop,
                    "F": F,
                    "pareto_history": pareto_pop_list,
                    "all_pop_history": all_pop_list,
                    "metrics_history": self.metrics_history,
                    "section_stats_history": self.section_stats_history,
                    "current_methods": self.current_methods,
                    "hypervolume_manager": self.hypervolume_manager,
                }
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(state, f)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: failed to save checkpoint at iteration {it}: {e}")
            except KeyboardInterrupt:
                # Ensure checkpoint on manual stop
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(state, f)
                raise

            except BaseException:
                # Ensure checkpoint on any unforeseen fatal error
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(state, f)
                raise

            except SystemExit:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(state, f)
                raise

            # End of iteration

        # محاسبه نهایی هایپرولیوم با نقطه مرجع ثابت
        # print("\nFinal calculation of hypervol with a fixed reference point...")
        # final_hypervolumes = self.hypervolume_manager.calculate_final_hypervolumes()

        # به‌روزرسانی metrics با هایپرولیوم نهایی
        self.metrics.update_metrics(pareto_pop_list, all_pop_list=all_pop_list)
        # self.metrics.hypervolume_history = final_hypervolumes

        # Ensure pareto_pop is defined even if no fronts exist
        if not pareto_pop:
            if len(F) > 0 and len(F[0]) > 0:
                pareto_pop = [pop[i] for i in F[0]]
            else:
                pareto_pop = []

        return {
            "pop": pop,
            "F": F,
            "pareto_pop": pareto_pop,
            "pareto_history": pareto_pop_list,
            "metrics": self.metrics,
            "hypervolume_manager": self.hypervolume_manager,
            # 'diagnostics': self.diagnostics
        }

    def dominates(self, p, q):
        """Check if p dominates q"""
        if p["constriant_violation"] > 0 and q["constriant_violation"] > 0:
            return p["constriant_violation"] < q["constriant_violation"]
        elif p["constriant_violation"] > 0 and q["constriant_violation"] == 0:
            return False
        elif p["constriant_violation"] == 0 and q["constriant_violation"] > 0:
            return True
        elif p["constriant_violation"] == 0 and q["constriant_violation"] == 0:
            return all(p["cost"] <= q["cost"]) and any(p["cost"] < q["cost"])

    def non_dominated_sorting(self, pop):
        """Non-dominated sorting (unchanged from original)"""
        pop_size = len(pop)
        domination_set = [[] for _ in range(pop_size)]
        dominated_count = [0 for _ in range(pop_size)]
        F = [[]]

        for i in range(pop_size):
            for j in range(i + 1, pop_size):
                if self.dominates(pop[i], pop[j]):
                    domination_set[i].append(j)
                    dominated_count[j] += 1
                elif self.dominates(pop[j], pop[i]):
                    domination_set[j].append(i)
                    dominated_count[i] += 1

            if dominated_count[i] == 0:
                pop[i]["rank"] = 0
                F[0].append(i)

        k = 0
        while True:
            Q = []
            for i in F[k]:
                for j in domination_set[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        pop[j]["rank"] = k + 1
                        Q.append(j)

            if not Q:
                break

            F.append(Q)
            k += 1

        return pop, F

    def calc_crowding_distance(self, pop, F):
        """Calculate crowding distance (unchanged from original)"""
        parto_count = len(F)
        n_obj = len(pop[0]["cost"])

        for k in range(parto_count):
            costs = np.array([pop[i]["normal_cost"] for i in F[k]])
            n = len(F[k])
            d = np.zeros((n, n_obj))

            for j in range(n_obj):
                idx = np.argsort(costs[:, j])
                d[idx[0], j] = np.inf
                d[idx[-1], j] = np.inf

                for i in range(1, n - 1):
                    d[idx[i], j] = costs[idx[i + 1], j] - costs[idx[i - 1], j]
                    if costs[idx[-1], j] != costs[idx[0], j]:
                        d[idx[i], j] /= costs[idx[-1], j] - costs[idx[0], j]

            for i in range(n):
                pop[F[k][i]]["crowding_distance"] = sum(d[i, :])

        return pop

    def sort_population(self, pop):
        """Sort population by rank and crowding distance"""
        pop = sorted(pop, key=lambda x: (x["rank"], -x["crowding_distance"]))

        max_rank = pop[-1]["rank"]
        F = []
        for r in range(max_rank + 1):
            F.append([i for i in range(len(pop)) if pop[i]["rank"] == r])

        return pop, F

    def truncate_population(self, pop, F, pop_size=None):
        """Truncate population to specified size"""
        if pop_size is None:
            pop_size = self.pop_size

        if len(pop) <= pop_size:
            return pop, F

        pop = pop[:pop_size]

        for k in range(len(F)):
            F[k] = [i for i in F[k] if i < pop_size]

        return pop, F

    def select_elite(self, pop, F):
        """
        انتخاب افراد نخبه از جمعیت فعلی
        بهترین افراد از رتبه اول (Pareto front) انتخاب می‌شوند
        """
        elite = []

        # ابتدا تمام افراد رتبه اول را اضافه کن
        if len(F) > 0 and len(F[0]) > 0:
            pareto_front = [pop[i] for i in F[0]]

            # اگر تعداد افراد رتبه اول کمتر از تعداد نخبگان مورد نیاز است
            if len(pareto_front) <= self.n_elite:
                elite.extend(pareto_front)
                remaining = self.n_elite - len(pareto_front)

                # از رتبه‌های بعدی نیز انتخاب کن
                for rank in range(1, len(F)):
                    if remaining <= 0:
                        break
                    rank_individuals = [pop[i] for i in F[rank]]
                    # مرتب‌سازی بر اساس crowding distance
                    rank_individuals.sort(
                        key=lambda x: x["crowding_distance"], reverse=True
                    )

                    take = min(remaining, len(rank_individuals))
                    elite.extend(rank_individuals[:take])
                    remaining -= take
            else:
                # اگر تعداد افراد رتبه اول بیشتر از نیاز است، بهترین‌ها را انتخاب کن
                pareto_front.sort(key=lambda x: x["crowding_distance"], reverse=True)
                elite = pareto_front[: self.n_elite]

        return elite

    def count_survived_offspring(self, elites, offspring_candidates, new_population):
        """
        شمارش تعداد فرزندانی که از نسل جاری تولید شده‌اند و در جمعیت جدید باقی مانده‌اند.
        تعریف:
        - فرزندان همان افرادی هستند که بعد از مرحله crossover/mutation در لیست offspring_candidates (popm) قرار دارند.
        - نخبگان (elites) افراد حفظ‌شده از نسل قبل هستند و نباید در شمارش لحاظ شوند.
        - new_population جمعیت نهایی پس از ترکیب نخبگان با فرزندان و انتخاب/مرتب‌سازی است.

        روش شمارش:
        - برای هر عضو new_population بررسی می‌کنیم آیا کروموزومش مساوی با یکی از کروموزوم‌های offspring_candidates است.
        - تطابق را با مقایسه عمیق ساختار کروموزوم (لیست‌ها و آرایه‌ها) انجام می‌دهیم.
        - به ازای هر کروموزوم فرزند فقط یک بار شمرده می‌شود تا تکرار حساب نشود.
        """
        def chromo_equal(c1, c2):
            if c1 is None or c2 is None:
                return False
            # part1, part2, part3: lists
            for i in range(3):
                if c1[i] != c2[i]:
                    return False
            # parts 4..7: numpy arrays, compare with array_equal
            for i in range(3, 7):
                if not np.array_equal(np.array(c1[i]), np.array(c2[i])):
                    return False
            return True

        # build multiset of offspring chromosomes (allow duplicates) using consumed flags
        consumed = [False] * len(offspring_candidates)
        count = 0
        # skip first len(elites) of new_population since they are elites by construction
        for ind in new_population[len(elites):]:
            ch = ind.get("chromosome")
            # try find a matching offspring not yet consumed
            for idx, off in enumerate(offspring_candidates):
                if not consumed[idx] and chromo_equal(ch, off.get("chromosome")):
                    consumed[idx] = True
                    count += 1
                    break
        return count

    def crowding_tournament_selection(self, pop, tournament_size=2):
        """
        انتخاب والد با استفاده از متد selection انتخاب شده توسط AI
        یا استفاده از متد پیش‌فرض
        """
        # بررسی اینکه آیا متد selection تعریف شده است
        selection_method_key = self.current_methods.get("selection_method", "default")
        
        if (
            selection_method_key != "default"
            and selection_method_key in self.selection_methods
        ):
            # استفاده از متد انتخابی AI
            selection_func = self.selection_methods[selection_method_key]
            selection_args = self.current_methods.get("selection_args", {})
            return selection_func(pop, **selection_args)
        else:
            # استفاده از متد پیش‌فرض (NSGA-II binary tournament)
            idx1, idx2 = np.random.choice(
                range(len(pop)), size=tournament_size, replace=False
            )

            p1, p2 = pop[idx1], pop[idx2]

            # اگر rank متفاوت، rank کمتر برنده
            if p1["rank"] < p2["rank"]:
                return p1
            elif p2["rank"] < p1["rank"]:
                return p2
            else:
                # اگر rank یکسان، crowding distance بیشتر برنده (حفظ تنوع)
                return p1 if p1["crowding_distance"] > p2["crowding_distance"] else p2

    def collect_metrics(self, pop, F, generation):
        """
        جمع‌آوری شاخص‌های عملکرد برای ارسال به AI

        Parameters:
        -----------
        pop: جمعیت فعلی
        F: لیست فرانت‌های Pareto
        generation: نسل فعلی

        Returns:
        --------
        Dict: شاخص‌های عملکرد
        """
        current_metrics = {}

        # محاسبه شاخص‌های اصلی
        if len(F) > 0 and len(F[0]) > 0:
            pareto_pop = [pop[i] for i in F[0]]
            pareto_costs = np.array([ind["cost"] for ind in pareto_pop])

            # ذخیره داده‌های نسل فعلی در hypervolume manager
            self.hypervolume_manager.store_generation_data(generation, pareto_costs)

            # محاسبه هایپرولیوم موقت با نقطه مرجع تخمینی
            preliminary_hv = self.hypervolume_manager.calculate_preliminary_hypervolume(
                generation
            )
            current_metrics["hypervolume"] = preliminary_hv

            # Spacing
            if hasattr(self.metrics, "spacing"):
                normal_pareto_costs = np.array(
                    [ind["normal_cost"] for ind in pareto_pop]
                )
                current_metrics["spacing"] = self.metrics.spacing(normal_pareto_costs)
            else:
                current_metrics["spacing"] = 0.0
            # Spread (Delta)
            if hasattr(self.metrics, "spread"):
                normal_pareto_costs = np.array(
                    [ind["normal_cost"] for ind in pareto_pop]
                )
                current_metrics["spread"] = self.metrics.spread(normal_pareto_costs)
            else:
                current_metrics["spread"] = 0.0

            # تعداد راه‌حل‌های Pareto
            current_metrics["pareto_count"] = len(pareto_pop)

            # میانگین crowding distance
            crowding_distances = np.array([ind["crowding_distance"] for ind in pareto_pop])
            current_metrics["avg_crowding_distance"] = (
                np.mean(crowding_distances[crowding_distances<np.inf]) if crowding_distances.any() else 0.0
            )
        else:
            # مقادیر پیش‌فرض اگر Pareto front خالی باشد
            current_metrics.update(
                {
                    "hypervolume": 0.0,
                    "spacing": 0.0,
                    "spread": 0.0,
                    "pareto_count": 0,
                    "avg_crowding_distance": 0.0,
                }
            )

        # محاسبه min_obj, mean_obj, std_obj از کل جمعیت
        if len(pop) > 0:
            all_pop_costs = np.array([ind["cost"] for ind in pop])
            current_metrics["mean_objectives"] = np.mean(all_pop_costs, axis=0).tolist()
            current_metrics["min_objectives"] = np.min(all_pop_costs, axis=0).tolist()
            current_metrics["std_objectives"] = np.std(all_pop_costs, axis=0).tolist()
        else:
            current_metrics["mean_objectives"] = [0.0, 0.0, 0.0]
            current_metrics["min_objectives"] = [0.0, 0.0, 0.0]
            current_metrics["std_objectives"] = [0.0, 0.0, 0.0]

        # شاخص‌های اضافی
        current_metrics["generation"] = generation
        current_metrics["population_size"] = len(pop)
        current_metrics["elitism_rate"] = self.elitism_rate  # اضافه کردن نرخ نخبه‌گرایی
        # آمار بخشی (section_stats) طبق Prompt 2
        section_stats = self.compute_section_stats(pop)
        
        # اضافه کردن مقادیر section_stats به تاریخچه (به فرمت dict با لیست مقادیر)
        for part_key, part_stats in section_stats.items():
            if part_key not in self.section_stats_history:
                self.section_stats_history[part_key] = {}
            for stat_key, stat_value in part_stats.items():
                if stat_key not in self.section_stats_history[part_key]:
                    self.section_stats_history[part_key][stat_key] = []
                self.section_stats_history[part_key][stat_key].append(stat_value)

        return current_metrics

    def _is_history_initialized(self):
        """
        بررسی می‌کند که آیا history مقداردهی اولیه شده است یا نه
        """
        # اگر حداقل یکی از history ها غیرخالی باشد، یعنی history initialize شده است
        if (self.current_methods["history"]["selection_method_history"] or
            self.current_methods["history"]["global_crossover_probability_history"] or
            self.current_methods["history"]["global_mutation_probability_history"]):
            return True
        
        # بررسی history های part
        for part in ["part1", "part2", "part3_p1", "part3_p2", "part4", "part5", "part6", "part7"]:
            if (self.current_methods["history"]["crossover_methods_history"][part] or
                self.current_methods["history"]["mutation_methods_history"][part]):
                return True
        
        return False

    def _save_initial_methods_to_history(self):
        """
        ذخیره مقادیر اولیه متدها در history
        این متد فقط یک بار در ابتدای اجرا فراخوانی می‌شود
        """
        # ذخیره crossover methods اولیه
        for part in ["part1", "part2", "part3_p1", "part3_p2", "part4", "part5", "part6", "part7"]:
            method_key = self.current_methods["crossover_methods"][part]
            if method_key != "default" and method_key in self.crossover_methods:
                method_name = self.crossover_methods[method_key].__name__
            else:
                method_name = "default"
            self.current_methods["history"]["crossover_methods_history"][part].append(method_name)
        
        # ذخیره mutation methods اولیه
        mutation_args_per_part = self.current_methods.get("mutation_args_per_part", {})
        for part in ["part1", "part2", "part3_p1", "part3_p2", "part4", "part5", "part6", "part7"]:
            method_key = self.current_methods["mutation_methods"][part]
            if method_key != "default" and method_key in self.mutation_methods:
                method_name = self.mutation_methods[method_key].__name__
            else:
                method_name = "default"
            
            # ذخیره نام متد به همراه آرگومان‌های ورودی
            part_args = mutation_args_per_part.get(part, {}) if isinstance(mutation_args_per_part, dict) else {}
            # فقط آرگومان‌هایی که None نیستند را ذخیره می‌کنیم
            args_dict = {k: v for k, v in part_args.items() if v is not None} if isinstance(part_args, dict) else {}
            mutation_record = {
                "method_name": method_name,
                "args": args_dict
            }
            self.current_methods["history"]["mutation_methods_history"][part].append(mutation_record)
        
        # ذخیره selection method اولیه
        selection_key = self.current_methods["selection_method"]
        if selection_key != "default" and selection_key in self.selection_methods:
            selection_name = getattr(self.selection_methods[selection_key], "__name__", "default")
        else:
            selection_name = "default"
        
        # ذخیره نام متد به همراه آرگومان‌های ورودی
        selection_args = self.current_methods.get("selection_args", {})
        # فقط آرگومان‌هایی که None نیستند را ذخیره می‌کنیم
        args_dict = {k: v for k, v in selection_args.items() if v is not None}
        selection_record = {
            "method_name": selection_name,
            "args": args_dict
        }
        self.current_methods["history"]["selection_method_history"].append(selection_record)
        
        # ذخیره احتمالات اولیه
        self.current_methods["history"]["global_crossover_probability_history"].append(
            round(self.current_methods["global_crossover_probability"], 4)
        )
        self.current_methods["history"]["global_mutation_probability_history"].append(
            round(self.current_methods["global_mutation_probability"], 4)
        )
        
        # ذخیره احتمالات part اولیه
        for part in ["part1", "part2", "part3_p1", "part3_p2", "part4", "part5", "part6", "part7"]:
            crossover_prob = self.current_methods["crossover_part_probability"].get(part)
            if crossover_prob is not None:
                self.current_methods["history"]["crossover_part_probability_history"][part].append(round(crossover_prob, 4))
            
            mutation_prob = self.current_methods["mutation_part_probability"].get(part)
            if mutation_prob is not None:
                self.current_methods["history"]["mutation_part_probability_history"][part].append(round(mutation_prob, 4))

    def update_genetic_operators(self, recommendations):
        """
        به‌روزرسانی متدهای ژنتیک بر اساس پیشنهادات AI

        Parameters:
        -----------
        recommendations: پیشنهادات AI
        """
        if not recommendations or "recommendations" not in recommendations:
            print("⚠️ AI recommendations are invalid")
            return

        rec = recommendations["recommendations"]
        # self.current_methods = deepcopy(rec)
        # به‌روزرسانی متدهای ترکیب برای هر بخش
        if "crossover_methods" in rec:
            for part in [
                "part1",
                "part2",
                "part3_p1",
                "part3_p2",
                "part4",
                "part5",
                "part6",
                "part7",
            ]:
                if part in rec["crossover_methods"]:
                    method_name = rec["crossover_methods"][part]
                    # پیدا کردن کلید متد (شماره آن در dictionary)
                    found = False
                    for key, method in self.crossover_methods.items():
                        if (
                            hasattr(method, "__name__")
                            and method.__name__ == method_name
                        ):
                            self.current_methods["crossover_methods"][part] = key
                            # ذخیره نام متد جدید در history
                            self.current_methods["history"]["crossover_methods_history"][part].append(method_name)
                            found = True
                            if self.verbose:
                                print(
                                    f"🔄 Crossover method for {part} updated: {method_name}, probability: {rec['crossover_part_probability'][part]}"
                                )
                            break
                    if not found:
                        print(f"⚠️ Invalid crossover method for {part}: {method_name}")

            if "global_crossover_probability" in rec:
                self.p_crossover = rec["global_crossover_probability"]
                self.current_methods["global_crossover_probability"] = self.p_crossover
                # ذخیره احتمال جدید در history
                self.current_methods["history"]["global_crossover_probability_history"].append(round(self.p_crossover, 4))
                if self.verbose:
                    print(f"📊 Crossover probability updated: {self.p_crossover}")
            if "crossover_part_probability" in rec:
                new_part_probs = rec["crossover_part_probability"]
                # ذخیره احتمالات جدید برای هر part در history
                for part in ["part1", "part2", "part3_p1", "part3_p2", "part4", "part5", "part6", "part7"]:
                    if part in new_part_probs and new_part_probs[part] is not None:
                        self.current_methods["history"]["crossover_part_probability_history"][part].append(round(new_part_probs[part], 4))
                self.current_methods["crossover_part_probability"] = new_part_probs
                if self.verbose:
                    print("🔄 Crossover part probabilities updated")

        # به‌روزرسانی متدهای جهش برای هر بخش
        if "mutation_methods" in rec:
            for part in [
                "part1",
                "part2",
                "part3_p1",
                "part3_p2",
                "part4",
                "part5",
                "part6",
                "part7",
            ]:
                if part in rec["mutation_methods"]:
                    method_name = rec["mutation_methods"][part]
                    # پیدا کردن کلید متد (شماره آن در dictionary)
                    found = False
                    for key, method in self.mutation_methods.items():
                        if (
                            hasattr(method, "__name__")
                            and method.__name__ == method_name
                        ):
                            self.current_methods["mutation_methods"][part] = key
                            # ذخیره نام متد جدید به همراه آرگومان‌های ورودی در history
                            mutation_args_per_part = self.current_methods.get("mutation_args_per_part", {})
                            part_args = mutation_args_per_part.get(part, {}) if isinstance(mutation_args_per_part, dict) else {}
                            # فقط آرگومان‌هایی که None نیستند را ذخیره می‌کنیم
                            args_dict = {k: v for k, v in part_args.items() if v is not None} if isinstance(part_args, dict) else {}
                            mutation_record = {
                                "method_name": method_name,
                                "args": args_dict
                            }
                            self.current_methods["history"]["mutation_methods_history"][part].append(mutation_record)
                            found = True
                            if self.verbose:
                                print(
                                    f"🧬 Mutation method for {part} updated: {method_name}, probability: {rec['mutation_part_probability'][part]}"
                                )
                            break
                    if not found:
                        print(f"⚠️ Invalid mutation method for {part}: {method_name}")

            # آرگومان‌های جهش به تفکیک پارت
            if "mutation_args_per_part" in rec:
                # اگر موجود بود ادغام می‌کنیم تا پارت‌هایی که ارسال نشدند حفظ شوند
                incoming = rec["mutation_args_per_part"] or {}
                if "mutation_args_per_part" not in self.current_methods or not isinstance(self.current_methods["mutation_args_per_part"], dict):
                    self.current_methods["mutation_args_per_part"] = {}
                for part_key in ["part1","part2","part3_p1","part3_p2","part4","part5","part6","part7"]:
                    if part_key in incoming and isinstance(incoming[part_key], dict):
                        self.current_methods["mutation_args_per_part"].setdefault(part_key, {})
                        self.current_methods["mutation_args_per_part"][part_key].update(incoming[part_key])
                        if self.verbose and incoming[part_key]:
                            print(f"⚙️ Mutation args for {part_key}: {self.current_methods['mutation_args_per_part'][part_key]}")

            if "global_mutation_probability" in rec:
                self.p_mutation = rec["global_mutation_probability"]
                self.current_methods["global_mutation_probability"] = self.p_mutation
                # ذخیره احتمال جدید در history
                self.current_methods["history"]["global_mutation_probability_history"].append(round(self.p_mutation, 4))
                if self.verbose:
                    print(f"📊 Mutation probability updated: {self.p_mutation}")
            if "mutation_part_probability" in rec:
                new_part_probs = rec["mutation_part_probability"]
                # ذخیره احتمالات جدید برای هر part در history
                for part in ["part1", "part2", "part3_p1", "part3_p2", "part4", "part5", "part6", "part7"]:
                    if part in new_part_probs and new_part_probs[part] is not None:
                        self.current_methods["history"]["mutation_part_probability_history"][part].append(round(new_part_probs[part], 4))
                self.current_methods["mutation_part_probability"] = new_part_probs
                if self.verbose:
                    print("🧬 Mutation part probabilities updated")

        # به‌روزرسانی متد انتخاب
        if "selection_method" in rec:
            selection_method_name = rec["selection_method"]
            # پیدا کردن کلید متد (شماره آن در dictionary)
            found = False
            for key, method in self.selection_methods.items():
                method_name = getattr(method, "__name__", "")
                if method_name == selection_method_name:
                    self.current_methods["selection_method"] = key
                    # ذخیره نام متد جدید به همراه آرگومان‌های ورودی در history
                    selection_args = self.current_methods.get("selection_args", {})
                    # فقط آرگومان‌هایی که None نیستند را ذخیره می‌کنیم
                    args_dict = {k: v for k, v in selection_args.items() if v is not None}
                    selection_record = {
                        "method_name": selection_method_name,
                        "args": args_dict
                    }
                    self.current_methods["history"]["selection_method_history"].append(selection_record)
                    found = True
                    if self.verbose:
                        print(f"🎯 Selection method updated: {selection_method_name}")
                    break
            if not found:
                print(f"⚠️ Invalid selection method: {selection_method_name}")

        # به‌روزرسانی نرخ نخبه‌گرایی
        if "selection_args" in rec:
            self.current_methods["selection_args"] = rec["selection_args"]
            if self.verbose:
                print(f"⚙️ Selection args updated: {self.current_methods['selection_args']}")

        if "elitism_rate" in rec:
            new_elitism_rate = rec["elitism_rate"]
            # اعتبارسنجی: باید بین 0.05 تا 0.3 باشد
            if 0.05 <= new_elitism_rate <= 0.3:
                self.elitism_rate = new_elitism_rate
                # محاسبه مجدد تعداد نخبگان
                self.n_elite = max(1, int(self.elitism_rate * self.pop_size))
                if self.verbose:
                    print(f"👑 Elitism rate updated: {self.elitism_rate:.4f} ({self.elitism_rate * 100:.1f}% → {self.n_elite} elite individuals)")
            else:
                print(f"⚠️ Invalid elitism_rate: {new_elitism_rate}. Must be between 0.05 and 0.3. Keeping current value: {self.elitism_rate:.4f}")

        # نمایش تحلیل AI
        if "analysis" in recommendations:
            print(f"🤖 AI Analysis: {recommendations['analysis']}")

        if "reasoning" in recommendations:
            print(f"💭 AI Reasoning: {recommendations['reasoning']}")

    def get_ai_recommendations(self, **prompt_args):
        """
        دریافت پیشنهادات AI برای بهبود الگوریتم

        Parameters:
        -----------
        generation: نسل فعلی
        metrics: شاخص‌های عملکرد

        Returns:
        --------
        Dict: پیشنهادات AI

        Raises:
        -------
        RuntimeError: در صورت خطا در دریافت یا اعتبارسنجی پیشنهادات AI
        """
        if not self.openrouter_client:
            raise RuntimeError("OpenRouter client is not initialized. Cannot get AI recommendations.")

        try:
            recommendations = self.openrouter_client.get_recommendations(
                **prompt_args
            )

            if recommendations:
                # اعتبارسنجی پیشنهادات
                if self.openrouter_client.validate_recommendations(recommendations):
                    return recommendations
                else:
                    error_msg = "❌ AI recommendations are invalid - format validation failed"
                    print(error_msg, '\n', recommendations)
                    traceback.print_exc()
                    raise RuntimeError(error_msg)
            else:
                error_msg = "❌ Failed to get recommendations from AI - received None or empty response"
                print(error_msg)
                raise RuntimeError(error_msg)

        except RuntimeError:
            # Re-raise RuntimeError as-is
            raise
        except Exception as e:
            error_msg = f"❌ Error communicating with AI: {str(e)}"
            print(error_msg)
            # 👇 این دستور تمام اطلاعات Traceback را چاپ می‌کند
            traceback.print_exc()
            raise RuntimeError(error_msg) from e
