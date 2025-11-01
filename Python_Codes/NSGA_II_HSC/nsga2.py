import numpy as np
from copy import deepcopy
import random
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


class NSGA2_Humanitarian:
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
        max_iter=100,
        pop_size=100,
        p_crossover=0.7,
        p_mutation=0.3,
        elitism_rate=0.1,
        verbose=True,
        openrouter_api_key=None,
        use_ai_optimization=False,
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
        """
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.elitism_rate = elitism_rate
        self.n_elite = max(1, int(elitism_rate * pop_size))  # حداقل یک فرد نخبه
        self.verbose = verbose
        self.use_ai_optimization = use_ai_optimization

        # تنظیم OpenRouter Client
        self.openrouter_client = None
        if use_ai_optimization and openrouter_api_key:
            self.openrouter_client = OpenRouterClient(openrouter_api_key, AI_MODEL)

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
            "global_crossover_probability": p_crossover,
            "global_mutation_probability": p_mutation,
            "mutation_inner_rate": 0.1,
            "crossover_part_probability": {
                "part1": None,
                "part2": None,
                "part3_p1": None,
                "part3_p2": None,
                "part4": None,
                "part5": None,
                "part6": None,
                "part7": None,
            },
            "mutation_part_probability": {
                "part1": None,
                "part2": None,
                "part3_p1": None,
                "part3_p2": None,
                "part4": None,
                "part5": None,
                "part6": None,
                "part7": None,
            },
            "history": {
                "crossover_history":{
                    "part1": [],
                "part2": [],
                "part3_p1": [],
                "part3_p2": [],
                "part4": [],
                "part5": [],
                "part6": [],
                "part7": [],
                    },
                    "mutation_history":{
                        "part1": [],
                        "part2": [],
                        "part3_p1": [],
                        "part3_p2": [],
                        "part4": [],
                        "part5": [],
                        "part6": [],
                        "part7": [],
                    },
                    "selection_history": [],
                    "global_crossover_probability_history": [],
                    "global_mutation_probability_history": [],
                    "mutation_inner_rate_history": [],
                }
        }

        # Problem dimensions
        self.da_id = damage_points_id
        self.idc_id = distribution_center_id
        self.n_shelters = len(shelter_id)
        self.n_distribution = len(distribution_center_id)
        self.n_damage_points = len(damage_points_id)
        self.n_hospitals = len(hospital_id)
        self.n_temp_medical = len(temporary_medical_id)

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
        # Instance برای استفاده از متدها
        self.selection_methods_instance = SelectionMethods()

        # تابع‌های wrapper برای متدهای selection که پارامتر اضافی دارند
        def make_selection_wrapper(method_name):
            def wrapper(pop):
                method = getattr(self.selection_methods_instance, method_name)
                if method_name == "crowded_binary_tournament":
                    return method(pop)
                elif method_name == "adaptive_k_tournament":
                    return method(pop, k=2, use_feas=False)
                elif method_name == "feasibility_first_tournament":
                    return method(pop, k=2)
                elif method_name == "rank_based_roulette":
                    return method(pop, power=1.0)
                elif method_name == "reference_biased_tournament":
                    # استفاده از reference points خالی
                    return method(pop, refs=[], k=2, tau=1.0)
                elif method_name == "age_diversity_tournament":
                    return method(pop, k=2, prefer_younger=True)
                elif method_name == "epsilon_dominance_tournament":
                    return method(pop, eps=0.01, k=2)
                else:
                    return method(pop)

            wrapper.__name__ = method_name  # ذخیره نام متد برای تشخیص بعدی
            return wrapper

        self.selection_methods = {
            "1": make_selection_wrapper("crowded_binary_tournament"),
            "2": make_selection_wrapper("adaptive_k_tournament"),
            "3": make_selection_wrapper("feasibility_first_tournament"),
            "4": make_selection_wrapper("rank_based_roulette"),
            "5": make_selection_wrapper("reference_biased_tournament"),
            "6": make_selection_wrapper("age_diversity_tournament"),
            "7": make_selection_wrapper("epsilon_dominance_tournament"),
        }

        self.metrics = ConvergenceMetrics()
        self.diagnostics = DiagnosticMetrics()

        # مدیریت هایپرولیوم با نقطه مرجع ثابت
        self.hypervolume_manager = HypervolumeManager(estimation_method="conservative")

        # برای ذخیره تاریخچه شاخص‌ها
        self.metrics_history = []
        # آمار بخشی نسل به نسل
        self.section_stats_history = []

    def _entropy(self, values):
        arr = np.array(values).flatten()
        if arr.size == 0:
            return 0.0
        unique, counts = np.unique(arr, return_counts=True)
        p = counts / counts.sum()
        # جلوگیری از log(0)
        p = p[p > 0]
        return float(-np.sum(p * np.log(p + 1e-12)))

    def _variance(self, values):
        arr = np.array(values).astype(float).flatten()
        if arr.size == 0:
            return 0.0
        return float(np.var(arr))

    def compute_section_stats(self, pop):
        """
        محاسبه آمار بخشی (entropy/variance) فقط برای هر بخش/سکشن (بدون نرخ تعمیر)
        """
        if not pop:
            return {}
        # جمع‌آوری مقادیر
        part1_vals, part2_vals = [], []
        part3_p1_vals, part3_p2_vals = [], []
        part4_vals, part5_vals, part6_vals, part7_vals = [], [], [], []
        for ind in pop:
            chrom = ind["chromosome"]
            if chrom is None:
                continue
            part1_vals.extend(chrom[0])
            part2_vals.extend(chrom[1])
            part3_p1_vals.extend(chrom[2][: self.n_damage_points])
            if self.n_shelters > self.n_damage_points:
                part3_p2_vals.extend(chrom[2][self.n_damage_points :])
            part4_vals.extend(chrom[3].flatten())
            part5_vals.extend(chrom[4].flatten())
            part6_vals.extend(chrom[5].flatten())
            part7_vals.extend(chrom[6].flatten())
        section_stats = {
            "part1": {
                "entropy": self._entropy(part1_vals),
            },
            "part2": {"variance": self._variance(part2_vals)},
            "part3_p1": {"entropy": self._entropy(part3_p1_vals)},
            "part3_p2": {
                "entropy": self._entropy(part3_p2_vals),
            },
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

        def get_prob(part):
            p = mpp.get(part)
            return float(p) if p is not None else 1.0

        # Part 1
        if np.random.rand() < get_prob("part1"):
            part1_method = self.current_methods["mutation_methods"]["part1"]
            if part1_method != "default" and part1_method in self.mutation_methods:
                mutated[0] = self.mutation_methods[part1_method](chromosome[0])
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
                inner_rate = self.current_methods.get("mutation_inner_rate", 0.1)
                try:
                    mutated[1] = self.mutation_methods[part2_method](
                        chromosome[1], mutation_rate=inner_rate
                    )
                except TypeError:
                    mutated[1] = self.mutation_methods[part2_method](chromosome[1])
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
                mutated_p1 = self.mutation_methods[part3_p1_method](
                    chromosome[2][: self.n_damage_points]
                )
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
                mutated_p2 = self.mutation_methods[part3_p2_method](
                    chromosome[2][self.n_damage_points :]
                )
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
                    inner_rate = self.current_methods.get("mutation_inner_rate", 0.1)
                    try:
                        mutated_flat = self.mutation_methods[method](
                            chromosome[part_idx], mutation_rate=inner_rate
                        )
                    except TypeError:
                        mutated_flat = self.mutation_methods[method](
                            chromosome[part_idx]
                        )
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
        checkpoint_path = problem.get("checkpoint_path", "nsga2_checkpoint.pkl")
        resume = bool(problem.get("resume", False))

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
                self.metrics_history = state.get("metrics_history", [])
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

        # Main loop
        for it in range(start_it, self.max_iter):
            # انتخاب افراد نخبه از نسل فعلی
            elite_individuals = self.select_elite(pop, F)

            # محاسبه تعداد افراد جدید مورد نیاز (کل جمعیت منهای نخبگان)
            n_new_individuals = self.pop_size - len(elite_individuals)

            # جمع‌آوری شاخص‌های عملکرد
            current_metrics = self.collect_metrics(pop, F, it)
            self.metrics_history.append(current_metrics)

            # درخواست پیشنهادات AI هر 5 نسل
            if (
                self.use_ai_optimization
                and self.openrouter_client
                and it > 0
                and it % 5 == 0
            ):
                print(f"\n🤖 Requesting AI recommendations for generation {it}...")
                recommendations = self.get_ai_recommendations(it, current_metrics)
                if recommendations:
                    self.update_genetic_operators(recommendations)
                else:
                    print("⚠️ Continuing with current methods")

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

            # اضافه کردن محاسبه metrics
            pareto_pop = [pop[i] for i in F[0]]
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
        print("\n🔄 Final calculation of hypervol with a fixed reference point...")
        final_hypervolumes = self.hypervolume_manager.calculate_final_hypervolumes()

        # به‌روزرسانی metrics با هایپرولیوم نهایی
        self.metrics.update_metrics(pareto_pop_list, all_pop_list=all_pop_list)
        self.metrics.hypervolume_history = final_hypervolumes

        # نمایش خلاصه هایپرولیوم
        hv_summary = self.hypervolume_manager.get_metrics_summary()
        if self.verbose:
            print("📊 Hypervolume Summary:")
            print(f"   Final Reference Point: {hv_summary['reference_point']}")
            print(f"   Final Hypervolume: {hv_summary['final_hypervolume']:.6f}")
            if hv_summary["hypervolume_improvement"] is not None:
                print(
                    f"   Hypervolume Improvement: {hv_summary['hypervolume_improvement']:.2f}%"
                )

        return {
            "pop": pop,
            "F": F,
            "pareto_pop": pareto_pop,
            "pareto_history": pareto_pop_list,
            "metrics": self.metrics,
            "hypervolume_manager": self.hypervolume_manager,
            "hypervolume_summary": hv_summary,
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
            return selection_func(pop)
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
        metrics = {}

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
            metrics["hypervolume"] = preliminary_hv

            # Spacing
            if hasattr(self.metrics, "spacing"):
                normal_pareto_costs = np.array(
                    [ind["normal_cost"] for ind in pareto_pop]
                )
                metrics["spacing"] = self.metrics.spacing(normal_pareto_costs)
            else:
                metrics["spacing"] = 0.0
            # Spread (Delta)
            if hasattr(self.metrics, "spread"):
                normal_pareto_costs = np.array(
                    [ind["normal_cost"] for ind in pareto_pop]
                )
                metrics["spread"] = self.metrics.spread(normal_pareto_costs)
            else:
                metrics["spread"] = 0.0

            # تعداد راه‌حل‌های Pareto
            metrics["pareto_count"] = len(pareto_pop)

            # میانگین crowding distance
            crowding_distances = [ind["crowding_distance"] for ind in pareto_pop]
            metrics["avg_crowding_distance"] = (
                np.mean(crowding_distances) if crowding_distances else 0.0
            )

            # محاسبه تنوع جمعیت با استفاده از متد diversity در convergence_metrics.py
            if hasattr(self.metrics, "diversity"):
                metrics["diversity"] = self.metrics.diversity(pareto_costs)
            else:
                metrics["diversity"] = 0.0
        else:
            # مقادیر پیش‌فرض اگر Pareto front خالی باشد
            metrics.update(
                {
                    "hypervolume": 0.0,
                    "spacing": 0.0,
                    "spread": 0.0,
                    "pareto_count": 0,
                    "avg_crowding_distance": 0.0,
                    "diversity": 0.0,
                }
            )

        # محاسبه min_obj, mean_obj, std_obj از کل جمعیت
        if len(pop) > 0:
            all_pop_costs = np.array([ind["cost"] for ind in pop])
            metrics["mean_objectives"] = np.mean(all_pop_costs, axis=0).tolist()
            metrics["min_objectives"] = np.min(all_pop_costs, axis=0).tolist()
            metrics["std_objectives"] = np.std(all_pop_costs, axis=0).tolist()
        else:
            metrics["mean_objectives"] = [0.0, 0.0, 0.0]
            metrics["min_objectives"] = [0.0, 0.0, 0.0]
            metrics["std_objectives"] = [0.0, 0.0, 0.0]

        # شاخص‌های اضافی
        metrics["generation"] = generation
        metrics["population_size"] = len(pop)
        metrics["elitism_rate"] = self.elitism_rate  # اضافه کردن نرخ نخبه‌گرایی
        # آمار بخشی (section_stats) طبق Prompt 2
        section_stats = self.compute_section_stats(pop)
        metrics["section_stats"] = section_stats

        return metrics

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
                            found = True
                            if self.verbose:
                                print(
                                    f"🔄 Crossover method for {part} updated: {method_name}, probability: {rec['crossover_methods']['crossover_part_probability'][part]}"
                                )
                            break
                    if not found:
                        print(f"⚠️ Invalid crossover method for {part}: {method_name}")

            if "global_crossover_probability" in rec["crossover_methods"]:
                self.p_crossover = rec["crossover_methods"]["global_crossover_probability"]
                self.current_methods["global_crossover_probability"] = self.p_crossover
                if self.verbose:
                    print(f"🔄 Crossover probability updated: {self.p_crossover}")
            if "crossover_part_probability" in rec["crossover_methods"]:
                self.current_methods["crossover_part_probability"] = rec["crossover_methods"][
                    "crossover_part_probability"
                ]
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
                            found = True
                            if self.verbose:
                                print(
                                    f"🔄 Mutation method for {part} updated: {method_name}, probability: {rec['mutation_methods']['mutation_part_probability'][part]}"
                                )
                            break
                    if not found:
                        print(f"⚠️ Invalid mutation method for {part}: {method_name}")

            if "global_mutation_probability" in rec["mutation_methods"]:
                self.p_mutation = rec["mutation_methods"]["global_mutation_probability"]
                self.current_methods["global_mutation_probability"] = self.p_mutation
                if self.verbose:
                    print(f"🔄 Mutation probability updated: {self.p_mutation}")
            if "mutation_rate" in rec["mutation_methods"]:
                self.current_methods["mutation_inner_rate"] = rec["mutation_methods"][
                    "mutation_inner_rate"
                ]
                if self.verbose:
                    print(
                        f"🔄 Inner mutation_rate updated: {self.current_methods['mutation_inner_rate']}"
                    )
            if "mutation_part_probability" in rec["mutation_methods"]:
                self.current_methods["mutation_part_probability"] = rec["mutation_methods"][
                    "mutation_part_probability"
                ]
                if self.verbose:
                    print("🔄 Mutation part probabilities updated")

        # به‌روزرسانی متد انتخاب
        if "selection_method" in rec:
            selection_method_name = rec["selection_method"]
            # پیدا کردن کلید متد (شماره آن در dictionary)
            found = False
            for key, method in self.selection_methods.items():
                method_name = getattr(method, "__name__", "")
                if method_name == selection_method_name:
                    self.current_methods["selection_method"] = key
                    found = True
                    if self.verbose:
                        print(f"🔄 Selection method updated: {selection_method_name}")
                    break
            if not found:
                print(f"⚠️ Invalid selection method: {selection_method_name}")

        # به‌روزرسانی نرخ نخبه‌گرایی
        if "elitism_rate" in rec:
            new_elitism_rate = rec["elitism_rate"]
            # اعتبارسنجی: باید بین 0.05 تا 0.3 باشد
            if 0.05 <= new_elitism_rate <= 0.3:
                self.elitism_rate = new_elitism_rate
                # محاسبه مجدد تعداد نخبگان
                self.n_elite = max(1, int(self.elitism_rate * self.pop_size))
                if self.verbose:
                    print(f"🔄 Elitism rate updated: {self.elitism_rate:.4f} ({self.elitism_rate * 100:.1f}% → {self.n_elite} elite individuals)")
            else:
                print(f"⚠️ Invalid elitism_rate: {new_elitism_rate}. Must be between 0.05 and 0.3. Keeping current value: {self.elitism_rate:.4f}")

        # نمایش تحلیل AI
        if "analysis" in recommendations:
            print(f"🤖 AI Analysis: {recommendations['analysis']}")

        if "reasoning" in recommendations:
            print(f"💭 AI Reasoning: {recommendations['reasoning']}")

    def get_ai_recommendations(self, generation, metrics):
        """
        دریافت پیشنهادات AI برای بهبود الگوریتم

        Parameters:
        -----------
        generation: نسل فعلی
        metrics: شاخص‌های عملکرد

        Returns:
        --------
        Dict: پیشنهادات AI یا None
        """
        if not self.openrouter_client:
            return None

        try:
            recommendations = self.openrouter_client.get_recommendations(
                generation, metrics, self.current_methods, self.metrics_history
            )

            if recommendations:
                # اعتبارسنجی پیشنهادات
                if self.openrouter_client.validate_recommendations(recommendations):
                    return recommendations
                else:
                    print("❌ AI recommendations are invalid")
                    return None
            else:
                print("❌ Failed to get recommendations from AI")
                return None

        except Exception:
            print("❌ Error communicating with AI:")
            # 👇 این دستور تمام اطلاعات Traceback را چاپ می‌کند
            traceback.print_exc()
            return None
