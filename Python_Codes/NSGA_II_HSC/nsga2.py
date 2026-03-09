import numpy as np
from copy import deepcopy
import random
from convergence_metrics import ConvergenceMetrics
from diagnostic_metrics import DiagnosticMetrics
import os
import pickle
from selection_methods import SelectionMethods


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

    def __init__(self, shelter_id, distribution_center_id, damage_points_id, hospital_id,
                 temporary_medical_id, max_iter=100, pop_size=100, p_crossover=0.7, p_mutation=0.3,
                 verbose=True, resume=True):
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
        verbose: Print iteration information (default: True)
        """
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.verbose = verbose
        self.resume = resume

        # Problem dimensions
        self.da_id = damage_points_id
        self.idc_id = distribution_center_id
        self.n_shelters = len(shelter_id)
        self.n_distribution = len(distribution_center_id)
        self.n_damage_points = len(damage_points_id)
        self.n_hospitals = len(hospital_id)
        self.n_temp_medical = len(temporary_medical_id)

        self.metrics = ConvergenceMetrics()
        self.diagnostics = DiagnosticMetrics()
        self.selection_methods = SelectionMethods()

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
        active_shelters = np.random.choice(
            self.n_shelters, n_active, replace=False)
        dist_assignment = [0] * self.n_shelters
        for idx in active_shelters:
            dist_assignment[idx] = random.randint(1, self.n_distribution)
        chromosome.append(dist_assignment)

        # Part 2: Flow values from distribution centers to shelters
        flow_values = []
        for i in range(self.n_shelters):
            # Use normal distribution with mean 0.6 and std 0.15
            # ratio = np.clip(np.random.normal(0.6, 0.15), 0, 1)
            ratio = np.random.random()
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
        part6 = np.zeros(
            (self.n_damage_points, self.n_hospitals + self.n_temp_medical))
        for i in range(self.n_damage_points):
            num_active = np.random.randint(
                1, self.n_hospitals + self.n_temp_medical)
            active_indices = random.sample(
                range(self.n_hospitals + self.n_temp_medical), num_active)
            weights = [np.random.random() for _ in range(num_active)]
            total = sum(weights)
            for idx, dest in enumerate(active_indices):
                part6[i][dest] = weights[idx] / total
        chromosome.append(part6)

        # Part 7: Percentage of minor injured people transported by ambulance
        part7 = np.zeros(
            (self.n_damage_points, self.n_hospitals + self.n_temp_medical))
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
            active_shelters = np.random.choice(
                self.n_shelters, n_active, replace=False)

            for idx in active_shelters:
                repaired_chromosome[0][idx] = random.randint(
                    1, self.n_distribution)

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
        child1[start:end+1] = parent1[start:end+1]
        child2[start:end+1] = parent2[start:end+1]

        # Fill the remaining positions in child1 with elements from parent2 not in the copied segment
        p2_idx = 0
        for i in range(size):
            if child1[i] is None:
                while parent2[p2_idx] in child1[start:end+1]:
                    p2_idx += 1
                child1[i] = parent2[p2_idx]
                p2_idx += 1

        # Fill the remaining positions in child2 with elements from parent1 not in the copied segment
        p1_idx = 0
        for i in range(size):
            if child2[i] is None:
                while parent1[p1_idx] in child2[start:end+1]:
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
        if sum(bool(x) for x in child1[0]) == 0:
            child1[0] = deepcopy(parent1[0])
            child1[1] = deepcopy(parent2[1])
        if sum(bool(x) for x in child2[0]) == 0:
            child2[0] = deepcopy(parent2[0])
            child2[1] = deepcopy(parent2[1])

        # Part 3: Use order crossover for damage points assignment (0 to n_damage_points)
        if self.n_damage_points > 0:
            part3_p1 = parent1[2][:self.n_damage_points]
            part3_p2 = parent2[2][:self.n_damage_points]
            child1_part3, child2_part3 = self.order_crossover(
                part3_p1, part3_p2)
            child1[2][:self.n_damage_points] = child1_part3
            child2[2][:self.n_damage_points] = child2_part3
        # For the rest (n_damage_points to n_shelters), swap as before
        for i in range(self.n_damage_points, self.n_shelters):
            if np.random.rand() < 0.5:
                child1[2][i], child2[2][i] = child2[2][i], child1[2][i]

        # Part 4
        mask_matrix = np.random.rand(
            self.n_damage_points, self.n_hospitals) < 0.5
        child1[3][mask_matrix] = parent2[3][mask_matrix]
        child2[3][mask_matrix] = parent1[3][mask_matrix]

        # Part 5
        point1 = np.random.choice(range(1, self.n_damage_points))
        point2 = np.random.choice(range(1, self.n_hospitals))
        child1[4][:point1, :point2], child2[4][:point1,
                                               :point2] = child2[4][:point1, :point2], child1[4][:point1, :point2]
        child1[4][point1:, point2:], child2[4][point1:,
                                               point2:] = child2[4][point1:, point2:], child1[4][point1:, point2:]

        # Part 6
        mask_matrix = np.random.rand(
            self.n_damage_points, self.n_hospitals + self.n_temp_medical) < 0.5
        child1[5][mask_matrix] = parent2[5][mask_matrix]
        child2[5][mask_matrix] = parent1[5][mask_matrix]

        # Part 7
        point1 = np.random.choice(range(1, self.n_damage_points))
        point2, point3 = sorted(random.sample(
            range(1, self.n_hospitals + self.n_temp_medical), k=2))
        child1[6][:point1, point2:point3], child2[6][:point1,
                                                     point2:point3] = child2[6][:point1, point2:point3], child1[6][:point1, point2:point3]
        child1[6][point1:, :point2], child2[6][point1:,
                                               :point2] = child2[6][point1:, :point2], child1[6][point1:, :point2]
        child1[6][point1:, point3:], child2[6][point1:,
                                               point3:] = child2[6][point1:, point3:], child1[6][point1:, point3:]

        # تعمیر کروموزوم‌ها (بخش‌های 1، 4، و 6)
        child1 = self.chromosome_repair(child1)
        child2 = self.chromosome_repair(child2)

        return child1, child2

    def mutate(self, chromosome, mutation_rate=0.1):
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
        mutated[1][i] = max(0, np.clip(
            mutated[1][i] + np.random.normal(0, 0.1), 0, 1))

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
                    mutated[4][i][j] + np.random.normal(0, 0.1), 0, 1)
            else:
                # if random.random() < 0.1:
                #     mutated[3][i][j] = 0
                # else:
                mutated[3][i][j] = max(0, np.clip(
                    mutated[3][i][j] + np.random.normal(0, 0.1), 0, 1))
                mutated[4][i][j] = np.clip(
                    mutated[4][i][j] + np.random.normal(0, 0.1), 0, 1)

        # Part 6 & 7
        for i in range(self.n_hospitals):
            j = random.randint(0, self.n_hospitals - 1)
            if mutated[5][i][j] == 0:
                # if random.random() < 0.1:
                mutated[5][i][j] = random.uniform(0, 0.2)
                mutated[6][i][j] = max(0, np.clip(
                    mutated[6][i][j] + np.random.normal(0, 0.1), 0, 1))
            else:
                # if random.random() < 0.1:
                #     mutated[5][i][j] = 0
                # else:
                mutated[5][i][j] = max(0, np.clip(
                    mutated[5][i][j] + np.random.normal(0, 0.1), 0, 1))
                mutated[6][i][j] = max(0, np.clip(
                    mutated[6][i][j] + np.random.normal(0, 0.1), 0, 1))
            k = random.randint(
                self.n_hospitals, self.n_hospitals + self.n_temp_medical - 1)
            if mutated[5][i][k] == 0:
                # if random.random() < 0.1:
                mutated[5][i][k] = random.uniform(0, 0.2)
                mutated[6][i][k] = max(0, np.clip(
                    mutated[6][i][k] + np.random.normal(0, 0.1), 0, 1))
            else:
                # if random.random() < 0.1:
                #     mutated[5][i][k] = 0
                # else:
                mutated[5][i][k] = max(0, np.clip(
                    mutated[5][i][k] + np.random.normal(0, 0.1), 0, 1))
                mutated[6][i][k] = max(0, np.clip(
                    mutated[6][i][k] + np.random.normal(0, 0.1), 0, 1))
        mutated = self.chromosome_repair(mutated)
        return mutated

    def run(self, problem, RUN_NUM, INIT_POP_TYPE):
        """
        Run NSGA-II for humanitarian logistics problem with checkpoint/resume support
        """
        # Extract problem info
        cost_function = problem['cost_function']
        if INIT_POP_TYPE == "random":
            checkpoint_path = os.path.join(os.path.dirname(__file__), "exports\\random_initial_pop_checkpoint", f"nsga2_checkpoint_{RUN_NUM}.pkl")
        elif INIT_POP_TYPE == "llm":
            checkpoint_path = os.path.join(os.path.dirname(__file__), "exports\gpt_initial_pop_checkpoint", f"nsga2_checkpoint_{RUN_NUM}.pkl")
        resume = self.resume

        # Ensure the checkpoint directory exists
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir and checkpoint_dir != ".":
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Empty individual
        empty_individual = {
            'chromosome': None,
            'cost': None,
            'normal_cost': None,
            'rank': None,
            'crowding_distance': None,
            'constriant_violation': 0
        }

        # Initialize or resume state
        pop = None
        F = None
        pareto_pop_list = []
        start_it = 0

        if resume and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'rb') as f:
                    state = pickle.load(f)
                start_it = int(state.get('iteration', -1)) + 1
                pop = state.get('pop')
                F = state.get('F')
                pareto_pop_list = state.get('pareto_history', [])
                # restore metrics if present
                if 'metrics' in state:
                    self.metrics = state['metrics']
                if self.verbose:
                    print(f"Resumed from checkpoint at iteration {start_it}")
            except Exception:
                if self.verbose:
                    print('Failed to load checkpoint. Starting fresh.')
                pop = None

        # if pop is None:
        #     # Initialize population
        #     pop = [deepcopy(empty_individual) for _ in range(self.pop_size)]
        #     for i in range(self.pop_size):
        #         pop[i]['chromosome'] = self.create_random_chromosome()
        #     chromosom_list = [p['chromosome'] for p in pop]
        #     costs, constriants_violation, normal_cost = cost_function(chromosom_list)
        #     for idx, i_cost in enumerate(costs):
        #         pop[idx]['cost'] = np.array(i_cost)
        #         pop[idx]['normal_cost'] = np.array(normal_cost[idx])
        #         pop[idx]['constriant_violation'] = constriants_violation[idx]

        #     # Non-dominated sorting
        #     pop, F = self.non_dominated_sorting(pop)
        #     # Calculate crowding distance
        #     pop = self.calc_crowding_distance(pop, F)
        #     # Sort population
        #     pop, F = self.sort_population(pop)
        
        if pop is None:
            pop = [deepcopy(empty_individual) for _ in range(self.pop_size)]
            # Try AI-generated initial population first if enabled
            ai_init_ok = False
            # If LLM-based init is disabled, try loading from initial_population.json
            try:
                import json
                init_path = os.path.join(os.path.dirname(__file__), "")
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
                        print("  Using random initialization instead")
            except Exception as e:
                # Any error -> ignore and fallback later
                if self.verbose:
                    print(f"⚠ Error loading initial_population.json: {e}")
                ai_init_ok = False
            # fallback if AI/file init disabled or failed
            if not ai_init_ok:
                print("Create random population...")
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
        pareto_pop = []  # safe default to avoid UnboundLocalError
        for it in range(start_it, self.max_iter):

            # Crossover
            popc = []
            n_crossover_pairs = self.pop_size // 2
            for _ in range(n_crossover_pairs):
                p1 = self.crowding_tournament_selection(pop)
                p2 = self.crowding_tournament_selection(pop)
                # selection_args ={"eps": 0.2, "k":2}
                # p1 = self.selection_methods.epsilon_dominance_tournament(pop, **selection_args)
                # p2 = self.selection_methods.epsilon_dominance_tournament(pop, **selection_args)
                if random.uniform(0, 1) < self.p_crossover:
                    c1_chrom, c2_chrom = self.crossover(
                        p1['chromosome'], p2['chromosome'])
                    c1 = deepcopy(empty_individual)
                    c2 = deepcopy(empty_individual)
                    c1['chromosome'] = c1_chrom
                    c2['chromosome'] = c2_chrom
                    popc.append(c1)
                    popc.append(c2)
                else:
                    popc.append(deepcopy(p1))
                    popc.append(deepcopy(p2))

            # اگر تعداد فرد باقی‌مانده فرد باشد، یک فرد اضافی تولید کن
            if len(popc) < self.pop_size:
                p = self.crowding_tournament_selection(pop)
                popc.append(deepcopy(p))

            # Mutation
            popm = []
            for i in range(len(popc)):
                if random.uniform(0, 1) < self.p_mutation:
                    m = deepcopy(empty_individual)
                    m['chromosome'] = self.mutate(popc[i]['chromosome'])
                    popm.append(m)
                else:
                    popm.append(popc[i])

            # ترکیب نخبگان با افراد جدید
            pop += popm

            # محاسبه cost برای افراد جدید (نخبگان قبلاً محاسبه شده‌اند)
            new_chromosomes = []
            for i in range(self.pop_size, len(pop)):
                if pop[i]['cost'] is None:
                    new_chromosomes.append(pop[i]['chromosome'])

            if new_chromosomes:
                costs, constraints_violation, normal_costs = cost_function(
                    new_chromosomes)
                new_idx = 0
                for i in range(self.pop_size, len(pop)):
                    if pop[i]['cost'] is None:
                        pop[i]['cost'] = np.array(costs[new_idx])
                        pop[i]['normal_cost'] = np.array(normal_costs[new_idx])
                        pop[i]['constriant_violation'] = constraints_violation[new_idx]
                        new_idx += 1

            # Non-dominated sorting
            pop, F = self.non_dominated_sorting(pop)

            # Calculate crowding distance
            pop = self.calc_crowding_distance(pop, F)

            # Sort population
            pop, F = self.sort_population(pop)

            # Truncate (اگر لازم باشد)
            pop, F = self.truncate_population(pop, F)

            # اضافه کردن محاس��ه metrics
            if len(F) > 0 and len(F[0]) > 0:
                pareto_pop = [pop[i] for i in F[0]]
            else:
                # fallback: use sorted population's first front if available
                sorted_pop, sorted_F = self.sort_population(pop)
                pareto_pop = [sorted_pop[i] for i in (sorted_F[0] if len(sorted_F) > 0 else [])]
            pareto_pop_list.append(deepcopy(pareto_pop))

            # Display iteration info
            if self.verbose:
                print(
                      f'    Iteration {it + 1}: Number of Pareto Members = {len(F[0])}')
            # Save checkpoint at the end of each successful iteration
            try:
                state = {
                    'iteration': it,
                    'pop': pop,
                    'F': F,
                    'pareto_history': pareto_pop_list,
                    'metrics': self.metrics,
                }
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(state, f)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: failed to save checkpoint at iteration {it}: {e}")

        # After loop end, update metrics and return
        self.metrics.update_metrics(pareto_pop_list)
        # Ensure pareto_pop is defined even if no fronts exist
        if not pareto_pop:
            if len(F) > 0 and len(F[0]) > 0:
                pareto_pop = [pop[i] for i in F[0]]
            else:
                pareto_pop = []
        return {
            'pop': pop,
            'F': F,
            'pareto_pop': pareto_pop,
            'pareto_history': pareto_pop_list,
            'metrics': self.metrics,
        }

    def dominates(self, p, q):
        """Check if p dominates q"""
        if p['constriant_violation'] > 0 and q['constriant_violation'] > 0:
            return p['constriant_violation'] < q['constriant_violation']
        elif p['constriant_violation'] > 0 and q['constriant_violation'] == 0:
            return False
        elif p['constriant_violation'] == 0 and q['constriant_violation'] > 0:
            return True
        elif p['constriant_violation'] == 0 and q['constriant_violation'] == 0:
            return all(p['cost'] <= q['cost']) and any(p['cost'] < q['cost'])

    def non_dominated_sorting(self, pop):
        """Non-dominated sorting (unchanged from original)"""
        pop_size = len(pop)
        domination_set = [[] for _ in range(pop_size)]
        dominated_count = [0 for _ in range(pop_size)]
        F = [[]]

        for i in range(pop_size):
            for j in range(i+1, pop_size):
                if self.dominates(pop[i], pop[j]):
                    domination_set[i].append(j)
                    dominated_count[j] += 1
                elif self.dominates(pop[j], pop[i]):
                    domination_set[j].append(i)
                    dominated_count[i] += 1

            if dominated_count[i] == 0:
                pop[i]['rank'] = 0
                F[0].append(i)

        k = 0
        while True:
            Q = []
            for i in F[k]:
                for j in domination_set[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        pop[j]['rank'] = k + 1
                        Q.append(j)

            if not Q:
                break

            F.append(Q)
            k += 1

        return pop, F

    def calc_crowding_distance(self, pop, F):
        """Calculate crowding distance (unchanged from original)"""
        parto_count = len(F)
        n_obj = len(pop[0]['cost'])

        for k in range(parto_count):
            costs = np.array([pop[i]['normal_cost'] for i in F[k]])
            n = len(F[k])
            d = np.zeros((n, n_obj))

            for j in range(n_obj):
                idx = np.argsort(costs[:, j])
                d[idx[0], j] = np.inf
                d[idx[-1], j] = np.inf

                for i in range(1, n-1):
                    d[idx[i], j] = costs[idx[i+1], j] - costs[idx[i-1], j]
                    if costs[idx[-1], j] != costs[idx[0], j]:
                        d[idx[i], j] /= costs[idx[-1], j] - costs[idx[0], j]

            for i in range(n):
                pop[F[k][i]]['crowding_distance'] = sum(d[i, :])

        return pop

    def sort_population(self, pop):
        """Sort population by rank and crowding distance"""
        pop = sorted(pop, key=lambda x: (x['rank'], -x['crowding_distance']))

        max_rank = pop[-1]['rank']
        F = []
        for r in range(max_rank + 1):
            F.append([i for i in range(len(pop)) if pop[i]['rank'] == r])

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

    def crowding_tournament_selection(self, pop, tournament_size=2):
        """انتخاب با ترجیح rank کمتر و تنوع بیشتر"""
        idx1, idx2 = np.random.choice(
            range(len(pop)), size=tournament_size, replace=False)

        p1, p2 = pop[idx1], pop[idx2]

        # اگر rank متفاوت، rank کمتر برنده
        if p1['rank'] < p2['rank']:
            return p1
        elif p2['rank'] < p1['rank']:
            return p2
        else:
            # اگر rank یکسان، crowding distance بیشتر برنده (حفظ تنوع)
            return p1 if p1['crowding_distance'] > p2['crowding_distance'] else p2
