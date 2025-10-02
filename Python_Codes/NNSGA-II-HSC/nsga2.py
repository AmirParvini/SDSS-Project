import numpy as np
from copy import deepcopy
from itertools import chain

class NSGA2_Humanitarian:
    """NSGA-II for humanitarian logistics optimization with custom chromosome structure"""

    def __init__(self, max_iter=100, pop_size=100, p_crossover=0.7, p_mutation=0.3, 
                 verbose=True, n_shelters=8, n_distribution=3, n_damage_points=5, 
                 n_hospitals=4, n_temp_medical=11, severe_injured=None, moderate_injured=None):
        """
        Constructor for humanitarian logistics NSGA-II
        
        Parameters:
        -----------
        n_shelters: Number of candidate shelter locations
        n_distribution: Number of distribution centers
        n_damage_points: Number of damage points
        n_hospitals: Number of hospitals
        n_temp_medical: Number of temporary medical centers
        severe_injured: Array of severe injured count at each damage point
        moderate_injured: Array of moderate injured count at each damage point
        """
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.verbose = verbose
        
        # Problem dimensions
        self.n_shelters = n_shelters
        self.n_distribution = n_distribution
        self.n_damage_points = n_damage_points
        self.n_hospitals = n_hospitals
        self.n_temp_medical = n_temp_medical
        
        # Injured counts (if not provided, generate random)
        if severe_injured is None:
            self.severe_injured = np.random.randint(10, 50, n_damage_points)
        else:
            self.severe_injured = np.array(severe_injured)
            
        if moderate_injured is None:
            self.moderate_injured = np.random.randint(20, 80, n_damage_points)
        else:
            self.moderate_injured = np.array(moderate_injured)

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
        n_active = np.random.randint(1, self.n_shelters + 1)
        active_shelters = np.random.choice(self.n_shelters, n_active, replace=False)
        dist_assignment = [0] * self.n_shelters
        for idx in active_shelters:
            dist_assignment[idx] = np.random.randint(1, self.n_distribution + 1)
        chromosome.append(dist_assignment)
        
        # Part 2: Flow values from distribution centers to shelters
        flow_values = []
        for i in range(self.n_shelters):
            if dist_assignment[i] > 0:
                # Use normal distribution with mean 0.6 and std 0.15
                ratio = np.clip(np.random.normal(0.6, 0.15), 0, 1)
                flow_values.append(ratio)
            else:
                flow_values.append(0)
        chromosome.append(flow_values)
        
        # Part 3: Damage point to shelter assignment
        damage_to_shelter = []
        for _ in range(self.n_shelters):
            if dist_assignment[_] > 0:
                # Assign to a random damage point
                damage_to_shelter.append(np.random.randint(1, self.n_damage_points + 1))
            else:
                # Shelter not active, set to 0
                damage_to_shelter.append(0)
        chromosome.append(damage_to_shelter)
        
        # Part 4: damaged_points to hospital assignment for transfer of severely injured
        part4 = np.zeros((self.n_damage_points, self.n_hospitals))
        for i in range(self.n_damage_points):
            num_active = np.random.randint(1, self.n_hospitals)
            active_indices = np.random.sample(range(self.n_hospitals), num_active)
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
            active_indices = np.random.sample(range(self.n_hospitals + self.n_temp_medical), num_active)
            weights = [np.random.random() for _ in range(num_active)]
            total = sum(weights)
            for idx, dest in enumerate(active_indices):
                part6[i][dest] = weights[idx] / total
        chromosome.append(part6)
        
        # Part 7: Percentage of minor injured people transported by ambulance
        part7 = np.zeros((self.n_damage_points, self.n_hospitals + self.n_temp_medical))
        for i in range(self.n_damage_points):
            for j in range(self.n_hospitals + self.n_temp_medical):
                if part6[i][j] > 0:
                    part7[i][j] = np.random.random()
        chromosome.append(part7)

        return chromosome

    # def repair_chromosome(self, chromosome):
    #     """
    #     Repair chromosome to ensure all constraints are satisfied
    #     """
    #     repaired = deepcopy(chromosome)
        
    #     # Ensure consistency between parts
    #     # If shelter is not selected (part 1 = 0), set corresponding values to 0
    #     for i in range(self.n_shelters):
    #         if repaired[0][i] == 0:
    #             repaired[1][i] = 0  # No flow
    #             repaired[2][i] = 0  # No damage point assignment
    #         else:
    #             # Ensure valid distribution center index
    #             if repaired[0][i] > self.n_distribution:
    #                 repaired[0][i] = np.random.randint(1, self.n_distribution + 1)
                
    #             # Ensure flow is positive for active shelters
    #             if repaired[1][i] <= 0:
    #                 repaired[1][i] = np.random.uniform(10, 200)
                
    #             # Ensure valid damage point assignment
    #             if repaired[2][i] <= 0 or repaired[2][i] > self.n_damage_points:
    #                 repaired[2][i] = np.random.randint(1, self.n_damage_points + 1)
        
    #     # Ensure valid hospital assignments
    #     for i in range(self.n_damage_points):
    #         if repaired[3][i] <= 0 or repaired[3][i] > self.n_hospitals:
    #             repaired[3][i] = np.random.randint(1, self.n_hospitals + 1)
        
    #     # Ensure binary values for temp medical centers
    #     for i in range(self.n_temp_medical):
    #         repaired[4][i] = 1 if repaired[4][i] > 0.5 else 0
        
    #     # Repair medical allocations (Parts 6, 7, 8)
    #     # Part 6: Severe injured to hospitals - cannot exceed total severe injured
    #     for i in range(self.n_damage_points):
    #         if repaired[5][i] > self.severe_injured[i]:
    #             repaired[5][i] = self.severe_injured[i]
    #         elif repaired[5][i] < 0:
    #             repaired[5][i] = 0
        
    #     # Part 7: Moderate injured to hospitals - cannot exceed total moderate injured
    #     for i in range(self.n_damage_points):
    #         if repaired[6][i] > self.moderate_injured[i]:
    #             repaired[6][i] = self.moderate_injured[i]
    #         elif repaired[6][i] < 0:
    #             repaired[6][i] = 0
        
    #     # Part 8: Moderate injured to temp centers
    #     # First, set inactive centers to 0
    #     for i in range(self.n_temp_medical):
    #         if repaired[4][i] == 0:
    #             repaired[7][i] = 0
    #         elif repaired[7][i] < 0:
    #             repaired[7][i] = 0
        
    #     # Ensure total moderate injured allocation doesn't exceed available
    #     for i in range(self.n_damage_points):
    #         # Calculate total moderate injured allocated from this damage point
    #         total_moderate_allocated = repaired[6][i]  # To hospitals
            
    #         # Add moderate injured sent to temp centers from this damage point
    #         # (This is simplified - in reality you'd need to track which damage point sends to which temp center)
            
    #         # If over-allocated, scale down proportionally
    #         if total_moderate_allocated > self.moderate_injured[i]:
    #             scale = self.moderate_injured[i] / total_moderate_allocated
    #             repaired[6][i] = int(repaired[6][i] * scale)
        
    #     return repaired

    def crossover(self, parent1, parent2):
        """
        Custom crossover for humanitarian logistics chromosome
        """
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)
        
        # Part-wise crossover with different strategies for each part
        
        # Part 1 & 2: Distribution assignment and flows (uniform crossover)
        for i in range(self.n_shelters):
            if np.random.rand() < 0.5:
                child1[0][i], child2[0][i] = child2[0][i], child1[0][i]
                child1[1][i], child2[1][i] = child2[1][i], child1[1][i]
        
        # Part 3: Damage to shelter (uniform crossover)
        for i in range(self.n_shelters):
            if np.random.rand() < 0.5:
                child1[2][i], child2[2][i] = child2[2][i], child1[2][i]
        
        # Part 4: Damage to hospital (two-point crossover)
        if self.n_damage_points > 2:
            points = sorted(np.random.choice(self.n_damage_points, 2, replace=False))
            temp = child1[3][points[0]:points[1]]
            child1[3][points[0]:points[1]] = child2[3][points[0]:points[1]]
            child2[3][points[0]:points[1]] = temp
        
        # Part 5: Temp medical centers (uniform crossover)
        for i in range(self.n_temp_medical):
            if np.random.rand() < 0.5:
                child1[4][i], child2[4][i] = child2[4][i], child1[4][i]
        
        # Part 6 & 7: Medical allocations for injured (arithmetic crossover)
        # This maintains more realistic values
        for i in range(self.n_damage_points):
            if np.random.rand() < 0.5:
                # Swap severe injured allocations
                child1[5][i], child2[5][i] = child2[5][i], child1[5][i]
                # Swap moderate to hospital allocations
                child1[6][i], child2[6][i] = child2[6][i], child1[6][i]
            else:
                # Arithmetic crossover for smoother values
                alpha = np.random.uniform(0.3, 0.7)
                temp1 = int(alpha * child1[5][i] + (1-alpha) * child2[5][i])
                temp2 = int((1-alpha) * child1[5][i] + alpha * child2[5][i])
                child1[5][i], child2[5][i] = temp1, temp2
                
                temp1 = int(alpha * child1[6][i] + (1-alpha) * child2[6][i])
                temp2 = int((1-alpha) * child1[6][i] + alpha * child2[6][i])
                child1[6][i], child2[6][i] = temp1, temp2
        
        # Part 8: Moderate to temp medical (uniform crossover)
        for i in range(self.n_temp_medical):
            if np.random.rand() < 0.5:
                child1[7][i], child2[7][i] = child2[7][i], child1[7][i]
        
        # Repair children to ensure validity
        child1 = self.repair_chromosome(child1)
        child2 = self.repair_chromosome(child2)
        
        return child1, child2

    def mutate(self, chromosome, mutation_rate=0.1):
        """
        Custom mutation for humanitarian logistics chromosome
        """
        mutated = deepcopy(chromosome)
        
        # Part 1: Distribution assignment mutation
        for i in range(self.n_shelters):
            if np.random.rand() < mutation_rate:
                if mutated[0][i] == 0:
                    # Activate shelter with probability
                    if np.random.rand() < 0.3:
                        mutated[0][i] = np.random.randint(1, self.n_distribution + 1)
                        mutated[1][i] = np.random.uniform(10, 200)
                        mutated[2][i] = np.random.randint(1, self.n_damage_points + 1)
                else:
                    # Change distribution center or deactivate
                    if np.random.rand() < 0.8:
                        mutated[0][i] = np.random.randint(1, self.n_distribution + 1)
                    else:
                        mutated[0][i] = 0
                        mutated[1][i] = 0
                        mutated[2][i] = 0
        
        # Part 2: Flow values mutation
        for i in range(self.n_shelters):
            if mutated[0][i] > 0 and np.random.rand() < mutation_rate:
                # Gaussian mutation for flow values
                mutated[1][i] = max(0, mutated[1][i] + np.random.normal(0, 20))
        
        # Part 3: Damage to shelter mutation
        for i in range(self.n_shelters):
            if mutated[0][i] > 0 and np.random.rand() < mutation_rate:
                mutated[2][i] = np.random.randint(1, self.n_damage_points + 1)
        
        # Part 4: Damage to hospital mutation
        for i in range(self.n_damage_points):
            if np.random.rand() < mutation_rate:
                mutated[3][i] = np.random.randint(1, self.n_hospitals + 1)
        
        # Part 5: Temp medical centers mutation (bit flip)
        for i in range(self.n_temp_medical):
            if np.random.rand() < mutation_rate:
                mutated[4][i] = 1 - mutated[4][i]
                # If activated, allocate some injured
                if mutated[4][i] == 1:
                    # Will be properly allocated in repair
                    pass
                else:
                    # If deactivated, clear allocations
                    mutated[7][i] = 0
        
        # Part 6: Severe injured mutation (small adjustments)
        for i in range(self.n_damage_points):
            if np.random.rand() < mutation_rate:
                # Small Gaussian perturbation
                change = int(np.random.normal(0, 5))
                mutated[5][i] = max(0, min(self.severe_injured[i], mutated[5][i] + change))
        
        # Part 7: Moderate to hospital mutation
        for i in range(self.n_damage_points):
            if np.random.rand() < mutation_rate:
                # Reallocate between hospital and temp centers
                if np.random.rand() < 0.5:
                    # Increase hospital allocation
                    increase = np.random.randint(1, 10)
                    mutated[6][i] = min(self.moderate_injured[i], mutated[6][i] + increase)
                else:
                    # Decrease hospital allocation (send more to temp)
                    decrease = np.random.randint(1, 10)
                    mutated[6][i] = max(0, mutated[6][i] - decrease)
        
        # Part 8: Moderate to temp medical mutation
        active_temps = [i for i in range(self.n_temp_medical) if mutated[4][i] == 1]
        if active_temps and np.random.rand() < mutation_rate:
            # Redistribute among active temp centers
            for i in range(len(active_temps)):
                if np.random.rand() < 0.3:
                    change = int(np.random.normal(0, 10))
                    mutated[7][active_temps[i]] = max(0, mutated[7][active_temps[i]] + change)
        
        return self.repair_chromosome(mutated)

    def run(self, problem):
        """
        Run NSGA-II for humanitarian logistics problem
        """
        # Extract problem info
        cost_function = problem['cost_function']
        
        # Number of offsprings
        n_crossover = 2 * int(self.p_crossover * self.pop_size / 2)
        n_mutation = int(self.p_mutation * self.pop_size)
        
        # Empty individual
        empty_individual = {
            'chromosome': None,
            'cost': None,
            'rank': None,
            'crowding_distance': None,
        }
        
        # Initialize population
        pop = [deepcopy(empty_individual) for _ in range(self.pop_size)]
        for i in range(self.pop_size):
            pop[i]['chromosome'] = self.create_random_chromosome()
            pop[i]['cost'] = cost_function(pop[i]['chromosome'])
        
        # Non-dominated sorting
        pop, F = self.non_dominated_sorting(pop)
        
        # Calculate crowding distance
        pop = self.calc_crowding_distance(pop, F)
        
        # Sort population
        pop, F = self.sort_population(pop)
        
        # Main loop
        for it in range(self.max_iter):
            # Crossover
            popc = []
            for _ in range(n_crossover // 2):
                parents_idx = np.random.choice(range(self.pop_size), size=2, replace=False)
                p1 = pop[parents_idx[0]]
                p2 = pop[parents_idx[1]]
                
                c1_chrom, c2_chrom = self.crossover(p1['chromosome'], p2['chromosome'])
                
                c1 = deepcopy(empty_individual)
                c2 = deepcopy(empty_individual)
                c1['chromosome'] = c1_chrom
                c2['chromosome'] = c2_chrom
                c1['cost'] = cost_function(c1['chromosome'])
                c2['cost'] = cost_function(c2['chromosome'])
                
                popc.extend([c1, c2])
            
            # Mutation
            popm = []
            for _ in range(n_mutation):
                p = pop[np.random.randint(self.pop_size)]
                m = deepcopy(empty_individual)
                m['chromosome'] = self.mutate(p['chromosome'])
                m['cost'] = cost_function(m['chromosome'])
                popm.append(m)
            
            # Merge populations
            pop = pop + popc + popm
            
            # Non-dominated sorting
            pop, F = self.non_dominated_sorting(pop)
            
            # Calculate crowding distance
            pop = self.calc_crowding_distance(pop, F)
            
            # Sort population
            pop, F = self.sort_population(pop)
            
            # Truncate
            pop, F = self.truncate_population(pop, F)
            
            # Display iteration info
            if self.verbose:
                print(f'Iteration {it + 1}: Number of Pareto Members = {len(F[0])}')
        
        # Get Pareto front
        pareto_pop = [pop[i] for i in F[0]]
        
        return {
            'pop': pop,
            'F': F,
            'pareto_pop': pareto_pop,
        }

    def dominates(self, p, q):
        """Check if p dominates q"""
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
            costs = np.array([pop[i]['cost'] for i in F[k]])
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