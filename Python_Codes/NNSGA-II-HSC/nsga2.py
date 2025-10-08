import numpy as np
from copy import deepcopy
import random
from convergence_metrics import ConvergenceMetrics

class NSGA2_Humanitarian:
    """NSGA-II for humanitarian logistics optimization with custom chromosome structure"""

    def __init__(self, shelter_id, distribution_center_id, damage_points_id, hospital_id,
                temporary_medical_id, max_iter=100, pop_size=100, p_crossover=0.7, p_mutation=0.3, 
                verbose=True):
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
        self.da_id = damage_points_id
        self.idc_id = distribution_center_id
        self.n_shelters = len(shelter_id)
        self.n_distribution = len(distribution_center_id)
        self.n_damage_points = len(damage_points_id)
        self.n_hospitals = len(hospital_id)
        self.n_temp_medical = len(temporary_medical_id)
        
        self.metrics = ConvergenceMetrics()

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
        damage_to_shelter:list  = deepcopy(self.da_id)
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
            active_indices = random.sample(range(self.n_hospitals + self.n_temp_medical), num_active)
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
            
        # Part 3
        for i in range(self.n_shelters):
            if np.random.rand() < 0.5:
                child1[2][i], child2[2][i] = child2[2][i], child1[2][i]
        
        # Part 4
        point1 = np.random.choice(range(1, self.n_damage_points))
        point2 = np.random.choice(range(1, self.n_hospitals))
        child1[3][:point1, :point2], child2[3][:point1, :point2] = child2[3][:point1, :point2], child1[3][:point1, :point2]
        child1[3][point1:, point2:], child2[3][point1:, point2:] = child2[3][point1:, point2:], child1[3][point1:, point2:]
        for i in child1[3]:
            if sum(bool(x) for x in i) == 0:
                child1[3] = deepcopy(parent1[3])
        for i in child2[3]:
            if sum(bool(x) for x in i) == 0:
                child2[3] = deepcopy(parent2[3])         
        # Part 5
        point1 = np.random.choice(range(1, self.n_damage_points))
        point2 = np.random.choice(range(1, self.n_hospitals))
        child1[4][:point1, :point2], child2[4][:point1, :point2] = child2[4][:point1, :point2], child1[4][:point1, :point2]
        child1[4][point1:, point2:], child2[4][point1:, point2:] = child2[4][point1:, point2:], child1[4][point1:, point2:]
            
        # Part 6
        point1 = np.random.choice(range(1, self.n_damage_points))
        point2 = np.random.choice(range(1, self.n_hospitals))
        point3 = np.random.choice(range(self.n_hospitals+1, self.n_temp_medical))
        child1[5][:point1, :point2], child2[5][:point1, :point2] = child2[5][:point1, :point2], child1[5][:point1, :point2]
        child1[5][point1:, point2:self.n_hospitals], child2[5][point1:, point2:self.n_hospitals] = child2[5][point1:, point2:self.n_hospitals], child1[5][point1:, point2:self.n_hospitals]
        child1[5][:point1, self.n_hospitals:point3], child2[5][:point1, self.n_hospitals:point3] = child2[5][:point1, self.n_hospitals:point3], child1[5][:point1, self.n_hospitals:point3]
        child1[5][point1:, point3:], child2[5][point1:, point3:] = child2[5][point1:, point3:], child1[5][point1:, point3:]
        
        # Part 7
        point1 = np.random.choice(range(1, self.n_damage_points))
        point2, point3 = sorted(random.sample(range(1, self.n_hospitals + self.n_temp_medical), k=2))
        child1[6][:point1, point2:point3], child2[6][:point1, point2:point3] = child2[6][:point1, point2:point3], child1[6][:point1, point2:point3]
        child1[6][point1:, :point2], child2[6][point1:, :point2] = child2[6][point1:, :point2], child1[6][point1:, :point2]
        child1[6][point1:, point3:], child2[6][point1:, point3:] = child2[6][point1:, point3:], child1[6][point1:, point3:]
        
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
        mutated[1][i] = max(0, np.clip(mutated[1][i] + np.random.normal(0, 0.1), 0, 1))
        
        # Part 3
        i = np.random.randint(0, self.n_shelters)
        if mutated[2][i] == 0:
            mutated[2][i] = random.choice(self.da_id)
        elif mutated[2][i] != 0 and i > self.n_damage_points:
            mutated[2][i] = 0
        elif mutated[2][i] != 0 and i < self.n_damage_points:
            mutated[2][i] = random.choice(self.da_id)
                
        # Part 4 & 5
        for i in range(self.n_damage_points):
            j = random.randint(0, self.n_hospitals - 1)
            if mutated[3][i][j] == 0:
                if random.random() < 0.1:
                    mutated[3][i][j] = random.uniform(0, 0.2)
                    mutated[4][i][j] = max(0, np.clip(mutated[4][i][j] + np.random.normal(0, 0.1), 0, 1))
            else:
                if random.random() < 0.1:
                    mutated[3][i][j] = 0
                else:
                    mutated[3][i][j] = max(0, np.clip(mutated[3][i][j] + np.random.normal(0, 0.1), 0, 1))
                    mutated[4][i][j] = max(0, np.clip(mutated[4][i][j] + np.random.normal(0, 0.1), 0, 1))
        
        # Part 6
        for i in range(self.n_hospitals):
            j = random.randint(0, self.n_hospitals - 1)
            if mutated[5][i][j] == 0:
                if random.random() < 0.1:
                    mutated[5][i][j] = random.uniform(0, 0.2)
                    mutated[6][i][j] = max(0, np.clip(mutated[6][i][j] + np.random.normal(0, 0.1), 0, 1))
            else:
                if random.random() < 0.1:
                    mutated[5][i][j] = 0
                else:
                    mutated[5][i][j] = max(0, np.clip(mutated[5][i][j] + np.random.normal(0, 0.1), 0, 1))
                    mutated[6][i][j] = max(0, np.clip(mutated[6][i][j] + np.random.normal(0, 0.1), 0, 1))
            k = random.randint(self.n_hospitals, self.n_hospitals + self.n_temp_medical - 1)
            if mutated[5][i][k] == 0:
                if random.random() < 0.1:
                    mutated[5][i][k] = random.uniform(0, 0.2)
                    mutated[6][i][k] = max(0, np.clip(mutated[6][i][k] + np.random.normal(0, 0.1), 0, 1))
            else:
                if random.random() < 0.1:
                    mutated[5][i][k] = 0
                else:
                    mutated[5][i][k] = max(0, np.clip(mutated[5][i][k] + np.random.normal(0, 0.1), 0, 1))
                    mutated[6][i][k] = max(0, np.clip(mutated[6][i][k] + np.random.normal(0, 0.1), 0, 1))
        
        return mutated

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
            print('iteration: ', it)
            popc = []
            for _ in range(n_crossover // 2):
                p1 = self.crowding_tournament_selection(pop)
                p2 = self.crowding_tournament_selection(pop)
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
            
            # اضافه کردن محاسبه metrics
            pareto_pop = [pop[i] for i in F[0]]
            self.metrics.update_metrics(pareto_pop, it)
            
            # Display iteration info
            if self.verbose:
                print(f'Iteration {it + 1}: Number of Pareto Members = {len(F[0])}')
                if it > 0:
                    print(f'   Hypervolume: {self.metrics.hypervolume_history[-1]:.6f}')
                    print(f'   Spacing: {self.metrics.spacing_history[-1]:.6f}')

        # Get Pareto front
        pareto_pop = [pop[i] for i in F[0]]
        
        return {
            'pop': pop,
            'F': F,
            'pareto_pop': pareto_pop,
            'metrics': self.metrics
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
    
    def crowding_tournament_selection(self, pop, tournament_size=2):
        """انتخاب با ترجیح rank کمتر و تنوع بیشتر"""
        idx1, idx2 = np.random.choice(range(len(pop)), size=tournament_size, replace=False)
        
        p1, p2 = pop[idx1], pop[idx2]
        
        # اگر rank متفاوت، rank کمتر برنده
        if p1['rank'] < p2['rank']:
            return p1
        elif p2['rank'] < p1['rank']:
            return p2
        else:
            # اگر rank یکسان، crowding distance بیشتر برنده (حفظ تنوع)
            return p1 if p1['crowding_distance'] > p2['crowding_distance'] else p2