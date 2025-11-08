import numpy as np
from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Callable
from abc import ABC, abstractmethod
import random
from convergence_metrics import ConvergenceMetrics
from diagnostic_metrics import DiagnosticMetrics


class IOptimizer(ABC):
    """Interface for optimization algorithms (Interface Segregation Principle)"""
    
    @abstractmethod
    def run(self, problem: Dict) -> Dict:
        pass
    
    @abstractmethod
    def create_random_solution(self) -> List:
        pass


class WolfPosition:
    """Single Responsibility: Manages wolf position and fitness (Single Responsibility Principle)"""
    
    def __init__(self, position: List, fitness: np.ndarray = None, 
                 normal_fitness: np.ndarray = None, constraint_violation: float = 0):
        self.position = position
        self.fitness = fitness if fitness is not None else np.array([np.inf, np.inf, np.inf])
        self.normal_fitness = normal_fitness if normal_fitness is not None else self.fitness
        self.constraint_violation = constraint_violation
        self.rank = None
        self.crowding_distance = 0
    
    def dominates(self, other: 'WolfPosition') -> bool:
        """Check if this wolf dominates another"""
        if self.constraint_violation > 0 and other.constraint_violation > 0:
            return self.constraint_violation < other.constraint_violation
        elif self.constraint_violation > 0 and other.constraint_violation == 0:
            return False
        elif self.constraint_violation == 0 and other.constraint_violation > 0:
            return True
        else:
            return all(self.fitness <= other.fitness) and any(self.fitness < other.fitness)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format compatible with metrics"""
        return {
            'chromosome': self.position,
            'cost': self.fitness,
            'normal_cost': self.normal_fitness,
            'rank': self.rank,
            'crowding_distance': self.crowding_distance,
            'constraint_violation': self.constraint_violation
        }


class ChromosomeHandler:
    """Single Responsibility: Handles chromosome operations (Open/Closed Principle)"""
    
    def __init__(self, n_shelters: int, n_distribution: int, n_damage_points: int,
                 n_hospitals: int, n_temp_medical: int):
        self.n_shelters = n_shelters
        self.n_distribution = n_distribution
        self.n_damage_points = n_damage_points
        self.n_hospitals = n_hospitals
        self.n_temp_medical = n_temp_medical
        self.da_id = list(range(3, 8))  # [3, 4, 5, 6, 7]
    
    def create_random(self) -> List:
        """Create random chromosome"""
        chromosome = []
        
        # Part 1: Distribution center assignment
        n_active = random.randint(1, self.n_shelters)
        active_shelters = np.random.choice(self.n_shelters, n_active, replace=False)
        dist_assignment = [0] * self.n_shelters
        for idx in active_shelters:
            dist_assignment[idx] = random.randint(1, self.n_distribution)
        chromosome.append(dist_assignment)
        
        # Part 2: Flow values
        flow_values = [np.clip(np.random.normal(0.6, 0.15), 0, 1) 
                      for _ in range(self.n_shelters)]
        chromosome.append(flow_values)
        
        # Part 3: Damage to shelter
        damage_to_shelter = deepcopy(self.da_id)
        np.random.shuffle(damage_to_shelter)
        for _ in range(self.n_damage_points, self.n_shelters):
            damage_to_shelter.append(np.random.choice(self.da_id))
        chromosome.append(damage_to_shelter)
        
        # Part 4: Severe injured to hospital
        part4 = self._create_allocation_matrix(self.n_damage_points, self.n_hospitals)
        chromosome.append(part4)
        
        # Part 5: Ambulance percentage for severe
        part5 = np.zeros((self.n_damage_points, self.n_hospitals))
        for i in range(self.n_damage_points):
            for j in range(self.n_hospitals):
                if part4[i][j] > 0:
                    part5[i][j] = np.random.random()
        chromosome.append(part5)
        
        # Part 6: Minor injured allocation
        part6 = self._create_allocation_matrix(self.n_damage_points, 
                                              self.n_hospitals + self.n_temp_medical)
        chromosome.append(part6)
        
        # Part 7: Ambulance percentage for minor
        part7 = np.zeros((self.n_damage_points, self.n_hospitals + self.n_temp_medical))
        for i in range(self.n_damage_points):
            for j in range(self.n_hospitals + self.n_temp_medical):
                if part6[i][j] > 0:
                    part7[i][j] = np.random.random()
        chromosome.append(part7)
        
        return chromosome
    
    def _create_allocation_matrix(self, rows: int, cols: int) -> np.ndarray:
        """Helper to create allocation matrices"""
        matrix = np.zeros((rows, cols))
        for i in range(rows):
            num_active = random.randint(1, cols)
            active_indices = random.sample(range(cols), num_active)
            weights = [np.random.random() for _ in range(num_active)]
            total = sum(weights)
            for idx, dest in enumerate(active_indices):
                matrix[i][dest] = weights[idx] / total
        return matrix
    
    def repair(self, chromosome: List) -> List:
        """Repair invalid chromosome parts"""
        repaired = deepcopy(chromosome)
        
        # Repair part 1
        if all(gene == 0 for gene in repaired[0]):
            n_active = random.randint(1, self.n_shelters)
            active_shelters = np.random.choice(self.n_shelters, n_active, replace=False)
            for idx in active_shelters:
                repaired[0][idx] = random.randint(1, self.n_distribution)
        
        # Repair part 4 and 6
        for part_idx in [3, 5]:
            if len(repaired) > part_idx:
                matrix = repaired[part_idx]
                rows, cols = matrix.shape
                for i in range(rows):
                    if np.sum(matrix[i, :]) == 0:
                        num_active = random.randint(1, cols)
                        active_indices = random.sample(range(cols), num_active)
                        weights = [np.random.random() for _ in range(num_active)]
                        total = sum(weights)
                        for idx, dest in enumerate(active_indices):
                            matrix[i][dest] = weights[idx] / total
        
        return repaired


class GWO_Humanitarian(IOptimizer):
    """Grey Wolf Optimizer for Multi-Objective Humanitarian Logistics"""
    
    def __init__(self, shelter_id: List[int], distribution_center_id: List[int],
                 damage_points_id: List[int], hospital_id: List[int],
                 temporary_medical_id: List[int], max_iter: int = 50,
                 pack_size: int = 100, verbose: bool = True):
        
        # Problem dimensions
        self.shelter_id = shelter_id
        self.distribution_center_id = distribution_center_id
        self.damage_points_id = damage_points_id
        self.hospital_id = hospital_id
        self.temporary_medical_id = temporary_medical_id
        
        # Algorithm parameters
        self.max_iter = max_iter
        self.pack_size = pack_size
        self.verbose = verbose
        
        # Initialize handlers
        self.chromosome_handler = ChromosomeHandler(
            len(shelter_id), len(distribution_center_id),
            len(damage_points_id), len(hospital_id), len(temporary_medical_id)
        )
        
        # Metrics
        self.metrics = ConvergenceMetrics()
        self.diagnostics = DiagnosticMetrics()
        
        # Leadership hierarchy
        self.alpha: Optional[WolfPosition] = None
        self.beta: Optional[WolfPosition] = None
        self.delta: Optional[WolfPosition] = None
    
    def create_random_solution(self) -> List:
        """Create random solution"""
        return self.chromosome_handler.create_random()
    
    def _evaluate_pack(self, wolves: List[WolfPosition], cost_function: Callable) -> None:
        """Evaluate fitness for all wolves"""
        chromosomes = [w.position for w in wolves if w.fitness[0] == np.inf]
        if chromosomes:
            costs, constraints, normal_costs = cost_function(chromosomes)
            idx = 0
            for wolf in wolves:
                if wolf.fitness[0] == np.inf:
                    wolf.fitness = np.array(costs[idx])
                    wolf.normal_fitness = np.array(normal_costs[idx])
                    wolf.constraint_violation = constraints[idx]
                    idx += 1
    
    def _update_leadership(self, wolves: List[WolfPosition]) -> None:
        """Update alpha, beta, and delta wolves using non-dominated sorting"""
        # Non-dominated sorting
        fronts = self._non_dominated_sorting(wolves)
        
        # Select leaders from first fronts
        leaders = []
        for front in fronts:
            leaders.extend([wolves[i] for i in front])
            if len(leaders) >= 3:
                break
        
        # Calculate crowding distance for leaders
        if leaders:
            self._calculate_crowding_distance(leaders)
            leaders.sort(key=lambda x: (-x.rank, -x.crowding_distance))
            
            self.alpha = leaders[0] if len(leaders) > 0 else wolves[0]
            self.beta = leaders[1] if len(leaders) > 1 else wolves[1]
            self.delta = leaders[2] if len(leaders) > 2 else wolves[2]
    
    def _non_dominated_sorting(self, wolves: List[WolfPosition]) -> List[List[int]]:
        """Fast non-dominated sorting"""
        n = len(wolves)
        domination_set = [[] for _ in range(n)]
        dominated_count = [0] * n
        fronts = [[]]
        
        for i in range(n):
            for j in range(i + 1, n):
                if wolves[i].dominates(wolves[j]):
                    domination_set[i].append(j)
                    dominated_count[j] += 1
                elif wolves[j].dominates(wolves[i]):
                    domination_set[j].append(i)
                    dominated_count[i] += 1
            
            if dominated_count[i] == 0:
                wolves[i].rank = 0
                fronts[0].append(i)
        
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in domination_set[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        wolves[j].rank = k + 1
                        next_front.append(j)
            k += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
        
        return fronts[:-1] if fronts[-1] == [] else fronts
    
    def _calculate_crowding_distance(self, wolves: List[WolfPosition]) -> None:
        """Calculate crowding distance for diversity preservation"""
        n = len(wolves)
        if n <= 2:
            for wolf in wolves:
                wolf.crowding_distance = np.inf
            return
        
        n_obj = len(wolves[0].fitness)
        for wolf in wolves:
            wolf.crowding_distance = 0
        
        for obj in range(n_obj):
            wolves.sort(key=lambda x: x.normal_fitness[obj])
            
            wolves[0].crowding_distance = np.inf
            wolves[-1].crowding_distance = np.inf
            
            f_max = wolves[-1].normal_fitness[obj]
            f_min = wolves[0].normal_fitness[obj]
            
            if f_max - f_min > 0:
                for i in range(1, n - 1):
                    wolves[i].crowding_distance += (
                        (wolves[i + 1].normal_fitness[obj] - wolves[i - 1].normal_fitness[obj]) /
                        (f_max - f_min)
                    )
    
    def _update_position(self, wolf: WolfPosition, a: float) -> List:
        """Update wolf position based on leaders"""
        new_position = []
        
        for part_idx in range(len(wolf.position)):
            if isinstance(wolf.position[part_idx], list):
                # Handle list parts
                new_part = []
                for i in range(len(wolf.position[part_idx])):
                    # Calculate coefficients
                    r1, r2 = np.random.random(2)
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    
                    r1, r2 = np.random.random(2)
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    
                    r1, r2 = np.random.random(2)
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    
                    # Calculate distances to leaders
                    D_alpha = abs(C1 * self.alpha.position[part_idx][i] - wolf.position[part_idx][i])
                    D_beta = abs(C2 * self.beta.position[part_idx][i] - wolf.position[part_idx][i])
                    D_delta = abs(C3 * self.delta.position[part_idx][i] - wolf.position[part_idx][i])
                    
                    # Update position
                    X1 = self.alpha.position[part_idx][i] - A1 * D_alpha
                    X2 = self.beta.position[part_idx][i] - A2 * D_beta
                    X3 = self.delta.position[part_idx][i] - A3 * D_delta
                    
                    new_val = (X1 + X2 + X3) / 3
                    
                    # Apply constraints based on part type
                    if part_idx == 0:  # Distribution assignment
                        new_val = int(np.clip(new_val, 0, self.chromosome_handler.n_distribution))
                    elif part_idx == 1:  # Flow values
                        new_val = np.clip(new_val, 0, 1)
                    elif part_idx == 2:  # Damage to shelter
                        if i < self.chromosome_handler.n_damage_points:
                            new_val = wolf.position[part_idx][i]  # Keep original for first n_damage_points
                        else:
                            new_val = int(np.clip(new_val, min(self.chromosome_handler.da_id),
                                                 max(self.chromosome_handler.da_id)))
                    
                    new_part.append(new_val)
                new_position.append(new_part)
                
            elif isinstance(wolf.position[part_idx], np.ndarray):
                # Handle matrix parts
                new_matrix = np.zeros_like(wolf.position[part_idx])
                rows, cols = wolf.position[part_idx].shape
                
                for i in range(rows):
                    for j in range(cols):
                        # Similar update process for matrices
                        r1, r2 = np.random.random(2)
                        A1 = 2 * a * r1 - a
                        C1 = 2 * r2
                        
                        r1, r2 = np.random.random(2)
                        A2 = 2 * a * r1 - a
                        C2 = 2 * r2
                        
                        r1, r2 = np.random.random(2)
                        A3 = 2 * a * r1 - a
                        C3 = 2 * r2
                        
                        D_alpha = abs(C1 * self.alpha.position[part_idx][i, j] - 
                                    wolf.position[part_idx][i, j])
                        D_beta = abs(C2 * self.beta.position[part_idx][i, j] - 
                                   wolf.position[part_idx][i, j])
                        D_delta = abs(C3 * self.delta.position[part_idx][i, j] - 
                                    wolf.position[part_idx][i, j])
                        
                        X1 = self.alpha.position[part_idx][i, j] - A1 * D_alpha
                        X2 = self.beta.position[part_idx][i, j] - A2 * D_beta
                        X3 = self.delta.position[part_idx][i, j] - A3 * D_delta
                        
                        new_matrix[i, j] = np.clip((X1 + X2 + X3) / 3, 0, 1)
                
                # Normalize rows for allocation matrices (parts 3 and 5)
                if part_idx in [3, 5]:
                    for i in range(rows):
                        row_sum = np.sum(new_matrix[i, :])
                        if row_sum > 0:
                            new_matrix[i, :] /= row_sum
                        else:
                            # If all zeros, randomly activate one
                            new_matrix[i, np.random.randint(cols)] = 1.0
                
                new_position.append(new_matrix)
            else:
                new_position.append(wolf.position[part_idx])
        
        return self.chromosome_handler.repair(new_position)
    
    def run(self, problem: Dict) -> Dict:
        """Run GWO algorithm"""
        cost_function = problem['cost_function']
        
        # Initialize wolf pack
        wolves = [WolfPosition(self.create_random_solution()) for _ in range(self.pack_size)]
        
        # Evaluate initial pack
        self._evaluate_pack(wolves, cost_function)
        
        # Initialize leaders
        self._update_leadership(wolves)
        
        # Storage for history
        pareto_history = []
        
        # Main loop
        for iteration in range(self.max_iter):
            # Linearly decrease a from 2 to 0
            a = 2 - iteration * (2 / self.max_iter)
            
            # Update each wolf position
            for i, wolf in enumerate(wolves):
                if wolf != self.alpha and wolf != self.beta and wolf != self.delta:
                    new_position = self._update_position(wolf, a)
                    wolves[i] = WolfPosition(new_position)
            
            # Evaluate new positions
            self._evaluate_pack(wolves, cost_function)
            
            # Update leadership
            self._update_leadership(wolves)
            
            # Get Pareto front
            fronts = self._non_dominated_sorting(wolves)
            pareto_wolves = [wolves[i] for i in fronts[0]] if fronts else []
            
            # Store history
            pareto_dict = [w.to_dict() for w in pareto_wolves]
            pareto_history.append(deepcopy(pareto_dict))
            
            # Update metrics
            if iteration % 5 == 0 or iteration == self.max_iter - 1:
                self.diagnostics.update_all_metrics(
                    population=[w.to_dict() for w in wolves],
                    pareto_pop=pareto_dict,
                    offspring_pop=None,
                    mutation_pop=None
                )
            
            # Display progress
            if self.verbose:
                best_f1 = min(w.fitness[0] for w in pareto_wolves) if pareto_wolves else np.inf
                best_f2 = min(w.fitness[1] for w in pareto_wolves) if pareto_wolves else np.inf
                best_f3 = min(w.fitness[2] for w in pareto_wolves) if pareto_wolves else np.inf
                
                print(f"Iteration {iteration + 1}/{self.max_iter}: "
                      f"Pareto Size = {len(pareto_wolves)}, "
                      f"Best [F1={best_f1:.2f}, F2={best_f2:.2f}, F3={best_f3:.2f}]")
        
        # Final metrics update
        self.metrics.update_metrics(pareto_history)
        
        # Prepare results
        final_pop = [w.to_dict() for w in wolves]
        final_pareto = [w.to_dict() for w in pareto_wolves]
        
        return {
            'pop': final_pop,
            'F': fronts,
            'pareto_pop': final_pareto,
            'pareto_history': pareto_history,
            'metrics': self.metrics,
            'diagnostics': self.diagnostics
        }


def run_gwo_optimization(main_instance):
    """Run GWO optimization using Main class instance"""
    
    print("\n" + "="*60)
    print("Starting GWO (Grey Wolf Optimizer)")
    print("="*60)
    
    # Problem definition
    problem = {
        'cost_function': main_instance.complex_humanitarian_cost
    }
    
    # Initialize GWO
    gwo = GWO_Humanitarian(
        shelter_id=main_instance.ec_id,
        distribution_center_id=main_instance.idc_id,
        damage_points_id=main_instance.da_id,
        hospital_id=main_instance.h_id,
        temporary_medical_id=main_instance.tmc_id,
        max_iter=50,
        pack_size=100,
        verbose=True
    )
    
    # Run optimization
    results = gwo.run(problem)
    
    # Extract results
    pareto_pop = results['pareto_pop']
    metrics = results['metrics']
    diagnostics = results['diagnostics']
    
    print("\n" + "="*60)
    print("GWO Optimization Complete")
    print("="*60)
    print(f"Final Pareto front size: {len(pareto_pop)}")
    
    if pareto_pop:
        pf_costs = np.array([ind['cost'] for ind in pareto_pop])
        print(f"Best F1 (Distance): {np.min(pf_costs[:, 0]):.2f}")
        print(f"Best F2 (Unmet Demand): {np.min(pf_costs[:, 1]):.2f}")
        print(f"Best F3 (Death Probability): {np.min(pf_costs[:, 2]):.2f}")
    
    # Plot metrics
    metrics.plot_convergence(save_path='gwo_convergence_metrics.png')
    metrics.print_summary()
    
    # Diagnose problems
    diagnostics.diagnose_problems()
    diagnostics.plot_diagnostic_metrics(save_path='gwo_diagnostic_metrics.png')
    
    return results


# Usage example
if __name__ == "__main__":
    # This would be called from main.py
    from main_updated import Main
    
    # Create Main instance
    m = Main()
    
    # Run GWO
    gwo_results = run_gwo_optimization(m)
    
    # Compare with NSGA-II if needed
    print("\n" + "="*60)
    print("GWO Implementation Complete!")
    print("Results saved with metrics and diagnostics")
    print("="*60)
