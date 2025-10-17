import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from typing import List, Dict, Tuple, Optional
import pickle
import json
from dataclasses import dataclass
from collections import defaultdict
import math
import random
import itertools


@dataclass
class EpsilonConstraintSolution:
    """نگهداری اطلاعات راه‌حل‌های ε-constraint"""
    chromosome: List
    objectives: np.ndarray
    epsilon_values: Dict[str, float]
    primary_objective: str
    constraint_violation: float = 0.0


class EpsilonConstraintHumanitarian:
    """
    پیاده‌سازی روش ε-Constraint برای مسئله لجستیک بشردوستانه
    
    این روش یکی از توابع هدف را به عنوان هدف اصلی انتخاب کرده و 
    بقیه را به صورت قیدهای ε تبدیل می‌کند.
    """
    
    def __init__(self, main_instance):
        """
        Parameters:
        -----------
        main_instance: شیء از کلاس Main که شامل تمام پارامترها و توابع مسئله است
        """
        self.main = main_instance
        self.shelter_id = main_instance.ec_id
        self.idc_id = main_instance.idc_id
        self.da_id = main_instance.da_id
        self.h_id = main_instance.h_id
        self.tmc_id = main_instance.tmc_id
        
        self.n_shelters = len(self.shelter_id)
        self.n_distribution = len(self.idc_id)
        self.n_damage_points = len(self.da_id)
        self.n_hospitals = len(self.h_id)
        self.n_temp_medical = len(self.tmc_id)
        
        # ذخیره راه‌حل‌ها
        self.solutions = []
        self.pareto_front = []
        
        # محدوده‌های توابع هدف (برای تعیین epsilon)
        self.objective_ranges = {
            'F1': {'min': 0, 'max': 1e6},  # Distance
            'F2': {'min': 0, 'max': 1e5},  # Unmet Demand
            'F3': {'min': 0, 'max': 1000}  # Death Probability
        }
        
    def encode_chromosome_to_vector(self, chromosome):
        """
        تبدیل کروموزوم ساختاریافته به بردار برای بهینه‌سازی
        """
        vector = []
        
        # Part 1: Distribution center assignment (discrete -> continuous [0,1])
        for val in chromosome[0]:
            if val == 0:
                vector.append(0)
            else:
                vector.append((val - 0.5) / self.n_distribution)
        
        # Part 2: Flow values [0,1]
        vector.extend(chromosome[1])
        
        # Part 3: Damage to shelter (discrete -> continuous)
        for val in chromosome[2]:
            if val == 0:
                vector.append(0)
            else:
                vector.append((val - min(self.da_id) + 0.5) / len(self.da_id))
        
        # Parts 4-7: Matrices (flatten)
        for matrix in chromosome[3:]:
            vector.extend(np.array(matrix).flatten())
            
        return np.array(vector)
    
    def decode_vector_to_chromosome(self, vector):
        """
        تبدیل بردار به کروموزوم ساختاریافته
        """
        chromosome = []
        idx = 0
        
        # Part 1: Distribution center assignment
        part1 = []
        for i in range(self.n_shelters):
            val = vector[idx]
            if val <= 0.01:  # Threshold for inactive shelter
                part1.append(0)
            else:
                part1.append(max(1, min(self.n_distribution, 
                                       int(val * self.n_distribution + 0.5))))
            idx += 1
        chromosome.append(part1)
        
        # Part 2: Flow values
        part2 = list(np.clip(vector[idx:idx+self.n_shelters], 0, 1))
        idx += self.n_shelters
        chromosome.append(part2)
        
        # Part 3: Damage to shelter
        part3 = []
        for i in range(self.n_shelters):
            val = vector[idx]
            if val <= 0.01:
                part3.append(0)
            else:
                da_idx = int(val * len(self.da_id) - 0.5)
                da_idx = max(0, min(len(self.da_id)-1, da_idx))
                part3.append(self.da_id[da_idx])
            idx += 1
        chromosome.append(part3)
        
        # Part 4: Damage to hospital for severe injured
        size4 = self.n_damage_points * self.n_hospitals
        matrix4 = vector[idx:idx+size4].reshape(self.n_damage_points, self.n_hospitals)
        matrix4 = np.clip(matrix4, 0, 1)
        # Normalize rows to sum to 1
        matrix4 = self.normalize_matrix_rows(matrix4)
        chromosome.append(matrix4.tolist())
        idx += size4
        
        # Part 5: Percentage by ambulance (severe)
        matrix5 = vector[idx:idx+size4].reshape(self.n_damage_points, self.n_hospitals)
        matrix5 = np.clip(matrix5, 0, 1)
        chromosome.append(matrix5.tolist())
        idx += size4
        
        # Part 6: Damage to hospital/TMC for minor injured
        size6 = self.n_damage_points * (self.n_hospitals + self.n_temp_medical)
        matrix6 = vector[idx:idx+size6].reshape(self.n_damage_points, 
                                                self.n_hospitals + self.n_temp_medical)
        matrix6 = np.clip(matrix6, 0, 1)
        matrix6 = self.normalize_matrix_rows(matrix6)
        chromosome.append(matrix6.tolist())
        idx += size6
        
        # Part 7: Percentage by ambulance (minor)
        matrix7 = vector[idx:idx+size6].reshape(self.n_damage_points, 
                                                self.n_hospitals + self.n_temp_medical)
        matrix7 = np.clip(matrix7, 0, 1)
        chromosome.append(matrix7.tolist())
        
        return chromosome
    
    def normalize_matrix_rows(self, matrix):
        """نرمال‌سازی سطرهای ماتریس"""
        normalized = matrix.copy()
        for i in range(matrix.shape[0]):
            row_sum = np.sum(matrix[i, :])
            if row_sum > 0:
                normalized[i, :] = matrix[i, :] / row_sum
            else:
                # اگر سطر صفر است، به صورت تصادفی مقداردهی کن
                n_active = random.randint(1, matrix.shape[1])
                active_indices = random.sample(range(matrix.shape[1]), n_active)
                weights = np.random.random(n_active)
                weights = weights / np.sum(weights)
                for idx, j in enumerate(active_indices):
                    normalized[i, j] = weights[idx]
        return normalized
    
    def objective_function(self, vector, primary_obj='F1', epsilon_constraints=None):
        """
        تابع هدف برای روش ε-constraint
        
        Parameters:
        -----------
        vector: بردار متغیرهای تصمیم
        primary_obj: تابع هدف اصلی ('F1', 'F2', 'F3')
        epsilon_constraints: دیکشنری حاوی محدودیت‌های epsilon برای سایر توابع
        
        Returns:
        --------
        objective_value: مقدار تابع هدف اصلی
        """
        # تبدیل بردار به کروموزوم
        chromosome = self.decode_vector_to_chromosome(vector)
        
        # محاسبه توابع هدف
        objectives, constraint_violation = self.main.complex_humanitarian_cost([chromosome])
        f1, f2, f3 = objectives[0]
        
        obj_values = {'F1': f1, 'F2': f2, 'F3': f3}
        
        # اضافه کردن پنالتی برای نقض قیدها
        penalty = 0
        
        # پنالتی برای نقض قید بودجه
        if constraint_violation[0] > 0:
            penalty += 1e6 * constraint_violation[0]
        
        # پنالتی برای نقض قیدهای epsilon
        if epsilon_constraints:
            for obj_name, epsilon_val in epsilon_constraints.items():
                if obj_name != primary_obj:
                    if obj_values[obj_name] > epsilon_val:
                        penalty += 1e5 * (obj_values[obj_name] - epsilon_val)
        
        return obj_values[primary_obj] + penalty
    
    def find_objective_ranges(self, n_samples=50):
        """
        یافتن محدوده هر تابع هدف با نمونه‌برداری تصادفی
        """
        print("Finding objective ranges through random sampling...")
        
        all_objectives = {'F1': [], 'F2': [], 'F3': []}
        
        for i in range(n_samples):
            # ایجاد کروموزوم تصادفی
            chromosome = self.create_random_chromosome()
            
            # محاسبه توابع هدف
            objectives, _ = self.main.complex_humanitarian_cost([chromosome])
            f1, f2, f3 = objectives[0]
            
            all_objectives['F1'].append(f1)
            all_objectives['F2'].append(f2)
            all_objectives['F3'].append(f3)
            
            if (i+1) % 10 == 0:
                print(f"  Sampled {i+1}/{n_samples} solutions")
        
        # بروزرسانی محدوده‌ها
        for obj_name in ['F1', 'F2', 'F3']:
            self.objective_ranges[obj_name]['min'] = min(all_objectives[obj_name])
            self.objective_ranges[obj_name]['max'] = max(all_objectives[obj_name])
            print(f"  {obj_name} range: [{self.objective_ranges[obj_name]['min']:.2f}, "
                  f"{self.objective_ranges[obj_name]['max']:.2f}]")
    
    def create_random_chromosome(self):
        """ایجاد کروموزوم تصادفی (کپی از NSGA2)"""
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
        np.random.shuffle(self.da_id)
        damage_to_shelter = copy.deepcopy(self.da_id)
        for _ in range(self.n_damage_points, self.n_shelters):
            damage_to_shelter.append(np.random.choice(self.da_id))
        chromosome.append(damage_to_shelter)
        
        # Part 4: Damage to hospital (severe)
        part4 = np.zeros((self.n_damage_points, self.n_hospitals))
        for i in range(self.n_damage_points):
            num_active = random.randint(1, self.n_hospitals)
            active_indices = random.sample(range(self.n_hospitals), num_active)
            weights = [np.random.random() for _ in range(num_active)]
            total = sum(weights)
            for idx, dest in enumerate(active_indices):
                part4[i][dest] = weights[idx] / total
        chromosome.append(part4)
        
        # Part 5: Percentage by ambulance (severe)
        part5 = np.zeros((self.n_damage_points, self.n_hospitals))
        for i in range(self.n_damage_points):
            for j in range(self.n_hospitals):
                if part4[i][j] > 0:
                    part5[i][j] = np.random.random()
        chromosome.append(part5)
        
        # Part 6: Damage to TMC/hospital (minor)
        part6 = np.zeros((self.n_damage_points, self.n_hospitals + self.n_temp_medical))
        for i in range(self.n_damage_points):
            num_active = np.random.randint(1, self.n_hospitals + self.n_temp_medical)
            active_indices = random.sample(range(self.n_hospitals + self.n_temp_medical), num_active)
            weights = [np.random.random() for _ in range(num_active)]
            total = sum(weights)
            for idx, dest in enumerate(active_indices):
                part6[i][dest] = weights[idx] / total
        chromosome.append(part6)
        
        # Part 7: Percentage by ambulance (minor)
        part7 = np.zeros((self.n_damage_points, self.n_hospitals + self.n_temp_medical))
        for i in range(self.n_damage_points):
            for j in range(self.n_hospitals + self.n_temp_medical):
                if part6[i][j] > 0:
                    part7[i][j] = np.random.random()
        chromosome.append(part7)
        
        return chromosome
    
    def solve_single_objective(self, primary_obj='F1', epsilon_constraints=None, 
                             maxiter=50, popsize=15):
        """
        حل مسئله تک‌هدفه با قیدهای epsilon
        
        Parameters:
        -----------
        primary_obj: تابع هدف اصلی
        epsilon_constraints: قیدهای epsilon برای سایر توابع
        maxiter: تعداد تکرار
        popsize: اندازه جمعیت
        """
        # تعیین محدوده متغیرها
        n_vars = (self.n_shelters * 3 + 
                 self.n_damage_points * self.n_hospitals * 2 +
                 self.n_damage_points * (self.n_hospitals + self.n_temp_medical) * 2)
        
        bounds = [(0, 1) for _ in range(n_vars)]
        
        # تابع هدف با قیدها
        def obj_func(x):
            return self.objective_function(x, primary_obj, epsilon_constraints)
        
        # اجرای الگوریتم differential evolution
        result = differential_evolution(
            obj_func,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            atol=1e-5,
            tol=0.01,
            seed=42,
            disp=False
        )
        
        # تبدیل نتیجه به کروموزوم
        best_chromosome = self.decode_vector_to_chromosome(result.x)
        
        # محاسبه توابع هدف
        objectives, constraint_violation = self.main.complex_humanitarian_cost([best_chromosome])
        
        return {
            'chromosome': best_chromosome,
            'objectives': objectives[0],
            'constraint_violation': constraint_violation[0],
            'success': result.success,
            'message': result.message
        }
    
    def run_epsilon_constraint(self, n_epsilon_levels=5, primary_objective='F1'):
        """
        اجرای کامل روش ε-constraint
        
        Parameters:
        -----------
        n_epsilon_levels: تعداد سطوح epsilon برای هر تابع هدف
        primary_objective: تابع هدف اصلی ('F1', 'F2', 'F3')
        """
        print("\n" + "="*60)
        print("Starting ε-Constraint Method")
        print("="*60)
        print(f"Primary Objective: {primary_objective}")
        print(f"Number of epsilon levels: {n_epsilon_levels}")
        
        # Step 1: یافتن محدوده توابع هدف
        self.find_objective_ranges(n_samples=30)
        
        # Step 2: ایجاد شبکه epsilon
        epsilon_grid = self.create_epsilon_grid(n_epsilon_levels)
        
        # Step 3: حل مسئله برای هر ترکیب epsilon
        print(f"\nSolving for {len(epsilon_grid)} epsilon combinations...")
        
        for idx, epsilon_values in enumerate(epsilon_grid):
            print(f"\nIteration {idx+1}/{len(epsilon_grid)}")
            print(f"  Epsilon constraints: {epsilon_values}")
            
            # حل مسئله
            solution = self.solve_single_objective(
                primary_obj=primary_objective,
                epsilon_constraints=epsilon_values,
                maxiter=30,
                popsize=10
            )
            
            # ذخیره راه‌حل
            if solution['constraint_violation'] == 0:  # فقط راه‌حل‌های موجه
                sol = EpsilonConstraintSolution(
                    chromosome=solution['chromosome'],
                    objectives=np.array(solution['objectives']),
                    epsilon_values=epsilon_values,
                    primary_objective=primary_objective,
                    constraint_violation=solution['constraint_violation']
                )
                self.solutions.append(sol)
                
                print(f"  Found solution: F1={solution['objectives'][0]:.2f}, "
                      f"F2={solution['objectives'][1]:.2f}, "
                      f"F3={solution['objectives'][2]:.4f}")
            else:
                print(f"  No feasible solution found (constraint violation: "
                      f"{solution['constraint_violation']:.4f})")
        
        # Step 4: استخراج Pareto Front
        self.extract_pareto_front()
        
        print(f"\n{'='*60}")
        print(f"Optimization completed!")
        print(f"Total solutions found: {len(self.solutions)}")
        print(f"Pareto optimal solutions: {len(self.pareto_front)}")
        print(f"{'='*60}")
        
        return self.pareto_front
    
    def create_epsilon_grid(self, n_levels):
        """
        ایجاد شبکه epsilon برای قیدها
        
        Parameters:
        -----------
        n_levels: تعداد سطوح برای هر تابع هدف
        """
        epsilon_grid = []
        
        # تعیین توابع هدف که باید به قید تبدیل شوند
        objectives = ['F1', 'F2', 'F3']
        constrained_objectives = [obj for obj in objectives if obj != 'F1']  # فرض: F1 هدف اصلی
        
        # ایجاد مقادیر epsilon برای هر تابع هدف
        epsilon_values = {}
        for obj in constrained_objectives:
            min_val = self.objective_ranges[obj]['min']
            max_val = self.objective_ranges[obj]['max']
            epsilon_values[obj] = np.linspace(min_val, max_val, n_levels)
        
        # ایجاد تمام ترکیب‌های ممکن
        if len(constrained_objectives) == 2:
            for e1 in epsilon_values[constrained_objectives[0]]:
                for e2 in epsilon_values[constrained_objectives[1]]:
                    epsilon_grid.append({
                        constrained_objectives[0]: e1,
                        constrained_objectives[1]: e2
                    })
        
        return epsilon_grid
    
    def extract_pareto_front(self):
        """استخراج جبهه پارتو از راه‌حل‌ها"""
        if not self.solutions:
            return
        
        # تبدیل به آرایه numpy
        objectives = np.array([sol.objectives for sol in self.solutions])
        
        # یافتن نقاط غیرمغلوب
        n_solutions = len(self.solutions)
        is_pareto = np.ones(n_solutions, dtype=bool)
        
        for i in range(n_solutions):
            if not is_pareto[i]:
                continue
            for j in range(n_solutions):
                if i == j:
                    continue
                # بررسی domination
                if (np.all(objectives[j] <= objectives[i]) and 
                    np.any(objectives[j] < objectives[i])):
                    is_pareto[i] = False
                    break
        
        # ذخیره Pareto solutions
        self.pareto_front = [self.solutions[i] for i in range(n_solutions) if is_pareto[i]]
    
    def plot_results(self, save_path='epsilon_constraint_results.png'):
        """
        رسم نتایج روش ε-constraint
        """
        if not self.pareto_front:
            print("No Pareto solutions found to plot!")
            return
        
        # استخراج objectives
        pf_objectives = np.array([sol.objectives for sol in self.pareto_front])
        all_objectives = np.array([sol.objectives for sol in self.solutions])
        
        fig = plt.figure(figsize=(20, 6))
        
        # Plot 1: 3D Pareto Front
        ax1 = fig.add_subplot(141, projection='3d')
        
        # نمایش همه راه‌حل‌ها
        ax1.scatter(all_objectives[:, 0], all_objectives[:, 1], all_objectives[:, 2],
                   c='lightgray', s=30, alpha=0.3, label='All Solutions')
        
        # نمایش Pareto front
        ax1.scatter(pf_objectives[:, 0], pf_objectives[:, 1], pf_objectives[:, 2],
                   c='red', s=50, alpha=0.8, edgecolors='black', label='Pareto Front')
        
        ax1.set_xlabel('F1: Distance')
        ax1.set_ylabel('F2: Unmet Demand')
        ax1.set_zlabel('F3: Death Probability')
        ax1.set_title('3D Pareto Front (ε-Constraint)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: F1 vs F2
        ax2 = fig.add_subplot(142)
        ax2.scatter(all_objectives[:, 0], all_objectives[:, 1],
                   c='lightgray', s=30, alpha=0.3, label='All')
        ax2.scatter(pf_objectives[:, 0], pf_objectives[:, 1],
                   c='red', s=50, alpha=0.8, edgecolors='black', label='Pareto')
        ax2.set_xlabel('F1: Distance')
        ax2.set_ylabel('F2: Unmet Demand')
        ax2.set_title('F1 vs F2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: F1 vs F3
        ax3 = fig.add_subplot(143)
        ax3.scatter(all_objectives[:, 0], all_objectives[:, 2],
                   c='lightgray', s=30, alpha=0.3, label='All')
        ax3.scatter(pf_objectives[:, 0], pf_objectives[:, 2],
                   c='red', s=50, alpha=0.8, edgecolors='black', label='Pareto')
        ax3.set_xlabel('F1: Distance')
        ax3.set_ylabel('F3: Death Probability')
        ax3.set_title('F1 vs F3')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: F2 vs F3
        ax4 = fig.add_subplot(144)
        ax4.scatter(all_objectives[:, 1], all_objectives[:, 2],
                   c='lightgray', s=30, alpha=0.3, label='All')
        ax4.scatter(pf_objectives[:, 1], pf_objectives[:, 2],
                   c='red', s=50, alpha=0.8, edgecolors='black', label='Pareto')
        ax4.set_xlabel('F2: Unmet Demand')
        ax4.set_ylabel('F3: Death Probability')
        ax4.set_title('F2 vs F3')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compare_with_nsga2(self, nsga2_pareto_front):
        """
        مقایسه نتایج ε-constraint با NSGA-II
        
        Parameters:
        -----------
        nsga2_pareto_front: آرایه numpy از نتایج NSGA-II
        """
        if not self.pareto_front:
            print("No ε-constraint solutions to compare!")
            return
        
        epsilon_objectives = np.array([sol.objectives for sol in self.pareto_front])
        
        fig = plt.figure(figsize=(15, 5))
        
        # 3D comparison
        ax = fig.add_subplot(131, projection='3d')
        
        # NSGA-II solutions
        ax.scatter(nsga2_pareto_front[:, 0], nsga2_pareto_front[:, 1], 
                  nsga2_pareto_front[:, 2],
                  c='blue', s=40, alpha=0.6, label='NSGA-II')
        
        # ε-constraint solutions
        ax.scatter(epsilon_objectives[:, 0], epsilon_objectives[:, 1], 
                  epsilon_objectives[:, 2],
                  c='red', s=40, alpha=0.6, marker='^', label='ε-Constraint')
        
        ax.set_xlabel('F1: Distance')
        ax.set_ylabel('F2: Unmet Demand')
        ax.set_zlabel('F3: Death Probability')
        ax.set_title('Method Comparison: 3D View')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # F1 vs F2 comparison
        ax2 = fig.add_subplot(132)
        ax2.scatter(nsga2_pareto_front[:, 0], nsga2_pareto_front[:, 1],
                   c='blue', s=40, alpha=0.6, label='NSGA-II')
        ax2.scatter(epsilon_objectives[:, 0], epsilon_objectives[:, 1],
                   c='red', s=40, alpha=0.6, marker='^', label='ε-Constraint')
        ax2.set_xlabel('F1: Distance')
        ax2.set_ylabel('F2: Unmet Demand')
        ax2.set_title('Method Comparison: F1 vs F2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Statistics comparison
        ax3 = fig.add_subplot(133)
        
        # محاسبه آمار
        stats_data = []
        methods = ['NSGA-II', 'ε-Constraint']
        objectives_data = [nsga2_pareto_front, epsilon_objectives]
        colors = ['blue', 'red']
        
        x_pos = np.arange(3)
        width = 0.35
        
        for i, (method, data) in enumerate(zip(methods, objectives_data)):
            means = np.mean(data, axis=0)
            ax3.bar(x_pos + i*width, means, width, label=method, color=colors[i], alpha=0.7)
        
        ax3.set_xlabel('Objectives')
        ax3.set_ylabel('Mean Value')
        ax3.set_title('Mean Objective Values Comparison')
        ax3.set_xticks(x_pos + width/2)
        ax3.set_xticklabels(['F1', 'F2', 'F3'])
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('epsilon_nsga2_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # چاپ آمار مقایسه‌ای
        print("\n" + "="*60)
        print("Comparison Statistics")
        print("="*60)
        print(f"\nNumber of solutions:")
        print(f"  NSGA-II: {len(nsga2_pareto_front)}")
        print(f"  ε-Constraint: {len(epsilon_objectives)}")
        
        print(f"\nObjective ranges:")
        for i, obj_name in enumerate(['F1', 'F2', 'F3']):
            print(f"\n{obj_name}:")
            print(f"  NSGA-II:      [{np.min(nsga2_pareto_front[:, i]):.2f}, "
                  f"{np.max(nsga2_pareto_front[:, i]):.2f}]")
            print(f"  ε-Constraint: [{np.min(epsilon_objectives[:, i]):.2f}, "
                  f"{np.max(epsilon_objectives[:, i]):.2f}]")
        
        # محاسبه hypervolume (تقریبی)
        ref_point = np.max(np.vstack([nsga2_pareto_front, epsilon_objectives]), axis=0) * 1.1
        
        hv_nsga2 = self.calculate_hypervolume_approx(nsga2_pareto_front, ref_point)
        hv_epsilon = self.calculate_hypervolume_approx(epsilon_objectives, ref_point)
        
        print(f"\nApproximate Hypervolume:")
        print(f"  NSGA-II:      {hv_nsga2:.4f}")
        print(f"  ε-Constraint: {hv_epsilon:.4f}")
        print("="*60)
    
    def calculate_hypervolume_approx(self, points, ref_point):
        """محاسبه تقریبی hypervolume با Monte Carlo"""
        n_samples = 10000
        count = 0
        
        for _ in range(n_samples):
            random_point = np.random.uniform(0, ref_point, size=3)
            
            # آیا این نقطه dominated است؟
            for p in points:
                if np.all(p <= random_point):
                    count += 1
                    break
        
        volume = (count / n_samples) * np.prod(ref_point)
        return volume
    
    def save_results(self, filename='epsilon_constraint_results.pkl'):
        """ذخیره نتایج در فایل"""
        results = {
            'solutions': self.solutions,
            'pareto_front': self.pareto_front,
            'objective_ranges': self.objective_ranges,
            'method': 'epsilon-constraint'
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        # ذخیره نتایج به صورت JSON (فقط objectives)
        json_results = {
            'n_solutions': len(self.solutions),
            'n_pareto': len(self.pareto_front),
            'objective_ranges': self.objective_ranges,
            'pareto_objectives': [sol.objectives.tolist() for sol in self.pareto_front]
        }
        
        with open(filename.replace('.pkl', '.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {filename} and {filename.replace('.pkl', '.json')}")


def run_epsilon_constraint_optimization(main_instance):
    """
    تابع اصلی برای اجرای بهینه‌سازی با روش ε-constraint
    
    Parameters:
    -----------
    main_instance: شیء از کلاس Main
    """
    print("\n" + "="*70)
    print("EPSILON-CONSTRAINT METHOD FOR HUMANITARIAN LOGISTICS")
    print("="*70)
    
    # ایجاد شیء ε-constraint
    epsilon_solver = EpsilonConstraintHumanitarian(main_instance)
    
    # اجرای روش ε-constraint
    # می‌توانید تعداد سطوح epsilon و تابع هدف اصلی را تغییر دهید
    pareto_solutions = epsilon_solver.run_epsilon_constraint(
        n_epsilon_levels=4,  # تعداد سطوح epsilon (4×4 = 16 ترکیب)
        primary_objective='F1'  # می‌تواند 'F1', 'F2', یا 'F3' باشد
    )
    
    # رسم نتایج
    epsilon_solver.plot_results(save_path='epsilon_constraint_results.png')
    
    # ذخیره نتایج
    epsilon_solver.save_results('epsilon_results.pkl')
    
    # نمایش خلاصه نتایج
    if pareto_solutions:
        pf_objectives = np.array([sol.objectives for sol in pareto_solutions])
        
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS SUMMARY")
        print("="*60)
        print(f"Total Pareto optimal solutions found: {len(pareto_solutions)}")
        print(f"\nBest values for each objective:")
        print(f"  Best F1 (Min Distance): {np.min(pf_objectives[:, 0]):.2f}")
        print(f"  Best F2 (Min Unmet Demand): {np.min(pf_objectives[:, 1]):.2f}")
        print(f"  Best F3 (Min Death Probability): {np.min(pf_objectives[:, 2]):.4f}")
        
        print(f"\nWorst values in Pareto front:")
        print(f"  Worst F1: {np.max(pf_objectives[:, 0]):.2f}")
        print(f"  Worst F2: {np.max(pf_objectives[:, 1]):.2f}")
        print(f"  Worst F3: {np.max(pf_objectives[:, 2]):.4f}")
        
        # یافتن راه‌حل متعادل (نزدیک‌ترین به نقطه ایده‌آل)
        ideal_point = np.min(pf_objectives, axis=0)
        nadir_point = np.max(pf_objectives, axis=0)
        
        # نرمال‌سازی
        normalized_objectives = (pf_objectives - ideal_point) / (nadir_point - ideal_point + 1e-10)
        
        # فاصله از نقطه ایده‌آل
        distances = np.linalg.norm(normalized_objectives, axis=1)
        balanced_idx = np.argmin(distances)
        balanced_solution = pareto_solutions[balanced_idx]
        
        print(f"\nBalanced solution (closest to ideal point):")
        print(f"  F1: {balanced_solution.objectives[0]:.2f}")
        print(f"  F2: {balanced_solution.objectives[1]:.2f}")
        print(f"  F3: {balanced_solution.objectives[2]:.4f}")
        print("="*60)
    
    return epsilon_solver

        