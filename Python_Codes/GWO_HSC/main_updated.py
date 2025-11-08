import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.widgets import Button
import datetime
from mpl_toolkits.mplot3d import Axes3D
from nsga2 import NSGA2_Humanitarian
from gwo_humanitarian import GWO_Humanitarian
# from graph_download import GraphDownload
# from pgsql_connector import PgsqlConnector
import requests
from collections import defaultdict
import math
# from geopy.distance import geodesic
import copy
import itertools
from convergence_metrics import ConvergenceMetrics
from tradeoff_analysis import TradeoffAnalysis
import seaborn as sns
import json


class Main():
    def __init__(self):
        response = requests.get("http://localhost:8000/api/v1/getparam")
        response = response.json()
        self.idc_id = [1, 2, 3]
        self.ec_id = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        self.da_id = [3, 4, 5, 6, 7]
        self.h_id = [1, 2, 3, 4]
        self.tmc_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        dist_idc_to_shelter = {}
        dist_da_to_h = {}
        dist_da_to_ec = {}
        dist_da_to_h_helicopter = {}
        dist_da_to_tmc = {}
        dist_da_to_tmc_helicopter = {}
        for i in response['pathes']['idc_ec_path']:
            dist_idc_to_shelter[f"{i['idc_id']},{i['ec_id']}"] = float(i['distance'])//1000
            
        for i in response['pathes']['da_h_path']:
            dist_da_to_h[f"{i['da_id']},{i['h_id']}"] = float(i['distance'])//1000
            dist_da_to_h_helicopter[f"{i['da_id']},{i['h_id']}"] = float(i['distance_helicopter'])//1000
            
        for i in response['pathes']['da_ec_dist']:
            dist_da_to_ec[f"{i['da_id']},{i['ec_id']}"] = float(i['distance'])
            
        for i in response['pathes']['da_tmc_path']:
            dist_da_to_tmc[f"{i['da_id']},{i['tmc_id']}"] = float(i['distance'])//1000
            dist_da_to_tmc_helicopter[f"{i['da_id']},{i['tmc_id']}"] = float(i['distance_helicopter'])//1000
        self.B = 1500000  # Budget
        t1 = 0.015  # Percentage distribution of severe injuries
        t2 = 0.067  # Percentage distribution of mild injuries
        Pua = 0.7  # Percentage of shelter area used
        rta = 17.5  # Area of relief tent (Square meter)
        rtc = 5  # Relief tent capacity (person)
        self.ambulance_speed = 20  # km/h
        self.helicopter_speed = 40  # km/h
        self.phi_min_s = 0
        self.phi_max_s = 0.9
        self.ks_s = 0.1
        self.tm_s = 10
        self.phi_min_m = 0
        self.phi_max_m = 0.9
        self.ks_m = 0.1
        self.tm_m = 20
        self.injured_treated_same_time = 50
        self.wt = 5
        self.affected_pop = {5: 30000, 6: 52500, 4: 4500, 7: 18000, 3: 4854}
        self.severe_injured = {key: value * t1 for key, value in self.affected_pop.items()}
        self.minor_injured = {key: value * t2 for key, value in self.affected_pop.items()}
        self.homeless = {
            key: self.affected_pop[key] - (self.severe_injured[key] + self.minor_injured[key])
            for key in self.affected_pop.keys()
        }
        self.demand = {key: value/5 for key, value in self.affected_pop.items()}
        ec_area = {8: 283762, 9: 133407, 10: 120000, 11: 18700, 12: 14000,
                13: 24000, 14: 20170, 15: 17400, 16: 25380, 17: 25374, 18: 58055}
        self.reliefpackage_volume = 0.6273
        self.cost = {
            'reliefpackage_cost': 108.76,
            'reliefpackage_transportation_cost': 30,
            'tmc_cost': 50000,
            'ec_cost': 50000,
            'ambulance': 50,
            'helicopter': 100
        }
        self.distance = {
            'dist_idc_to_shelter': dist_idc_to_shelter,
            'dist_da_to_ec': dist_da_to_ec,
            'dist_da_to_h': dist_da_to_h,
            'dist_da_to_h_helicopter': dist_da_to_h_helicopter,
            'dist_da_to_tmc': dist_da_to_tmc,
            'dist_da_to_tmc_helicopter': dist_da_to_tmc_helicopter
        }
        self.capacity = {
            'ambulance': {'injured_type1': 2, 'injured_type2': 4},
            'helicopter': {'injured_type1': 4, 'injured_type2': 12},
            'truck_type1': 6,
            'truck_type2': 12,
            'hospital': {1: 800, 2: 600, 3: 600, 4: 600},
            'tmc': {1: 300, 2: 300, 3: 300, 4: 300, 5: 300, 6: 300, 7: 300, 8: 300, 9: 300, 10: 300},
            'shelter': {key: (value * Pua / rta)*rtc for key, value in ec_area.items()}
        }

    def complex_humanitarian_cost(self, chromosome_list):
        F1, F2, F3, constriant_violation = [], [], [], []
        for chromosome in chromosome_list:
            idc_to_shelter = chromosome[0]
            flow_values = chromosome[1]
            damage_to_shelter = chromosome[2]
            damage_to_hospital = chromosome[3]
            severe_injured_to_hospital_by_ambulance = chromosome[4]
            damage_to_hospital_TMC = chromosome[5]
            minor_injured_to_hospital_TMC_by_ambulance = chromosome[6]

            sum_dist_idcs_to_ecs = 0
            sum_dist_da_to_ec = 0
            sum_ecs_cost = 0
            sum_tmcs_cost = 0
            da_ec_dist = 0
            transporting_reliefpackage_cost = 0
            unmet_demand = {}
            ec_capacity_shortage = 0
            hospital_capacity_shortage_for_severe_injured = 0
            hospital_capacity_shortage_for_minor_injured = 0
            tmc_capacity_shortage = 0
            death_prob = 0
            transporting_injured_cost = 0
            hospital_cap = copy.deepcopy(self.capacity['hospital'])
            tmc_cap = copy.deepcopy(self.capacity['tmc'])
            da_ec_alloc = defaultdict(list)
            
            shelters = [self.ec_id[idx] for idx, x in enumerate(idc_to_shelter) if x != 0]
            damaged_areas = [x for x in damage_to_shelter if x != 0]
            if len(shelters) == len(damaged_areas):
                for idx, b in enumerate(damaged_areas):
                    da_ec_alloc[b].append(shelters[idx])
                    sum_dist_da_to_ec += self.distance['dist_da_to_ec'][f"{b},{shelters[idx]}"]
            elif len(shelters) > len(damaged_areas):
                cycle = itertools.cycle(damaged_areas)
                damaged_areas = list(itertools.islice(cycle, len(shelters)))
                for idx, b in enumerate(damaged_areas):
                    da_ec_alloc[b].append(shelters[idx])
                    sum_dist_da_to_ec += self.distance['dist_da_to_ec'][f"{b},{shelters[idx]}"]
            elif len(shelters) < len(damaged_areas):
                cycle = itertools.cycle(shelters)
                shelters = list(itertools.islice(cycle, len(damaged_areas)))
                for idx, b in enumerate(damaged_areas):
                    da_ec_alloc[b].append(shelters[idx])
                    sum_dist_da_to_ec += self.distance['dist_da_to_ec'][f"{b},{shelters[idx]}"]
            demand, ec_capacity_shortage, da_ec_dist = self.allocate_population_under_capacity(da_ec_alloc)
            
            for idx, i in enumerate(idc_to_shelter):
                if i > 0:
                    sum_ecs_cost += self.cost['ec_cost']
                    sum_dist_idcs_to_ecs += self.distance['dist_idc_to_shelter'][f"{i},{self.ec_id[idx]}"]
                    transporting_reliefpackage_cost += sum_dist_idcs_to_ecs * \
                        self.cost['reliefpackage_transportation_cost'] * \
                        flow_values[idx] * demand[self.ec_id[idx]]
                    unmet_demand[self.ec_id[idx]] = demand[self.ec_id[idx]] * (1-flow_values[idx])

            for idx, i in enumerate(damage_to_hospital):
                i = np.array(i)/sum(i)
                for h_idx, j in enumerate(i):
                    if j > 0:
                        if self.severe_injured[self.da_id[idx]] * j > hospital_cap[self.h_id[h_idx]]:
                            hospital_capacity_shortage_for_severe_injured += round(self.severe_injured[self.da_id[idx]] * j) - hospital_cap[self.h_id[h_idx]]
                            hospital_cap[self.h_id[h_idx]] = 0
                        else:
                            hospital_cap[self.h_id[h_idx]] = hospital_cap[self.h_id[h_idx]] - round(self.severe_injured[self.da_id[idx]] * j)
                        transporting_injured_cost += (math.ceil(self.severe_injured[self.da_id[idx]] * j * severe_injured_to_hospital_by_ambulance[idx][h_idx]) / self.capacity['ambulance']['injured_type1']) * self.cost['ambulance'] + (
                            math.ceil(self.severe_injured[self.da_id[idx]] * j * (1 - severe_injured_to_hospital_by_ambulance[idx][h_idx])) / self.capacity['helicopter']['injured_type1']) * self.cost['helicopter']
                        t_ambulance = (self.distance['dist_da_to_h'][f"{self.da_id[idx]},{self.h_id[h_idx]}"] / self.ambulance_speed)*60
                        t_helicopter = (self.distance['dist_da_to_h_helicopter'][f"{self.da_id[idx]},{self.h_id[h_idx]}"] / self.ambulance_speed)*60
                        death_prob += (self.death_probability(self.phi_min_s, self.phi_max_s, self.ks_s, t_ambulance, self.tm_s) * round(self.severe_injured[self.da_id[idx]] * j)) + (
                        self.death_probability(self.phi_min_s, self.phi_max_s, self.ks_s, t_helicopter, self.tm_s) * round(self.severe_injured[self.da_id[idx]] * (1 - j))
                        )
            
            for idx, i in enumerate(damage_to_hospital_TMC):
                i = np.array(i)/sum(i)
                for h_idx, j in enumerate(i[:len(self.h_id)]):
                    if j > 0:
                        sum_tmcs_cost += self.cost['tmc_cost']
                        if self.minor_injured[self.da_id[idx]] * j > hospital_cap[self.h_id[h_idx]]:
                            hospital_capacity_shortage_for_minor_injured += round(self.minor_injured[self.da_id[idx]] * j) - hospital_cap[self.h_id[h_idx]]
                            hospital_cap[self.h_id[h_idx]] = 0
                        else:
                            hospital_cap[self.h_id[h_idx]] = hospital_cap[self.h_id[h_idx]] - round(self.minor_injured[self.da_id[idx]] * j)
                        transporting_injured_cost += (math.ceil(self.minor_injured[self.da_id[idx]] * j * minor_injured_to_hospital_TMC_by_ambulance[idx][h_idx]) / self.capacity['ambulance']['injured_type2']) * self.cost['ambulance'] +\
                            (math.ceil(self.minor_injured[self.da_id[idx]] * j * (1 - minor_injured_to_hospital_TMC_by_ambulance[idx][h_idx])) / self.capacity['helicopter']['injured_type2']) * self.cost['helicopter']
                        t_ambulance = (self.distance['dist_da_to_h'][f"{self.da_id[idx]},{self.h_id[h_idx]}"] / self.ambulance_speed)*60
                        t_helicopter = (self.distance['dist_da_to_h_helicopter'][f"{self.da_id[idx]},{self.h_id[h_idx]}"] / self.ambulance_speed)*60
                        death_prob += (self.death_probability(self.phi_min_s, self.phi_max_s, self.ks_s, t_ambulance, self.tm_s) * round(self.minor_injured[self.da_id[idx]] * j)) + (
                        self.death_probability(self.phi_min_s, self.phi_max_s, self.ks_s, t_helicopter, self.tm_s) * round(self.minor_injured[self.da_id[idx]] * (1 - j))
                        )  
                for h_tmc_idx, j in enumerate(i[len(self.h_id)+1:]):
                    if j > 0:
                        if self.minor_injured[self.da_id[idx]] * j > tmc_cap[self.tmc_id[h_tmc_idx]]:
                            tmc_capacity_shortage += round(self.minor_injured[self.da_id[idx]] * j) - tmc_cap[self.tmc_id[h_tmc_idx]]
                            tmc_cap[self.tmc_id[h_tmc_idx]] = 0
                        else:
                            tmc_cap[self.tmc_id[h_tmc_idx]] = tmc_cap[self.tmc_id[h_idx]] - round(self.minor_injured[self.da_id[idx]] * j)
                        transporting_injured_cost += (
                            (math.ceil(self.minor_injured[self.da_id[idx]] * j * minor_injured_to_hospital_TMC_by_ambulance[idx][h_tmc_idx]) / self.capacity['ambulance']['injured_type1'])) * self.cost['ambulance'] + (
                                math.ceil(self.minor_injured[self.da_id[idx]] * j * (1 - minor_injured_to_hospital_TMC_by_ambulance[idx][h_tmc_idx])) / self.capacity['helicopter']['injured_type1']) * self.cost['helicopter']
                        t_ambulance = (self.distance['dist_da_to_tmc'][f"{self.da_id[idx]},{self.tmc_id[h_tmc_idx]}"] / self.ambulance_speed)*60
                        t_helicopter = (self.distance['dist_da_to_tmc_helicopter'][f"{self.da_id[idx]},{self.tmc_id[h_tmc_idx]}"] / self.ambulance_speed)*60
                        death_prob += (self.death_probability(self.phi_min_m, self.phi_max_m, self.ks_m, t_ambulance, self.tm_m) * round(self.minor_injured[self.da_id[idx]] * j)) + (
                        self.death_probability(self.phi_min_m, self.phi_max_m, self.ks_m, t_helicopter, self.tm_m) * round(self.minor_injured[self.da_id[idx]] * (1 - j))
                        )
            
            sum_costs = transporting_reliefpackage_cost + transporting_injured_cost + sum_ecs_cost + sum_tmcs_cost
            if sum_costs > self.B:
                constriant_violation.append((sum_costs - self.B)/self.B)
            else:
                constriant_violation.append(0)
            hospital_capacity_shortage_death_prob_sever_injured = 0
            if hospital_capacity_shortage_for_severe_injured > 0:
                t = self.wt
                for i in range(hospital_capacity_shortage_for_severe_injured):
                    if i%self.injured_treated_same_time == 0:
                        t += self.wt
                    hospital_capacity_shortage_death_prob_sever_injured += self.death_probability(self.phi_min_s, self.phi_max_s, self.ks_s, t, self.tm_s)
            hospital_capacity_shortage_death_prob_minor_injured = 0
            if hospital_capacity_shortage_for_minor_injured > 0:
                t = self.wt
                for i in range(hospital_capacity_shortage_for_minor_injured):
                    if i%self.injured_treated_same_time == 0:
                        t += self.wt
                    hospital_capacity_shortage_death_prob_minor_injured += self.death_probability(self.phi_min_m, self.phi_max_m, self.ks_m, t, self.tm_m)
            tmc_capacity_shortage_death_prob = 0
            if tmc_capacity_shortage > 0:
                t = self.wt
                for i in range(tmc_capacity_shortage):
                    if i%self.injured_treated_same_time == 0:
                        t += self.wt
                    tmc_capacity_shortage_death_prob += self.death_probability(self.phi_min_m, self.phi_max_m, self.ks_m, t, self.tm_m)
            
            F1.append(da_ec_dist)
            F2.append(sum(unmet_demand.values()) + ec_capacity_shortage/5)
            F3.append(death_prob + hospital_capacity_shortage_death_prob_sever_injured + hospital_capacity_shortage_death_prob_minor_injured + tmc_capacity_shortage_death_prob)
            
        F1, F2, F3 = np.array(F1), np.array(F2), np.array(F3)
        F1_normalized = [(f1 - min(F1)) / (max(F1) - min(F1)) for f1 in F1]
        F2_normalized = [(f2 - min(F2)) / (max(F2) - min(F2)) for f2 in F2]
        F3_normalized = [(f3 - min(F3)) / (max(F3) - min(F3)) for f3 in F3]
        result_list_of_lists = [list(t) for t in zip(F1, F2, F3)]
        normalize_result_list_of_lists = [list(t) for t in zip(F1_normalized, F2_normalized, F3_normalized)]
        return result_list_of_lists, constriant_violation, normalize_result_list_of_lists

    def allocate_population_under_capacity(self, da_ec: dict):
        demand = {}
        pop_to_ec = []
        ec_capacity_shortage = []
        da_ec_dist = []
        for da_id, ec_id_list in da_ec.items():
            ec_id_list = list(dict.fromkeys(ec_id_list))
            ec_cap = [self.capacity['shelter'][v] for v in ec_id_list]
            da_ec_dist.append(sum([self.distance['dist_da_to_ec'][f"{da_id},{v}"] for v in ec_id_list]))
            totalcapacity = sum(ec_cap)
            alloc_ratio = [i/totalcapacity for i in ec_cap]
            for idx, ar in enumerate(alloc_ratio):
                if round(ar * self.affected_pop[da_id]) > ec_cap[idx]:
                    pop_to_ec.append(ec_cap[idx])
                    ec_capacity_shortage.append(round(ar * self.affected_pop[da_id]) - ec_cap[idx])
                else:
                    pop_to_ec.append(round(ar * self.affected_pop[da_id]))
                    ec_capacity_shortage.append(0)
            for (key, value) in enumerate(list(zip(ec_id_list, pop_to_ec))):
                demand[value[0]] = math.ceil(value[-1]/5)
        return demand, sum(ec_capacity_shortage), sum(da_ec_dist)

    def death_probability(self, phi_min, phi_max, ks, t, tm):
        return phi_min + (phi_max - phi_min)/(1 + math.exp(ks * (tm - t)))
    
    def visualize_results(self, results_nsga, results_gwo):
        """Visualize and compare results from both algorithms"""
        
        # Extract Pareto fronts
        pf_nsga = np.array([ind['cost'] for ind in results_nsga['pareto_pop']])
        pf_gwo = np.array([ind['cost'] for ind in results_gwo['pareto_pop']])
        
        # Create comparison plots
        fig = plt.figure(figsize=(18, 10))
        
        # 3D comparison
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.scatter(pf_nsga[:, 0], pf_nsga[:, 1], pf_nsga[:, 2], 
                   c='blue', s=50, alpha=0.6, label='NSGA-II')
        ax1.scatter(pf_gwo[:, 0], pf_gwo[:, 1], pf_gwo[:, 2], 
                   c='red', s=50, alpha=0.6, label='GWO')
        ax1.set_xlabel('F1: Distance')
        ax1.set_ylabel('F2: Unmet Demand')
        ax1.set_zlabel('F3: Death Probability')
        ax1.set_title('3D Pareto Front Comparison')
        ax1.legend()
        
        # 2D projections
        pairs = [(0, 1, 'F1 vs F2'), (0, 2, 'F1 vs F3'), (1, 2, 'F2 vs F3')]
        for idx, (i, j, title) in enumerate(pairs, 2):
            ax = fig.add_subplot(2, 3, idx)
            ax.scatter(pf_nsga[:, i], pf_nsga[:, j], c='blue', s=30, alpha=0.6, label='NSGA-II')
            ax.scatter(pf_gwo[:, i], pf_gwo[:, j], c='red', s=30, alpha=0.6, label='GWO')
            ax.set_xlabel(f'F{i+1}')
            ax.set_ylabel(f'F{j+1}')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Convergence comparison
        ax5 = fig.add_subplot(235)
        iterations_nsga = range(1, len(results_nsga['metrics'].normalized_hypervolume) + 1)
        iterations_gwo = range(1, len(results_gwo['metrics'].normalized_hypervolume) + 1)
        ax5.plot(iterations_nsga, results_nsga['metrics'].normalized_hypervolume, 
                'b-', label='NSGA-II', linewidth=2)
        ax5.plot(iterations_gwo, results_gwo['metrics'].normalized_hypervolume, 
                'r-', label='GWO', linewidth=2)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Hypervolume')
        ax5.set_title('Convergence Comparison')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # Performance metrics table
        ax6 = fig.add_subplot(236)
        ax6.axis('off')
        
        metrics_data = [
            ['Metric', 'NSGA-II', 'GWO'],
            ['Pareto Size', len(pf_nsga), len(pf_gwo)],
            ['Best F1', f'{np.min(pf_nsga[:, 0]):.2f}', f'{np.min(pf_gwo[:, 0]):.2f}'],
            ['Best F2', f'{np.min(pf_nsga[:, 1]):.2f}', f'{np.min(pf_gwo[:, 1]):.2f}'],
            ['Best F3', f'{np.min(pf_nsga[:, 2]):.2f}', f'{np.min(pf_gwo[:, 2]):.2f}'],
            ['Final HV', f'{results_nsga["metrics"].normalized_hypervolume[-1]:.4f}',
                        f'{results_gwo["metrics"].normalized_hypervolume[-1]:.4f}']
        ]
        
        table = ax6.table(cellText=metrics_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax6.set_title('Performance Comparison', y=0.8, fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def run_algorithm(self, algorithm='both', max_iter=50, pop_size=100):
        """
        Run optimization algorithms
        
        Parameters:
        -----------
        algorithm: str, 'nsga', 'gwo', or 'both'
        max_iter: int, maximum iterations
        pop_size: int, population/pack size
        """
        
        problem = {'cost_function': self.complex_humanitarian_cost}
        
        results = {}
        
        if algorithm in ['nsga', 'both']:
            print("\n" + "="*60)
            print("Running NSGA-II Optimization")
            print("="*60)
            
            nsga = NSGA2_Humanitarian(
                max_iter=max_iter,
                pop_size=pop_size,
                p_crossover=0.9,
                p_mutation=0.1,
                elitism_rate=0.1,
                verbose=True,
                shelter_id=self.ec_id,
                distribution_center_id=self.idc_id,
                damage_points_id=self.da_id,
                hospital_id=self.h_id,
                temporary_medical_id=self.tmc_id
            )
            
            results['nsga'] = nsga.run(problem)
            
            # Plot NSGA-II metrics
            results['nsga']['metrics'].plot_convergence(save_path='nsga_convergence.png')
            results['nsga']['metrics'].print_summary()
        
        if algorithm in ['gwo', 'both']:
            print("\n" + "="*60)
            print("Running GWO Optimization")
            print("="*60)
            
            gwo = GWO_Humanitarian(
                shelter_id=self.ec_id,
                distribution_center_id=self.idc_id,
                damage_points_id=self.da_id,
                hospital_id=self.h_id,
                temporary_medical_id=self.tmc_id,
                max_iter=max_iter,
                pack_size=pop_size,
                verbose=True
            )
            
            results['gwo'] = gwo.run(problem)
            
            # Plot GWO metrics
            results['gwo']['metrics'].plot_convergence(save_path='gwo_convergence.png')
            results['gwo']['metrics'].print_summary()
        
        # Compare algorithms if both were run
        if algorithm == 'both':
            self.visualize_results(results['nsga'], results['gwo'])
            
            print("\n" + "="*60)
            print("ALGORITHM COMPARISON SUMMARY")
            print("="*60)
            
            pf_nsga = np.array([ind['cost'] for ind in results['nsga']['pareto_pop']])
            pf_gwo = np.array([ind['cost'] for ind in results['gwo']['pareto_pop']])
            
            print(f"\nNSGA-II:")
            print(f"  - Pareto solutions: {len(pf_nsga)}")
            print(f"  - Best F1: {np.min(pf_nsga[:, 0]):.2f}")
            print(f"  - Best F2: {np.min(pf_nsga[:, 1]):.2f}")
            print(f"  - Best F3: {np.min(pf_nsga[:, 2]):.2f}")
            
            print(f"\nGWO:")
            print(f"  - Pareto solutions: {len(pf_gwo)}")
            print(f"  - Best F1: {np.min(pf_gwo[:, 0]):.2f}")
            print(f"  - Best F2: {np.min(pf_gwo[:, 1]):.2f}")
            print(f"  - Best F3: {np.min(pf_gwo[:, 2]):.2f}")
        
        return results


# Usage
if __name__ == "__main__":
    m = Main()
    
    # Run both algorithms and compare
    results = m.run_algorithm(algorithm='both', max_iter=50, pop_size=100)
    
    # Or run single algorithm:
    # results = m.run_algorithm(algorithm='nsga', max_iter=50, pop_size=100)
    # results = m.run_algorithm(algorithm='gwo', max_iter=50, pop_size=100)
