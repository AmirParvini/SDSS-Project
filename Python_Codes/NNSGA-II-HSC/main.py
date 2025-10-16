import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.widgets import Button
import datetime
from mpl_toolkits.mplot3d import Axes3D
from nsga2 import NSGA2_Humanitarian
from graph_download import GraphDownload
from pgsql_connector import PgsqlConnector
import requests
from collections import defaultdict
import math
from geopy.distance import geodesic
import copy
import itertools
from convergence_metrics import ConvergenceMetrics

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
        rta = 17.5  # Area of ​​relief tent (Square meter)
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
        self.injured_treated_same_time = 50 # Maximum number of injured people treated at the same time
        self.wt = 5 # The amount of waiting time in the treatment queue for the maximum number of injured people who can be treated simultaneously.
        self.affected_pop = {5: 30000, 6: 52500, 4: 4500, 7: 18000,
                        3: 4854}  # the affected population (scenario 1)
        self.severe_injured = {key: value * t1 for key, value in self.affected_pop.items()}
        self.minor_injured = {key: value * t2 for key, value in self.affected_pop.items()}
        self.homeless = {
            key: self.affected_pop[key] - (self.severe_injured[key] + self.minor_injured[key])
            for key in self.affected_pop.keys()
        }
        self.demand = {key: value/5 for key, value in self.affected_pop.items()}
        ec_area = {8: 283762, 9: 133407, 10: 120000, 11: 18700, 12: 14000,
                13: 24000, 14: 20170, 15: 17400, 16: 25380, 17: 25374, 18: 58055}
        self.reliefpackage_volume = 0.6273  # Cubic meter
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
            'ambulance': {'injured_type1': 2, 'injured_type2': 4},  # person
            'helicopter': {'injured_type1': 4, 'injured_type2': 12},  # person
            'truck_type1': 6,  # Cubic meter
            'truck_type2': 12,  # Cubic meter
            'hospital': {1: 800, 2: 600, 3: 600, 4: 600},  # person
            'tmc': {1: 300, 2: 300, 3: 300, 4: 300, 5: 300, 6: 300, 7: 300, 8: 300, 9: 300, 10: 300},  # person
            'shelter': {key: (value * Pua / rta)*rtc for key, value in ec_area.items()} # person
        }


    def complex_humanitarian_cost(self, chromosome_list):
        # global idc_id, ec_id, da_id, h_id, tmc_id, ambulance_speed, helicopter_speed, \
        #     dist_idc_to_shelter, dist_da_to_h, dist_da_to_ec, dist_da_to_tmc, dist_da_to_h_helicopter, dist_da_to_tmc_helicopter, cost, severe_injured, minor_injured
        """
        More realistic cost function with actual problem parameters
        
        Parameters:
        -----------
        distances: dict with distance matrices
        demands: dict with demand values for shelters and medical facilities
        capacities: dict with capacity constraints
        """
        F1, F2, F3, constriant_violation = [], [], [], []
        for chromosome in chromosome_list:
            idc_to_shelter = chromosome[0]
            flow_values = chromosome[1]
            damage_to_shelter = chromosome[2]
            damage_to_hospital = chromosome[3]
            severe_injured_to_hospital_by_ambulance = chromosome[4]
            damage_to_hospital_TMC = chromosome[5]
            minor_injured_to_hospital_TMC_by_ambulance = chromosome[6]

            # calculating TransportingReliefPackageCost, UnmetDemand
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
            
            # part 3
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
            
            for idx, i in enumerate(idc_to_shelter): # part 1, 2
                if i > 0:
                    sum_ecs_cost += self.cost['ec_cost']
                    sum_dist_idcs_to_ecs += self.distance['dist_idc_to_shelter'][f"{i},{self.ec_id[idx]}"]
                    transporting_reliefpackage_cost += sum_dist_idcs_to_ecs * \
                        self.cost['reliefpackage_transportation_cost'] * \
                        flow_values[idx] * demand[self.ec_id[idx]]
                    unmet_demand[self.ec_id[idx]] = demand[self.ec_id[idx]] * \
                        (1-flow_values[idx])

            for idx, i in enumerate(damage_to_hospital): #part 4, 5
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
            
            for idx, i in enumerate(damage_to_hospital_TMC): # part 6, 7
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
            # F1.append(transporting_reliefpackage_cost + transporting_injured_cost + sum_ecs_cost + sum_tmcs_cost)
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
            # print(
            #     'transporting_reliefpackage_cost', transporting_reliefpackage_cost, '\n'
            #     'transporting_injured_cost', transporting_injured_cost, '\n'
            #     'sum_ecs_cost', sum_ecs_cost, '\n'
            #     'sum_tmcs_cost', sum_tmcs_cost, '\n'
            #     'ec_capacity_shortage', ec_capacity_shortage, '\n'
            #     'hospital_capacity_shortage', hospital_capacity_shortage, '\n'
            #     'tmc_capacity_shortage', tmc_capacity_shortage, '\n'
            #     'unmet_demand', unmet_demand, '\n'
            #     'death_prob', death_prob
            # )
        F1, F2, F3 = np.array(F1), np.array(F2), np.array(F3)
        F1_normalized = [(f1 - min(F1)) / (max(F1) - min(F1)) for f1 in F1]
        F2_normalized = [(f2 - min(F2)) / (max(F2) - min(F2)) for f2 in F2]
        F3_normalized = [(f3 - min(F3)) / (max(F3) - min(F3)) for f3 in F3]
        result_list_of_lists = [list(t) for t in zip(F1, F2, F3)]
        print(sum([1 for c in constriant_violation if c > 0]))
        # print('F1:', F2)
        return result_list_of_lists, constriant_violation


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
            # if self.affected_pop[da_id] > totalcapacity:
            #     ec_capacity_shortage.append(self.affected_pop[da_id]-totalcapacity)
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
    


    # Parameters
    # gd = GraphDownload()
    # G = gd.download_tehran_district6_graph()
    # dbname="sdss"
    # user="postgres"
    # password="AP@pgsql79"
    # host="localhost"
    # port="5432"
    # pgc = PgsqlConnector(dbname, user, password, host, port)
    # idc_to_shelter, da_to_ec, da_to_tmc, da_to_h= pgc.data_fetch()
    # for ite in idc_to_shelter:
    #     short_path = gd._route(G, ite["source_coord"], ite["target_coord"])
    #     distance_meters = geodesic(ite['source_coord'], ite['target_coord']).m
    #     pgc.data_send(ite["source_id"], ite["target_id"], "idc_ec_path", short_path['geometry']['coordinates'], short_path['properties']['weight'], distance_helicopter=distance_meters)
        # dist_idc_to_shelter.append(short_path['properties']['weight'])
    # for dte in da_to_ec:
    #     distance_meters = geodesic(dte['source_coord'], dte['target_coord']).m
    #     pgc.data_send(dte["source_id"], dte["target_id"], "da_ec_dist", distance=distance_meters)
    # for dth in da_to_h:
    #     short_path = gd._route(G, dth["source_coord"], dth["target_coord"])
    #     distance_meters = geodesic(dth['source_coord'], dth['target_coord']).m
    #     pgc.data_send(dth["source_id"], dth["target_id"], "da_h_path", short_path['geometry']['coordinates'], short_path['properties']['weight'], distance_helicopter=distance_meters)
    #     # dist_da_to_h.append(short_path['properties']['weight'])
    # for dtt in da_to_tmc:
    #     short_path = gd._route(G, dtt["source_coord"], dtt["target_coord"])
    #     distance_meters = geodesic(dtt['source_coord'], dtt['target_coord']).m
    #     pgc.data_send(dtt["source_id"], dtt["target_id"], "da_tmc_path", short_path['geometry']['coordinates'], short_path['properties']['weight'], distance_helicopter=distance_meters)
        # dist_da_to_tmc.append(short_path['properties']['weight'])
        

    def main(self):
        # Problem Definition
        problem = {
            'cost_function': self.complex_humanitarian_cost  # Use simple or complex function
        }

        # Initialize Algorithm with problem dimensions
        alg = NSGA2_Humanitarian(
            max_iter=40,
            pop_size=150,
            p_crossover=0.9,
            p_mutation=0.1,
            verbose=True,
            shelter_id = self.ec_id,
            distribution_center_id = self.idc_id,
            damage_points_id = self.da_id,
            hospital_id = self.h_id,
            temporary_medical_id = self.tmc_id
        )

        # Solve the Problem
        results = alg.run(problem)
        pop = results['pop']
        F = results['F']
        pareto_pop = results['pareto_pop']
        metrics = results['metrics'] 
        # diagnostics = results['diagnostics']
        
        # رسم نمودارهای همگرایی
        metrics.plot_convergence(save_path='convergence_metrics.png')
        metrics.print_summary()
        # diagnostics.diagnose_problems()
        # diagnostics.plot_diagnostic_metrics(save_path='diagnostic_metrics.png')
        
        # Plot Results
        fig = plt.figure(figsize=(20, 5))

        # Extract Pareto Front costs
        pf_costs = np.array([ind['cost'] for ind in pareto_pop])

        # Plot 1: F1 vs F2
        ax1 = fig.add_subplot(131)
        sc1 = ax1.scatter(pf_costs[:, 0], pf_costs[:, 1], c='blue', s=50, alpha=0.6, edgecolors='black')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('F1: Total da_ec_distance', fontsize=12)
        ax1.set_ylabel('F2: Unmet Demand', fontsize=12)
        ax1.set_title('Pareto Front: F1 vs F2', fontsize=14)

        if len(pf_costs) > 0:
            best_cost_idx = np.argmin(pf_costs[:, 0])
            ax1.scatter(pf_costs[best_cost_idx, 0], pf_costs[best_cost_idx, 1],
                        c='red', s=100, marker='*', label='Min F1')
            best_demand_idx = np.argmin(pf_costs[:, 1])
            ax1.scatter(pf_costs[best_demand_idx, 0], pf_costs[best_demand_idx, 1],
                        c='green', s=100, marker='*', label='Min F2')
            ax1.legend()

        # Plot 2: F1 vs F3
        ax2 = fig.add_subplot(132)
        sc2 = ax2.scatter(pf_costs[:, 0], pf_costs[:, 2], c='blue', s=50, alpha=0.6, edgecolors='black')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('F1: Total da_ec_distance', fontsize=12)
        ax2.set_ylabel('F3: Death Probability', fontsize=12)
        ax2.set_title('Pareto Front: F1 vs F3', fontsize=14)

        if len(pf_costs) > 0:
            best_cost_idx = np.argmin(pf_costs[:, 0])
            ax2.scatter(pf_costs[best_cost_idx, 0], pf_costs[best_cost_idx, 2],
                        c='red', s=100, marker='*', label='Min F1')
            best_death_idx = np.argmin(pf_costs[:, 2])
            ax2.scatter(pf_costs[best_death_idx, 0], pf_costs[best_death_idx, 2],
                        c='orange', s=100, marker='*', label='Min F3')
            ax2.legend()

        # Plot 3: F2 vs F3
        ax3 = fig.add_subplot(133)
        sc3 = ax3.scatter(pf_costs[:, 1], pf_costs[:, 2], c='blue', s=50, alpha=0.6, edgecolors='black')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('F2: Unmet Demand', fontsize=12)
        ax3.set_ylabel('F3: Death Probability', fontsize=12)
        ax3.set_title('Pareto Front: F2 vs F3', fontsize=14)

        if len(pf_costs) > 0:
            best_demand_idx = np.argmin(pf_costs[:, 1])
            ax3.scatter(pf_costs[best_demand_idx, 1], pf_costs[best_demand_idx, 2],
                        c='green', s=100, marker='*', label='Min F2')
            best_death_idx = np.argmin(pf_costs[:, 2])
            ax3.scatter(pf_costs[best_death_idx, 1], pf_costs[best_death_idx, 2],
                        c='orange', s=100, marker='*', label='Min F3')
            ax3.legend()

        plt.tight_layout()
        plt.show()

        # 3D Animated Pareto Front across generations
        pareto_history = results.get('pareto_history', [])
        costs_history = [np.array([ind['cost'] for ind in gen]) for gen in pareto_history]
        # Compute stable axis limits across all generations
        if len(costs_history) > 0:
            all_costs = np.vstack([c for c in costs_history if c.size > 0]) if any(c.size > 0 for c in costs_history) else None
        else:
            all_costs = None
        fig_anim = plt.figure(figsize=(12, 9))
        ax_anim = fig_anim.add_subplot(111, projection='3d')
        scat = ax_anim.scatter([], [], [], c='blue', s=50, alpha=0.6, edgecolors='black', label='Pareto Solutions')
        ax_anim.set_xlabel('F1: Total da_ec_distance', fontsize=12)
        ax_anim.set_ylabel('F2: Unmet Demand', fontsize=12)
        ax_anim.set_zlabel('F3: Death Probability', fontsize=12)
        ax_anim.set_title('3D Pareto Front (Animated by Generation)', fontsize=14)
        ax_anim.grid(True, alpha=0.3)
        if all_costs is not None and all_costs.size > 0:
            x_min, x_max = np.min(all_costs[:, 0]), np.max(all_costs[:, 0])
            y_min, y_max = np.min(all_costs[:, 1]), np.max(all_costs[:, 1])
            z_min, z_max = np.min(all_costs[:, 2]), np.max(all_costs[:, 2])
            ax_anim.set_xlim(x_min, x_max)
            ax_anim.set_ylim(y_min, y_max)
            ax_anim.set_zlim(z_min, z_max)
        ax_anim.legend()

        # Animation control state
        is_paused = {'value': False}

        def init():
            scat._offsets3d = ([], [], [])
            return (scat,)

        def update(frame_idx):
            if frame_idx < len(costs_history) and costs_history[frame_idx].size > 0:
                frame_costs = costs_history[frame_idx]
                xs, ys, zs = frame_costs[:, 0], frame_costs[:, 1], frame_costs[:, 2]
                scat._offsets3d = (xs, ys, zs)
                ax_anim.set_title(f'3D Pareto Front (Generation {frame_idx + 1}/{len(costs_history)})')
            else:
                scat._offsets3d = ([], [], [])
            return (scat,)

        if len(costs_history) > 0:
            anim = FuncAnimation(fig_anim, update, init_func=init, frames=len(costs_history), interval=400, blit=False, repeat=True)

            # Play/Pause button
            btn_ax = fig_anim.add_axes([0.8, 0.02, 0.1, 0.05])
            btn_playpause = Button(btn_ax, 'Play/Pause')

            def on_playpause_clicked(event):
                if is_paused['value']:
                    anim.event_source.start()
                    is_paused['value'] = False
                else:
                    anim.event_source.stop()
                    is_paused['value'] = True

            btn_playpause.on_clicked(on_playpause_clicked)

            # Save button (tries GIF then MP4)
            btn_save_ax = fig_anim.add_axes([0.67, 0.02, 0.1, 0.05])
            btn_save = Button(btn_save_ax, 'Save')

            def on_save_clicked(event):
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                # Try GIF with Pillow
                try:
                    gif_path = f'pareto_animation_{timestamp}.gif'
                    writer = PillowWriter(fps=max(1, int(1000/anim.event_source.interval)))
                    anim.save(gif_path, writer=writer)
                    print(f'Saved GIF: {gif_path}')
                    return
                except Exception as e:
                    print(f'GIF save failed: {e}')
                # Try MP4 with ffmpeg
                try:
                    mp4_path = f'pareto_animation_{timestamp}.mp4'
                    writer = FFMpegWriter(fps=max(1, int(1000/anim.event_source.interval)))
                    anim.save(mp4_path, writer=writer)
                    print(f'Saved MP4: {mp4_path}')
                except Exception as e:
                    print(f'MP4 save failed: {e}')

            btn_save.on_clicked(on_save_clicked)

            plt.tight_layout()
            plt.show()

        # Display some statistics
        print("\n" + "="*50)
        print("Optimization Results Summary")
        print("="*50)
        print(f"Number of Pareto optimal solutions: {len(pareto_pop)}")
        print(f"Best F1 (Total da_ec_distance): {np.min(pf_costs[:, 0])}")
        print(f"Best F2 (Unmet Demand): {np.min(pf_costs[:, 1])}")
        print(f"Best F3 (Death Probability): {np.min(pf_costs[:, 2])}")
        # Extract Pareto Front costs
        # pf_costs = np.array([ind['cost'] for ind in pareto_pop])

        # Plot Results
        # plt.figure(figsize=(10, 6))
        # plt.scatter(pf_costs[:, 0], pf_costs[:, 1], c='blue', s=50, alpha=0.6, edgecolors='black')
        # plt.grid(True, alpha=0.3)
        # plt.xlabel('Total Cost (f1)', fontsize=12)
        # plt.ylabel('Unmet Demand / Response Time (f2)', fontsize=12)
        # plt.title('Pareto Front - Humanitarian Logistics Optimization', fontsize=14)

        # # Annotate some solutions
        # if len(pf_costs) > 0:
        #     # Mark best cost solution
        #     best_cost_idx = np.argmin(pf_costs[:, 0])
        #     plt.scatter(pf_costs[best_cost_idx, 0], pf_costs[best_cost_idx, 1],
        #                 c='red', s=100, marker='*', label='Min Cost')

        #     # Mark best service solution
        #     best_service_idx = np.argmin(pf_costs[:, 1])
        #     plt.scatter(pf_costs[best_service_idx, 0], pf_costs[best_service_idx, 1],
        #                 c='green', s=100, marker='*', label='Min Unmet Demand')

        #     plt.legend()

        # plt.tight_layout()
        # plt.show()

        # # Display some statistics
        # print("\n" + "="*50)
        # print("Optimization Results Summary")
        # print("="*50)
        # print(f"Number of Pareto optimal solutions: {len(pareto_pop)}")
        # print(f"Best cost: {np.min(pf_costs[:, 0]):.2f}")
        # print(f"Best service (min unmet demand): {np.min(pf_costs[:, 1]):.2f}")

        # # Display a sample solution
        # if len(pareto_pop) > 0:
        #     print("\n" + "="*50)
        #     print("Sample Pareto Optimal Solution (Balanced)")
        #     print("="*50)

            # Find a balanced solution (closest to the middle of Pareto front)
            # normalized_costs = (pf_costs - pf_costs.min(axis=0)) / (pf_costs.max(axis=0) - pf_costs.min(axis=0) + 1e-10)
            # distances_to_ideal = np.linalg.norm(normalized_costs, axis=1)
            # balanced_idx = np.argmin(distances_to_ideal)

            # balanced_solution = pareto_pop[balanced_idx]
            # chromosome = balanced_solution['chromosome']

            # print(f"Objectives: f1={balanced_solution['cost'][0]:.2f}, f2={balanced_solution['cost'][1]:.2f}")
            # print(f"\nActive Shelters: {sum([1 if x > 0 else 0 for x in chromosome[0]])}/{len(chromosome[0])}")
            # print(f"Active Temporary Medical Centers: {sum(chromosome[4])}/{len(chromosome[4])}")
            # print(f"Total Flow: {sum(chromosome[1]):.2f}")

            # # Show distribution assignments
            # print("\nShelter Assignments:")
            # for i, (dc, flow) in enumerate(zip(chromosome[0], chromosome[1])):
            #     if dc > 0:
            #         print(f"  Shelter {i+1}: Supplied by DC{dc}, Flow={flow:.2f}")

            # # Show medical allocations
            # print("\nMedical Allocations:")
            # print(f"Severe Injured to Hospitals: {sum(chromosome[5])}/{sum(severe_injured)}")
            # print(f"Moderate Injured to Hospitals: {sum(chromosome[6])}/{sum(moderate_injured)}")
            # print(f"Moderate Injured to Temp Medical: {sum(chromosome[7])}/{sum(moderate_injured)}")

            # # Show detailed medical allocation
            # print("\nDetailed Medical Allocation by Damage Point:")
            # for i in range(alg.n_damage_points):
            #     print(f"  Damage Point {i+1}:")
            #     print(f"    Severe: {chromosome[5][i]}/{severe_injured[i]} to Hospital {chromosome[3][i]}")
            #     print(f"    Moderate: {chromosome[6][i]} to Hospital, "
            #           f"{moderate_injured[i] - chromosome[6][i]} to Temp Medical")

            # # Show active temp medical centers
            # print("\nActive Temporary Medical Centers:")
            # for i, (active, allocated) in enumerate(zip(chromosome[4], chromosome[7])):
            #     if active:
            #         print(f"  Temp Medical {i+1}: {allocated} patients")
m = Main()
m.main()