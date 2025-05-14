import geopandas as gpd
import numpy as np
import pprint
import random as rn
from itertools import chain
import math
import multiprocessing
import copy
import random as rn
import json
import sys
import unicodedata
from scipy.stats import truncnorm


sys.stdout.reconfigure(encoding='utf-8')
data = sys.stdin.buffer.read()
data = json.loads(data)
parameters = data['config_parameters']
nodes_data = data['nodes_data']
nodes_dist_data = data['nodes_dist_data']
CMD_nodes_data = nodes_data[0]
LDC_nodes_data = nodes_data[1]
EC_nodes_data = nodes_data[2]
CMD_to_LDC_dist_data = nodes_dist_data[0]
LDC_to_EC_dist_data = nodes_dist_data[1]
commodity_demands_unit = data['commodity_demands_unit']
APR = float(data['APR'])

# Parameters for standard configuration
NT = parameters['NT']  # total number of LDC facilities available
L = parameters['L']  # Lambda: scaling factor for total number of facilities opened
A = parameters['A']  # Alpha: maximum number of neighborhoods that a neighborhood can serve
G = parameters['G']  # Gamma: penalty for unit unmet demand
O = parameters['O']  # Omega: penalty for unit overcapacity
k = ["Water", "Food", "Medical Kit", "Shelter"]  # commodities type
cd = dict(zip(k, [commodity_demands_unit["Water(unit-pp)"], commodity_demands_unit["Food(unit-pp)"], commodity_demands_unit["MedicalKit(unit-pp)"], commodity_demands_unit["Shelter(unit-pp)"]]))  # commodities demand of an injured
V = parameters['V']*sum(cd.values()) # Capacity of LDC


CMD_Coordinate = [] # [[x1,y1], [x2,y2], ...]
LDC_Coordinate = [] # [[x1,y1], [x2,y2], ...]
EC_Coordinate = [] # [[x1,y1], [x2,y2], ...]
Pop_Blocks = [] # [[x1,y1,pop1], [x2,y2,pop2], ...]
No_CMD = 0 # number of Crises Management Databases
No_LDC = 0 # number of Local Distribution Centers
No_EC = 0 # number of Evacuation Centers
CMD_name = []
LDC_name = []
c_LDC = [] # capacities of LDC
EC_name = []
EC_pop = []
EC_demand = [] # demands of EC
EC_water_demand = []
EC_food_demand = []
EC_medicalkit_demand = []
EC_shelter_demand = []
elite_Chromosom = []
elites_fitness = math.inf
bad_chromosom_index = 0
genetic_best_solution = np.nan

def normalize_str(s: list):
    return unicodedata.normalize('NFKC', s.strip())

for CMD_node_data in CMD_nodes_data:
    CMD_Coordinate.append([CMD_node_data['X'], CMD_node_data['Y']])
    CMD_name.append(CMD_node_data['Name'])
    No_CMD += 1
for LDC_node_data in LDC_nodes_data:
    LDC_Coordinate.append([LDC_node_data['X'], LDC_node_data['Y']])
    LDC_name.append(LDC_node_data['Name'])
    c_LDC.append(V)
    No_LDC += 1
for EC_node_data in EC_nodes_data:
    EC_Coordinate.append([EC_node_data['X'], EC_node_data['Y']])
    EC_name.append(EC_node_data['Name'])
    EC_pop.append(EC_node_data["Population"])
    EC_water_demand.append(EC_node_data["Population"]*cd["Water"])
    EC_food_demand.append(EC_node_data["Population"]*cd["Food"])
    EC_medicalkit_demand.append(EC_node_data["Population"]*cd["Medical Kit"])
    EC_shelter_demand.append(EC_node_data["Population"]*cd["Shelter"])
    EC_demand.append(EC_node_data["Population"]*sum(cd.values()))
    No_EC += 1

CMD_name = [normalize_str(s) for s in CMD_name]
LDC_name = [normalize_str(s) for s in LDC_name]
EC_name = [normalize_str(s) for s in EC_name]

cmd_ldc_ODMatrix = np.full((No_CMD, No_LDC), np.nan)
for CLdist in CMD_to_LDC_dist_data:
    if normalize_str(CLdist['CMD_name']) in CMD_name and normalize_str(CLdist['LDC_name']) in LDC_name:
        cmd_ldc_ODMatrix[CMD_name.index(normalize_str(CLdist['CMD_name'])), LDC_name.index(normalize_str(CLdist['LDC_name']))] = CLdist['Distance']
# print(json.dumps(cmd_ldc_ODMatrix.tolist(), ensure_ascii=False))
ldc_ec_ODMatrix = np.full((No_LDC, No_EC), np.nan)
for LEdist in LDC_to_EC_dist_data:
    if normalize_str(LEdist['LDC_name']) in LDC_name and normalize_str(LEdist['EC_name']) in EC_name:
        ldc_ec_ODMatrix[LDC_name.index(normalize_str(LEdist['LDC_name'])), EC_name.index(normalize_str(LEdist['EC_name']))] = LEdist['Distance']
# print(json.dumps(cmd_ldc_ODMatrix.tolist(), ensure_ascii=False))


nan_cols_cmd_ldc_ODMatrix = np.where(np.all(np.isnan(cmd_ldc_ODMatrix), axis=0))[0]
nan_rows_cmd_ldc_ODMatrix = np.where(np.all(np.isnan(cmd_ldc_ODMatrix), axis=1))[0]
nan_cols_ldc_ec_ODMatrix = np.where(np.all(np.isnan(ldc_ec_ODMatrix), axis=0))[0]
nan_rows_ldc_ec_ODMatrix = np.where(np.all(np.isnan(ldc_ec_ODMatrix), axis=1))[0]
cmd_invalid_index = list(nan_rows_cmd_ldc_ODMatrix)
ldc_invalid_index = list(nan_rows_ldc_ec_ODMatrix)
ec_invalid_index = list(nan_cols_ldc_ec_ODMatrix)

# print(json.dumps(f"cmd_invalid_index: {cmd_invalid_index}\nldc_invalid_index: {ldc_invalid_index}\nec_invalid_index: {ec_invalid_index}"))

if len(cmd_invalid_index) > 0:
    cmd_ldc_ODMatrix = np.delete(cmd_ldc_ODMatrix, cmd_invalid_index, axis=0)
    CMD_name = np.delete(CMD_name, cmd_invalid_index)
    CMD_Coordinate = np.delete(CMD_Coordinate, cmd_invalid_index)
    No_CMD = len(CMD_name)
if len(ldc_invalid_index) > 0:
    cmd_ldc_ODMatrix = np.delete(cmd_ldc_ODMatrix, ldc_invalid_index, axis=1)
    ldc_ec_ODMatrix = np.delete(ldc_ec_ODMatrix, ldc_invalid_index, axis=0)
    LDC_name = np.delete(LDC_name, ldc_invalid_index)
    LDC_Coordinate = np.delete(LDC_Coordinate, ldc_invalid_index)
    No_LDC = len(LDC_name)
    c_LDC = [V]*No_LDC
if len(ec_invalid_index) > 0:
    ldc_ec_ODMatrix = np.delete(ldc_ec_ODMatrix, ec_invalid_index, axis=1)
    EC_name = np.delete(EC_name, ec_invalid_index)
    EC_Coordinate = np.delete(EC_Coordinate, ec_invalid_index)
    No_EC = len(EC_name)
    EC_demand = np.delete(EC_demand, ec_invalid_index)
    EC_water_demand = np.delete(EC_water_demand, ec_invalid_index)
    EC_food_demand = np.delete(EC_food_demand, ec_invalid_index)
    EC_medicalkit_demand = np.delete(EC_medicalkit_demand, ec_invalid_index)
    EC_shelter_demand = np.delete(EC_shelter_demand, ec_invalid_index)

mu_pop = np.mean(EC_pop)
std_pop = np.std(EC_pop)
a, b = min(EC_pop), max(EC_pop)
a_std = (a - mu_pop) / std_pop
b_std = (b - mu_pop) / std_pop

# Initial Population    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
def generate_initial_population(pop_size):
    population = []
    for ـ in range(pop_size):
        Chromosom_part1 = []
        Chromosom_part2 = []
        # تولید مقدار تصادفی کالا برای هر LDC
        Pop_random_values = truncnorm.rvs(a_std, b_std, loc=mu_pop*2, scale=std_pop, size=No_LDC)
        Commodities_random_values = list(np.array(Pop_random_values)*sum(cd.values()))
        Commodities_random_values_filter = [V if crv>V else crv for crv in Commodities_random_values]
        for i in range(No_LDC):
            value = rn.randint(0, 1)
            if value == 0 or i in ldc_invalid_index:
                Chromosom_part1.append(0)
            else:
                cmd_rnd = rn.choice(range(No_CMD))
                commodity_amount = Commodities_random_values_filter[i]
                Chromosom_part1.append([cmd_rnd, commodity_amount])
        Chromosom_part2 = rn.sample(range(No_EC), No_EC)
        population.append(Chromosom_part1 + Chromosom_part2)
    return population
# Population = generate_initial_population(10)
# print(json.dumps(Population, ensure_ascii=False))
# Fitness    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
#fitness
used_CMD_index = []
opened_LDC_index = []
unmet_demand = np.NaN
additional_inventory = np.NaN
sum_dist = np.NaN
sum_opened_LDC = np.NaN
CMDs_to_LDCs_allocation = {f"{CMD_name[i]}":[] for i in range(No_CMD)}
LDCs_to_ECs_dist_allocation = {f"{LDC_name[i]}":[] for i in range(No_LDC)}

def fitness(pop):
    global elite_Chromosom, elites_fitness, bad_chromosom_index, unmet_demand, additional_inventory, sum_dist, sum_opened_LDC, CMDs_to_LDCs_allocation, LDCs_to_ECs_dist_allocation
    fitness_list = []
    for chromosom in pop:
        CMDs_to_LDCs_allocation = {f"{CMD_name[i]}":[] for i in range(No_CMD)}
        LDCs_to_ECs_dist_allocation = {f"{LDC_name[i]}":[] for i in range(No_LDC)}
        used_CMD_index = []
        opened_LDC_index = []
        EC_demand_update = EC_demand.copy()
        used_LDC_inventory_list = []
        CMDs_to_LDCs_dist = []
        LDCs_to_ECs_dist = []
        for index, i in enumerate(chromosom[:No_LDC]):
            if type(i) == list:
                opened_LDC_index.append(index)
                used_LDC_inventory_list.append(i[1])
                used_CMD_index.append(i[0])
        if sum(EC_demand_update) - sum(used_LDC_inventory_list) > 0:
            unmet_demand = sum(EC_demand_update) - sum(used_LDC_inventory_list)
            additional_inventory = 0
        elif sum(EC_demand_update) - sum(used_LDC_inventory_list) < 0:
            unmet_demand = 0
            additional_inventory = sum(used_LDC_inventory_list) - sum(EC_demand_update)
        else:
            unmet_demand = 0
            additional_inventory = 0
        # print(f"used CMDs index: {used_CMD_index}\nopened LDCs index: {opened_LDC_index}\nused LDCs inventory: {used_LDC_inventory_list}\nused Ecs index: {used_EC_index}\nused ECs demand: {EC_demand_update}\n")
        for i in range(len(used_CMD_index)):
            CMDs_to_LDCs_dist.append(cmd_ldc_ODMatrix[used_CMD_index[i]][opened_LDC_index[i]])
            CMDs_to_LDCs_allocation[f"{CMD_name[used_CMD_index[i]]}"].append((LDC_name[opened_LDC_index[i]], used_LDC_inventory_list[i]))
        # print(f"CMDs_to_LDCs_dist: {CMDs_to_LDCs_dist}")
        k = 0
        NOT_DEMAND = False
        for i in range(A):
            for j in range(len(opened_LDC_index)):
                K = k
                if used_LDC_inventory_list[j] == 0:
                    continue
                while EC_demand_update[k] == 0:
                    k += 1
                    if k == No_EC:
                        k = 0
                    if k == K:
                        NOT_DEMAND = True
                        break
                if NOT_DEMAND == True:
                    break
                if used_LDC_inventory_list[j] > EC_demand_update[chromosom[No_LDC:][k]]:
                    used_LDC_inventory_list[j] -= EC_demand_update[chromosom[No_LDC:][k]]
                    EC_demand_update[k] = 0
                    LDCs_to_ECs_dist.append(ldc_ec_ODMatrix[opened_LDC_index[j]][chromosom[No_LDC:][k]])
                    LDCs_to_ECs_dist_allocation[f"{LDC_name[opened_LDC_index[j]]}"].append((EC_name[chromosom[No_LDC:][k]], EC_demand_update[chromosom[No_LDC:][k]]))
                elif used_LDC_inventory_list[j] < EC_demand_update[chromosom[No_LDC:][k]]:
                    EC_demand_update[chromosom[No_LDC:][k]] -= used_LDC_inventory_list[j]
                    used_LDC_inventory_list[j] = 0
                    LDCs_to_ECs_dist.append(ldc_ec_ODMatrix[opened_LDC_index[j]][chromosom[No_LDC:][k]])
                    LDCs_to_ECs_dist_allocation[f"{LDC_name[opened_LDC_index[j]]}"].append((EC_name[chromosom[No_LDC:][k]], used_LDC_inventory_list[j]))
                elif used_LDC_inventory_list[j] == EC_demand_update[chromosom[No_LDC:][k]]:
                    used_LDC_inventory_list[j] = 0
                    EC_demand_update[chromosom[No_LDC:][k]] = 0
                    LDCs_to_ECs_dist.append(ldc_ec_ODMatrix[opened_LDC_index[j]][chromosom[No_LDC:][k]])
                    LDCs_to_ECs_dist_allocation[f"{LDC_name[opened_LDC_index[j]]}"].append((EC_name[chromosom[No_LDC:][k]], used_LDC_inventory_list[j]))
                k += 1
                if k == No_EC:
                        k = 0
            if NOT_DEMAND == True:
                break
        # print(f"LDCs_to_ECs_dist: {LDCs_to_ECs_dist}")

        sum_dist = sum(CMDs_to_LDCs_dist) + sum(LDCs_to_ECs_dist)
        sum_opened_LDC = len(opened_LDC_index)
        fitness_list.append(sum_dist + sum_opened_LDC*1000 + unmet_demand*10 + additional_inventory)
        # print(f"\nsum_dist: {sum_dist}\nsum_opened_LDC: {sum_opened_LDC}\nunmet_demand: {unmet_demand}\nadditional_inventory: {additional_inventory}")
        # print(f"EC_demand_update: {EC_demand_update}\nused_LDC_inventory_list: {used_LDC_inventory_list}\n")
    bad_chromosom_index = fitness_list.index(max(fitness_list))
    if min(fitness_list) < elites_fitness:
        elites_fitness = min(fitness_list)
        elite_Chromosom = pop[fitness_list.index(min(fitness_list))]
    return fitness_list

# fitness_list = fitness(Population)
# print(json.dumps(fitness_list))
# print(CMDs_to_LDCs_allocation)
# print(LDCs_to_ECs_dist_allocation)


# Split Rank Selection
def SRS_Selection(ChromosomsFitness: list):
    global Ranking, ProbsRange
    selectedchromosomforcrossover = []
    SelectedChromosomForCrossOver = []
    ChromosomsFitnessSorted = sorted(ChromosomsFitness, reverse=True)
    # print(ChromosomsFitnessSorted)
    ChromosomsIndexByFitness = [ChromosomsFitness.index(i) for i in ChromosomsFitnessSorted]
    for _ in Ranking:
        r = rn.uniform(0,1)
        for index, j in enumerate(ProbsRange):
            if r > j[0] and r <= j[1]:
                selectedchromosomforcrossover.append(ChromosomsIndexByFitness[index])
                if len(selectedchromosomforcrossover) == 2:
                    SelectedChromosomForCrossOver.append(selectedchromosomforcrossover)
                    selectedchromosomforcrossover = []
                    break
    return SelectedChromosomForCrossOver

# selcted_chromosom = SRS_Selection(fitness_list)
# print(json.dumps(selcted_chromosom))


# Two_point_OX_Crossover
def Two_point_OX_Crossover(pop, selectedcrossover, probability):
    childs = []
    for i1, i2 in selectedcrossover:
        if rn.uniform(0, 1) <= probability:
            for p1, p2 in [(pop[i1], pop[i2]), (pop[i2], pop[i1])]:
                # part 1 (Two_point crossover)
                ch_part1, p1_part1, p2_part1 = [-1] * No_LDC, p1[:No_LDC], p1[:No_LDC]
                rn1, rn2 = rn.randint(0, No_LDC), rn.randint(No_LDC//2, No_LDC)
                # rn1, rn2 = min(rn1, rn2), max(rn1, rn2)
                ch_part1[rn1:rn2] = p1_part1[rn1:rn2]
                ch_part1[:rn1], ch_part1[rn2:] = p2_part1[:rn1], p2_part1[rn2:]

                # part 2 (OX crossover)
                ch_part2, p1_part2, p2_part2 = [-1] * No_EC, p1[No_LDC:], p2[No_LDC:]
                rn1, rn2 = rn.randint(0, No_LDC//2-1), rn.randint(No_LDC//2, No_LDC)
                ch_part2[rn1:rn2] = p1_part2[rn1:rn2]
                Chain = chain(range(rn1), range(rn2, No_EC))
                j = 0
                for i in Chain:
                    if i < rn1 or i >= rn2:
                        while p2_part2[j] in ch_part2:
                            j += 1
                        ch_part2[i] = p2_part2[j]
                    j += 1
                childs.append(ch_part1 + ch_part2)
        else:
            childs.extend([pop[i1], pop[i2]])
    return childs
# childs = Two_point_OX_Crossover(Population, selcted_chromosom, 1)
# print(json.dumps(childs))
# Two_point_OX_Crossover


# Mutation
def Mutation(childs: list, probability):
    childsaftermutation = []
    for i in childs:
        r = rn.uniform(0,1)
        if r <= probability:
            # part 1
            rnd0, rnd1 = rn.sample(list(range(No_LDC)),2)
            if type(i[rnd0]) == list:
                r = rn.choice([-1,1])
                i[rnd0][1] += r*10000*sum(cd.values())
                if i[rnd0][1] > V:
                    i[rnd0][1] = V
                elif i[rnd0][1] < 0:
                    i[rnd0][1] = 0
            i[rnd0], i[rnd1] = i[rnd1], i[rnd0]
            # part 2
            rnd0, rnd1 = rn.sample(list(range(No_LDC, No_EC)),2)
            i[rnd0], i[rnd1] = i[rnd1], i[rnd0]

            childsaftermutation.append(i)
        else:
            childsaftermutation.append(i)
    return childsaftermutation


Ranking = []
ChromosomsProb =[]
ProbsRange = []
def Generation(repeat):
    global genetic_best_solution, Ranking, ChromosomsProb, ProbsRange, elite_Chromosom, bad_chromosom_index
    Population = generate_initial_population(200)
    minfit = []
    Ranking = range(1,len(Population)+1)
    K = len(Ranking)
    for i in Ranking:
        if i <= K/2:
            ChromosomsProb.append((12*i)/(5*K*(K+2)))
        else:
            ChromosomsProb.append((28*i)/(5*K*(3*K+2)))
    for index, i in enumerate(ChromosomsProb):
        if index == 0:
            ProbsRange.append([0,i])
        else:
            ProbsRange.append([ProbsRange[index-1][1], ProbsRange[index-1][1] + (i)])
    for i in range(repeat):
        Fitness = fitness(Population)
        minfit.append(min(Fitness) if (min(Fitness) <= elites_fitness) else elites_fitness)
        SelectedChromosoms = SRS_Selection(Fitness)
        Childs = Two_point_OX_Crossover(Population, SelectedChromosoms, 0.8)
        childsaftermutation = Mutation(Childs, 0.2)
        Population = childsaftermutation
        Population[bad_chromosom_index] = elite_Chromosom
    genetic_best_solution = Population[fitness(Population).index(min(fitness(Population)))]
    return genetic_best_solution
genetic_best_solution = Generation(2000)
fitness([genetic_best_solution])
print(json.dumps(genetic_best_solution, ensure_ascii=False))
print(json.dumps(f"Unmet Demand = {unmet_demand}\nAdditional Inventory = {additional_inventory}\nSum Distance = {sum_dist}\nSum Opened LDC = {sum_opened_LDC}"))

output = {
    # "LDC_Coordinate": LDC_Coordinate,
    # "LDC_name": LDC_name,
    # "c_LDC": c_LDC,
    # "No_LDC": No_LDC
}

json_data = json.dumps(output)
# print(json_data)
