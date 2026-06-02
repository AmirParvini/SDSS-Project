import numpy as np
import random as rn
from itertools import chain
import utm
import math
import time
import copy
import random as rn
import json
import sys
import unicodedata
from RLagent_NewModel import rlagent
from scipy.stats import truncnorm

start_time = time.time()
sys.stdout.reconfigure(encoding="utf-8")
data = sys.stdin.buffer.read()
data = json.loads(data)
parameters = data["config_parameters"]
nodes_data = data["nodes_data"]
nodes_dist_data = data["nodes_dist_data"]
CMD_nodes_data = nodes_data[0]
LDC_nodes_data = nodes_data[1]
EC_nodes_data = nodes_data[2]
CMD_to_LDC_dist_data = nodes_dist_data[0]
LDC_to_EC_dist_data = nodes_dist_data[1]
commodity_demands_unit = data["commodity_demands_unit"]
APR = float(data["APR"])

# Parameters for standard configuration
NT = parameters["NT"]  # total number of LDC facilities available
L = parameters["L"] # scaling factor for total number of LDC opened
A = parameters["A"] # Alpha: maximum number of neighborhoods that a neighborhood can serve
G = parameters["G"] # Gamma: penalty for unit unmet demand
O = parameters["O"] # Omega: penalty for unit overcapacity
k = ["Water", "Food", "Medical Kit", "Shelter"]  # commodities type
cd = dict(zip(k,[commodity_demands_unit["Water(unit-pp)"], commodity_demands_unit["Food(unit-pp)"], commodity_demands_unit["MedicalKit(unit-pp)"], commodity_demands_unit["Shelter(unit-pp)"]]))  # commodities demand of an injured
V = parameters["V"]  # Capacity of LDC


CMD_Coordinate = []  # [[x1,y1], [x2,y2], ...]
LDC_Coordinate = []  # [[x1,y1], [x2,y2], ...]
EC_Coordinate = []  # [[x1,y1], [x2,y2], ...]
Pop_Blocks = []  # [[x1,y1,pop1], [x2,y2,pop2], ...]
No_CMD = 0  # number of Crises Management Databases
No_LDC = 0  # number of Local Distribution Centers
No_EC = 0  # number of Evacuation Centers
CMD_name = []
LDC_name = []
c_LDC = []  # capacities of LDC
EC_name = []
EC_pop = []
EC_demand = []  # demands of EC
EC_water_demand = []
EC_food_demand = []
EC_medicalkit_demand = []
EC_shelter_demand = []
elite_Chromosom = []
elite_fitness = math.inf
bad_chromosom_index = 0


def normalize_str(s: list):
    return unicodedata.normalize("NFKC", s.strip())


for CMD_node_data in CMD_nodes_data:
    lat, lon = utm.to_latlon(CMD_node_data["X"], CMD_node_data["Y"], 39, "N")
    CMD_Coordinate.append([lat, lon])
    CMD_name.append(CMD_node_data["Name"])
    No_CMD += 1
for LDC_node_data in LDC_nodes_data:
    lat, lon = utm.to_latlon(LDC_node_data["X"], LDC_node_data["Y"], 39, "N")
    LDC_Coordinate.append([lat, lon])
    LDC_name.append(LDC_node_data["Name"])
    c_LDC.append(V)
    No_LDC += 1
for EC_node_data in EC_nodes_data:
    lat, lon = utm.to_latlon(EC_node_data["X"], EC_node_data["Y"], 39, "N")
    EC_Coordinate.append([lat, lon])
    EC_name.append(EC_node_data["Name"])
    EC_pop.append(EC_node_data["Population"])
    EC_water_demand.append(round(EC_node_data["Population"] * cd["Water"], 2))
    EC_food_demand.append(round(EC_node_data["Population"] * cd["Food"], 2))
    EC_medicalkit_demand.append(round(EC_node_data["Population"] * cd["Medical Kit"], 2))
    EC_shelter_demand.append(round(EC_node_data["Population"] * cd["Shelter"], 2))
    EC_demand.append(round(EC_node_data["Population"] * APR * sum(cd.values()), 2))
    No_EC += 1

CMD_name = [normalize_str(s) for s in CMD_name]
LDC_name = [normalize_str(s) for s in LDC_name]
EC_name = [normalize_str(s) for s in EC_name]

cmd_ldc_ODMatrix = np.full((No_CMD, No_LDC), np.nan)
for CLdist in CMD_to_LDC_dist_data:
    if (
        normalize_str(CLdist["CMD_name"]) in CMD_name
        and normalize_str(CLdist["LDC_name"]) in LDC_name
    ):
        cmd_ldc_ODMatrix[
            CMD_name.index(normalize_str(CLdist["CMD_name"])),
            LDC_name.index(normalize_str(CLdist["LDC_name"])),
        ] = CLdist["Distance"]
ldc_ec_ODMatrix = np.full((No_LDC, No_EC), np.nan)
for LEdist in LDC_to_EC_dist_data:
    if (
        normalize_str(LEdist["LDC_name"]) in LDC_name
        and normalize_str(LEdist["EC_name"]) in EC_name
    ):
        ldc_ec_ODMatrix[
            LDC_name.index(normalize_str(LEdist["LDC_name"])),
            EC_name.index(normalize_str(LEdist["EC_name"])),
        ] = LEdist["Distance"]

nan_cols_cmd_ldc_ODMatrix = np.where(
    np.all(np.isnan(cmd_ldc_ODMatrix), axis=0))[0]
nan_rows_cmd_ldc_ODMatrix = np.where(
    np.all(np.isnan(cmd_ldc_ODMatrix), axis=1))[0]
nan_cols_ldc_ec_ODMatrix = np.where(
    np.all(np.isnan(ldc_ec_ODMatrix), axis=0))[0]
nan_rows_ldc_ec_ODMatrix = np.where(
    np.all(np.isnan(ldc_ec_ODMatrix), axis=1))[0]
cmd_invalid_index = list(nan_rows_cmd_ldc_ODMatrix)
ldc_invalid_index = list(nan_rows_ldc_ec_ODMatrix)
ec_invalid_index = list(nan_cols_ldc_ec_ODMatrix)

if len(cmd_invalid_index) > 0:
    cmd_ldc_ODMatrix = np.delete(cmd_ldc_ODMatrix, cmd_invalid_index, axis=0)
    CMD_name = list(np.delete(CMD_name, cmd_invalid_index))
    CMD_Coordinate = np.delete(
        CMD_Coordinate, cmd_invalid_index, axis=0).tolist()
    No_CMD = len(CMD_name)
if len(ldc_invalid_index) > 0:
    cmd_ldc_ODMatrix = np.delete(cmd_ldc_ODMatrix, ldc_invalid_index, axis=1)
    ldc_ec_ODMatrix = np.delete(ldc_ec_ODMatrix, ldc_invalid_index, axis=0)
    LDC_name = list(np.delete(LDC_name, ldc_invalid_index))
    LDC_Coordinate = np.delete(
        LDC_Coordinate, ldc_invalid_index, axis=0).tolist()
    No_LDC = len(LDC_name)
    c_LDC = [V] * No_LDC
if len(ec_invalid_index) > 0:
    ldc_ec_ODMatrix = np.delete(ldc_ec_ODMatrix, ec_invalid_index, axis=1)
    EC_name = list(np.delete(EC_name, ec_invalid_index))
    EC_Coordinate = np.delete(EC_Coordinate, ec_invalid_index, axis=0).tolist()
    No_EC = len(EC_name)
    EC_demand = np.delete(EC_demand, ec_invalid_index)
    EC_water_demand = np.delete(EC_water_demand, ec_invalid_index)
    EC_food_demand = np.delete(EC_food_demand, ec_invalid_index)
    EC_medicalkit_demand = np.delete(EC_medicalkit_demand, ec_invalid_index)
    EC_shelter_demand = np.delete(EC_shelter_demand, ec_invalid_index)

mu_pop = round(np.mean(EC_pop), 2)
std_pop = round(np.std(EC_pop), 2)
a, b = round(min(EC_pop), 2), round(max(EC_pop), 2)
a_std = round((a - mu_pop) / std_pop, 2)
b_std = round((b - mu_pop) / std_pop, 2)


# Initial Population    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
def generate_initial_population(pop_size):
    population = []
    for ـ in range(pop_size):
        Chromosom_part1 = []
        Chromosom_part2 = []
        # تولید مقدار تصادفی کالا برای هر LDC
        # Pop_random_values = truncnorm.rvs(a_std, b_std, loc=mu_pop * 2, scale=std_pop, size=No_LDC)
        # Commodities_random_values = list(np.array(Pop_random_values) * sum(cd.values()))
        # Commodities_random_values_filter = [V if crv > V else crv for crv in Commodities_random_values]
        nt = 0
        for i in range(No_LDC):
            value = rn.randint(0, 1)
            if value == 0:
                Chromosom_part1.append(0)
            else:
                cmd_rnd = rn.choice(range(No_CMD))
                # commodity_amount = Commodities_random_values_filter[i]
                if nt < NT:
                    Chromosom_part1.append([cmd_rnd, V])
                    nt += 1
                else:
                    Chromosom_part1.append(0)
        Chromosom_part2 = rn.sample(range(No_EC), No_EC)
        population.append(Chromosom_part1 + Chromosom_part2)
    return population
# Initial Population    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -

# Fitness    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
used_CMD_index = []
opened_LDC_index = []
unmet_demand = np.NaN
additional_inventory = np.NaN
sum_dist = np.NaN
sum_opened_LDC = np.NaN
CMDs_to_LDCs_allocation = {}
LDCs_to_ECs_allocation = {}
LDC_name_additional_inventory = {}
EC_name_unmetdemand = {}

def fitness(pop):
    global unmet_demand, additional_inventory, sum_dist, sum_opened_LDC, CMDs_to_LDCs_allocation, LDCs_to_ECs_allocation, LDC_name_additional_inventory, EC_name_unmetdemand
    fitness_list = []
    for chromosom in pop:
        unmet_demand_penalty = 0
        additional_inventory_penalty = 0
        CMDs_to_LDCs_allocation = {f"{CMD_name[i]}": [] for i in range(No_CMD)}
        LDCs_to_ECs_allocation = {f"{LDC_name[i]}": [] for i in range(No_LDC)}
        LDC_name_additional_inventory = {}
        EC_name_unmetdemand = {}
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
        for i in range(len(used_CMD_index)):
            CMDs_to_LDCs_dist.append(cmd_ldc_ODMatrix[used_CMD_index[i]][opened_LDC_index[i]])
            CMDs_to_LDCs_allocation[f"{CMD_name[used_CMD_index[i]]}"]\
            .append([LDC_name[opened_LDC_index[i]],[used_LDC_inventory_list[i] / sum(cd.values()) * c for c in cd.values()]])
        k = 0
        NOT_DEMAND = False
        for i in range(A):
            for j in range(len(opened_LDC_index)):
                K = k
                if used_LDC_inventory_list[j] == 0:
                    continue
                while EC_demand_update[chromosom[No_LDC:][k]] == 0:
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
                    LDCs_to_ECs_dist.append(ldc_ec_ODMatrix[opened_LDC_index[j]][chromosom[No_LDC:][k]])
                    LDCs_to_ECs_allocation[f"{LDC_name[opened_LDC_index[j]]}"]\
                    .append([EC_name[chromosom[No_LDC:][k]],[EC_demand_update[chromosom[No_LDC:][k]]/sum(cd.values())*c for c in cd.values()]])
                    EC_demand_update[chromosom[No_LDC:][k]] = 0
                    k += 1
                elif used_LDC_inventory_list[j] < EC_demand_update[chromosom[No_LDC:][k]]:
                    EC_demand_update[chromosom[No_LDC:][k]] -= used_LDC_inventory_list[j]
                    LDCs_to_ECs_dist.append(ldc_ec_ODMatrix[opened_LDC_index[j]][chromosom[No_LDC:][k]])
                    LDCs_to_ECs_allocation[f"{LDC_name[opened_LDC_index[j]]}"]\
                    .append([EC_name[chromosom[No_LDC:][k]],[used_LDC_inventory_list[j] /sum(cd.values())*c for c in cd.values()]])
                    used_LDC_inventory_list[j] = 0
                elif used_LDC_inventory_list[j]== EC_demand_update[chromosom[No_LDC:][k]]:
                    used_LDC_inventory_list[j] = 0
                    EC_demand_update[chromosom[No_LDC:][k]] = 0
                    LDCs_to_ECs_dist.append(ldc_ec_ODMatrix[opened_LDC_index[j]][chromosom[No_LDC:][k]])
                    LDCs_to_ECs_allocation[f"{LDC_name[opened_LDC_index[j]]}"]\
                    .append([EC_name[chromosom[No_LDC:][k]],[used_LDC_inventory_list[j] /sum(cd.values()) *c for c in cd.values()]])
                    k += 1
                if k == No_EC:
                    k = 0
            if NOT_DEMAND == True:
                break
        LDC_additional_inventory_index_value = [[index, value] for index, value in enumerate(used_LDC_inventory_list) if value != 0]
        EC_unmetdemand_index_value = [[index, value] for index, value in enumerate(EC_demand_update) if value != 0]
        for i, v in LDC_additional_inventory_index_value:
            v /= sum(cd.values())
            LDC_name_additional_inventory[f'{LDC_name[opened_LDC_index[i]]}'] = [v*cd["Water"], v*cd["Food"], v*cd["Medical Kit"], v*cd["Shelter"]]
        for i, v in EC_unmetdemand_index_value:
            v /= sum(cd.values())
            EC_name_unmetdemand[f'{EC_name[i]}'] = [v*cd["Water"], v*cd["Food"], v*cd["Medical Kit"], v*cd["Shelter"]]
        unmet_demand = sum(EC_demand_update)
        additional_inventory = sum(used_LDC_inventory_list)
        if unmet_demand > 0:
            unmet_demand_penalty = unmet_demand/10
        if additional_inventory > 0 and unmet_demand == 0:
            additional_inventory_penalty = additional_inventory_penalty/10
        sum_dist = sum(CMDs_to_LDCs_dist) + sum(LDCs_to_ECs_dist)
        sum_opened_LDC = len(opened_LDC_index)
        fitness_list.append(sum_dist + sum_opened_LDC * L + unmet_demand * G + additional_inventory*O + unmet_demand_penalty + additional_inventory_penalty)
    return fitness_list
# Fitness    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -


# Split Rank Selection    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
def SRS_Selection(ChromosomsFitness: list):
    selectedchromosomforcrossover = []
    SelectedChromosomForCrossOver = []
    ChromosomsFitnessSorted = sorted(ChromosomsFitness, reverse=True)
    ChromosomsIndexByFitness = [
        ChromosomsFitness.index(i) for i in ChromosomsFitnessSorted
    ]
    for _ in Ranking:
        r = rn.uniform(0, 1)
        for index, j in enumerate(ProbsRange):
            if r > j[0] and r <= j[1]:
                selectedchromosomforcrossover.append(
                    ChromosomsIndexByFitness[index])
                if len(selectedchromosomforcrossover) == 2:
                    SelectedChromosomForCrossOver.append(
                        selectedchromosomforcrossover)
                    selectedchromosomforcrossover = []
                    break
    return SelectedChromosomForCrossOver
# Split Rank Selection    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -

# Two_point_OX_Crossover    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
def Two_point_OX_Crossover(pop, selectedcrossover, probability):
    # childs = []
    # for i1, i2 in selectedcrossover:
    #     if rn.uniform(0, 1) <= probability:
    #         for p1, p2 in [(pop[i1], pop[i2]), (pop[i2], pop[i1])]:
    #             # part 1 (Two_point crossover)
    #             ch_part1, p1_part1, p2_part1 = [-1] * No_LDC, p1[:No_LDC], p1[:No_LDC]
    #             rn1, rn2 = rn.sample(range(No_LDC), 2)
    #             rn1, rn2 = min(rn1, rn2), max(rn1, rn2)
    #             ch_part1[rn1:rn2] = p1_part1[rn1:rn2]
    #             ch_part1[:rn1], ch_part1[rn2:] = p2_part1[:rn1], p2_part1[rn2:]
    #             removing_index__list = [] # برای چک کردن محدودیت تعدا مراکز توزیع
    #             for index, i in enumerate(ch_part1):
    #                 if type(i) == list:
    #                     removing_index__list.append(index)
    #             if len(removing_index__list) > NT:
    #                 e = len(removing_index__list) - NT
    #                 rand_remove = rn.sample(removing_index__list, k=e)
    #                 for i in rand_remove:
    #                     ch_part1[i] = 0

    #             # part 2 (OX crossover)
    #             ch_part2, p1_part2, p2_part2 = [-1] * No_EC, p1[No_LDC:], p2[No_LDC:]
    #             rn1, rn2 = rn.sample(range(No_EC), 2)
    #             rn1, rn2 = min(rn1, rn2), max(rn1, rn2)
    #             ch_part2[rn1:rn2] = p1_part2[rn1:rn2]
    #             Chain = chain(range(rn1), range(rn2, No_EC))
    #             j = 0
    #             for i in Chain:
    #                 if i < rn1 or i >= rn2:
    #                     while p2_part2[j] in ch_part2:
    #                         j += 1
    #                     ch_part2[i] = p2_part2[j]
    #                 j += 1
    #             childs.append(ch_part1 + ch_part2)
    #     else:
    #         childs.extend([pop[i1], pop[i2]])
    # return childs

    childs = []
    for i1, i2 in selectedcrossover:
        if rn.uniform(0, 1) <= probability:
            p1, p2 = pop[i1], pop[i2]

            # --- Child 1 ---
            # پارت اول از Two-point crossover
            ch1_part1 = [-1] * No_LDC
            rn1, rn2 = sorted([rn.randint(0, No_LDC), rn.randint(0, No_LDC)])
            ch1_part1[rn1:rn2] = copy.deepcopy(p1[rn1:rn2])
            for i in list(range(0, rn1)) + list(range(rn2, No_LDC)):
                ch1_part1[i] = p2[i]
            removing_index__list = [] # برای چک کردن محدودیت تعدا مراکز توزیع
            for index, i in enumerate(ch1_part1):
                if type(i) == list:
                    removing_index__list.append(index)
            if len(removing_index__list) > NT:
                e = len(removing_index__list) - NT
                rand_remove = rn.sample(removing_index__list, k=e)
                for i in rand_remove:
                    ch1_part1[i] = 0
            # پارت دوم فقط کپی از پارت دوم والد اول
            ch1_part2 = p1[No_LDC:]
            child1 = ch1_part1 + ch1_part2
            # --- Child 2 ---
            # پارت اول فقط کپی از پارت اول والد دوم
            ch2_part1 = p2[:No_LDC]
            # پارت دوم از OX crossover
            p1_part2 = p1[No_LDC:]
            p2_part2 = p2[No_LDC:]
            ch2_part2 = [-1] * No_EC
            rn1, rn2 = sorted([rn.randint(0, No_EC), rn.randint(0, No_EC)])
            ch2_part2[rn1:rn2] = p2_part2[rn1:rn2]
            j = 0
            for i in list(range(0, rn1)) + list(range(rn2, No_EC)):
                while p1_part2[j] in ch2_part2:
                    j += 1
                ch2_part2[i] = p1_part2[j]
                j += 1

            child2 = ch2_part1 + ch2_part2
            childs.extend([child1, child2])
        else:
            # No crossover → direct copy
            childs.extend([pop[i1], pop[i2]])
    return childs

# Two_point_OX_Crossover    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -


# Mutation    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
def Mutation(childs: list, probability):
    childsaftermutation = []
    for i in childs:
        r = rn.uniform(0, 1)
        if r <= probability:
            mp = rn.uniform(0, 1)
            if mp <= 0.35:
                # part 1 (Add/Reduction Commodity)
                no_zero_count = sum(item != 0 for item in i[:No_LDC])
                rnd0, rnd1 = rn.sample(list(range(No_LDC)), 2)
                if type(i[rnd0]) == list:
                    if (unmet_demand > 0 and additional_inventory == 0) or (unmet_demand > additional_inventory):
                        i[rnd0][1] += parameters['V']/100 * sum(cd.values())
                        if i[rnd0][1] > parameters['V']:
                            i[rnd0][1] = parameters['V']
                    elif (additional_inventory > 0 and unmet_demand == 0) or (unmet_demand < additional_inventory):
                        i[rnd0][1] -= c/100 * sum(cd.values())
                        if i[rnd0][1] < 0:
                            i[rnd0][1] = 0
                elif i[rnd0] == 0 and no_zero_count < NT:
                    if unmet_demand > 0 and additional_inventory == 0:
                        Pop_random_values = truncnorm.rvs(a_std, b_std, loc=mu_pop * 2, scale=std_pop, size=1)
                        Commodities_random_values = Pop_random_values[0] * sum(cd.values())
                        Commodities_random_values_filter = parameters['V'] if Commodities_random_values > parameters['V'] else Commodities_random_values
                        i[rnd0] = [rn.choice(range(No_CMD)), Commodities_random_values_filter]
                    elif additional_inventory > 0 and unmet_demand == 0:
                        i[rnd1] = 0
            elif 3.5 < mp <= 7:
                # part1 (Swap)
                i[rnd0], i[rnd1] = i[rnd1], i[rnd0]
            else:
                # part 2 (Swap)
                rnd0, rnd1 = rn.sample(list(range(No_LDC, No_EC)), 2)
                i[rnd0], i[rnd1] = i[rnd1], i[rnd0]
            childsaftermutation.append(i)
        else:
            childsaftermutation.append(i)
    return childsaftermutation
# Mutation    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -

# Generating    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
Ranking = []
ChromosomsProb = []
ProbsRange = []

def Generation(repeat, popsize):
    global Ranking, ChromosomsProb, ProbsRange, elite_Chromosom, elite_fitness, bad_chromosom_index
    Population = generate_initial_population(popsize)
    Ranking = range(1, len(Population) + 1)
    K = len(Ranking)
    for i in Ranking:
        if i <= K / 2:
            ChromosomsProb.append((12 * i) / (5 * K * (K + 2)))
        else:
            ChromosomsProb.append((28 * i) / (5 * K * (3 * K + 2)))
    for index, i in enumerate(ChromosomsProb):
        if index == 0:
            ProbsRange.append([0, i])
        else:
            ProbsRange.append([ProbsRange[index - 1][1], ProbsRange[index - 1][1] + (i)])
    for i in range(repeat):
        if i == 0:
            c, m = agent.initAction()
        else:
            c, m = agent.decide()
        Fitness = fitness(Population)
        agent.observe(Population, Fitness)
        agent.updateQlearning()
        bad_chromosom_index = Fitness.index(max(Fitness))
        if min(Fitness) < elite_fitness:
            elite_fitness = min(Fitness)
            elite_Chromosom = copy.deepcopy(Population[Fitness.index(min(Fitness))])
        SelectedChromosoms = SRS_Selection(Fitness)
        Childs = Two_point_OX_Crossover(Population, SelectedChromosoms, c)
        childsaftermutation = Mutation(Childs, m)
        Population = childsaftermutation
        Population[bad_chromosom_index] = copy.deepcopy(elite_Chromosom)
    lastfitness = fitness(Population)
    if min(lastfitness) < elite_fitness:
            elite_fitness = min(lastfitness)
            elite_Chromosom = copy.deepcopy(Population[lastfitness.index(min(lastfitness))])
    best_solution = copy.deepcopy(elite_Chromosom)
    return best_solution

# Generating    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -


# Tabu_Search    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
def generate_neighbors(current_solution, moves_iteration):
    # no_zero_count = sum(item != 0 for item in current_solution[:No_LDC])
    neighbors = []
    moves = []  # برای ذخیره حرکت‌ها
    # حرکت نوع 1: جابجایی در بخش اول
    rn1 = rn.sample(range(No_LDC), moves_iteration)
    rn2 = rn.sample(range(No_LDC), moves_iteration)
    for i in range(moves_iteration):
        neighbor = copy.deepcopy(current_solution)
        neighbor[rn1[i]], neighbor[rn2[i]] = neighbor[rn2[i]], neighbor[rn1[i]]
        neighbors.append(neighbor)
        moves.append(("par1_swap", i, rn2[i]))
    # حرکت نوع 2: جابجایی در بخش دوم
    rn1 = rn.sample(
        range(No_LDC, len(current_solution[No_LDC:])), moves_iteration)
    rn2 = rn.sample(
        range(No_LDC, len(current_solution[No_LDC:])), moves_iteration)
    for _ in range(moves_iteration):
        neighbor = copy.deepcopy(current_solution)
        neighbor[rn1[i]], neighbor[rn2[i]] = neighbor[rn2[i]], neighbor[rn1[i]]
        neighbors.append(neighbor)
        moves.append(("par2_swap", rn1[i], rn2[i]))
    # حرکت نوع 3: افزودن یا حذف LDC
    # Pop_random_values = truncnorm.rvs(a_std, b_std, loc=mu_pop * 2, scale=std_pop, size=moves_iteration)
    # Commo_random_values = list(np.array(Pop_random_values) * sum(cd.values()))
    # for i in range(moves_iteration):
    #     k = rn.choice(range(No_LDC))
    #     if current_solution[k] == 0 and no_zero_count < NT:
    #         neighbor = copy.deepcopy(current_solution)
    #         neighbor[k] = [
    #             rn.choice(range(No_CMD)),
    #             Commo_random_values[i] if Commo_random_values[i] < V else V,
    #         ]
    #         neighbors.append(neighbor)
    #         moves.append(("ChangeLDC", k, neighbor[k]))
    #     else:
    #         neighbor = copy.deepcopy(current_solution)
    #         neighbor[k] = 0
    #         neighbors.append(neighbor)
    #         moves.append(("ChangeLDC", k, 0))
    #     # حرکت نوع 4: افزودن یا کاهش کالا
    #     r = rn.sample(range(No_LDC), moves_iteration)
    #     for i in r:
    #         if type(current_solution[i]) == list:
    #             neighbor = copy.deepcopy(current_solution)
    #             for j in [-1,1]:
    #                 neighbor[i][1] += j * parameters["V"]/10 * sum(cd.values())
    #                 if neighbor[i][1] > V:
    #                     neighbor[i][1] = V
    #                 elif neighbor[i][1] < 0:
    #                     neighbor[i][1] = 0
    #                 neighbors.append(neighbor)
    #                 moves.append(("ChangeCommodity", i, neighbor[i][1]))
    return neighbors, moves


def tabu_search(initial_solution, max_iterations, tabu_tenure):
    current_solution = initial_solution
    best_solution = copy.deepcopy(current_solution)
    best_cost = fitness([best_solution])
    tabu_list = []  # لیست ممنوعه برای ذخیره حرکت‌ها
    bc = []
    for m in range(max_iterations):
        neighbors, moves = generate_neighbors(current_solution, 10)
        # فیلتر کردن همسایه‌های مجاز
        valid_neighbors = []
        valid_moves = []
        for neighbor, move in zip(neighbors, moves):
            if move not in tabu_list:
                valid_neighbors.append(neighbor)
                valid_moves.append(move)
        if not valid_neighbors:
            break  # هیچ همسایه مجازی وجود ندارد
        # ارزیابی همسایه‌ها
        neighbors_costs = [fitness([n]) for n in valid_neighbors]
        min_cost = min(neighbors_costs)
        min_index = neighbors_costs.index(min_cost)
        # به‌روزرسانی جواب فعلی
        current_solution = valid_neighbors[min_index]
        current_move = valid_moves[min_index]
        # به‌روزرسانی بهترین جواب
        if min_cost < best_cost:
            best_solution = copy.deepcopy(current_solution)
            best_cost = min_cost
        bc.append(best_cost)
        # به‌روزرسانی لیست ممنوعه
        tabu_list.append(current_move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
    return best_solution, best_cost
# Tabu_Search    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -

# Parameters of Reinforcement Learning
cRange = np.array(range(1, 10))/10
mRange = np.array(range(1, 10))/10
alpha = 0.7
gamma = 0.1
epsilon = 0.3
c = 0.5
m = 0.5
agent = rlagent(alpha, gamma, epsilon, cRange, mRange, c, m)
Iterations, popsize = 1000, 250
genetic_best_solution = Generation(Iterations, popsize)
tabu_best_solution, best_cost = tabu_search(genetic_best_solution, 4000, 12)
fitness([tabu_best_solution])
Allocation_Clusters = [CMDs_to_LDCs_allocation, LDCs_to_ECs_allocation]
end_time = time.time()



SolutionInformation = {
    "Solution Status": "Optimal solution found",
    "Iterations": Iterations,
    "Solution Time": end_time - start_time,
    "Total Distance": sum_dist,
    "Total LDC opened": sum_opened_LDC,
    "Unmet Demand": unmet_demand,
    "Additional Inventory": additional_inventory,
}

filtered_CMDs_to_LDCs_allocation = {
    k: v for k, v in CMDs_to_LDCs_allocation.items() if v
}
filtered_LDCs_to_ECs_allocation = {
    k: v for k, v in LDCs_to_ECs_allocation.items() if v
}


NodesCoordinate = {"CMDs_coordinate": {},
                   "LDCs_coordinate": {}, "ECs_coordinate": {}}
for k, v in filtered_CMDs_to_LDCs_allocation.items():
    NodesCoordinate["CMDs_coordinate"][k] = CMD_Coordinate[CMD_name.index(k)]
for k, v in filtered_LDCs_to_ECs_allocation.items():
    NodesCoordinate["LDCs_coordinate"][k] = LDC_Coordinate[LDC_name.index(k)]
    for i in v:
        NodesCoordinate["ECs_coordinate"][i[0]] = EC_Coordinate[EC_name.index(i[0])]

Additional_unmetdemand = {
    'LDC_name_additional_inventory': LDC_name_additional_inventory,
    'EC_name_unmetdemand': EC_name_unmetdemand
}

output = {
    "SolutionInformation": SolutionInformation,
    "CMDs_to_LDCs_Allocation": filtered_CMDs_to_LDCs_allocation,
    "LDCs_to_ECs_Allocation": filtered_LDCs_to_ECs_allocation,
    "NodesCoordinate": NodesCoordinate,
    "Additional_unmetdemand": Additional_unmetdemand
}

json_data = json.dumps(output)
print(json_data)
sys.stdout.flush()
