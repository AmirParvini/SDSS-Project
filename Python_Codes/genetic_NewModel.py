import numpy as np
import random as rn
import math
import copy
import random as rn
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import geopandas as gpd
from RLagent import rlagent
import time


start_time_genetic = time.time()
end_time_genetic = np.nan
start_time_tabu = np.nan
end_time_tabu = np.nan
# Parameters of Reinforcement Learning
cRange = np.array(range(1, 10))/10
mRange = np.array(range(1, 10))/10
alpha = 0.7
gamma = 0.1
epsilon = 0.3
c = 0.5
m = 0.5
agent = rlagent(alpha, gamma, epsilon, cRange, mRange, c, m)

# Parameters for standard configuration
APR = 0.95
NT = 100
L = 1000  # Lambda: scaling factor for total number of facilities opened
A = 10 # Alpha: maximum number of neighborhoods that a neighborhood can serve
G = 0.5  # Gamma: scale for unit unmet demand
O = 1 # Omega: scale for unit additional inventory
k = ["Water", "Food", "Medical Kit", "Shelter"]  # commodities type
cd = dict(zip(k, [0.833, 1, 0.111, 0.2])) # commodities demand of an injured in 1day
c = 60000
C = c # Capacity of LDC

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


# Reading the CMD File
def read_CMD_file(path: str):
    global CMD_Coordinate, CMD_name, No_CMD
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    for i, line in enumerate(lines, start=1):
        if 1 < i:
            CMD_Attributes = line.split(",")
            CMD_name.append(CMD_Attributes[1])
            CMD_Coordinate.append(
                [float(CMD_Attributes[-2]), float(CMD_Attributes[-1])]
            )
    No_CMD = len(CMD_Coordinate)
    # print(f"Number of CMDs: {No_CMD} \nCMD_Name: {CMD_name} \nCMD_Coordinate: {CMD_Coordinate}\n\n")
    file.close()


# Reading the LDC File
def read_LDC_file(path: str):
    global LDC_Coordinate, LDC_name, No_LDC, c_LDC
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    for i, line in enumerate(lines, start=1):
        if 1 < i:
            LDC_Attributes = line.split(",")
            LDC_name.append(LDC_Attributes[5])
            LDC_Coordinate.append(
                [float(LDC_Attributes[-2]), float(LDC_Attributes[-1])]
            )
    No_LDC = len(LDC_Coordinate)
    c_LDC = [C] * No_LDC
    # print(f"Number of LDCs: {No_LDC} \nLDC_Name: {LDC_name} \nLDC_Coordinate: {LDC_Coordinate}\n\n")
    file.close()


# Reading the EC File
def read_EC_file(path: str):
    global EC_Coordinate, EC_name, No_EC
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    for i, line in enumerate(lines, start=1):
        if 1 < i:
            EC_Attributes = line.split(",")
            EC_name.append(EC_Attributes[2])
            EC_Coordinate.append(
                [float(EC_Attributes[-2]), float(EC_Attributes[-1])])
    No_EC = len(EC_Coordinate)
    # print(f"Number of ECs: {No_EC} \nCE_Name: {EC_name} \nCE_Coordinate: {EC_Coordinate}\n\n")
    file.close()


def read_Pop_Blocks_file(path: str):
    global Pop_Blocks
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    for i, line in enumerate(lines, start=1):
        if 1 < i:
            Pop_Blocks_Attributes = line.split(",")
            Pop_Blocks.append(
                [
                    float(Pop_Blocks_Attributes[-2]),
                    float(Pop_Blocks_Attributes[-1]),
                    float(Pop_Blocks_Attributes[37]),
                ]
            )
    # print(f"Number of Pop_Blocks: {No_Pop_Blocks}\n Pop_Blocks: {Pop_Blocks}\n\n")
    file.close()


read_CMD_file("C:\\Users\\Amir\\Desktop\\SDSS-Project\\Data\\CMD_Points.txt")
read_LDC_file("C:\\Users\\Amir\\Desktop\\SDSS-Project\\Data\\LDC_Points.txt")
read_EC_file("C:\\Users\\Amir\\Desktop\\SDSS-Project\\Data\\EC_Points.txt")
read_Pop_Blocks_file(
    "C:\\Users\\Amir\\Desktop\\SDSS-Project\\Data\\Reg4PopBlock_Points.txt"
)

Pop_Blocks = np.array(Pop_Blocks)
EC_Coordinate = np.array(EC_Coordinate)
distances = cdist(Pop_Blocks[:, :2], EC_Coordinate, metric="euclidean")
# پیدا کردن اندیس نزدیک‌ترین مرکز برای هر نقطه
assigned_centers = np.argmin(distances, axis=1)
# محاسبه مجموع جمعیت برای هر مرکز
EC_pop = np.bincount(assigned_centers, weights=Pop_Blocks[:, 2])
assignments = {i: [] for i in range(158)}
for point_idx, center_idx in enumerate(assigned_centers):
    assignments[center_idx].append(point_idx)
# چاپ یا ذخیره تخصیص‌ها
# for center_idx, point_indices in assignments.items():
# print(f"EC {center_idx}: PopBlock {point_indices}")
EC_water_demand = [round(cd["Water"] * p * APR, 2) for p in EC_pop]
EC_food_demand = [round(cd["Food"] * p * APR, 2) for p in EC_pop]
EC_medicalkit_demand = [round(cd["Medical Kit"] * p * APR, 2) for p in EC_pop]
EC_shelter_demand = [round(cd["Shelter"] * p * APR, 2) for p in EC_pop]
EC_demand = [round(w + f + m + s, 2) for w, f, m, s in zip(EC_water_demand, EC_food_demand, EC_medicalkit_demand, EC_shelter_demand)]
# print(f"\nwater demand of ECs: {EC_water_demand}\nfood demand of ECs: {EC_food_demand}\nmedicalkit demand of ECs: {EC_medicalkit_demand}\nshelter demand of ECs: {EC_shelter_demand}\n\nvd_EC: {EC_demand}")

cmd_ldc_ODMatrix = []
ldc_ec_ODMatrix = []


def CMD_LDC_ODMatrix(path: str):
    global cmd_ldc_ODMatrix, No_CMD, No_LDC
    cmd_ldc_ODMatrix = np.full((No_CMD, No_LDC), np.nan)
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    file.close()
    for i, line in enumerate(lines, start=1):
        if 1 < i:
            Attributes = line.split(",")
            OriginID = Attributes[2]
            DestinationID = Attributes[3]
            cmd_ldc_ODMatrix[int(OriginID) - 1][int(DestinationID) - 1] = float(
                Attributes[-1]
            )


CMD_LDC_ODMatrix(
    "C:\\Users\\Amir\\Desktop\\SDSS-Project\\Data\\OD_Matrix(CMDs_to_LDCs).txt"
)
# pprint.pprint(cmd_ldc_ODMatrix)


def LDC_EC_ODMatrix(path: str):
    global ldc_ec_ODMatrix, No_LDC, No_EC
    ldc_ec_ODMatrix = np.full((No_LDC, No_EC), np.nan)
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    file.close()
    for i, line in enumerate(lines, start=1):
        if 1 < i:
            Attributes = line.split(",")
            OriginID = Attributes[2]
            DestinationID = Attributes[3]
            ldc_ec_ODMatrix[int(OriginID) - 1][int(DestinationID) - 1] = float(
                Attributes[-1]
            )


LDC_EC_ODMatrix(
    "C:\\Users\\Amir\\Desktop\\SDSS-Project\\Data\\OD_Matrix(LDCs_to_ECs).txt"
)
np.set_printoptions(threshold=np.inf)
# print(cmd_ldc_ODMatrix)


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
    c_LDC = [C] * No_LDC
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
a, b = round(min(EC_pop),2 ), round(max(EC_pop), 2)
a_std = round((a - mu_pop) / std_pop, 2)
b_std = round((b - mu_pop) / std_pop, 2)


# Initial Population    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
def generate_initial_population(pop_size):
    population = []
    for ـ in range(pop_size):
        Chromosom_part1 = []
        Chromosom_part2 = []
        # تولید مقدار تصادفی کالا برای هر LDC
        Pop_random_values = truncnorm.rvs(a_std, b_std, loc=mu_pop * 2, scale=std_pop, size=No_LDC)
        Commodities_random_values = list(np.array(Pop_random_values) * sum(cd.values()))
        Commodities_random_values_filter = [C if crv > C else crv for crv in Commodities_random_values]
        nt = 0
        for i in range(No_LDC):
            value = rn.randint(0, 1)
            if value == 0:
                Chromosom_part1.append(0)
            else:
                if nt < NT:
                    cmd_rnd = rn.choice(range(No_CMD))
                    commodity_amount = Commodities_random_values_filter[i]
                    Chromosom_part1.append([cmd_rnd, C])
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

def fitness(pop):
    global unmet_demand, additional_inventory, sum_dist, sum_opened_LDC, CMDs_to_LDCs_allocation, LDCs_to_ECs_allocation
    fitness_list = []
    for chromosom in pop:
        unmet_demand_penalty = 0
        additional_inventory_penalty = 0 
        CMDs_to_LDCs_allocation = {f"{CMD_name[i]}": [] for i in range(No_CMD)}
        LDCs_to_ECs_allocation = {
            f"{LDC_name[i]}": [] for i in range(No_LDC)}
        used_CMD_index = []
        opened_LDC_index = []
        EC_demand_update = copy.deepcopy(EC_demand)
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
                elif used_LDC_inventory_list[j] == EC_demand_update[chromosom[No_LDC:][k]]:
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
        unmet_demand = sum(EC_demand_update)
        additional_inventory = sum(used_LDC_inventory_list)
        # print(f'unmet_demand: {unmet_demand}\n additional_inventory: {additional_inventory}')
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
                        i[rnd0][1] += c/100 * sum(cd.values())
                        if i[rnd0][1] > C:
                            i[rnd0][1] = C
                    elif (additional_inventory > 0 and unmet_demand == 0) or (unmet_demand < additional_inventory):
                        i[rnd0][1] -= c/100 * sum(cd.values())
                        if i[rnd0][1] < 0:
                            i[rnd0][1] = 0
                elif i[rnd0] == 0 and no_zero_count < NT:
                    if unmet_demand > 0 and additional_inventory == 0:
                        Pop_random_values = truncnorm.rvs(a_std, b_std, loc=mu_pop * 2, scale=std_pop, size=1)
                        Commodities_random_values = Pop_random_values[0] * sum(cd.values())
                        Commodities_random_values_filter = C if Commodities_random_values > C else Commodities_random_values
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
    global end_time_genetic, Ranking, ChromosomsProb, ProbsRange, elite_Chromosom, elite_fitness, bad_chromosom_index
    Population = generate_initial_population(popsize)
    minfit = []
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
        print(f"genetic iteration: {i}")
        if i == 0:
            c, m = agent.initAction()
        else:
            c, m = agent.decide()
            # print("Crossover Prob = ", c, "Mutation prob = " ,m)
        Fitness = fitness(Population)
        agent.observe(Population, Fitness)
        agent.updateQlearning()
        bad_chromosom_index = Fitness.index(max(Fitness))
        if min(Fitness) < elite_fitness:
            elite_fitness = min(Fitness)
            elite_Chromosom = copy.deepcopy(Population[Fitness.index(min(Fitness))])
        minfit.append(min(Fitness) if (min(Fitness) <= elite_fitness) else elite_fitness)
        SelectedChromosoms = SRS_Selection(Fitness)
        Childs = Two_point_OX_Crossover(Population, SelectedChromosoms, c)
        childsaftermutation = Mutation(Childs, m)
        Population = childsaftermutation
        Population[bad_chromosom_index] = copy.deepcopy(elite_Chromosom)
    lastfitness = fitness(Population)
    if min(lastfitness) < elite_fitness:
            elite_fitness = min(lastfitness)
            elite_Chromosom = copy.deepcopy(Population[lastfitness.index(min(lastfitness))])
    minfit.append(elite_fitness)
    best_solution = copy.deepcopy(elite_Chromosom)
    g = range(len(minfit))
    fitness([best_solution])
    end_time_genetic = time.time() 
    plt.plot(g, minfit)
    plt.title(f"min_fitness = {minfit[-1]}")
    param_text = (
        f'Learning rate: {alpha}\n'
        f'Discount rate: {gamma}\n'
        f'Exploration rate: {epsilon}'
            # f'Population Size: {popsize}\n'
            #   f'Generations: {repeat}\n'
            #   f'NT: {NT}\n'
            #   f'L: {L}\n'
            #   f'A: {A}\n'
            #   f'G: {G}\n'
            #   f'O: {O}\n'
            #   f'C: {C}\n'
            #   f'-----------\n'
            #   f"Unmet Demand = {round(unmet_demand,2)}\nAdditional Inventory = {round(additional_inventory,2)}\nSum Distance = {round(sum_dist,2)}\nSum Opened LDC = {sum_opened_LDC}"
              )
    plt.text(0.7, 0.95, param_text, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', lw=1, alpha=0.5))
    plt.tight_layout()
    plt.xlabel("iteration")
    plt.ylabel("fitness")
    plt.show()
    return best_solution
# Generating    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -


# Tabu_Search    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
def generate_neighbors(current_solution, moves_iteration):
    # no_zero_count = sum(item != 0 for item in current_solution[:No_LDC])
    neighbors = [] # جواب های تغییر یافته
    moves = [] # حرکتهایی که جواب ها رو تغییر میده
    # حرکت نوع 1: جابجایی در بخش اول
    for i in range(moves_iteration):
        rn1, rn2 = rn.sample(range(No_LDC), 2)
        neighbor = copy.deepcopy(current_solution)
        neighbor[rn1], neighbor[rn2] = neighbor[rn2], neighbor[rn1]
        neighbors.append(neighbor)
        moves.append(("par1_swap", rn1, rn2))
    # حرکت نوع 2: جابجایی در بخش دوم
    for _ in range(moves_iteration):
        rn1, rn2 = rn.sample(range(No_LDC), 2)
        neighbor = copy.deepcopy(current_solution)
        neighbor[rn1], neighbor[rn2] = neighbor[rn2], neighbor[rn1]
        neighbors.append(neighbor)
        moves.append(("par2_swap", rn1, rn2))
    # حرکت نوع 3: افزودن یا حذف LDC
    # Pop_random_values = truncnorm.rvs(a_std, b_std, loc=mu_pop * 2, scale=std_pop, size=moves_iteration)
    # Commo_random_values = list(np.array(Pop_random_values) * sum(cd.values()))
    # for i in range(moves_iteration):
    #     k = rn.choice(range(No_LDC))
    #     if current_solution[k] == 0 and no_zero_count < NT:
    #         neighbor = copy.deepcopy(current_solution)
    #         neighbor[k] = [
    #             rn.choice(range(No_CMD)),
    #             Commo_random_values[i] if Commo_random_values[i] < C else C,
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
    #                 neighbor[i][1] += j*2000*sum(cd.values())
    #                 if neighbor[i][1] > C:
    #                     neighbor[i][1] = C
    #                 elif neighbor[i][1] < 0:
    #                     neighbor[i][1] = 0
    #                 neighbors.append(neighbor)
    #                 moves.append(("ChangeCommodity", i, neighbor[i][1]))
    return neighbors, moves


def tabu_search(initial_solution, max_iterations, tabu_tenure):
    global end_time_tabu
    current_solution = initial_solution
    best_solution = copy.deepcopy(current_solution)
    best_cost = fitness([best_solution])[0]
    tabu_list = []  # لیست ممنوعه برای ذخیره حرکت‌ها
    bc = []
    for m in range(max_iterations):
        print(f"Tabu iteration: {m}")
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
        neighbors_costs = [fitness([n])[0] for n in valid_neighbors]
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
    end_time_tabu = time.time()
    plt.plot(range(1, max_iterations + 1), bc)
    plt.title(f"min_fitness = {best_cost}")
    plt.xlabel("iteration")
    plt.ylabel("fitness")
    plt.show()
    return best_solution, best_cost
# Tabu_Search    -   -   -   -   -   -   -   -   -   -   -   -   -   -   -

Iterations, popsize = 500, 250
genetic_best_solution = Generation(Iterations, popsize)
print(f"\ngenetic_best_fitness: {fitness([genetic_best_solution])[0]}\ngenetic_best_solution: {genetic_best_solution}")
print(f"Unmet Demand = {unmet_demand}\nAdditional Inventory = {additional_inventory}\nSum Distance = {sum_dist}\nSum Opened LDC = {sum_opened_LDC}")
# genetic_best_solution = generate_initial_population(1)[0]
start_time_tabu = time.time()
tabu_best_solution, best_cost = tabu_search(genetic_best_solution, 4000, 12)
print(f"Tabu_Search_Best_fitness: {fitness([tabu_best_solution])}\nTabu_Search Best Solution : {tabu_best_solution}")
print(f"Unmet Demand = {unmet_demand}\nAdditional Inventory = {additional_inventory}\nSum Distance = {sum_dist}\nSum Opened LDC = {sum_opened_LDC}"
      ,f'Solution Time = {(end_time_genetic-start_time_genetic)+(end_time_tabu-start_time_tabu)}')
Allocation_Clusters = [CMDs_to_LDCs_allocation, LDCs_to_ECs_allocation]


SolutionInformation = {
    "Solution Status": "Optimal solution found",
    "Iterations": Iterations,
    "Total Distance": sum_dist,
    "Total LDC opened": sum_opened_LDC,
    "Unmet Demand": unmet_demand,
    "Additional Inventory": additional_inventory,
}

filtered_CMDs_to_LDCs_allocation = {
    k: C for k, C in CMDs_to_LDCs_allocation.items() if C
}
filtered_LDCs_to_ECs_allocation = {
    k: C for k, C in LDCs_to_ECs_allocation.items() if C
}

# print(f"\nfiltered_CMDs_to_LDCs_allocation: {filtered_CMDs_to_LDCs_allocation}\n")
# print(
#     f"\nfiltered_LDCs_to_ECs_allocation: {filtered_LDCs_to_ECs_allocation}\n"
# )

NodesCoordinate = {"CMDs_coordinate": {},
                   "LDCs_coordinate": {}, "ECs_coordinate": {}}
for k, C in filtered_CMDs_to_LDCs_allocation.items():
    NodesCoordinate["CMDs_coordinate"][k] = CMD_Coordinate[CMD_name.index(k)]
for k, C in filtered_LDCs_to_ECs_allocation.items():
    NodesCoordinate["LDCs_coordinate"][k] = LDC_Coordinate[LDC_name.index(k)]
    for i in C:
        # print(json.dumps(i[0], ensure_ascii=False))
        NodesCoordinate["ECs_coordinate"][i[0]] = EC_Coordinate[EC_name.index(i[0])]


output = {
    "SolutionInformation": SolutionInformation,
    "CMDs_to_LDCs_Allocation": filtered_CMDs_to_LDCs_allocation,
    "LDCs_to_ECs_Allocation": filtered_LDCs_to_ECs_allocation,
    "NodesCoordinate": NodesCoordinate,
}

# json_data = json.dumps(output)
# print(output)




def BestChromosom_Plot():
    global CMDs_to_LDCs_allocation, LDCs_to_ECs_allocation
    # display shp file
    shapefile_path = "C:\\Users\\Amir\\Desktop\\SDSS-Project\\Data\\Mahallat\\Reg_4.shp"
    polygons = gpd.read_file(shapefile_path)
    # print(polygons.head())
    polygons.plot(edgecolor="black", facecolor="lightblue", figsize=(10, 8))
    plt.title("Polygon Map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # display nodes
    x_cmd = []
    y_cmd = []
    x_ldc = []
    y_ldc = []
    x_ec = []
    y_ec = []
    for i in CMD_Coordinate:
        x_cmd.append(i[0])
        y_cmd.append(i[1])
    plt.scatter(x_cmd, y_cmd, c="r", s=15, marker="s")
    for n in range(No_CMD):
        plt.text(x_cmd[n], y_cmd[n], n, fontsize=7, ha="right", va="bottom")
    for i in LDC_Coordinate:
        x_ldc.append(i[0])
        y_ldc.append(i[1])
    plt.scatter(x_ldc, y_ldc, s=10)
    for n in range(No_LDC):
        plt.text(x_ldc[n], y_ldc[n], n, fontsize=6, ha="right", va="bottom")
    for i in EC_Coordinate:
        x_ec.append(i[0])
        y_ec.append(i[1])
    plt.scatter(x_ec, y_ec, s=2)
    for n in range(No_EC):
        plt.text(x_ec[n], y_ec[n], n, fontsize=5, ha="right", va="bottom")

    # display allocation nodes
    for kv in zip(CMDs_to_LDCs_allocation.keys(), CMDs_to_LDCs_allocation.values()):
        key = int(kv[0])
        value = kv[1]
        for C in value:
            plt.plot(
                [x_cmd[key], x_ldc[C]],
                [y_cmd[key], y_ldc[C]],
                color="blue",
                linewidth=1,
            )
    for kv in zip(
        LDCs_to_ECs_allocation.keys(), LDCs_to_ECs_allocation.values()
    ):
        key = int(kv[0])
        value = kv[1]
        for C in value:
            plt.plot(
                [x_ldc[key], x_ec[C]], [y_ldc[key], y_ec[C]], color="red", linewidth=0.3
            )