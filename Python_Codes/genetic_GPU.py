import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.markers as mks
import numpy as np
import pprint
import random as rn
from itertools import chain
import math
import pandas as pd
import torch
import itertools

# Parameters for standard configuration
NT = 400  # total number of TDR facilities available
ST = 0.975 # safety level threshold
M = 10**6  # total demands threshold
L = 1000  # Lambda: scaling factor for total number of facilities opened
A = 10  # Alpha: maximum number of neighborhoods that a neighborhood can serve
B = 10  # Beta: maximum number of neighborhoods that a neighborhood can be served by
G = 15  # Gamma: penalty for unit unmet demand
k = ["Water", "Food", "Medical Kit"]  # commodities type
v = dict(zip(k, [1, 1, 0.5]))  # unit volume of commodity k
w = dict(zip(k, [1, 0.25, 0.25]))  # unit weight of commodity k
cd = dict(zip(k, [2.5, 3, 0.333]))  # commodities demand of an injured in 1day
V = 34560*5 # Volume Capacity of TDR
W = 30480 # Weight Capacity of TDR


# Reading the File
n = [] # maximum number of TDR facilities that can be allocated in neighborhood i
s = [] # safety level of neighborhood i
elite_Chromosom = []
elites_fitness = math.inf
bad_chromosom_index = 0
x_nodes = []
y_nodes = []
No_nodes = 0
No_supplierNodes = 0
No_demandNodes = 0
supplier_nodes_index = []
demand_nodes_index = []
v_capacity = []
w_capacity = []
v_demands = []
w_demands = []
dist = [] # distance between neighborhoods i and j

def readfile_torch(path: str):
    global No_nodes, No_supplierNodes, No_demandNodes, supplier_nodes_index, demand_nodes_index, n, v_demands
    with open(path, "r", encoding='utf-8') as file:
        lines = file.readlines()
    # print(lines)
    for i, line in enumerate(lines, start=1):
        if 2 < i:
            NodesAttributes = line.split(",")
            x_nodes.append(float(NodesAttributes[7]))
            y_nodes.append(float(NodesAttributes[8]))
            population = int(NodesAttributes[4])
            No_facility = int(NodesAttributes[6])
            n.append(No_facility)
            s.append(float(NodesAttributes[5]))
            v_capacity.append(V*(No_facility))
            w_capacity.append(W*No_facility)
            v_demands.append(sum([x * y * population for x, y in zip(list(v.values()), list(cd.values()))]))
            w_demands.append(sum([x * y * population for x, y in zip(list(w.values()), list(cd.values()))]))
    print("x_nodes = ", x_nodes, "\ny_nodes = ", y_nodes, "\nv_capacity = ",
        v_capacity, "\nw_capacity = ", w_capacity, "\nv_demands = ",
        v_demands, "\nw_demands =", w_demands, "\nSafty Levels = ", s)
    
    print("\nSumVolumeCapacities = ", sum(v_capacity), "\nSumWeightCapacities = ", sum(w_capacity),
          "\nSumVolumeDemands = ", sum(v_demands), "\nSumWeightDemands = ", sum(w_demands))

    No_nodes = len(x_nodes)
    supplier_nodes_index = [index for index, s in enumerate(s) if s >= ST]
    demand_nodes_index = [index for index, s in enumerate(s) if s < ST]
    No_supplierNodes = len(supplier_nodes_index)
    No_demandNodes = len(demand_nodes_index)
    print("\nNum of Nodes = ",No_nodes,
          "\nNum of Spplier Nodes = ", No_supplierNodes,
          "\nNum of demand Nodes = ", No_demandNodes,
          "\nSupplier Nodes Index = ", supplier_nodes_index,
          "\nDemand Nodes Index = ", demand_nodes_index)



readfile_torch("/kaggle/input/neighborhood-data/Neighborhood_Data.txt")




def DistMatrix_torch():
    coords = torch.stack((torch.tensor(x_nodes), torch.tensor(y_nodes)), dim=1)
    distmatrix = torch.cdist(coords, coords, p=2).round(decimals=2)
    return distmatrix.numpy()  # در صورت نیاز به numpy array

distmatrix = DistMatrix_torch()
print(distmatrix)



# Initial Population
def generate_initial_population_torch(pop_size):
    Population = torch.empty((pop_size, No_nodes), dtype=torch.int32)

    for idx in range(pop_size):
        Chromosom = torch.empty(No_nodes, dtype=torch.int32)
        selectable = torch.tensor(demand_nodes_index).clone()
        
        # پر کردن گره‌های تقاضا
        permuted_indices = torch.randperm(len(selectable))
        Chromosom[:len(selectable)] = selectable[permuted_indices]
        
        # پر کردن گره‌های تأمین
        for i, supplier in enumerate(supplier_nodes_index, start=len(demand_nodes_index)):
            Chromosom[i] = torch.randint(min(n), n[supplier] + 1, (1,)).item()

        Population[idx] = Chromosom

    return Population.numpy()  # اگر نیاز به numpy باشد

Population = generate_initial_population_torch(500)



# Fitness
def distance_torch(pop: torch.Tensor):
    distance_fitness = []
    
    # تقاضای هر تأمین‌کننده
    d_supp = torch.tensor([v_demands[s] for s in supplier_nodes_index], dtype=torch.float32)
    
    # ظرفیت هر تأمین‌کننده پس از تخصیص
    c_supp = torch.tensor([i * V for i in pop[No_demandNodes:]], dtype=torch.float32) - d_supp
    
    # تقاضای گره‌های تقاضا
    d = torch.tensor([v_demands[i] for i in demand_nodes_index], dtype=torch.float32)
    
    j = 0
    for i in range(No_supplierNodes):
        while c_supp[i] >= d[j]:
            distance_fitness.append(distmatrix[supplier_nodes_index[i], pop[j]].item())
            c_supp[i] -= d[j]
            d[j] = 0
            j += 1
            if j == No_demandNodes:
                return sum(distance_fitness)
        
        d[j] -= c_supp[i]
        c_supp[i] = 0

    return sum(distance_fitness)
def fitness_torch(population: torch.Tensor):
    global elite_Chromosom, elites_fitness, bad_chromosom_index
    
    # محاسبه مجموع امکانات و ظرفیت کل
    facilities_fitness = population[:, No_demandNodes:].sum(1)
    total_capacity = (population[:, No_demandNodes:] * V).sum(1)
    
    # محاسبه تقاضای برآورده نشده
    facilities_fitness = torch.tensor(facilities_fitness)
    total_capacity = torch.tensor(total_capacity)
    unmet_demand = torch.clamp(torch.tensor(sum(v_demands)) - total_capacity, min=0)
    
    # محاسبه تناسب فاصله
    distance_fitness_values = torch.tensor([distance_torch(pop) for pop in population])
    
    # محاسبه مقدار نهایی Fitness
    Fitness = facilities_fitness * L + unmet_demand * G + distance_fitness_values
    
    # پیدا کردن کروموزوم بد و بهترین کروموزوم نخبه
    bad_chromosom_index = torch.argmax(Fitness).item()
    
    if Fitness.min().item() < elites_fitness:
        elites_fitness = Fitness.min().item()
        elite_Chromosom = population[torch.argmin(Fitness)].tolist()
    
    return Fitness.tolist()



# SRS Selection
def compute_probabilities(K):
    Ranking = torch.arange(1, K + 1, dtype=torch.float32)
    ChromosomsProb = torch.where(
        Ranking <= K / 2, 
        (12 * Ranking) / (5 * K * (K + 2)), 
        (28 * Ranking) / (5 * K * (3 * K + 2))
    )
    ProbsRange = torch.cumsum(ChromosomsProb, 0)
    return ProbsRange
def SRS_Selection_torch(ChromosomsFitness: list):
    K = len(ChromosomsFitness)
    ProbsRange = compute_probabilities(K)

    # مرتب‌سازی براساس Fitness
    ChromosomsFitnessSorted, ChromosomsIndexByFitness = torch.sort(torch.tensor(ChromosomsFitness), descending=True)
    
    # تولید مقادیر تصادفی و انتخاب کروموزوم‌ها
    r = torch.rand(K)
    selected_indices = torch.searchsorted(ProbsRange, r)
    
    # تشکیل زوج‌های کروموزومی برای Crossover
    SelectedChromosomForCrossOver = selected_indices.reshape(-1, 2).tolist()

    return [[ChromosomsIndexByFitness[i].item(), ChromosomsIndexByFitness[j].item()] for i, j in SelectedChromosomForCrossOver]




# New Crossover
def New_Crossover_torch(selectedcrossover):
    childs = []
    for i1, i2 in selectedcrossover:
        if torch.rand(1).item() <= 0.8:  # احتمال اجرای کراس‌اور

            for p1, p2 in [(Population[i1], Population[i2]), (Population[i2], Population[i1])]:
                p1 = torch.tensor(p1).clone().detach()  # تبدیل به `torch.Tensor`
                p2 = torch.tensor(p2).clone().detach()

                child = torch.full((No_demandNodes,), -1, dtype=torch.int32)  # مقداردهی اولیه

                # انتخاب دو نقطه تصادفی برای کراس‌اور
                rn1 = torch.randint(1, No_demandNodes // 3, (1,)).item()
                rn2 = torch.randint(2 * No_demandNodes // 3, No_demandNodes - 2, (1,)).item()

                # پیدا کردن المان‌های مشترک بین `p1` و `p2` در محدوده انتخابی
                mask = torch.isin(p1[rn1:rn2+1], p2[rn1:rn2+1])
                joint = torch.arange(rn1, rn2+1)[mask.nonzero(as_tuple=True)[0]]

                # مقداردهی `child` با مقادیر مشترک
                child[joint] = p1[joint].to(torch.int32)

                # مقداردهی همسایه‌های مشترک در صورتی که در `child` مقدار نداشته باشند
                for j in joint:
                    if child[j - 1] == -1 and p2[j - 1] not in child:
                        child[j - 1] = p2[j - 1]
                    if child[j + 1] == -1 and p2[j + 1] not in child:
                        child[j + 1] = p2[j + 1]

                # مقداردهی بقیه قسمت‌های `child`
                c = 0
                for index, l in enumerate(child):
                    if l == -1:
                        # print(f"index = {index} && c = {c}")
                        while p2[c] in child:
                            c += 1
                            if c == No_nodes:
                                break
                        if c == No_nodes:
                                break
                        child[index] = p2[c]
                        c += 1
                child = torch.cat([child, torch.tensor(p1[No_demandNodes:], dtype=torch.int32)], dim=0)  # استفاده از `torch.cat`
                childs.append(child.tolist())
        else:
            # اگر کراس‌اور انجام نشود، والدین را بدون تغییر اضافه کن
            childs.extend([Population[i1], Population[i2]])

    return childs






# Mutation
def Mutation_torch(childs: torch.Tensor):
    mutation_mask = torch.rand(len(childs)) <= 0.2  # انتخاب کروموزوم‌هایی که جهش خواهند داشت
    mutated_childs = childs.clone()  # ایجاد یک کپی از `childs` برای جلوگیری از تغییر مستقیم داده‌ها
    
    for i, mutate in enumerate(mutation_mask):
        if mutate:
            rnd = torch.randint(0, No_demandNodes, (2,))
            mutated_childs[i, rnd[0]], mutated_childs[i, rnd[1]] = mutated_childs[i, rnd[1]], mutated_childs[i, rnd[0]]
    
    return mutated_childs



def Generation_torch(repeat: int):
    global Population
    g = torch.arange(1, repeat + 1)  # برداری‌سازی اعداد ۱ تا `repeat`
    minfit = torch.empty(repeat)  # ذخیره حداقل مقدار `fitness` در هر نسل
    
    for i in range(repeat):
        
        Fitness = torch.tensor(fitness_torch(Population))
        # ثبت مقدار کمترین `fitness`
        minfit[i] = min(Fitness) if min(Fitness) <= elites_fitness else elites_fitness

        # انتخاب کروموزوم‌ها برای `crossover`
        SelectedChromosoms = SRS_Selection_torch(Fitness.tolist())

        # تولید نسل جدید
        Childs = New_Crossover_torch(SelectedChromosoms)
        print(Childs)
        childsaftermutation = Mutation_torch(torch.tensor(Childs))

        # بروزرسانی `Population`
        Population = childsaftermutation.clone()
        Population[bad_chromosom_index] = torch.tensor(elite_Chromosom)
    
    # پیدا کردن بهترین راه‌حل
    best_fitness_values = torch.tensor(fitness_torch(Population))
    best_solution = Population[torch.argmin(best_fitness_values)]
    
    
    plt.plot(g.numpy(), minfit.numpy())
    plt.title(f'min_fitness = {minfit[-1].item()}')
    plt.show()
    return best_solution

# اجرای تابع برای ۶۰۰۰ نسل
best_solution = Generation_torch(6000)

# نمایش بهترین راه‌حل و اطلاعات تکمیلی
print("Best Solution =", best_solution.numpy())
print("\nSumDemand =", sum(v_demands),
      "\nSumSupply =", sum(V * n for n in best_solution[No_demandNodes:].tolist()))
print("\nTotal Facilities Opened =", sum(best_solution[No_demandNodes:].tolist()),
      "\nUnmet Demand =", sum(v_demands) - sum(V * n for n in best_solution[No_demandNodes:].tolist()),
      "\nTotal Distance =", distance_torch(best_solution.tolist()))
