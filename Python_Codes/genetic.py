from RLagent import rlagent
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.markers as mks
import numpy as np
import pprint
import random as rn
from itertools import chain
import math
import pandas as pd
import copy


# Parameters of Reinforcement Learning
cRange = np.array(range(1, 10))/10
mRange = np.array(range(1, 10))/10
alpha = 0.7
gamma = 0.1
epsilon = 0.3
c = 0.7
m = 0.3
agent = rlagent(alpha, gamma, epsilon, cRange, mRange, c, m)



# Parameters for standard configuration
APR = 0.9
NT = 100 # total number of TDR facilities available
ST = 0.95 # safety level threshold
M = 10**6 # total demands threshold
L = 5000 # Lambda: scaling factor for total number of facilities opened
A = 10 # Alpha: maximum number of neighborhoods that a neighborhood can serve
B = 10 # Beta: maximum number of neighborhoods that a neighborhood can be served by
G = 15 # Gamma: penalty for unit unmet demand
k = ["Water", "Food", "Medical Kit"]  # commodities type
v = dict(zip(k, [1, 1, 0.5]))  # unit volume of commodity k
w = dict(zip(k, [1, 0.25, 0.25]))  # unit weight of commodity k
cd = dict(zip(k, [0.833, 1, 0.111]))  # commodities demand of an injured in 1day
V = 120000 # Volume Capacity of TDR
W = 120000 # Weight Capacity of TDR


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
supplier_nodes_name = []
demand_nodes_index = []
v_capacity = []
w_capacity = []
v_demands = []
v_demands_water = []
v_demands_food = []
v_demands_medicalkit = []
w_demands = []
dist = [] # distance between neighborhoods i and j
genetic_best_solution = np.nan
Allocation_Clusters = {}

def readfile(path: str):
    global No_nodes, No_supplierNodes, No_demandNodes, supplier_nodes_index, demand_nodes_index, n, v_demands
    with open(path, "r", encoding='utf-8') as file:
        lines = file.readlines()
    # print(lines)
    for i, line in enumerate(lines, start=1):
        if 2 < i:
            NodesAttributes = line.split(",")
            supplier_nodes_name.append(NodesAttributes[1])
            x_nodes.append(float(NodesAttributes[7]))
            y_nodes.append(float(NodesAttributes[8]))
            population = int(NodesAttributes[4])*APR
            No_facility = int(NodesAttributes[6])
            n.append(No_facility)
            s.append(float(NodesAttributes[5]))
            v_capacity.append(V*(No_facility))
            w_capacity.append(W*No_facility)
            v_demands.append(sum([x * y * population for x, y in zip(list(v.values()), list(cd.values()))]))
            v_demands_water.append(v['Water'] * cd['Water'] * population)
            v_demands_food.append(v['Food'] * cd['Food'] * population)
            v_demands_medicalkit.append(v['Medical Kit'] * cd['Medical Kit'] * population)
            w_demands.append(sum([x * y * population for x, y in zip(list(w.values()), list(cd.values()))]))
            
    No_nodes = len(x_nodes)
    supplier_nodes_index = [index for index, s in enumerate(s) if s >= ST]
    demand_nodes_index = [index for index, s in enumerate(s) if s < ST]
    No_supplierNodes = len(supplier_nodes_index)
    No_demandNodes = len(demand_nodes_index)
    
    print("x_nodes = ", x_nodes, "\n\ny_nodes = ", y_nodes, "\n\nv_capacity = ",
        v_capacity, "\n\nw_capacity = ", w_capacity, "\n\nv_demands = ",
        v_demands, "\n\nw_demands =", w_demands, "\n\nSafty Levels = ", s)
    
    print("\nSumVolumeCapacities = ", sum(v_capacity[i] for i in supplier_nodes_index), "\nSumWeightCapacities = ", sum(w_capacity[i] for i in supplier_nodes_index),
          "\nSumVolumeDemands = ", sum(v_demands), "\nSumWeightDemands = ", sum(w_demands))
    
    print("\nNum of Nodes = ",No_nodes,
          "\nNum of Spplier Nodes = ", No_supplierNodes,
          "\nNum of demand Nodes = ", No_demandNodes,
          "\nSum of Facilities = ", sum([n[i] for i in supplier_nodes_index]),
          "\nSupplier Nodes Index = ", supplier_nodes_index,
          "\nDemand Nodes Index = ", demand_nodes_index)
    

readfile("C:\\Users\\Amir\\Desktop\\SDSS-Project\\Data\\Neighborhood_Data.txt")    





def DistMatrix():
    distmatrix = np.empty((No_nodes, No_nodes), dtype=float)
    for i in range(No_nodes):
        for j in range(No_nodes):
            distmatrix[i][j] = round(math.sqrt( pow(x_nodes[i]-x_nodes[j],2) + pow(y_nodes[i]-y_nodes[j],2)), 2)
    return distmatrix
distmatrix = DistMatrix()
pprint.pprint(distmatrix)



# Initial Population
def generate_initial_population(pop_size):
    Population = []
    for ـ in range(pop_size):
        Chromosom = []
        selectable = demand_nodes_index.copy()
        for i in range(No_nodes):
            if (i <= No_demandNodes-1):
                j = rn.choice(selectable)
                selectable.remove(j)
            else:
                j = rn.choice(range(0,n[supplier_nodes_index[i-No_demandNodes]]+1))
            Chromosom.append(j)

        Population.append(Chromosom)
    return Population
Population = generate_initial_population(200)
# print(Population)





# Fitness
def distance(pop: list):
    distance_fitness = []
    d_supp = np.array([v_demands[s] for s in supplier_nodes_index])
    c_supp = np.array([i*V  for i in pop[No_demandNodes:]]) - d_supp
    d = np.array([v_demands[i] for i in demand_nodes_index])
    j = 0
    end = False
    for i in range(No_supplierNodes):
        while c_supp[i] >= d[j]:
            distance_fitness.append(distmatrix[supplier_nodes_index[i], pop[j]])
            c_supp[i] -= d[j]
            d[j] = 0
            j += 1
            if j == No_demandNodes:
                end = True
                break
        if end == True:
            break
        d[j] -= c_supp[i]
        c_supp[i] = 0
        distance_fitness.append(distmatrix[supplier_nodes_index[i], pop[j]])
    return sum(distance_fitness)
def fitness(population: list):
    global elite_Chromosom, elites_fitness, bad_chromosom_index
    if len(np.array(population).shape) == 1:
        population = [population]
    Fitness = []
    for pop in population:
        
        sum_facilities_fitness = sum(pop[No_demandNodes:])
        
        d = sum(v_demands)
        c = sum([V*i for i in pop[No_demandNodes:]])
        
        # unmet_demand_fitness = abs(d-c)
        
        if d-c > 0:
            penalty = (d-c)/10
            unmet_demand_fitness = (d-c)
        else:
            unmet_demand_fitness = 0
            penalty = abs(d-c)/10
        
        distance_fitness = distance(pop)
         
        Fitness.append(sum_facilities_fitness*L + unmet_demand_fitness*G + penalty + distance_fitness*10)
    bad_chromosom_index = Fitness.index(max(Fitness))
    if min(Fitness) < elites_fitness:
        elites_fitness = min(Fitness)
        elite_Chromosom = Population[Fitness.index(min(Fitness))]
    return Fitness
# Fitness = fitness(Population)
# print(Fitness)



# SRS Selection
Ranking = range(1,len(Population)+1)
K = len(Ranking)
ChromosomsProb =[]
ProbsRange = []
# for i in Ranking:
#     if i <= K/2:
#         ChromosomsProb.append((12*i)/(5*K*(K+2)))
#     else:
#         ChromosomsProb.append((28*i)/(5*K*(3*K+2)))
# for index, i in enumerate(ChromosomsProb):
#     if index == 0:
#         ProbsRange.append([0,i])
#     else:
#         ProbsRange.append([ProbsRange[index-1][1], ProbsRange[index-1][1] + (i)])
def SRS_Selection(ChromosomsFitness: list, l1, l2):
    ChromosomsProb =[]
    ProbsRange = []
    for i in Ranking:
        if i <= K/2:
            ChromosomsProb.append(l1*(8*i)/(K*(K+2)))
        else:
            ChromosomsProb.append(l2*(8*i)/(K*(3*K+2)))
    for index, i in enumerate(ChromosomsProb):
        if index == 0:
            ProbsRange.append([0,i])
        else:
            ProbsRange.append([ProbsRange[index-1][1], ProbsRange[index-1][1] + (i)])
    selectedchromosomforcrossover = []
    SelectedChromosomForCrossOver = []
    ChromosomsFitnessSorted = sorted(ChromosomsFitness, reverse=True)
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
# selectedChromosom = SRS_Selection(Fitness)
# print(selectedChromosom)


# Tournament Selection
def TournamentSelection(ChromosomsFitness: list):
    ps = len(ChromosomsFitness)//2
    TournamentSize = 10
    SelectedChromosomForCrossOver = []
    selectedchromosomforcrossover = []
    for _ in range(ps):
        for i in range(2):
            tournament = [rn.choice(range(ps)) for _ in range(TournamentSize)]
            tournamentFitness = [ChromosomsFitness[i] for i in tournament]
            if rn.uniform(0, 1) < 0.7:
                winner = tournament[tournamentFitness.index(max(tournamentFitness))]
            else:
                winner = tournament[tournamentFitness.index(rn.choice(tournamentFitness))]
            selectedchromosomforcrossover.append(winner)
        SelectedChromosomForCrossOver.append(selectedchromosomforcrossover)
        selectedchromosomforcrossover = []
    return SelectedChromosomForCrossOver




# New Crossover
def New_Crossover(selectedcrossover, crossoverprob):
    childs = []
    for i1, i2 in selectedcrossover:
        if rn.uniform(0, 1) <= crossoverprob:
            for p1, p2 in [(Population[i1], Population[i2]), (Population[i2], Population[i1])]:
                child = [-1] * No_demandNodes
                rn1, rn2 = rn.randint(1, No_demandNodes // 2), rn.randint(No_demandNodes // 2, No_demandNodes - 2)
                
                joint = [r for r in range(rn1, rn2 + 1) if p1[r] in p2[rn1:rn2 + 1]]
                for r in joint:
                    child[r] = p1[r]
                
                for j in joint:
                    if child[j-1] == -1 and p2[j-1] not in child:
                        child[j-1] = p2[j-1]
                    if child[j+1] == -1 and p2[j+1] not in child:
                        child[j+1] = p2[j+1]

                c = 0
                for index, l in enumerate(child):
                    if l == -1:
                        while p2[c] in child:
                            c += 1
                        child[index] = p2[c]
                        c += 1
                childs.append(child + p1[No_demandNodes:])
        else:
            childs.extend([Population[i1], Population[i2]])

    return childs


# Partial Mapped Crossover
def PMX_Crossover(selectedcrossover, p):
    childs = []
    for i in selectedcrossover:
        r = rn.uniform(0,1)
        if r <= p:
            p1 = Population[i[0]]
            p2 = Population[i[1]]
            child1 = [1]*len(p1)
            child2 = [1]*len(p1)
            crossoverindex = rn.sample(list(range(No_demandNodes)),2)
            bp1 = p1[min(crossoverindex):max(crossoverindex)+1]
            bp2 = p2[min(crossoverindex):max(crossoverindex)+1]
            child1[min(crossoverindex):max(crossoverindex)+1] = bp2.copy()
            child2[min(crossoverindex):max(crossoverindex)+1] = bp1.copy()
            Chain = chain(range(min(crossoverindex)), range(max(crossoverindex)+1, len(child1)))
            for j in Chain:
                chekpoint1 = p1[j]
                chekpoint2 = p2[j]
                m = 0
                ch = False
                while chekpoint1 in bp2:
                    m += 1
                    chekpoint1 = bp1[bp2.index(chekpoint1)]
                    if m > len(bp2):
                        ch = True
                        child1[j] = p2[j]
                        break
                if ch == False:
                    child1[j] = chekpoint1
                m = 0
                ch = False
                while chekpoint2 in bp1:
                    m += 1
                    chekpoint2 = bp2[bp1.index(chekpoint2)]
                    if m > len(bp1):
                        ch = True
                        child2[j] = p1[j]
                        break
                if ch == False:
                    child2[j] = chekpoint2
                
                
            for index, i in enumerate(p2[No_demandNodes:]):
                child1[No_demandNodes+index] = i
                child2[No_demandNodes+index] = p1[No_demandNodes+index]
                
            childs.append(child1)
            childs.append(child2)
            
        else:
            childs.append(Population[i[0]])
            childs.append(Population[i[1]])
    return childs




# Mutation
def Mutation(childs: list, mutationprob):
    childsaftermutation = []
    for i in childs:
        r = rn.uniform(0,1)
        if r <= mutationprob:
            rnd = rn.sample(list(range(No_demandNodes)),2)
            rnd0 = i[rnd[0]]
            rnd1 = i[rnd[1]]
            i[rnd[0]] = rnd1
            i[rnd[1]] = rnd0
            childsaftermutation.append(i)
        else:
            childsaftermutation.append(i)
    return childsaftermutation
# mutation = Mutation(childs)
# print(mutation)



# Generation
def Generation(repeat):
    global Population, genetic_best_solution
    g = range(1,repeat+1)
    minfit = []
    for i in range(repeat):
        
        # if i == 0:
        #     c, m = agent.initAction()
        # else:
        #     c, m = agent.decide()
        #     print("Crossover Prob = ", c, "Mutation prob = " ,m)
        
        Fitness = fitness(Population)
        # agent.observe(Population, Fitness)
        # agent.updateQlearning()
        minfit.append(min(Fitness) if (min(Fitness) <= elites_fitness) else elites_fitness)
        if repeat < 50:
            l1 = 0.5
            l2 = 0.5
        else:
            l1 = 0.2
            l2 = 0.8
        SelectedChromosoms = SRS_Selection(Fitness, l1 ,l2)
        Childs = New_Crossover(SelectedChromosoms, 0.95)
        childsaftermutation = Mutation(Childs, 0.2)
        Population = childsaftermutation
        Population[bad_chromosom_index] = elite_Chromosom
    # Chromosom_Plot(InitialChromosoms[Fitness(InitialChromosoms).index(min(Fitness(InitialChromosoms)))])
    best_solution = Population[fitness(Population).index(min(fitness(Population)))]
    
    # df = pd.DataFrame(agent.Q)
    # تغییر تنظیمات برای نمایش همه سطرها و ستون‌ها
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # نمایش DataFrame
    # print(df)
    
    plt.plot(g,minfit)
    plt.title(f'min_fitness = {minfit[-1]}')
    plt.show()
    # agent.df_normalplot(agent.df_list)
    # agent.diversity_plot(repeat,agent.div_list)
    return best_solution
genetic_best_solution = Generation(20)
print("Best Solution = ",genetic_best_solution)
print("\nSumDemand = ", sum(v_demands), "\nSumSupply = ", sum(V*n for n in genetic_best_solution[No_demandNodes:]))
print("\nTotal Facilities opened = ",sum(genetic_best_solution[No_demandNodes:]),
      "\nUnmet Demand = ", sum(v_demands)-sum(V*n for n in genetic_best_solution[No_demandNodes:]),
      "\nTotal Distance = ",distance(genetic_best_solution))



def chromosomplot(bestsolution):
# display shp file
    shapefile_path = "C:\\Users\\Amir\\Desktop\\SDSS-Project\\Data\\Mahallat\\Reg1_3_4.shp"
    polygons = gpd.read_file(shapefile_path)
    # print(polygons.head())
    polygons.plot(edgecolor='black', facecolor='lightblue', figsize=(10, 8))
    plt.title("Polygon Map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
# display nodes
    x_d = []
    y_d = []
    x_s = []
    y_s = []
    for i in demand_nodes_index:
        x_d.append(x_nodes[i])
        y_d.append(y_nodes[i])
    plt.scatter(x_d, y_d, s=5)
    for i in supplier_nodes_index:
        x_s.append(x_nodes[i])
        y_s.append(y_nodes[i])
    plt.scatter(x_s, y_s, c='r', s=15, marker='s')
    
    for n, txt in enumerate(range(No_nodes)):
        plt.text(x_nodes[n], y_nodes[n], n, fontsize=6, ha='right', va='bottom')
    
# display allocating nodes
    d_supp = np.array([v_demands[s] for s in supplier_nodes_index])
    c_supp = np.array([i*V  for i in bestsolution[No_demandNodes:]]) - d_supp
    d = np.array([v_demands[i] for i in demand_nodes_index])
    j = 0
    end = False
    for i in range(No_supplierNodes):
        x = []
        y = []
        # print(f"c_supp = {c_supp} \n d = {d} \n")
        while c_supp[i] >= d[j]:
            c_supp[i] -= d[j]
            d[j] = 0
            # print(f"c_supp = {c_supp} \n d = {d} \n")
            x.append(x_nodes[supplier_nodes_index[i]])
            y.append(y_nodes[supplier_nodes_index[i]])
            x.append(x_nodes[bestsolution[j]])
            y.append(y_nodes[bestsolution[j]])
            plt.plot(x,y)
            x= []
            y = []
            j += 1
            if j == No_demandNodes:
                end = True
                break
        if end == True:
            break
        d[j] -= c_supp[i]
        c_supp[i] = 0
        # print(f"c_supp = {c_supp} \n d = {d} \n")
        x.append(x_nodes[supplier_nodes_index[i]])
        y.append(y_nodes[supplier_nodes_index[i]])
        x.append(x_nodes[bestsolution[j]])
        y.append(y_nodes[bestsolution[j]])
        plt.plot(x,y)
    plt.show()
    
    
 
 
 
def generate_neighbors(current_solution):
    neighbors = []
    moves = []  # برای ذخیره حرکت‌ها

    # حرکت نوع 1: جابجایی در بخش اول
    for _ in range(No_demandNodes):  # تولید ۱۰ همسایه
        neighbor = copy.deepcopy(current_solution)
        i, j = rn.sample(range(No_demandNodes), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        neighbors.append(neighbor)
        moves.append(("swap", i, j))

    # حرکت نوع 2: تغییر تعداد تسهیلات
    for k in range(No_supplierNodes):
        for change in [-1, 1]:
            neighbor = copy.deepcopy(current_solution)
            current_facilities = neighbor[No_demandNodes + k]
            new_facilities = current_facilities + change
            if 1 <= new_facilities <= n[supplier_nodes_index[k]]:
                neighbor[No_demandNodes + k] = new_facilities
                neighbors.append(neighbor)
                moves.append(("facility", k, change))

    return neighbors, moves


def tabu_search(initial_solution, max_iterations, tabu_tenure):
    current_solution = initial_solution
    best_solution = copy.deepcopy(current_solution)
    best_cost = fitness(best_solution)

    tabu_list = []  # لیست ممنوعه برای ذخیره حرکت‌ها
    bc = []
    for _ in range(max_iterations):
        neighbors, moves = generate_neighbors(current_solution)

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
        neighbors_costs = [fitness(n) for n in valid_neighbors]
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
            
    plt.plot(range(1,max_iterations+1),bc)
    plt.title(f'min_fitness = {best_cost}')
    plt.show()

    return best_solution, best_cost


def allocation_clusters(best_solution):
    Allocation_Clusters = {}
    d_supp = np.array([v_demands[s] for s in supplier_nodes_index])
    c_supp = np.array([i*V  for i in best_solution[No_demandNodes:]]) - d_supp
    d = np.array([v_demands[i] for i in demand_nodes_index])
    j = 0
    end = False
    for i in range(No_supplierNodes):
        dni = []
        while c_supp[i] >= d[j]:
            dni.append(best_solution[j])
            c_supp[i] -= d[j]
            d[j] = 0
            j += 1
            if j == No_demandNodes:
                end = True
                break
        if end == True:
            break
        dni.append(best_solution[j])
        Allocation_Clusters[f'{supplier_nodes_index[i]}'] = dni
        d[j] -= c_supp[i]
        c_supp[i] = 0
    return Allocation_Clusters


best_solution, best_cost = tabu_search(genetic_best_solution, 30, 10)
Allocation_Clusters = allocation_clusters(best_solution)


InventoryDesitions = {}
for index, i in enumerate(Allocation_Clusters.keys()):
    sum_vd_water = sum([v_demands_water[k] for k in Allocation_Clusters[i]])
    sum_vd_food = sum([v_demands_food[k] for k in Allocation_Clusters[i]])
    sum_vd_medicalkit = sum([v_demands_medicalkit[k] for k in Allocation_Clusters[i]])
    values = [best_solution[No_demandNodes + index], sum_vd_water, sum_vd_food, sum_vd_medicalkit]
    InventoryDesitions[supplier_nodes_name[int(i)]] = values

print(f"best_solution = {best_solution} \n best_cost = {best_cost}")
print("Best Solution = ",best_solution)
print("\nSumDemand = ", sum(v_demands), "\nSumSupply = ", sum(V*n for n in best_solution[No_demandNodes:]))
print("\nTotal Facilities opened = ",sum(best_solution[No_demandNodes:]),
      "\nUnmet Demand = ", sum(v_demands)-sum(V*n for n in best_solution[No_demandNodes:]),
      "\nTotal Distance = ",distance(best_solution))

 
chromosomplot(best_solution)    