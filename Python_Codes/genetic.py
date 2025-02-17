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


# Parameters of Reinforcement Learning
cRange = np.array(range(1, 10))/10
mRange = np.array(range(1, 10))/10
alpha = 0.7
gamma = 0.1
epsilon = 0.3
c = 0.8
m = 0.2
agent = rlagent(alpha, gamma, epsilon, cRange, mRange, c, m)



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
def readfile(path: str):
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
            if (i <= len(demand_nodes_index)-1):
                j = rn.choice(selectable)
                selectable.remove(j)
            else:
                j = rn.choice(range(min(n),n[supplier_nodes_index[i-len(demand_nodes_index)]]+1))
            Chromosom.append(j)

        Population.append(Chromosom)
    return Population
Population = generate_initial_population(300)
# print(Population)





# Fitness
def distance(pop: list):
    distance_fitness = []
    d_supp = np.array([v_demands[s] for s in supplier_nodes_index])
    c_supp = np.array([i*V  for i in pop[No_demandNodes:]])- d_supp
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
    return sum(distance_fitness)
def fitness(population: list):
    global elite_Chromosom, elites_fitness
    Fitness = []
    for pop in population:
        
        sum_facilities_fitness = sum(pop[No_demandNodes: len(pop)])
        
        d = sum(v_demands)
        c = sum([V*i for i in pop[No_demandNodes:]])
        if d-c > 0:
            unmet_demand_fitness = (d-c)
        else:
            unmet_demand_fitness = 0
        
        distance_fitness = distance(pop)

            
        Fitness.append(sum_facilities_fitness*L + unmet_demand_fitness*G + distance_fitness)
    bad_chromosom_index = Population.index(Population[Fitness.index(max(Fitness))])
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
def SRS_Selection(ChromosomsFitness: list):
    selectedchromosomforcrossover = []
    SelectedChromosomForCrossOver = []
    # if elites_fitness not in ChromosomsFitness: 
    #     Population[rn.randint(0,len(Population)-1)] = elite_Chromosom
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
    # print(f"SelectedChromosomForCrossOver length = {len(SelectedChromosomForCrossOver)}")
    return SelectedChromosomForCrossOver
# selectedChromosom = SRS_Selection(Fitness)
# print(selectedChromosom)




# PMX Crossover
def PMX_Crossover(selectedcrossover, crossoverprob):
    childs = []
    for i in selectedcrossover:
        r = rn.uniform(0,1)
        if r <= crossoverprob:
            p1 = Population[i[0]]
            p2 = Population[i[1]]
            child1 = [1]*len(p1)
            child2 = [1]*len(p1)
            crossoverindex = rn.sample(list(range(len(demand_nodes_index))),2)
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
                
                
            for index, i in enumerate(p2[len(demand_nodes_index):]):
                child1[len(demand_nodes_index)+index] = i
                child2[len(demand_nodes_index)+index] = p1[len(demand_nodes_index)+index]
                
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
            rnd = rn.sample(list(range(len(demand_nodes_index))),2)
            rnd0 = i[rnd[0]]
            rnd1 = i[rnd[1]]
            i[rnd[0]] = rnd1
            i[rnd[1]] = rnd0
            childsaftermutation.append(i)
            
            # rnd = rn.sample(list(range(len(demand_nodes_index), len(i))),2)
            # rnd0 = i[rnd[0]]
            # rnd1 = i[rnd[1]]
            # if n[rnd[1]] > rnd0 and n[rnd[0]] > rnd1:
            #     i[rnd[0]] = rnd1
            #     i[rnd[1]] = rnd0
            # childsaftermutation.append(i)
        else:
            childsaftermutation.append(i)
    return childsaftermutation
# mutation = Mutation(childs)
# print(mutation)



# Generation
def Generation(repeat):
    global Population
    g = range(1,repeat+1)
    minfit = []
    for i in range(repeat):
        
        if i == 0:
            c, m = agent.initAction()
        else:
            c, m = agent.decide()
            # print("Crossover Prob = ", c, "Mutation prob = " ,m)
        
        Fitness = fitness(Population)
        agent.observe(Population, Fitness)
        agent.updateQlearning()
        minfit.append(min(Fitness) if (min(Fitness) <= elites_fitness) else elites_fitness)
        SelectedChromosoms = SRS_Selection(Fitness)
        Childs = PMX_Crossover(SelectedChromosoms, c)
        childsaftermutation = Mutation(Childs, m)
        Population = childsaftermutation
        Population[bad_chromosom_index] = elite_Chromosom
    # Chromosom_Plot(InitialChromosoms[Fitness(InitialChromosoms).index(min(Fitness(InitialChromosoms)))])
    best_solution = Population[fitness(Population).index(min(fitness(Population)))]
    
    df = pd.DataFrame(agent.Q)
    # تغییر تنظیمات برای نمایش همه سطرها و ستون‌ها
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # نمایش DataFrame
    print(df)
    
    plt.plot(g,minfit)
    plt.title(f'min_fitness = {minfit[-1]}')
    plt.figure(1)
    plt.show()
    # agent.df_normalplot(agent.df_list)
    agent.diversity_plot(repeat,agent.div_list)
    return best_solution
best_solution = Generation(4000)
print("Best Solution = ",best_solution)
print("\nSumDemand = ", sum(v_demands), "\nSumSupply = ", sum(V*n for n in best_solution[No_demandNodes:]))
print("\nTotal Facilities opened = ",sum(best_solution[No_demandNodes:]),
      "\nUnmet Demand = ", sum(v_demands)-sum(V*n for n in best_solution[No_demandNodes:]),
      "\nTotal Distance = ",distance(best_solution))



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
    c_supp = np.array([i*V  for i in best_solution[No_demandNodes:]]) - d_supp
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
    plt.figure(2)
    plt.show()
chromosomplot(best_solution)    