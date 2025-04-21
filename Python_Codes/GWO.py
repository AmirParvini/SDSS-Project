import numpy as np
import random
import math
import matplotlib.pyplot as plt
import geopandas as gpd

# Parameters for standard configuration
NT = 400  # total number of TDR facilities available
ST = 0.975  # safety level threshold
M = 10**6  # total demands threshold
L = 5000  # Lambda: scaling factor for total number of facilities opened
A = 10  # Alpha: maximum number of neighborhoods that a neighborhood can serve
B = 10  # Beta: maximum number of neighborhoods that a neighborhood can be served by
G = 15  # Gamma: penalty for unit unmet demand
k = ["Water", "Food", "Medical Kit"]  # commodities type
v = dict(zip(k, [1, 1, 0.5]))  # unit volume of commodity k
w = dict(zip(k, [1, 0.25, 0.25]))  # unit weight of commodity k
cd = dict(zip(k, [2.5, 3, 0.333]))  # commodities demand of an injured in 1day
V = 120000  # Volume Capacity of TDR
W = 30480*5  # Weight Capacity of TDR

# Global variables
n = []  # maximum number of TDR facilities that can be allocated in neighborhood i
s = []  # safety level of neighborhood i
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
distmatrix = []

def readfile(path: str):
    global No_nodes, No_supplierNodes, No_demandNodes, supplier_nodes_index, demand_nodes_index, n, v_demands
    with open(path, "r", encoding='utf-8') as file:
        lines = file.readlines()
    for i, line in enumerate(lines, start=1):
        if 2 < i:
            NodesAttributes = line.split(",")
            x_nodes.append(float(NodesAttributes[7]))
            y_nodes.append(float(NodesAttributes[8]))
            population = int(NodesAttributes[4])
            No_facility = int(NodesAttributes[6])
            n.append(No_facility)
            s.append(float(NodesAttributes[5]))
            v_capacity.append(V * No_facility)
            w_capacity.append(W * No_facility)
            v_demands.append(sum([x * y * population for x, y in zip(list(v.values()), list(cd.values()))]))
            w_demands.append(sum([x * y * population for x, y in zip(list(w.values()), list(cd.values()))]))
    
    No_nodes = len(x_nodes)
    supplier_nodes_index = [index for index, s in enumerate(s) if s >= ST]
    demand_nodes_index = [index for index, s in enumerate(s) if s < ST]
    No_supplierNodes = len(supplier_nodes_index)
    No_demandNodes = len(demand_nodes_index)

def DistMatrix():
    distmatrix = np.empty((No_nodes, No_nodes), dtype=float)
    for i in range(No_nodes):
        for j in range(No_nodes):
            distmatrix[i][j] = round(math.sqrt(pow(x_nodes[i] - x_nodes[j], 2) + pow(y_nodes[i] - y_nodes[j], 2)), 2)
    return distmatrix

# Initial Population
def generate_initial_population(pop_size):
    Population = []
    for _ in range(pop_size):
        Chromosom = []
        selectable = demand_nodes_index.copy()
        for i in range(No_nodes):
            if i < No_demandNodes:
                j = random.choice(selectable)
                selectable.remove(j)
            else:
                j = random.choice(range(0, n[supplier_nodes_index[i - No_demandNodes]] + 1))
            Chromosom.append(j)
        Population.append(Chromosom)
    return Population

# Fitness Functions
def distance(pop: list):
    distance_fitness = []
    d_supp = np.array([v_demands[s] for s in supplier_nodes_index])
    c_supp = np.array([i * V for i in pop[No_demandNodes:]]) - d_supp
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
        if end:
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
        c = sum([V * i for i in pop[No_demandNodes:]])
        if d - c > 0:
            penalty = (d - c) / 10
            unmet_demand_fitness = (d - c)
        else:
            unmet_demand_fitness = 0
            penalty = abs(d - c) / 10
        distance_fitness = distance(pop)
        Fitness.append(sum_facilities_fitness * L + unmet_demand_fitness * G + penalty + distance_fitness * 10)
    bad_chromosom_index = Fitness.index(max(Fitness))
    if min(Fitness) < elites_fitness:
        elites_fitness = min(Fitness)
        elite_Chromosom = population[Fitness.index(min(Fitness))]
    return Fitness

# GWO Parameters
pop_size = 50
max_iter = 2000

# GWO Helper Functions
def order_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    start, end = sorted(random.sample(range(size), 2))
    child[start:end + 1] = parent1[start:end + 1]
    p2_idx = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent2[p2_idx]
            p2_idx += 1
    return child

def gwo_update_demand(wolf, alpha, beta, delta):
    """Update demand allocation using Order Crossover"""
    child1 = order_crossover(wolf[:No_demandNodes], alpha[:No_demandNodes])
    child2 = order_crossover(wolf[:No_demandNodes], beta[:No_demandNodes])
    child3 = order_crossover(wolf[:No_demandNodes], delta[:No_demandNodes])
    children = [child1, child2, child3]
    fitnesses = [fitness([child + wolf[No_demandNodes:]])[0] for child in children]
    best_child = children[np.argmin(fitnesses)]
    return best_child + wolf[No_demandNodes:]

def gwo_update_facility(wolf, alpha, beta, delta, a):
    """Update facility allocation"""
    facility_part = wolf[No_demandNodes:].copy()
    for i in range(No_supplierNodes):
        r1, r2 = random.random(), random.random()
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_alpha = abs(C1 * alpha[No_demandNodes + i] - wolf[No_demandNodes + i])
        X1 = alpha[No_demandNodes + i] - A1 * D_alpha

        r1, r2 = random.random(), random.random()
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = abs(C2 * beta[No_demandNodes + i] - wolf[No_demandNodes + i])
        X2 = beta[No_demandNodes + i] - A2 * D_beta

        r1, r2 = random.random(), random.random()
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_delta = abs(C3 * delta[No_demandNodes + i] - wolf[No_demandNodes + i])
        X3 = delta[No_demandNodes + i] - A3 * D_delta

        new_value = (X1 + X2 + X3) / 3
        facility_part[i] = min(max(round(new_value), 0), n[supplier_nodes_index[i]])
    return wolf[:No_demandNodes] + facility_part

# GWO Algorithm
def gwo_optimization():
    readfile("C:\\Users\\Amir\\Desktop\\SDSS-Project\\Data\\Neighborhood_Data.txt")
    global distmatrix
    distmatrix = DistMatrix()
    
    # Initialize population
    population = generate_initial_population(pop_size)
    fitness_values = fitness(population)
    sorted_indices = np.argsort(fitness_values)
    alpha = population[sorted_indices[0]].copy()
    beta = population[sorted_indices[1]].copy()
    delta = population[sorted_indices[2]].copy()
    best_fitness = fitness_values[sorted_indices[0]]
    fitness_history = [best_fitness]

    for iter in range(max_iter):
        a = 2 - 2 * iter / max_iter  # Linearly decrease 'a' from 2 to 0
        for i in range(pop_size):
            wolf = population[i].copy()
            wolf = gwo_update_demand(wolf, alpha, beta, delta)
            wolf = gwo_update_facility(wolf, alpha, beta, delta, a)
            population[i] = wolf

        fitness_values = fitness(population)
        sorted_indices = np.argsort(fitness_values)
        if fitness_values[sorted_indices[0]] < best_fitness:
            best_fitness = fitness_values[sorted_indices[0]]
            alpha = population[sorted_indices[0]].copy()
        beta = population[sorted_indices[1]].copy()
        delta = population[sorted_indices[2]].copy()
        fitness_history.append(best_fitness)

    # Plot fitness convergence
    plt.plot(range(max_iter + 1), fitness_history)
    plt.title(f'GWO Best Fitness = {best_fitness}')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.show()
    return alpha, best_fitness

def chromosomplot(bestsolution):
    shapefile_path = "C:\\Users\\Amir\\Desktop\\SDSS-Project\\Data\\Mahallat\\Reg1_3_4.shp"
    polygons = gpd.read_file(shapefile_path)
    polygons.plot(edgecolor='black', facecolor='lightblue', figsize=(10, 8))
    plt.title("Polygon Map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    x_d, y_d = [], []
    x_s, y_s = [], []
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
    
    d_supp = np.array([v_demands[s] for s in supplier_nodes_index])
    c_supp = np.array([i * V for i in bestsolution[No_demandNodes:]]) - d_supp
    d = np.array([v_demands[i] for i in demand_nodes_index])
    j = 0
    end = False
    for i in range(No_supplierNodes):
        x, y = [], []
        while c_supp[i] >= d[j]:
            c_supp[i] -= d[j]
            d[j] = 0
            x.append(x_nodes[supplier_nodes_index[i]])
            y.append(y_nodes[supplier_nodes_index[i]])
            x.append(x_nodes[bestsolution[j]])
            y.append(y_nodes[bestsolution[j]])
            plt.plot(x, y)
            x, y = [], []
            j += 1
            if j == No_demandNodes:
                end = True
                break
        if end:
            break
        d[j] -= c_supp[i]
        c_supp[i] = 0
        x.append(x_nodes[supplier_nodes_index[i]])
        y.append(y_nodes[supplier_nodes_index[i]])
        x.append(x_nodes[bestsolution[j]])
        y.append(y_nodes[bestsolution[j]])
        plt.plot(x, y)
    plt.show()

# Run GWO
gwo_best_solution, gwo_best_fitness = gwo_optimization()

print("GWO Best Solution:", gwo_best_solution)
print("Fitness:", gwo_best_fitness)
print("Sum Demand:", sum(v_demands))
print("Sum Supply:", sum(V * n for n in gwo_best_solution[No_demandNodes:]))
print("Total Facilities Opened:", sum(gwo_best_solution[No_demandNodes:]))
print("Unmet Demand:", sum(v_demands) - sum(V * n for n in gwo_best_solution[No_demandNodes:]))
print("Total Distance:", distance(gwo_best_solution))
chromosomplot(gwo_best_solution)