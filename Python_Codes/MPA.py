import numpy as np
import random
import math
import random as rn
import matplotlib.pyplot as plt
import geopandas as gpd

# Parameters for standard configuration
NT = 400  # total number of TDR facilities available
ST = 0.975 # safety level threshold
M = 10**6  # total demands threshold
L = 5000  # Lambda: scaling factor for total number of facilities opened
A = 10  # Alpha: maximum number of neighborhoods that a neighborhood can serve
B = 10  # Beta: maximum number of neighborhoods that a neighborhood can be served by
G = 15  # Gamma: penalty for unit unmet demand
k = ["Water", "Food", "Medical Kit"]  # commodities type
v = dict(zip(k, [1, 1, 0.5]))  # unit volume of commodity k
w = dict(zip(k, [1, 0.25, 0.25]))  # unit weight of commodity k
cd = dict(zip(k, [2.5, 3, 0.333]))  # commodities demand of an injured in 1day
V = 120000 # Volume Capacity of TDR
W = 30480*5 # Weight Capacity of TDR


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
genetic_best_solution = np.nan

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
# pprint.pprint(distmatrix)



# Initial Population
def generate_initial_population(pop_size):
    Population = []
    for ـ in range(pop_size):
        Chromosom = []
        selectable = demand_nodes_index.copy()
        for i in range(No_nodes):
            if (i < No_demandNodes):
                j = rn.choice(selectable)
                selectable.remove(j)
            else:
                j = rn.choice(range(0,n[supplier_nodes_index[i-No_demandNodes]]+1))
            Chromosom.append(j)

        Population.append(Chromosom)
    return Population
Population = generate_initial_population(50)
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

# MPA Parameters
pop_size = 50
max_iter = 2000
FADs = 0.2  # Fish Aggregating Devices effect
P = 0.5     # Probability for Levy flight

# Helper Functions
def levy_flight():
    """Generate a Levy flight step size."""
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma)
    v = np.random.normal(0, 1)
    step = u / abs(v) ** (1 / beta)
    return step

def apply_levy_to_permutation(chromosome):
    """Apply Levy flight to demand allocation part."""
    demand_part = chromosome[:No_demandNodes].copy()
    step = int(abs(levy_flight()) * 5)  # Scale Levy step
    for _ in range(min(step, No_demandNodes // 2)):
        i, j = random.sample(range(No_demandNodes), 2)
        demand_part[i], demand_part[j] = demand_part[j], demand_part[i]
    return demand_part + chromosome[No_demandNodes:]

def apply_brownian_to_permutation(chromosome):
    """Apply Brownian motion to demand allocation part."""
    demand_part = chromosome[:No_demandNodes].copy()
    i, j = random.sample(range(No_demandNodes), 2)
    demand_part[i], demand_part[j] = demand_part[j], demand_part[i]
    return demand_part + chromosome[No_demandNodes:]

def apply_levy_to_integer(chromosome):
    """Apply Levy flight to facility part."""
    facility_part = chromosome[No_demandNodes:].copy()
    step = int(abs(levy_flight()) * 3)  # Scale Levy step
    idx = random.randint(0, No_supplierNodes - 1)
    facility_part[idx] = min(max(facility_part[idx] + step, 0), n[supplier_nodes_index[idx]])
    return chromosome[:No_demandNodes] + facility_part

def apply_brownian_to_integer(chromosome):
    """Apply Brownian motion to facility part."""
    facility_part = chromosome[No_demandNodes:].copy()
    idx = random.randint(0, No_supplierNodes - 1)
    change = random.choice([-1, 1])
    facility_part[idx] = min(max(facility_part[idx] + change, 0), n[supplier_nodes_index[idx]])
    return chromosome[:No_demandNodes] + facility_part

def local_search_permutation(chromosome):
    """Local search for demand allocation."""
    demand_part = chromosome[:No_demandNodes].copy()
    best_fitness = fitness([demand_part + chromosome[No_demandNodes:]])[0]
    for i in range(No_demandNodes - 1):
        new_demand = demand_part.copy()
        new_demand[i], new_demand[i + 1] = new_demand[i + 1], new_demand[i]
        new_chrom = new_demand + chromosome[No_demandNodes:]
        new_fitness = fitness([new_chrom])[0]
        if new_fitness < best_fitness:
            demand_part = new_demand
            best_fitness = new_fitness
    return demand_part + chromosome[No_demandNodes:]

def random_reinitialization():
    """Generate a new random chromosome."""
    return generate_initial_population(1)[0]

def order_crossover(parent1, parent2):
    """Order Crossover (OX) for permutation part."""
    size = len(parent1)
    child = [-1] * size
    start, end = sorted(random.sample(range(size), 2))
    child[start:end+1] = parent1[start:end+1]
    p2_idx = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent2[p2_idx]
            p2_idx += 1
    return child

# MPA Algorithm
def mpa_optimization():
    # Initialize population
    population = generate_initial_population(pop_size)
    best_solution = None
    best_fitness = float('inf')
    fitness_history = []

    for iter in range(max_iter):
        # Evaluate fitness
        fitness_values = fitness(population)
        current_best_idx = np.argmin(fitness_values)
        if fitness_values[current_best_idx] < best_fitness:
            best_fitness = fitness_values[current_best_idx]
            best_solution = population[current_best_idx].copy()
        
        fitness_history.append(best_fitness)
        new_population = []

        if iter < max_iter / 3:  # Phase 1: Exploration
            for i in range(pop_size):
                chrom = population[i].copy()
                if random.random() < P:
                    chrom = apply_levy_to_permutation(chrom)
                    chrom = apply_levy_to_integer(chrom)
                else:
                    chrom = apply_brownian_to_permutation(chrom)
                    chrom = apply_brownian_to_integer(chrom)
                new_population.append(chrom)

        elif iter < 2 * max_iter / 3:  # Phase 2: Mixed
            for i in range(pop_size):
                chrom = population[i].copy()
                if random.random() < 0.5:
                    # Apply Order Crossover for demand part
                    demand_child = order_crossover(chrom[:No_demandNodes], best_solution[:No_demandNodes])
                    chrom[:No_demandNodes] = demand_child
                    # For facility part, blend as before
                    for j in range(No_supplierNodes):
                        if random.random() < 0.5:
                            chrom[No_demandNodes + j] = best_solution[No_demandNodes + j]
                else:
                    chrom = apply_brownian_to_permutation(chrom)
                    chrom = apply_brownian_to_integer(chrom)
                new_population.append(chrom)

        else:  # Phase 3: Exploitation
            for i in range(pop_size):
                chrom = population[i].copy()
                chrom = local_search_permutation(chrom)
                chrom = apply_brownian_to_integer(chrom)
                new_population.append(chrom)

        # Apply FADs effect
        for i in range(pop_size):
            if random.random() < FADs:
                new_population[i] = random_reinitialization()

        # Update population
        population = new_population

    # Plot fitness convergence
    plt.plot(range(max_iter), fitness_history)
    plt.title(f'MPA Best Fitness = {best_fitness}')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.show()

    return best_solution, best_fitness

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
    
# Run MPA
mpa_best_solution, mpa_best_fitness = mpa_optimization()

print("MPA Best Solution:", mpa_best_solution)
print("Fitness:", mpa_best_fitness)
print("Sum Demand:", sum(v_demands))
print("Sum Supply:", sum(V * n for n in mpa_best_solution[No_demandNodes:]))
print("Total Facilities Opened:", sum(mpa_best_solution[No_demandNodes:]))
print("Unmet Demand:", sum(v_demands) - sum(V * n for n in mpa_best_solution[No_demandNodes:]))
print("Total Distance:", distance(mpa_best_solution))
chromosomplot(mpa_best_solution)