import sys
import json
import math
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
    data = sys.stdin.buffer.read()
    data = json.loads(data)
    parameters = data['parameters']
    nodes_data = data['nodes_data']


    # Parameters for standard configuration
    NT = parameters['NT']  # total number of TDR facilities available
    ST = parameters['ST'] # safety level threshold
    M = parameters['M']  # total demands threshold
    L = parameters['L']  # Lambda: scaling factor for total number of facilities opened
    A = parameters['A']  # Alpha: maximum number of neighborhoods that a neighborhood can serve
    B = parameters['B']  # Beta: maximum number of neighborhoods that a neighborhood can be served by
    G = parameters['G']  # Gamma: penalty for unit unmet demand
    k = ["Water", "Food", "Medical Kit"]  # commodities type
    v = dict(zip(k, [1, 1, 0.5]))  # unit volume of commodity k
    w = dict(zip(k, [1, 0.25, 0.25]))  # unit weight of commodity k
    cd = dict(zip(k, [2.5, 3, 0.333]))  # commodities demand of an injured in 1day
    V = parameters['V'] # Volume Capacity of TDR
    W = parameters['W'] # Weight Capacity of TDR



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



    for nd in nodes_data:
        n.append(nd['NodeFacalities'])
        s.append(nd['NodeSaftyLevel'])
        x_nodes.append(nd['XCoordinate'])
        y_nodes.append(nd['YCoordinate'])
        population = nd['NodePopulation']
        v_demands.append(sum([x * y * population for x, y in zip(list(v.values()), list(cd.values()))]))
        w_demands.append(sum([x * y * population for x, y in zip(list(w.values()), list(cd.values()))]))
        v_capacity.append(V*(nd['NodeFacalities']))
        w_capacity.append(W*(nd['NodeFacalities']))
    No_nodes = len(x_nodes)
    supplier_nodes_index = [index for index, s in enumerate(s) if s >= ST]
    demand_nodes_index = [index for index, s in enumerate(s) if s < ST]
    No_supplierNodes = len(supplier_nodes_index)
    No_demandNodes = len(demand_nodes_index)

    print("x_nodes = ", x_nodes, "\n\ny_nodes = ", y_nodes, "\n\nv_capacity = ",
            v_capacity, "\n\nw_capacity = ", w_capacity, "\n\nv_demands = ",
            v_demands, "\n\nw_demands =", w_demands, "\n\nSafty Levels = ", s, flush=True)

    print("\nSumVolumeCapacities = ", sum(v_capacity[i] for i in supplier_nodes_index), "\nSumWeightCapacities = ", sum(w_capacity[i] for i in supplier_nodes_index),
            "\nSumVolumeDemands = ", sum(v_demands), "\nSumWeightDemands = ", sum(w_demands), flush=True)

    print("\nNum of Nodes = ",No_nodes,
            "\nNum of Spplier Nodes = ", No_supplierNodes,
            "\nNum of demand Nodes = ", No_demandNodes,
            "\nSum of Facilities = ", sum([n[i] for i in supplier_nodes_index]),
            "\nSupplier Nodes Index = ", supplier_nodes_index,
            "\nDemand Nodes Index = ", demand_nodes_index, flush=True)

except Exception as e:
    print(e)
