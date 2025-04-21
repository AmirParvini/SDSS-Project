import sys
import json
import math
import utm
import numpy as np

try:

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
    zone_number = 39
    zone_letter = 'N'

    def readfile(path: str):
        global No_nodes, No_supplierNodes, No_demandNodes, supplier_nodes_index, demand_nodes_index, n, v_demands
        with open(path, "r", encoding='utf-8') as file:
            lines = file.readlines()
        # print(lines)
        for i, line in enumerate(lines, start=1):
            if 2 < i:
                NodesAttributes = line.split(",")
                # latitude, longitude = utm.to_latlon(float(NodesAttributes[7]), float(NodesAttributes[8]), zone_number, zone_letter)
                # x_nodes.append(longitude)
                # y_nodes.append(latitude)
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
        
        # print("\nSumVolumeCapacities = ", sum(v_capacity[i] for i in supplier_nodes_index), "\nSumWeightCapacities = ", sum(w_capacity[i] for i in supplier_nodes_index),
        #     "\nSumVolumeDemands = ", sum(v_demands), "\nSumWeightDemands = ", sum(w_demands))
        
        # print("\nNum of Nodes = ",No_nodes,
        #     "\nNum of Spplier Nodes = ", No_supplierNodes,
        #     "\nNum of demand Nodes = ", No_demandNodes,
        #     "\nSum of Facilities = ", sum([n[i] for i in supplier_nodes_index]),
        #     "\nSupplier Nodes Index = ", supplier_nodes_index,
        #     "\nDemand Nodes Index = ", demand_nodes_index)
        

    readfile("C:\\Users\\Amir\\Desktop\\SDSS-Project\\Data\\Neighborhood_Data.txt")   

except Exception as e:
    print(e)
