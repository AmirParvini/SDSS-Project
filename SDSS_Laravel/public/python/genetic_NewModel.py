import geopandas as gpd
import numpy as np
import pprint
import random as rn
from itertools import chain
import math
import multiprocessing
import copy
import random
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
data = sys.stdin.buffer.read()
data = json.loads(data)
parameters = data['config_parameters']
nodes_data = data['nodes_data']
CMD_nodes_data = nodes_data[0]
LDC_nodes_data = nodes_data[1]
EC_nodes_data = nodes_data[2]
commodity_demands_unit = data['commodity_demands_unit']
APR = float(data['APR'])

# Parameters for standard configuration
NT = parameters['NT']  # total number of TDR facilities available
L = parameters['L']  # Lambda: scaling factor for total number of facilities opened
A = parameters['A']  # Alpha: maximum number of neighborhoods that a neighborhood can serve
G = parameters['G']  # Gamma: penalty for unit unmet demand
O = parameters['O']  # Omega: penalty for unit overcapacity
k = ["Water", "Food", "Medical Kit", "Shelter"]  # commodities type
cd = dict(zip(k, [commodity_demands_unit["Water(unit-pp)"], commodity_demands_unit["Food(unit-pp)"], commodity_demands_unit["MedicalKit(unit-pp)"], commodity_demands_unit["Shelter(unit-pp)"]]))  # commodities demand of an injured
V = parameters['V'] # Volume Capacity of TDR


CMD_Coordinate = [] # [[x1,y1], [x2,y2], ...]
LDC_Coordinate = [] # [[x1,y1], [x2,y2], ...]
EC_Coordinate = [] # [[x1,y1], [x2,y2], ...]
Pop_Blocks = [] # [[x1,y1,pop1], [x2,y2,pop2], ...]
No_CMD = 0 # number of Crises Management Databases
No_LDC = 0 # number of Local Distribution Centers
No_EC = 0 # number of Evacuation Centers
CMD_name = []
LDC_name = []
EC_name = []
c_LDC = [] # capacities of LDC
EC_demand = [] # demands of EC
EC_water_demand = []
EC_food_demand = []
EC_medicalkit_demand = []
EC_shelter_demand = []
elite_Chromosom = []
elites_fitness = math.inf
bad_chromosom_index = 0
genetic_best_solution = np.nan


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
    EC_water_demand.append(EC_node_data["Population"]*cd["Water"])
    EC_food_demand.append(EC_node_data["Population"]*cd["Food"])
    EC_medicalkit_demand.append(EC_node_data["Population"]*cd["Medical Kit"])
    EC_shelter_demand.append(EC_node_data["Population"]*cd["Shelter"])
    EC_demand.append(EC_water_demand[-1]+EC_food_demand[-1]+EC_medicalkit_demand[-1]+EC_shelter_demand[-1])
    No_EC += 1


# print(json.dumps(EC_name))




output = {
    # "LDC_Coordinate": LDC_Coordinate,
    # "LDC_name": LDC_name,
    # "c_LDC": c_LDC,
    # "No_LDC": No_LDC
}

json_data = json.dumps(output)
print(json_data)