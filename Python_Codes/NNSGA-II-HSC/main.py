import numpy as np
import matplotlib.pyplot as plt
from nsga2 import NSGA2_Humanitarian
from graph_download import GraphDownload
from pgsql_connector import PgsqlConnector

# Example cost function for humanitarian logistics
def humanitarian_cost_function(chromosome):
    """
    Calculate objectives for humanitarian logistics problem
    
    Returns:
    --------
    numpy array with objective values [f1, f2]
    f1: Total transportation and operation cost
    f2: Total unmet demand or response time
    """
    
    # Extract chromosome parts
    dist_assignment = chromosome[0]
    flow_values = chromosome[1]
    damage_to_shelter = chromosome[2]
    damage_to_hospital = chromosome[3]
    temp_medical = chromosome[4]
    severe_to_hospital = chromosome[5]
    moderate_to_hospital = chromosome[6]
    moderate_to_temp = chromosome[7]
    
    # Objective 1: Total cost (example calculation)
    # Transportation cost from distribution centers to shelters
    transport_cost = sum(flow_values) * 0.5  # Simplified cost per unit
    
    # Shelter operation cost
    shelter_cost = sum([100 if x > 0 else 0 for x in dist_assignment])
    
    # Temporary medical center cost
    temp_medical_cost = sum(temp_medical) * 200
    
    medical_transport_cost = (sum(severe_to_hospital) * 10 + 
                             sum(moderate_to_hospital) * 5 + 
                             sum(moderate_to_temp) * 3)
    
    f1 = transport_cost + shelter_cost + temp_medical_cost + medical_transport_cost
    
    # Objective 2: Service quality (minimize unmet demand or maximize coverage)
    # Count active shelters
    active_shelters = sum([1 if x > 0 else 0 for x in dist_assignment])
    
    # Penalty for inactive shelters (unserved areas)
    unserved_penalty = (len(dist_assignment) - active_shelters) * 150
    
    # Penalty for low flow values
    flow_penalty = sum([max(0, 100 - f) for f in flow_values if f > 0])
    
    # Penalty for unallocated injured (simplified - you should use actual injured counts)
    # This is where you'd use the actual severe_injured and moderate_injured arrays
    total_severe = 200  # Example total
    total_moderate = 350  # Example total
    
    severe_unallocated = max(0, total_severe - sum(severe_to_hospital))
    moderate_unallocated = max(0, total_moderate - sum(moderate_to_hospital) - sum(moderate_to_temp))
    
    medical_penalty = severe_unallocated * 50 + moderate_unallocated * 20
    
    f2 = unserved_penalty + flow_penalty + medical_penalty
    
    return np.array([f1, f2])

# Alternative more complex cost function
def complex_humanitarian_cost(chromosome, distances=None, demands=None, capacities=None):
    """
    More realistic cost function with actual problem parameters
    
    Parameters:
    -----------
    distances: dict with distance matrices
    demands: dict with demand values for shelters and medical facilities
    capacities: dict with capacity constraints
    """
    
    dist_assignment = chromosome[0]
    flow_values = chromosome[1]
    damage_to_shelter = chromosome[2]
    damage_to_hospital = chromosome[3]
    temp_medical = chromosome[4]
    
    # If no data provided, generate random realistic data
    if distances is None:
        n_dist = 3
        n_shelters = len(dist_assignment)
        n_damage = 5
        n_hospitals = 4
        
        # Random distance matrix from distribution centers to shelters
        dist_to_shelter = np.random.uniform(10, 100, (n_dist, n_shelters))
        
        # Random distance from damage points to shelters
        damage_to_shelter_dist = np.random.uniform(5, 50, (n_damage, n_shelters))
        
        # Random distance from damage points to hospitals
        damage_to_hospital_dist = np.random.uniform(5, 60, (n_damage, n_hospitals))
    
    if demands is None:
        # Random demands for each shelter
        shelter_demands = np.random.uniform(50, 200, len(dist_assignment))
        
        # Random number of homeless and injured at damage points
        homeless_count = np.random.randint(20, 100, 5)
        severe_injured = np.random.randint(5, 20, 5)
        moderate_injured = np.random.randint(10, 30, 5)
    
    # Objective 1: Total transportation and setup cost
    f1 = 0
    
    # Transportation cost from distribution to shelters
    for i, dc in enumerate(dist_assignment):
        if dc > 0:
            f1 += dist_to_shelter[dc-1, i] * flow_values[i] * 0.1
    
    # Fixed cost for opening shelters
    f1 += sum([500 if x > 0 else 0 for x in dist_assignment])
    
    # Fixed cost for temporary medical centers
    f1 += sum(temp_medical) * 800
    
    # Objective 2: Total response time or unmet demand
    f2 = 0
    
    # Response time for homeless to shelters
    for i, shelter_idx in enumerate(damage_to_shelter):
        if shelter_idx > 0:
            f2 += damage_to_shelter_dist[i % 5, shelter_idx-1] * homeless_count[i % 5] * 0.01
    
    # Response time for injured to hospitals
    for i, hospital_idx in enumerate(damage_to_hospital):
        if hospital_idx > 0:
            f2 += damage_to_hospital_dist[i, hospital_idx-1] * severe_injured[i] * 0.02
    
    # Penalty for unmet demand
    total_demand = sum(shelter_demands)
    total_supply = sum(flow_values)
    if total_supply < total_demand:
        f2 += (total_demand - total_supply) * 10
    
    return np.array([f1, f2])

# Parameters
dist_idc_to_ec = []
dist_da_to_h = []
dist_da_to_tmc = []
gd = GraphDownload()
G = gd.download_tehran_district6_graph()
dbname="sdss"
user="postgres"
password="AP@pgsql79"
host="localhost"
port="5432"
pgc = PgsqlConnector(dbname, user, password, host, port)
idc_to_ec, da_to_ec, da_to_tmc, da_to_h= pgc.data_fetch()
# for ite in idc_to_ec:
#     short_path = gd._route(G, ite["source_coord"], ite["target_coord"])
#     pgc.data_send(ite["source_id"], ite["target_id"], short_path['geometry']['coordinates'], short_path['properties']['weight'], "idc_ec_path")
#     dist_idc_to_ec.append(short_path['properties']['weight'])
# for dte in pgc.da_to_ec:
#     short_path = gd._route(G, dte["source_coord"], dte["target_coord"])
#     pgc.data_send(dte["source_coord"], dte["target_coord"], short_path['geometry']['coordinates'], short_path['properties']['weight'], "da_ec_path")
for dth in da_to_h:
    short_path = gd._route(G, dth["source_coord"], dth["target_coord"])
    pgc.data_send(dth["source_id"], dth["target_id"], short_path['geometry']['coordinates'], short_path['properties']['weight'], "da_h_path")
    dist_da_to_h.append(short_path['properties']['weight'])
# for dtt in da_to_tmc:
#     short_path = gd._route(G, dtt["source_coord"], dtt["target_coord"])
#     pgc.data_send(dtt["source_id"], dtt["target_id"], short_path['geometry']['coordinates'], short_path['properties']['weight'], "da_tmc_path")
#     dist_da_to_tmc.append(short_path['properties']['weight'])
t1 = 0.015 # Percentage distribution of severe injuries
t2 = 0.067 # Percentage distribution of mild injuries
Pua = 0.07 # Percentage of shelter area used
rtc = 5 # Capacity of each relief tent
affected_pop = [30000, 52500, 4500, 18000, 4854] # the affected population (scenario 1)
severe_injured = np.array(affected_pop) * t1
minor_injured = np.array(affected_pop) * t2
homeless = affected_pop - (severe_injured + minor_injured)
demand = np.array(affected_pop)/5
ec_area = [283762, 133407, 120000, 18700, 14000, 24000, 20170, 17400, 25380, 25374, 58055]
reliefpackage_volume = 0.6273 # Cubic meter
cost = {
    'reliefpackage_cost': 108.76,
    'reliefpackage_transportation_cost': 30,
    'tmc_cost': 50000
}
distance = {
    'dist_idc_to_ec': dist_idc_to_ec, # OD matrix of distribution centers and shelters
    'dist_da_to_ec': [], # OD matrix of damaged area and shelters
    'dist_da_to_h': dist_da_to_h, # OD matrix of damaged area centers and hospitals
    'dist_da_to_tmc': dist_da_to_tmc # OD matrix of damaged area centers and temporary medical centers
}
print(distance)
capacity = {
    'ambulance_type1': {'injured_typ1': 2, 'injured_typ2': 4}, # person
    'helicopter': {'injured_typ1': 4, 'injured_typ2': 12}, # person
    'truck_type1': 6, # Cubic meter
    'truck_type2': 12, # Cubic meter
    'hospital': [800, 600, 600, 600], # person
    'tmc': [300, 300, 600, 300, 600, 300, 300, 600, 600, 300], # person
    'shelter': np.array(ec_area) * Pua / rtc
}


# Problem Definition
problem = {
    'cost_function': complex_humanitarian_cost  # Use simple or complex function
}

# Initialize Algorithm with problem dimensions
alg = NSGA2_Humanitarian(
    max_iter=50,
    pop_size=100,
    p_crossover=0.7,
    p_mutation=0.3,
    verbose=True,
    n_shelters=8,
    n_distribution=3,
    n_damage_points=5,
    n_hospitals=4,
    n_temp_medical=11,
)

# # Solve the Problem
# results = alg.run(problem)
# pop = results['pop']
# F = results['F']
# pareto_pop = results['pareto_pop']

# # Extract Pareto Front costs
# pf_costs = np.array([ind['cost'] for ind in pareto_pop])

# # Plot Results
# plt.figure(figsize=(10, 6))
# plt.scatter(pf_costs[:, 0], pf_costs[:, 1], c='blue', s=50, alpha=0.6, edgecolors='black')
# plt.grid(True, alpha=0.3)
# plt.xlabel('Total Cost (f1)', fontsize=12)
# plt.ylabel('Unmet Demand / Response Time (f2)', fontsize=12)
# plt.title('Pareto Front - Humanitarian Logistics Optimization', fontsize=14)

# # Annotate some solutions
# if len(pf_costs) > 0:
#     # Mark best cost solution
#     best_cost_idx = np.argmin(pf_costs[:, 0])
#     plt.scatter(pf_costs[best_cost_idx, 0], pf_costs[best_cost_idx, 1], 
#                 c='red', s=100, marker='*', label='Min Cost')
    
#     # Mark best service solution
#     best_service_idx = np.argmin(pf_costs[:, 1])
#     plt.scatter(pf_costs[best_service_idx, 0], pf_costs[best_service_idx, 1], 
#                 c='green', s=100, marker='*', label='Min Unmet Demand')
    
#     plt.legend()

# plt.tight_layout()
# plt.show()

# # Display some statistics
# print("\n" + "="*50)
# print("Optimization Results Summary")
# print("="*50)
# print(f"Number of Pareto optimal solutions: {len(pareto_pop)}")
# print(f"Best cost: {np.min(pf_costs[:, 0]):.2f}")
# print(f"Best service (min unmet demand): {np.min(pf_costs[:, 1]):.2f}")

# # Display a sample solution
# if len(pareto_pop) > 0:
#     print("\n" + "="*50)
#     print("Sample Pareto Optimal Solution (Balanced)")
#     print("="*50)
    
#     # Find a balanced solution (closest to the middle of Pareto front)
#     normalized_costs = (pf_costs - pf_costs.min(axis=0)) / (pf_costs.max(axis=0) - pf_costs.min(axis=0) + 1e-10)
#     distances_to_ideal = np.linalg.norm(normalized_costs, axis=1)
#     balanced_idx = np.argmin(distances_to_ideal)
    
#     balanced_solution = pareto_pop[balanced_idx]
#     chromosome = balanced_solution['chromosome']
    
#     print(f"Objectives: f1={balanced_solution['cost'][0]:.2f}, f2={balanced_solution['cost'][1]:.2f}")
#     print(f"\nActive Shelters: {sum([1 if x > 0 else 0 for x in chromosome[0]])}/{len(chromosome[0])}")
#     print(f"Active Temporary Medical Centers: {sum(chromosome[4])}/{len(chromosome[4])}")
#     print(f"Total Flow: {sum(chromosome[1]):.2f}")
    
#     # Show distribution assignments
#     print("\nShelter Assignments:")
#     for i, (dc, flow) in enumerate(zip(chromosome[0], chromosome[1])):
#         if dc > 0:
#             print(f"  Shelter {i+1}: Supplied by DC{dc}, Flow={flow:.2f}")
    
#     # Show medical allocations
#     print("\nMedical Allocations:")
#     print(f"Severe Injured to Hospitals: {sum(chromosome[5])}/{sum(severe_injured)}")
#     print(f"Moderate Injured to Hospitals: {sum(chromosome[6])}/{sum(moderate_injured)}")
#     print(f"Moderate Injured to Temp Medical: {sum(chromosome[7])}/{sum(moderate_injured)}")
    
#     # Show detailed medical allocation
#     print("\nDetailed Medical Allocation by Damage Point:")
#     for i in range(alg.n_damage_points):
#         print(f"  Damage Point {i+1}:")
#         print(f"    Severe: {chromosome[5][i]}/{severe_injured[i]} to Hospital {chromosome[3][i]}")
#         print(f"    Moderate: {chromosome[6][i]} to Hospital, "
#               f"{moderate_injured[i] - chromosome[6][i]} to Temp Medical")
    
#     # Show active temp medical centers
#     print("\nActive Temporary Medical Centers:")
#     for i, (active, allocated) in enumerate(zip(chromosome[4], chromosome[7])):
#         if active:
#             print(f"  Temp Medical {i+1}: {allocated} patients")