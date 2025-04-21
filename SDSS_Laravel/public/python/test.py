import sys
import json
import math
import numpy as np
import utm
import time

try:
    st = time.time()
    sys.stdout.reconfigure(encoding='utf-8')
    data = sys.stdin.buffer.read()
    data = json.loads(data)
    parameters = data['config_parameters']
    nodes_data = data['nodes_data']
    commodity_demands = data['commodity_demands']
    APR = float(data['APR'])


    out = {
        "out": APR
    }
    json_data = json.dumps(out)
    print(json_data)
    # print("lat = ", lat, "\n\nlon = ", lon, "\n\nv_capacity = ",
    #         v_capacity, "\n\nw_capacity = ", w_capacity, "\n\nv_demands = ",
    #         v_demands, "\n\nw_demands =", w_demands, "\n\nSafty Levels = ", s
    #         )

    # print("\nSumVolumeCapacities = ", sum(v_capacity[i] for i in supplier_nodes_index), "\nSumWeightCapacities = ", sum(w_capacity[i] for i in supplier_nodes_index),
    #         "\nSumVolumeDemands = ", sum(v_demands), "\nSumWeightDemands = ", sum(w_demands))

    # print("\nNum of Nodes = ",No_nodes,
    #         "\nNum of Spplier Nodes = ", No_supplierNodes,
    #         "\nNum of demand Nodes = ", No_demandNodes,
    #         "\nSum of Facilities = ", sum([n[i] for i in supplier_nodes_index]),
    #         "\nSupplier Nodes Index = ", supplier_nodes_index,
    #         "\nDemand Nodes Index = ", demand_nodes_index)

except Exception as e:
    error_output = {
        "error": str(e)
    }
    print(json.dumps(error_output))
