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
commodity_demands_unit = data['commodity_demands_unit']
APR = float(data['APR'])