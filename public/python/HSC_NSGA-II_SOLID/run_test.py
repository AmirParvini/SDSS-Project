import sys
import json

from hsc_optimizer.config.scenario import StdinData, StdinConfig
from hsc_optimizer.config.data_loader import InMemoryDistanceLoader
from hsc_optimizer.domain.problem_data import Distances

stdin_data = StdinData.get_data()
stdinconfig = StdinConfig(data_dict=stdin_data)
distances = InMemoryDistanceLoader(stdin_data['distances'])
# print(json.dumps(stdinconfig.severe_ratio, ensure_ascii=False, indent=2))
from hsc_optimizer.config.scenario import build_problem_data
problem = build_problem_data(stdinconfig, distances.load())
print(json.dumps(str(problem.n_shelters), ensure_ascii=False, indent=2))

# problem_data_input = {
#     'dc_id': problem.dc_id,
#     'ec_id': problem.ec_id,
#     'da_id': problem.da_id,
#     'h_id': problem.h_id,
#     'tmc_id': problem.tmc_id,
#     'affected_pop': problem.affected_pop,
#     'severe_injured': problem.severe_injured,
#     'minor_injured': problem.minor_injured,
#     'homeless': problem.homeless,
#     'cost': problem.cost,
#     'distance': problem.distance,
#     'capacity': problem.capacity,
#     'ambulance_speed': problem.ambulance_speed,
#     'helicopter_speed': problem.helicopter_speed,
#     'severe_death_model': problem.severe_death_model,
#     'moderate_death_model': problem.moderate_death_model,
#     'injured_treated_same_time': problem.injured_treated_same_time,
#     'waiting_time': problem.waiting_time,
#     'budget': problem.budget
# }

# print(json.dumps(problem_data_input, ensure_ascii=False, indent=2))