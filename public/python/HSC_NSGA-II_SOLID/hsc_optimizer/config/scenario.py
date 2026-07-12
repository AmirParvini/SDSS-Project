"""Scenario configuration.

The original ``Main.__init__`` mixed raw constants, derived demographics and
file I/O in one place.  Here the *inputs* live in a declarative
:class:`ScenarioConfig` (easy to change, serialise or swap per scenario) and a
builder turns that config plus a distance source into an immutable
:class:`~hsc_optimizer.domain.problem_data.ProblemData`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Union

from ..domain.problem_data import (
    DeathModelParameters,
    Distances,
    FacilityCapacities,
    ProblemData,
)

import sys
import json
import math

@dataclass
class ScenarioConfig:
    """Declarative description of a scenario's raw inputs.

    Defaults reproduce "scenario 1" from the original ``main.py``.
    """

    # Entity ids
    idc_id: List[int] = field(default_factory=lambda: [1, 2, 3])
    ec_id: List[int] = field(
        default_factory=lambda: [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    da_id: List[int] = field(default_factory=lambda: [3, 4, 5, 6, 7])
    h_id: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    tmc_id: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Budget & injury distribution ratios
    budget: float = 1_500_000
    severe_ratio: float = 0.015            # t1: share of severe injuries
    minor_ratio: float = 0.067             # t2: share of mild injuries

    # Shelter geometry
    shelter_area_usage: float = 0.7        # Pua: usable fraction of shelter area
    relief_tent_area: float = 17.5         # rta: m^2 per tent
    relief_tent_capacity: int = 5          # rtc: persons per tent

    # Vehicle speeds (km/h)
    ambulance_speed: float = 20
    helicopter_speed: float = 40

    # Death-probability model (severe / moderate)
    severe_death_model: DeathModelParameters = field(
        default_factory=lambda: DeathModelParameters(phi_min=0, phi_max=0.9, ks=0.1, tm=10))
    moderate_death_model: DeathModelParameters = field(
        default_factory=lambda: DeathModelParameters(phi_min=0, phi_max=0.9, ks=0.1, tm=20))

    # Treatment queue parameters
    injured_treated_same_time: int = 50
    waiting_time: float = 5

    # Affected population per damage area
    affected_pop: Dict[int, float] = field(
        default_factory=lambda: {5: 30000, 6: 52500, 4: 4500, 7: 18000, 3: 4854})

    # Shelter areas (m^2) used to derive shelter capacity
    ec_area: Dict[int, float] = field(default_factory=lambda: {
        8: 283762, 9: 133407, 10: 120000, 11: 18700, 12: 14000,
        13: 24000, 14: 20170, 15: 17400, 16: 25380, 17: 25374, 18: 58055,
    })

    relief_package_volume: float = 0.6273  # m^3

    cost: Dict[str, float] = field(default_factory=lambda: {
        'reliefpackage_cost': 108.76,
        'reliefpackage_transportation_cost': 30,
        'tmc_cost': 50000,
        'ec_cost': 50000,
        'ambulance': 50,
        'helicopter': 100,
    })

    ambulance_capacity: Dict[str, int] = field(
        default_factory=lambda: {'injured_type1': 2, 'injured_type2': 4})
    helicopter_capacity: Dict[str, int] = field(
        default_factory=lambda: {'injured_type1': 4, 'injured_type2': 12})
    hospital_capacity: Dict[int, float] = field(
        default_factory=lambda: {1: 800, 2: 600, 3: 600, 4: 600})
    tmc_capacity: Dict[int, float] = field(default_factory=lambda: {
        1: 300, 2: 300, 3: 300, 4: 300, 5: 300,
        6: 300, 7: 300, 8: 300, 9: 300, 10: 300,
    })

class StdinData:
    def __init__(self):
        pass

    def get_data():
        data = sys.stdin.read()
        data_dict = json.loads(data)
        return data_dict

class StdinConfig:

    def __init__(self, data_dict: Dict):
        data_dict = data_dict
        self.dd = data_dict
        self.hsc_parameters = data_dict['hsc_parameters']
        self.nodes_id = data_dict['nodes_id']
        self.affected_pop = data_dict['affected_pops']
        self.ec_area = data_dict['shelters_area']
        self.hospital_capacity = data_dict['hospitals_capacity']
        self.tmc_capacity = data_dict['tmcs_capacity']

        # Entity ids
        self.dc_id = self.nodes_id['dc']
        self.ec_id = self.nodes_id['ec']
        self.da_id = self.nodes_id['da']
        self.h_id = self.nodes_id['h']
        self.tmc_id = self.nodes_id['tmc']

        # Budget & injury distribution ratios
        self.budget = self.hsc_parameters['budget']
        self.severe_ratio = self.hsc_parameters['t2']
        self.minor_ratio = self.hsc_parameters['t1']

        # Shelter geometry
        self.shelter_area_usage = self.hsc_parameters['pua']
        self.relief_tent_area = self.hsc_parameters['rta']
        self.relief_tent_capacity = self.hsc_parameters['rtc']

        # Vehicle speeds
        self.ambulance_speed = self.hsc_parameters['gv_speed']
        self.helicopter_speed = self.hsc_parameters['av_speed']

        # Death-probability models
        self.severe_death_model = DeathModelParameters(
            self.hsc_parameters['phi_min_s'], self.hsc_parameters['phi_max_s'], ks=0.1, tm=10)
        self.moderate_death_model = DeathModelParameters(
            self.hsc_parameters['phi_min_m'], self.hsc_parameters['phi_max_m'], ks=0.1, tm=20)

        # Treatment queue parameters
        self.injured_treated_same_time = self.hsc_parameters['itst']
        self.waiting_time = self.hsc_parameters['wt']
        self.relief_package_volume = 0.6273  # m^3

        # Cost dict
        self.cost = {
            'reliefpackage_cost': self.hsc_parameters['rp_cost'],
            'reliefpackage_transportation_cost': self.hsc_parameters['rpt_cost'],
            'tmc_cost': self.hsc_parameters['tmc_cost'],
            'ec_cost': self.hsc_parameters['shelter_cost'],
            'ambulance': self.hsc_parameters['gv_cost'],
            'helicopter': self.hsc_parameters['av_cost'],
        }

        # Capacity dicts
        self.ambulance_capacity = {
            'injured_type1': self.hsc_parameters['gv_severe_capacity'],
            'injured_type2': self.hsc_parameters['gv_moderate_capacity'],
        }
        self.helicopter_capacity = {
            'injured_type1': self.hsc_parameters['av_severe_capacity'],
            'injured_type2': self.hsc_parameters['av_moderate_capacity'],
        }       

def build_problem_data(config: StdinConfig, distances: Distances) -> ProblemData:
    """Assemble an immutable :class:`ProblemData` from config + distances."""
    severe_injured = {k: round(v * config.severe_ratio) for k, v in config.affected_pop.items()}
    minor_injured = {k: round(v * config.minor_ratio) for k, v in config.affected_pop.items()}
    homeless = {
        k: config.affected_pop[f'{k}'] - (severe_injured[f'{k}'] + minor_injured[f'{k}'])
        for k in config.affected_pop
    }

    shelter_capacity = {
        ec_id: math.floor((area * config.shelter_area_usage / config.relief_tent_area)
        * config.relief_tent_capacity)
        for ec_id, area in config.ec_area.items()
    }

    capacities = FacilityCapacities(
        ambulance=dict(config.ambulance_capacity),
        helicopter=dict(config.helicopter_capacity),
        hospital=dict(config.hospital_capacity),
        tmc=dict(config.tmc_capacity),
        shelter=shelter_capacity,
    )

    return ProblemData(
        dc_id=list(config.dc_id),
        ec_id=list(config.ec_id),
        da_id=list(config.da_id),
        h_id=list(config.h_id),
        tmc_id=list(config.tmc_id),
        affected_pop=dict(config.affected_pop),
        severe_injured=severe_injured,
        minor_injured=minor_injured,
        homeless=homeless,
        cost=dict(config.cost),
        distance=distances,
        capacity=capacities,
        ambulance_speed=config.ambulance_speed,
        helicopter_speed=config.helicopter_speed,
        severe_death_model=config.severe_death_model,
        moderate_death_model=config.moderate_death_model,
        injured_treated_same_time=config.injured_treated_same_time,
        waiting_time=config.waiting_time,
        budget=config.budget,
    )
