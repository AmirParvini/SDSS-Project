# -*- coding: utf-8 -*-
# TDR Facility Allocation - Complete Reformulation

import numpy as np
from typing import Tuple


class Problem():
    """
    TDR Facility Allocation with proper formulation
    """
    
    def __init__(self):
        # Fixed problem size for testing
        self.n_neighborhoods = 5
        self.n_commodities = 2
        
        # Decision variables: facilities in each neighborhood
        self._n_vars = self.n_neighborhoods
        
        # Three objectives
        self._n_objs = 3
        
        # Constraints: max per neighborhood + total max
        self._n_constrs = self.n_neighborhoods + 1
        
        # Initialize problem data
        self._setup_problem_data()
        
        # Variable bounds (0 to 3 facilities per neighborhood)
        self._vars_lb = np.zeros(self._n_vars)
        self._vars_ub = np.ones(self._n_vars) * 3
    
    def _setup_problem_data(self):
        """Setup realistic problem data"""
        np.random.seed(123)  # For reproducibility
        
        n = self.n_neighborhoods
        c = self.n_commodities
        
        # Realistic distance matrix (km)
        self.distances = np.array([
            [0,  10, 15, 20, 25],
            [10, 0,  12, 18, 22],
            [15, 12, 0,  8,  14],
            [20, 18, 8,  0,  10],
            [25, 22, 14, 10, 0]
        ])
        
        # Demand for each commodity at each neighborhood
        self.demand = np.array([
            [30, 25],  # Neighborhood 1
            [40, 35],  # Neighborhood 2
            [35, 30],  # Neighborhood 3
            [45, 40],  # Neighborhood 4
            [50, 45]   # Neighborhood 5
        ])
        
        # Supply capacity per facility
        self.supply_per_facility = 40
        
        # Maximum facilities
        self.max_per_neighborhood = 3
        self.max_total = 10
        
        # Objective weights
        self.facility_cost = 100     # Cost per facility
        self.transport_cost = 1      # Cost per unit-km
        self.shortage_penalty = 50   # Penalty per unit shortage
    
    def get_individual_result(self, individual: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate solution properly
        """
        # Round to integers and clip to bounds
        facilities = np.round(individual).astype(int)
        facilities = np.clip(facilities, 0, self.max_per_neighborhood)
        
        n = self.n_neighborhoods
        c = self.n_commodities
        
        # Calculate supply capacity at each location
        supply_capacity = facilities * self.supply_per_facility
        
        # Allocate supplies using nearest allocation
        total_transport = 0
        total_shortage = 0
        
        for j in range(n):  # For each demand point
            for k in range(c):  # For each commodity
                remaining_demand = self.demand[j, k]
                
                # Sort supply points by distance
                supply_points = []
                for i in range(n):
                    if supply_capacity[i] > 0:
                        supply_points.append((i, self.distances[i, j]))
                supply_points.sort(key=lambda x: x[1])
                
                # Allocate from nearest suppliers
                for i, dist in supply_points:
                    if remaining_demand <= 0:
                        break
                    
                    # Amount to supply (proportional allocation)
                    available = supply_capacity[i] / c  # Divide equally among commodities
                    supply_amount = min(remaining_demand, available)
                    
                    total_transport += supply_amount * dist
                    remaining_demand -= supply_amount
                
                # Record shortage
                if remaining_demand > 0:
                    total_shortage += remaining_demand
        
        # Calculate objectives
        obj1 = self.facility_cost * np.sum(facilities)
        obj2 = self.transport_cost * total_transport
        obj3 = self.shortage_penalty * total_shortage
        
        objectives = np.array([obj1, obj2, obj3])
        
        # Calculate constraint violations
        violations = []
        
        # Max facilities per neighborhood
        for i in range(n):
            violations.append(max(0, facilities[i] - self.max_per_neighborhood))
        
        # Total facilities constraint
        violations.append(max(0, np.sum(facilities) - self.max_total))
        
        return objectives, np.array(violations)
    
    @property
    def n_vars(self):
        return self._n_vars
    
    @property
    def n_objs(self):
        return self._n_objs
    
    @property
    def n_constrs(self):
        return self._n_constrs
    
    @property
    def vars_lb(self):
        return self._vars_lb
    
    @property
    def vars_ub(self):
        return self._vars_ub