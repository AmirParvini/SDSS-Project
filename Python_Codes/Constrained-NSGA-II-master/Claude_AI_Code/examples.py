# -*- coding: utf-8 -*-

import numpy as np
from problem import Problem
from nsga2 import ConstrainedNSGA2
import matplotlib.pyplot as plt


def analyze_problem(problem):
    """Analyze problem characteristics"""
    print("=" * 60)
    print("Problem Analysis")
    print("=" * 60)
    
    total_demand = np.sum(problem.demand)
    max_supply = problem.max_total * problem.supply_per_facility
    
    print(f"Total demand: {total_demand}")
    print(f"Maximum possible supply: {max_supply}")
    print(f"Supply/Demand ratio: {max_supply/total_demand:.2f}")
    
    print("\nDemand by neighborhood:")
    for i in range(problem.n_neighborhoods):
        print(f"  Neighborhood {i+1}: {np.sum(problem.demand[i])}")
    
    print("\nDistance matrix:")
    print(problem.distances)
    
    # Test different allocations
    print("\n" + "=" * 60)
    print("Testing Different Allocations")
    print("=" * 60)
    
    test_allocations = [
        ("No facilities", [0, 0, 0, 0, 0]),
        ("Central only", [0, 0, 2, 0, 0]),
        ("Distributed", [1, 1, 1, 1, 1]),
        ("Corner heavy", [2, 0, 0, 0, 2]),
        ("Maximum", [2, 2, 2, 2, 2])
    ]
    
    for name, alloc in test_allocations:
        obj, viol = problem.get_individual_result(np.array(alloc))
        print(f"\n{name}: {alloc}")
        print(f"  Facility cost: {obj[0]:.0f}")
        print(f"  Transport cost: {obj[1]:.1f}")
        print(f"  Shortage cost: {obj[2]:.0f}")
        print(f"  Total cost: {np.sum(obj):.1f}")
        print(f"  Constraint violation: {np.sum(viol):.1f}")


def run_optimization():
    """Run NSGA-II optimization"""
    
    problem = Problem()
    
    # Analyze problem first
    analyze_problem(problem)
    
    print("\n" + "=" * 60)
    print("Running NSGA-II Optimization")
    print("=" * 60)
    
    # Run with better initialization
    nsga2 = ConstrainedNSGA2(
        problem,
        pop_size=100,
        max_gen=200,
        etac=20,
        etam=20,
        pc=0.9,
        pm=1/problem.n_vars,  # Standard mutation rate
        if_plot_front=True
    )
    
    population = nsga2.run()
    
    # Extract Pareto optimal solutions
    n_vars = problem.n_vars
    n_objs = problem.n_objs
    
    # Find rank 0 solutions (Pareto front)
    rank_col = n_vars + n_objs + 1
    pareto_mask = population[:, rank_col] == 0
    pareto_solutions = population[pareto_mask]
    
    print(f"\nFound {len(pareto_solutions)} Pareto optimal solutions")
    
    if len(pareto_solutions) > 0:
        # Sort by total cost
        costs = []
        for sol in pareto_solutions:
            facilities = np.round(sol[:n_vars]).astype(int)
            obj, _ = problem.get_individual_result(facilities)
            costs.append(np.sum(obj))
        
        sorted_idx = np.argsort(costs)
        
        print("\n" + "=" * 60)
        print("Best Pareto Solutions (by total cost)")
        print("=" * 60)
        
        for rank, idx in enumerate(sorted_idx[:10]):
            sol = pareto_solutions[idx]
            facilities = np.round(sol[:n_vars]).astype(int)
            obj, viol = problem.get_individual_result(facilities)
            
            print(f"\nRank {rank+1}:")
            print(f"  Allocation: {facilities} (Total: {np.sum(facilities)})")
            print(f"  Costs: Facility={obj[0]:.0f}, Transport={obj[1]:.1f}, Shortage={obj[2]:.0f}")
            print(f"  Total cost: {np.sum(obj):.1f}")
            
            if np.sum(viol) > 0:
                print(f"  WARNING: Constraint violation = {np.sum(viol)}")
    
    return population, problem


if __name__ == "__main__":
    np.random.seed(42)
    population, problem = run_optimization()
    
    # Additional analysis
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    
    # Get unique solutions
    n_vars = problem.n_vars
    unique_allocations = set()
    
    for individual in population:
        alloc = tuple(np.round(individual[:n_vars]).astype(int))
        unique_allocations.add(alloc)
    
    print(f"Unique allocations found: {len(unique_allocations)}")
    print("\nSample unique allocations:")
    for i, alloc in enumerate(list(unique_allocations)[:5]):
        print(f"  {alloc}")