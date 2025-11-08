#!/usr/bin/env python3
"""
Test script for comparing NSGA-II and GWO algorithms on humanitarian logistics problem
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_updated import Main
import numpy as np
import matplotlib.pyplot as plt
import time


def test_algorithms():
    """Test and compare both algorithms"""
    
    print("\n" + "="*70)
    print(" MULTI-OBJECTIVE HUMANITARIAN LOGISTICS OPTIMIZATION ")
    print(" Comparing NSGA-II vs GWO (Grey Wolf Optimizer) ")
    print("="*70)
    
    # Initialize main class
    print("\nInitializing problem...")
    m = Main()
    
    # Configuration
    MAX_ITER = 30  # Reduced for faster testing
    POP_SIZE = 50  # Reduced for faster testing
    
    print(f"\nConfiguration:")
    print(f"  - Max iterations: {MAX_ITER}")
    print(f"  - Population size: {POP_SIZE}")
    print(f"  - Objectives: F1 (Distance), F2 (Unmet Demand), F3 (Death Probability)")
    
    # Test NSGA-II
    print("\n" + "-"*70)
    print("Testing NSGA-II...")
    print("-"*70)
    
    start_time = time.time()
    results_nsga = m.run_algorithm(algorithm='nsga', max_iter=MAX_ITER, pop_size=POP_SIZE)
    nsga_time = time.time() - start_time
    
    # Test GWO
    print("\n" + "-"*70)
    print("Testing GWO...")
    print("-"*70)
    
    start_time = time.time()
    results_gwo = m.run_algorithm(algorithm='gwo', max_iter=MAX_ITER, pop_size=POP_SIZE)
    gwo_time = time.time() - start_time
    
    # Extract results
    pf_nsga = np.array([ind['cost'] for ind in results_nsga['nsga']['pareto_pop']])
    pf_gwo = np.array([ind['cost'] for ind in results_gwo['gwo']['pareto_pop']])
    
    # Print comparison summary
    print("\n" + "="*70)
    print(" FINAL COMPARISON RESULTS ")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'NSGA-II':>15} {'GWO':>15}")
    print("-"*60)
    print(f"{'Execution Time (s)':<30} {nsga_time:>15.2f} {gwo_time:>15.2f}")
    print(f"{'Pareto Front Size':<30} {len(pf_nsga):>15} {len(pf_gwo):>15}")
    print(f"{'Best F1 (Distance)':<30} {np.min(pf_nsga[:, 0]):>15.2f} {np.min(pf_gwo[:, 0]):>15.2f}")
    print(f"{'Best F2 (Unmet Demand)':<30} {np.min(pf_nsga[:, 1]):>15.2f} {np.min(pf_gwo[:, 1]):>15.2f}")
    print(f"{'Best F3 (Death Probability)':<30} {np.min(pf_nsga[:, 2]):>15.2f} {np.min(pf_gwo[:, 2]):>15.2f}")
    
    # Calculate hypervolume if available
    if hasattr(results_nsga['nsga']['metrics'], 'normalized_hypervolume'):
        hv_nsga = results_nsga['nsga']['metrics'].normalized_hypervolume[-1]
        hv_gwo = results_gwo['gwo']['metrics'].normalized_hypervolume[-1]
        print(f"{'Final Hypervolume':<30} {hv_nsga:>15.4f} {hv_gwo:>15.4f}")
    
    # Calculate spacing if available
    if hasattr(results_nsga['nsga']['metrics'], 'normalized_spacing'):
        sp_nsga = results_nsga['nsga']['metrics'].normalized_spacing[-1]
        sp_gwo = results_gwo['gwo']['metrics'].normalized_spacing[-1]
        print(f"{'Final Spacing':<30} {sp_nsga:>15.4f} {sp_gwo:>15.4f}")
    
    # Determine winner for each metric
    print("\n" + "="*70)
    print(" PERFORMANCE ANALYSIS ")
    print("="*70)
    
    winners = {}
    
    # Speed
    if nsga_time < gwo_time:
        winners['speed'] = 'NSGA-II'
        speed_diff = ((gwo_time - nsga_time) / gwo_time) * 100
        print(f"✓ Speed Winner: NSGA-II ({speed_diff:.1f}% faster)")
    else:
        winners['speed'] = 'GWO'
        speed_diff = ((nsga_time - gwo_time) / nsga_time) * 100
        print(f"✓ Speed Winner: GWO ({speed_diff:.1f}% faster)")
    
    # Pareto front size
    if len(pf_nsga) > len(pf_gwo):
        winners['diversity'] = 'NSGA-II'
        print(f"✓ Diversity Winner: NSGA-II ({len(pf_nsga)} solutions)")
    else:
        winners['diversity'] = 'GWO'
        print(f"✓ Diversity Winner: GWO ({len(pf_gwo)} solutions)")
    
    # Objective values
    if np.min(pf_nsga[:, 0]) < np.min(pf_gwo[:, 0]):
        winners['F1'] = 'NSGA-II'
    else:
        winners['F1'] = 'GWO'
    
    if np.min(pf_nsga[:, 1]) < np.min(pf_gwo[:, 1]):
        winners['F2'] = 'NSGA-II'
    else:
        winners['F2'] = 'GWO'
    
    if np.min(pf_nsga[:, 2]) < np.min(pf_gwo[:, 2]):
        winners['F3'] = 'NSGA-II'
    else:
        winners['F3'] = 'GWO'
    
    print(f"✓ Best F1 Winner: {winners['F1']}")
    print(f"✓ Best F2 Winner: {winners['F2']}")
    print(f"✓ Best F3 Winner: {winners['F3']}")
    
    # Overall winner
    nsga_wins = sum(1 for v in winners.values() if v == 'NSGA-II')
    gwo_wins = sum(1 for v in winners.values() if v == 'GWO')
    
    print("\n" + "="*70)
    if nsga_wins > gwo_wins:
        print(f" OVERALL WINNER: NSGA-II ({nsga_wins}/{len(winners)} metrics)")
    elif gwo_wins > nsga_wins:
        print(f" OVERALL WINNER: GWO ({gwo_wins}/{len(winners)} metrics)")
    else:
        print(f" RESULT: TIE ({nsga_wins}/{len(winners)} metrics each)")
    print("="*70)
    
    # Create simple comparison plot
    create_simple_comparison_plot(pf_nsga, pf_gwo)
    
    return results_nsga, results_gwo


def create_simple_comparison_plot(pf_nsga, pf_gwo):
    """Create a simple comparison plot"""
    
    fig = plt.figure(figsize=(15, 5))
    
    # F1 vs F2
    ax1 = fig.add_subplot(131)
    ax1.scatter(pf_nsga[:, 0], pf_nsga[:, 1], c='blue', s=30, alpha=0.6, label='NSGA-II')
    ax1.scatter(pf_gwo[:, 0], pf_gwo[:, 1], c='red', s=30, alpha=0.6, label='GWO')
    ax1.set_xlabel('F1: Distance')
    ax1.set_ylabel('F2: Unmet Demand')
    ax1.set_title('F1 vs F2 Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # F1 vs F3
    ax2 = fig.add_subplot(132)
    ax2.scatter(pf_nsga[:, 0], pf_nsga[:, 2], c='blue', s=30, alpha=0.6, label='NSGA-II')
    ax2.scatter(pf_gwo[:, 0], pf_gwo[:, 2], c='red', s=30, alpha=0.6, label='GWO')
    ax2.set_xlabel('F1: Distance')
    ax2.set_ylabel('F3: Death Probability')
    ax2.set_title('F1 vs F3 Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # F2 vs F3
    ax3 = fig.add_subplot(133)
    ax3.scatter(pf_nsga[:, 1], pf_nsga[:, 2], c='blue', s=30, alpha=0.6, label='NSGA-II')
    ax3.scatter(pf_gwo[:, 1], pf_gwo[:, 2], c='red', s=30, alpha=0.6, label='GWO')
    ax3.set_xlabel('F2: Unmet Demand')
    ax3.set_ylabel('F3: Death Probability')
    ax3.set_title('F2 vs F3 Comparison')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.suptitle('NSGA-II vs GWO: Pareto Front Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('quick_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    try:
        results = test_algorithms()
        print("\n✓ All tests completed successfully!")
        print("✓ Check generated plots: ")
        print("  - nsga_convergence.png")
        print("  - gwo_convergence.png")
        print("  - quick_comparison.png")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
