"""
اسکریپت اجرای روش ε-Constraint
برای مسئله لجستیک بشردوستانه
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
dir_path = Path(__file__).parent.parent / 'NSGA_II_HSC'
sys.path.append(str(dir_path))
from main import Main
from epsilon_constraint_humanitarian import EpsilonConstraintHumanitarian
import time
import pandas as pd
import json


def run_epsilon_constraint_analysis():
    """
    اجرای کامل روش ε-Constraint با تحلیل نتایج
    """
    print("\n" + "="*70)
    print("ε-CONSTRAINT METHOD FOR HUMANITARIAN LOGISTICS")
    print("="*70)
    
    # ایجاد instance از مسئله
    print("\nInitializing problem instance...")
    main_instance = Main()
    
    # ================ اجرای ε-Constraint ================
    print("\n" + "-"*70)
    print("Running ε-Constraint Method...")
    print("-"*70)
    
    start_time = time.time()
    
    # ایجاد ε-constraint solver
    epsilon_solver = EpsilonConstraintHumanitarian(main_instance)
    
    # اجرا با تنظیمات مختلف برای سه حالت (با اهداف اصلی متفاوت)
    results_by_objective = {}
    
    # حالت 1: F1 به عنوان هدف اصلی
    print("\n>>> Optimization with F1 (Distance) as primary objective...")
    epsilon_solver_f1 = EpsilonConstraintHumanitarian(main_instance)
    pareto_f1 = epsilon_solver_f1.run_epsilon_constraint(
        n_epsilon_levels=4,
        primary_objective='F1'
    )
    results_by_objective['F1'] = epsilon_solver_f1
    
    # حالت 2: F2 به عنوان هدف اصلی
    print("\n>>> Optimization with F2 (Unmet Demand) as primary objective...")
    epsilon_solver_f2 = EpsilonConstraintHumanitarian(main_instance)
    pareto_f2 = epsilon_solver_f2.run_epsilon_constraint(
        n_epsilon_levels=4,
        primary_objective='F2'
    )
    results_by_objective['F2'] = epsilon_solver_f2
    
    # حالت 3: F3 به عنوان هدف اصلی
    print("\n>>> Optimization with F3 (Death Probability) as primary objective...")
    epsilon_solver_f3 = EpsilonConstraintHumanitarian(main_instance)
    pareto_f3 = epsilon_solver_f3.run_epsilon_constraint(
        n_epsilon_levels=4,
        primary_objective='F3'
    )
    results_by_objective['F3'] = epsilon_solver_f3
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"{'='*70}")
    
    # ================ تحلیل و مقایسه نتایج ================
    analyze_results(results_by_objective)
    
    # ================ رسم نمودارها ================
    plot_epsilon_results(results_by_objective)
    
    # ================ ذخیره نتایج ================
    save_all_results(results_by_objective, total_time)
    
    return results_by_objective


def analyze_results(results_dict):
    """
    تحلیل نتایج ε-Constraint با اهداف اصلی مختلف
    """
    print("\n" + "="*70)
    print("ANALYSIS OF RESULTS")
    print("="*70)
    
    # جدول خلاصه نتایج
    summary_data = {
        'Primary Objective': [],
        'Total Solutions': [],
        'Pareto Solutions': [],
        'F1 Min': [],
        'F2 Min': [],
        'F3 Min': [],
        'F1 Range': [],
        'F2 Range': [],
        'F3 Range': []
    }
    
    for obj_name, solver in results_dict.items():
        if len(solver.pareto_front) > 0:
            pf_objectives = np.array([sol.objectives for sol in solver.pareto_front])
            
            summary_data['Primary Objective'].append(obj_name)
            summary_data['Total Solutions'].append(len(solver.solutions))
            summary_data['Pareto Solutions'].append(len(solver.pareto_front))
            summary_data['F1 Min'].append(f"{np.min(pf_objectives[:, 0]):.2f}")
            summary_data['F2 Min'].append(f"{np.min(pf_objectives[:, 1]):.2f}")
            summary_data['F3 Min'].append(f"{np.min(pf_objectives[:, 2]):.4f}")
            summary_data['F1 Range'].append(f"[{np.min(pf_objectives[:, 0]):.0f}, {np.max(pf_objectives[:, 0]):.0f}]")
            summary_data['F2 Range'].append(f"[{np.min(pf_objectives[:, 1]):.0f}, {np.max(pf_objectives[:, 1]):.0f}]")
            summary_data['F3 Range'].append(f"[{np.min(pf_objectives[:, 2]):.3f}, {np.max(pf_objectives[:, 2]):.3f}]")
    
    df_summary = pd.DataFrame(summary_data)
    print("\nSummary Table:")
    print(df_summary.to_string(index=False))
    
    # بهترین راه‌حل برای هر هدف
    print("\n" + "-"*70)
    print("Best Solutions for Each Objective:")
    print("-"*70)
    
    # جمع‌آوری همه راه‌حل‌های پارتو
    all_pareto_solutions = []
    for solver in results_dict.values():
        all_pareto_solutions.extend(solver.pareto_front)
    
    if all_pareto_solutions:
        all_objectives = np.array([sol.objectives for sol in all_pareto_solutions])
        
        # بهترین F1
        best_f1_idx = np.argmin(all_objectives[:, 0])
        print(f"\nBest F1 (Distance):")
        print(f"  F1 = {all_objectives[best_f1_idx, 0]:.2f}")
        print(f"  F2 = {all_objectives[best_f1_idx, 1]:.2f}")
        print(f"  F3 = {all_objectives[best_f1_idx, 2]:.4f}")
        
        # بهترین F2
        best_f2_idx = np.argmin(all_objectives[:, 1])
        print(f"\nBest F2 (Unmet Demand):")
        print(f"  F1 = {all_objectives[best_f2_idx, 0]:.2f}")
        print(f"  F2 = {all_objectives[best_f2_idx, 1]:.2f}")
        print(f"  F3 = {all_objectives[best_f2_idx, 2]:.4f}")
        
        # بهترین F3
        best_f3_idx = np.argmin(all_objectives[:, 2])
        print(f"\nBest F3 (Death Probability):")
        print(f"  F1 = {all_objectives[best_f3_idx, 0]:.2f}")
        print(f"  F2 = {all_objectives[best_f3_idx, 1]:.2f}")
        print(f"  F3 = {all_objectives[best_f3_idx, 2]:.4f}")
        
        # راه‌حل متوازن (کمترین فاصله از نقطه ایده‌آل)
        ideal_point = np.min(all_objectives, axis=0)
        nadir_point = np.max(all_objectives, axis=0)
        normalized_objectives = (all_objectives - ideal_point) / (nadir_point - ideal_point + 1e-10)
        distances_to_ideal = np.linalg.norm(normalized_objectives, axis=1)
        balanced_idx = np.argmin(distances_to_ideal)
        
        print(f"\nBalanced Solution (closest to ideal point):")
        print(f"  F1 = {all_objectives[balanced_idx, 0]:.2f}")
        print(f"  F2 = {all_objectives[balanced_idx, 1]:.2f}")
        print(f"  F3 = {all_objectives[balanced_idx, 2]:.4f}")


def plot_epsilon_results(results_dict):
    """
    رسم نمودارهای جامع نتایج ε-Constraint
    """
    fig = plt.figure(figsize=(20, 12))
    
    # رنگ‌ها برای هر هدف اصلی
    colors = {'F1': 'red', 'F2': 'blue', 'F3': 'green'}
    markers = {'F1': 'o', 'F2': 's', 'F3': '^'}
    
    # --- نمودار 1: 3D Pareto Front (همه با هم)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    for obj_name, solver in results_dict.items():
        if len(solver.pareto_front) > 0:
            pf_objectives = np.array([sol.objectives for sol in solver.pareto_front])
            ax1.scatter(pf_objectives[:, 0], pf_objectives[:, 1], pf_objectives[:, 2],
                       c=colors[obj_name], s=40, alpha=0.6, 
                       label=f'Primary: {obj_name}', marker=markers[obj_name])
    
    ax1.set_xlabel('F1: Distance')
    ax1.set_ylabel('F2: Unmet Demand')
    ax1.set_zlabel('F3: Death Probability')
    ax1.set_title('3D Pareto Fronts - All Primary Objectives')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- نمودارهای 2D برای مقایسه
    projections = [
        (0, 1, 'F1: Distance', 'F2: Unmet Demand', 2),
        (0, 2, 'F1: Distance', 'F3: Death Probability', 3),
        (1, 2, 'F2: Unmet Demand', 'F3: Death Probability', 4)
    ]
    
    for i, j, xlabel, ylabel, subplot_idx in projections:
        ax = fig.add_subplot(2, 3, subplot_idx)
        
        for obj_name, solver in results_dict.items():
            if len(solver.pareto_front) > 0:
                pf_objectives = np.array([sol.objectives for sol in solver.pareto_front])
                ax.scatter(pf_objectives[:, i], pf_objectives[:, j],
                          c=colors[obj_name], s=30, alpha=0.6,
                          label=f'{obj_name}', marker=markers[obj_name])
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{xlabel} vs {ylabel}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # --- نمودار 5: تعداد راه‌حل‌ها
    ax5 = fig.add_subplot(2, 3, 5)
    
    objectives = []
    total_sols = []
    pareto_sols = []
    
    for obj_name, solver in results_dict.items():
        objectives.append(obj_name)
        total_sols.append(len(solver.solutions))
        pareto_sols.append(len(solver.pareto_front))
    
    x_pos = np.arange(len(objectives))
    width = 0.35
    
    bars1 = ax5.bar(x_pos - width/2, total_sols, width, label='Total Solutions', alpha=0.7)
    bars2 = ax5.bar(x_pos + width/2, pareto_sols, width, label='Pareto Solutions', alpha=0.7)
    
    ax5.set_xlabel('Primary Objective')
    ax5.set_ylabel('Number of Solutions')
    ax5.set_title('Solution Count by Primary Objective')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(objectives)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # اضافه کردن مقادیر روی میله‌ها
    for bar in bars1:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # --- نمودار 6: محدوده اهداف
    ax6 = fig.add_subplot(2, 3, 6)
    
    # داده‌ها برای box plot
    all_f1, all_f2, all_f3 = [], [], []
    
    for solver in results_dict.values():
        if len(solver.pareto_front) > 0:
            pf_objectives = np.array([sol.objectives for sol in solver.pareto_front])
            all_f1.extend(pf_objectives[:, 0])
            all_f2.extend(pf_objectives[:, 1])
            all_f3.extend(pf_objectives[:, 2])
    
    if all_f1:
        # نرمال‌سازی برای نمایش بهتر
        normalized_data = [
            (np.array(all_f1) - np.min(all_f1)) / (np.max(all_f1) - np.min(all_f1) + 1e-10),
            (np.array(all_f2) - np.min(all_f2)) / (np.max(all_f2) - np.min(all_f2) + 1e-10),
            (np.array(all_f3) - np.min(all_f3)) / (np.max(all_f3) - np.min(all_f3) + 1e-10)
        ]
        
        bp = ax6.boxplot(normalized_data, labels=['F1', 'F2', 'F3'], patch_artist=True)
        
        # رنگ‌آمیزی box plot
        colors_bp = ['lightcoral', 'lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors_bp):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax6.set_ylabel('Normalized Value [0,1]')
        ax6.set_title('Distribution of Objective Values (Normalized)')
        ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('ε-Constraint Method Results Analysis', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig('epsilon_constraint_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_all_results(results_dict, total_time):
    """
    ذخیره تمام نتایج در فایل‌های مختلف
    """
    # ذخیره نتایج JSON
    all_results = {
        'method': 'epsilon-constraint',
        'total_execution_time': total_time,
        'results_by_primary_objective': {}
    }
    
    for obj_name, solver in results_dict.items():
        if len(solver.pareto_front) > 0:
            pf_objectives = np.array([sol.objectives for sol in solver.pareto_front])
            all_objectives = np.array([sol.objectives for sol in solver.solutions])
            
            all_results['results_by_primary_objective'][obj_name] = {
                'total_solutions': len(solver.solutions),
                'pareto_solutions': len(solver.pareto_front),
                'pareto_front': pf_objectives.tolist(),
                'all_solutions': all_objectives.tolist(),
                'statistics': {
                    'F1': {
                        'min': float(np.min(pf_objectives[:, 0])),
                        'max': float(np.max(pf_objectives[:, 0])),
                        'mean': float(np.mean(pf_objectives[:, 0])),
                        'std': float(np.std(pf_objectives[:, 0]))
                    },
                    'F2': {
                        'min': float(np.min(pf_objectives[:, 1])),
                        'max': float(np.max(pf_objectives[:, 1])),
                        'mean': float(np.mean(pf_objectives[:, 1])),
                        'std': float(np.std(pf_objectives[:, 1]))
                    },
                    'F3': {
                        'min': float(np.min(pf_objectives[:, 2])),
                        'max': float(np.max(pf_objectives[:, 2])),
                        'mean': float(np.mean(pf_objectives[:, 2])),
                        'std': float(np.std(pf_objectives[:, 2]))
                    }
                }
            }
    
    # ذخیره در فایل JSON
    with open('epsilon_constraint_all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # ذخیره نتایج پارتو در CSV برای تحلیل‌های بعدی
    all_pareto_data = []
    for obj_name, solver in results_dict.items():
        for sol in solver.pareto_front:
            all_pareto_data.append({
                'Primary_Objective': obj_name,
                'F1_Distance': sol.objectives[0],
                'F2_UnmetDemand': sol.objectives[1],
                'F3_DeathProbability': sol.objectives[2]
            })
    
    if all_pareto_data:
        df_pareto = pd.DataFrame(all_pareto_data)
        df_pareto.to_csv('epsilon_constraint_pareto_solutions.csv', index=False)
    
    print("\n" + "-"*70)
    print("Results saved to:")
    print("  - epsilon_constraint_all_results.json")
    print("  - epsilon_constraint_pareto_solutions.csv")
    print("  - epsilon_constraint_analysis.png")
    print("-"*70)


# ================ اجرای برنامه ================

if __name__ == "__main__":
    # اجرای کامل روش ε-Constraint
    results = run_epsilon_constraint_analysis()
    
    print("\n" + "="*70)
    print("ε-CONSTRAINT OPTIMIZATION COMPLETED!")
    print("="*70)
    print("\nYou can now analyze the results from the generated files.")
    print("To run with different settings, modify the parameters in")
    print("run_epsilon_constraint_analysis() function.")
