import multiprocessing as mp
import matplotlib
# Backend is not forced to 'Agg' to allow plt.show() to work.
# The plotting is done in the main process, so GUI blocking is not an issue.
import matplotlib.pyplot as plt
import numpy as np
from llm_nsga2 import LLM_NSGA2_Humanitarian
from nsga2 import NSGA2_Humanitarian
from convergence_metrics import ConvergenceMetrics
import time
import traceback

def run_llm_nsga2(problem, result_queue, algorithm_name):
    """Run LLM-enhanced NSGA-II algorithm"""
    try:
        # Initialize algorithm parameters
        shelter_id = problem.get('shelter_id', list(range(1, 6)))  # 5 shelters
        distribution_center_id = problem.get('distribution_center_id', list(range(1, 4)))  # 3 distribution centers
        damage_points_id = problem.get('damage_points_id', list(range(1, 4)))  # 3 damage points
        hospital_id = problem.get('hospital_id', list(range(1, 3)))  # 2 hospitals
        temporary_medical_id = problem.get('temporary_medical_id', list(range(1, 3)))  # 2 temporary medical centers

        from ai_config import get_ai_config
        ai_config = get_ai_config()

        # Initialize LLM NSGA-II
        llm_nsga2 = LLM_NSGA2_Humanitarian(
            shelter_id=shelter_id,
            distribution_center_id=distribution_center_id,
            damage_points_id=damage_points_id,
            hospital_id=hospital_id,
            temporary_medical_id=temporary_medical_id,
            max_iter=200,  # Reduced for faster execution
            pop_size=150,
            p_crossover=0.9,
            p_mutation=0.1,
            elitism_rate=0.1,
            verbose=True,  # Disable verbose output for parallel execution
            openrouter_api_key=ai_config['api_key'],
            use_ai_optimization=ai_config['use_optimization'],
            llm_iter=10,
            use_llm_init_pop = False)

        start_time = time.time()
        result = llm_nsga2.run(problem)
        end_time = time.time()

        # Extract metrics history
        metrics: ConvergenceMetrics = result.get('metrics', {})

        result_data = {
            'algorithm': algorithm_name,
            'hypervolume': metrics.normalized_hypervolume,
            'spacing': metrics.normalized_spacing,
            'diversity': metrics.normalized_diversity,
            'min_objs': metrics.min_objectives_history,
            'mean_objs':metrics.mean_objectives_history,
            'pareto_count': metrics.n_pareto_history,
            'generations': len(metrics.normalized_hypervolume),
            'execution_time': end_time - start_time,
            'final_hypervolume': float(metrics.normalized_hypervolume[-1])
        }

        result_queue.put(result_data)
        print(f"[SUCCESS] {algorithm_name} completed in {result_data['execution_time']:.2f} seconds")

    except Exception as e:
        error_data = {
            'algorithm': algorithm_name,
            'error': str(e),
            'hypervolume': [],
            'spacing': [],
            'diversity': [],
            'min_objs': [],
            'mean_objs': [],
            'pareto_count': 0,
            'generations': [],
            'execution_time': 0,
            'final_hypervolume': 0
        }
        result_queue.put(error_data)
        print(f"[ERROR] in {__file__} {algorithm_name} failed:")
        traceback.print_exc()

def run_standard_nsga2(problem, result_queue, algorithm_name):
    """Run standard NSGA-II algorithm"""
    try:
        # Initialize algorithm parameters
        shelter_id = problem.get('shelter_id', list(range(1, 6)))  # 5 shelters
        distribution_center_id = problem.get('distribution_center_id', list(range(1, 4)))  # 3 distribution centers
        damage_points_id = problem.get('damage_points_id', list(range(1, 4)))  # 3 damage points
        hospital_id = problem.get('hospital_id', list(range(1, 3)))  # 2 hospitals
        temporary_medical_id = problem.get('temporary_medical_id', list(range(1, 3)))  # 2 temporary medical centers

        # Initialize standard NSGA-II
        nsga2 = NSGA2_Humanitarian(
            shelter_id=shelter_id,
            distribution_center_id=distribution_center_id,
            damage_points_id=damage_points_id,
            hospital_id=hospital_id,
            temporary_medical_id=temporary_medical_id,
            max_iter=200,  # Reduced for faster execution
            pop_size=150,
            p_crossover=0.9,
            p_mutation=0.1,
            elitism_rate=0.1,
            verbose=False  # Disable verbose output for parallel execution
        )

        start_time = time.time()
        result = nsga2.run(problem)
        end_time = time.time()
        
        # Extract metrics history
        metrics: ConvergenceMetrics = result.get('metrics', {})

        result_data = {
            'algorithm': algorithm_name,
            'hypervolume': metrics.normalized_hypervolume,
            'spacing': metrics.normalized_spacing,
            'diversity': metrics.normalized_diversity,
            'min_objs': metrics.min_objectives_history,
            'mean_objs':metrics.mean_objectives_history,
            'pareto_count': metrics.n_pareto_history,
            'generations': len(metrics.normalized_hypervolume),
            'execution_time': end_time - start_time,
            'final_hypervolume': float(metrics.normalized_hypervolume[-1])
        }

        result_queue.put(result_data)
        print(f"[SUCCESS] {algorithm_name} completed in {result_data['execution_time']:.2f} seconds")

    except Exception as e:
        error_data = {
            'algorithm': algorithm_name,
            'error': str(e),
            'hypervolume': [],
            'spacing': [],
            'diversity': [],
            'pareto_count': 0,
            'generations': [],
            'execution_time': 0,
            'final_hypervolume': 0
        }
        result_queue.put(error_data)
        print(f"[ERROR] in {__file__} {algorithm_name} failed:")
        traceback.print_exc()

class RealCostFunction:
    """Real humanitarian cost function for multiprocessing"""
    def __init__(self):
        # Initialize the Main class to get the cost function
        from main import Main
        self.main_instance = Main()

    def __call__(self, chromosomes):
        """Use the real complex_humanitarian_cost function"""
        return self.main_instance.complex_humanitarian_cost(chromosomes)

def create_real_problem():
    """Create a real humanitarian logistics problem"""
    return {
        'cost_function': RealCostFunction(),
        'shelter_id': list(range(8, 19)),  # 11 shelters (8-18)
        'distribution_center_id': list(range(1, 4)),  # 3 distribution centers
        'damage_points_id': list(range(3, 8)),  # 5 damage points (3-7)
        'hospital_id': list(range(1, 5)),  # 4 hospitals
        'temporary_medical_id': list(range(1, 11)),  # 10 temporary medical centers
        'resume': True
    }

def plot_comparison(results):
    """Plot comparison of metrics between the two algorithms"""
    if len(results) != 2:
        print("Error in __file__: Need results from both algorithms")
        return

    llm_result = None
    standard_result = None

    for result in results:
        if 'LLM' in result['algorithm']:
            llm_result = result
        else:
            standard_result = result

    if not llm_result or not standard_result:
        print("Error in __file__: Could not identify LLM and standard results")
        return

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('NSGA-II Algorithms Comparison: LLM-Enhanced vs Standard', fontsize=14)

    # Hypervolume comparison
    ax1 = axes[0, 0]
    if llm_result['hypervolume'] and standard_result['hypervolume']:
        min_len = min(len(llm_result['hypervolume']), len(standard_result['hypervolume']))
        ax1.plot(range(min_len), llm_result['hypervolume'][:min_len],
                label=f"LLM-Enhanced (Final: {llm_result['final_hypervolume']:.4f})",
                color='blue', linewidth=2)
        ax1.plot(range(min_len), standard_result['hypervolume'][:min_len],
                label=f"Standard (Final: {standard_result['final_hypervolume']:.4f})",
                color='red', linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Hypervolume')
        ax1.set_title('Hypervolume Comparison')
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add data point markers
        ax1.scatter(range(min_len), llm_result['hypervolume'][:min_len],
                   color='blue', s=20, alpha=0.7)
        ax1.scatter(range(min_len), standard_result['hypervolume'][:min_len],
                   color='red', s=20, alpha=0.7)

    # Spacing comparison
    ax2 = axes[0, 1]
    if llm_result['spacing'] and standard_result['spacing']:
        min_len = min(len(llm_result['spacing']), len(standard_result['spacing']))
        ax2.plot(range(min_len), llm_result['spacing'][:min_len],
                label='LLM-Enhanced', color='blue', linewidth=2)
        ax2.plot(range(min_len), standard_result['spacing'][:min_len],
                label='Standard', color='red', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Spacing')
        ax2.set_title('Spacing Metric Comparison')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add data point markers
        ax2.scatter(range(min_len), llm_result['spacing'][:min_len],
                   color='blue', s=20, alpha=0.7)
        ax2.scatter(range(min_len), standard_result['spacing'][:min_len],
                   color='red', s=20, alpha=0.7)

    # Diversity comparison
    ax3 = axes[1, 0]
    if llm_result['diversity'] and standard_result['diversity']:
        min_len = min(len(llm_result['diversity']), len(standard_result['diversity']))
        ax3.plot(range(min_len), llm_result['diversity'][:min_len],
                label='LLM-Enhanced', color='blue', linewidth=2)
        ax3.plot(range(min_len), standard_result['diversity'][:min_len],
                label='Standard', color='red', linewidth=2)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Diversity')
        ax3.set_title('Population Diversity Comparison')
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Add data point markers
        ax3.scatter(range(min_len), llm_result['diversity'][:min_len],
                   color='blue', s=20, alpha=0.7)
        ax3.scatter(range(min_len), standard_result['diversity'][:min_len],
                   color='red', s=20, alpha=0.7)

    # Pareto front size comparison
    ax4 = axes[1, 1]
    if llm_result['pareto_count'] and standard_result['pareto_count']:
        min_len = min(len(llm_result['pareto_count']), len(standard_result['pareto_count']))
        ax4.plot(range(min_len), llm_result['pareto_count'][:min_len],
                label='LLM-Enhanced', color='blue', linewidth=2)
        ax4.plot(range(min_len), standard_result['pareto_count'][:min_len],
                label='Standard', color='red', linewidth=2)
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Pareto Front Size')
        ax4.set_title('Pareto Front Size Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Add data point markers
        ax4.scatter(range(min_len), llm_result['pareto_count'][:min_len],
                   color='blue', s=20, alpha=0.7)
        ax4.scatter(range(min_len), standard_result['pareto_count'][:min_len],
                   color='red', s=20, alpha=0.7)

    plt.tight_layout()
    plt.show(block=True)
    # Add execution time information
    execution_info = f"""
    Execution Times:
    LLM-Enhanced NSGA-II: {llm_result['execution_time']:.2f} seconds
    Standard NSGA-II: {standard_result['execution_time']:.2f} seconds

    Final Hypervolume:
    LLM-Enhanced: {llm_result['final_hypervolume']:.4f}
    Standard: {standard_result['final_hypervolume']:.4f}
    """

    plt.figtext(0.02, 0.02, execution_info, fontsize=9,
                verticalalignment='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))

    # Always save to file to avoid interactive GUI blocking
    # Do not block; close the figure to free resources
    plt.close(fig)

    # Print summary
    print("\n" + "="*60)
    print("NSGA-II ALGORITHMS COMPARISON SUMMARY")
    print("="*60)
    print(f"LLM-Enhanced NSGA-II Execution Time: {llm_result['execution_time']:.2f} seconds")
    print(f"Standard NSGA-II Execution Time: {standard_result['execution_time']:.2f} seconds")
    print(f"LLM-Enhanced Final Hypervolume: {llm_result['final_hypervolume']:.4f}")
    print(f"Standard Final Hypervolume: {standard_result['final_hypervolume']:.4f}")

    if llm_result['final_hypervolume'] > standard_result['final_hypervolume']:
        print("[WINNER] LLM-Enhanced NSGA-II achieved better final hypervolume!")
    elif llm_result['final_hypervolume'] < standard_result['final_hypervolume']:
        print("[WINNER] Standard NSGA-II achieved better final hypervolume!")
    else:
        print("[TIE] Both algorithms achieved similar final hypervolume!")

def plot_objective_trends(results):
    """Plot comparison of min and mean objective values over generations."""
    llm_result = next((r for r in results if 'LLM' in r['algorithm']), None)
    standard_result = next((r for r in results if 'Standard' in r['algorithm']), None)

    if not llm_result or not standard_result:
        print("Could not find both LLM and Standard results for objective plotting.")
        return

    # Check if data exists
    if not llm_result.get('min_objs') or not standard_result.get('min_objs'):
        print("Objective data ('min_objs' or 'mean_objs') not found in results.")
        return

    num_objectives = len(llm_result['min_objs'][0])
    fig, axes = plt.subplots(2, num_objectives, figsize=(7 * num_objectives, 10), sharex=True)
    fig.suptitle('Objective Value Trends Comparison', fontsize=16)
    
    # Plot Min Objectives
    for i in range(num_objectives):
        ax = axes[0, i]
        llm_min_obj = [gen[i] for gen in llm_result['min_objs']]
        std_min_obj = [gen[i] for gen in standard_result['min_objs']]
        min_len = min(len(llm_min_obj), len(std_min_obj))

        ax.plot(range(min_len), llm_min_obj[:min_len], label='LLM-Enhanced', color='blue', linestyle='-')
        ax.plot(range(min_len), std_min_obj[:min_len], label='Standard', color='red', linestyle='-')
        ax.set_title(f'Min Objective {i+1}')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.4)
        ax.legend()

    # Plot Mean Objectives
    for i in range(num_objectives):
        ax = axes[1, i]
        llm_mean_obj = [gen[i] for gen in llm_result['mean_objs']]
        std_mean_obj = [gen[i] for gen in standard_result['mean_objs']]
        min_len = min(len(llm_mean_obj), len(std_mean_obj))

        ax.plot(range(min_len), llm_mean_obj[:min_len], label='LLM-Enhanced', color='cyan', linestyle='--')
        ax.plot(range(min_len), std_mean_obj[:min_len], label='Standard', color='magenta', linestyle='--')
        ax.set_title(f'Mean Objective {i+1}')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.4)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    """Main function to run parallel comparison"""
    print("Starting Parallel NSGA-II Comparison")
    print("="*50)

    # Create problem definition
    # NOTE: Using real humanitarian logistics problem
    problem = create_real_problem()

    # Create a queue to collect results
    result_queue = mp.Queue()

    # Create processes for both algorithms
    llm_process = mp.Process(
        target=run_llm_nsga2,
        args=(problem, result_queue, "LLM-Enhanced NSGA-II")
    )

    standard_process = mp.Process(
        target=run_standard_nsga2,
        args=(problem, result_queue, "Standard NSGA-II")
    )

    # Start both processes
    print("Starting parallel execution...")
    start_time = time.time()

    llm_process.start()
    standard_process.start()

    # Collect results first, then join. This is a more robust pattern.
    results = []
    for _ in range(2): # We expect two results
        try:
            # Wait for a result to appear in the queue
            result = result_queue.get(timeout=3600) # Generous 1-hour timeout
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Did not receive a result from a process: {e}")
            traceback.print_exc()
            break # Exit loop if a result is not received

    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

    # Now that results are collected, join the processes
    llm_process.join(timeout=60)
    standard_process.join(timeout=60)

    # If any process is still alive after collecting results, terminate it
    if llm_process.is_alive():
        print("[WARN] LLM-Enhanced process did not terminate after collecting results. Forcing termination...")
        llm_process.terminate()
        llm_process.join()
    if standard_process.is_alive():
        print("[WARN] Standard process did not terminate after collecting results. Forcing termination...")
        standard_process.terminate()
        standard_process.join()

    if len(results) == 2:
        print("Generating comparison plots...")
        plot_comparison(results)
        plot_objective_trends(results)
    else:
        print(f"in __file__ Expected 2 results, got {len(results)}")
        for result in results:
            print(f"Result: {result.get('algorithm', 'Unknown')} - Error: {result.get('error', 'None')}")

if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.set_start_method('spawn', force=True)
    main()