import multiprocessing as mp
import matplotlib
# Backend is not forced to 'Agg' to allow plt.show() to work.
# The plotting is done in the main process, so GUI blocking is not an issue.
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.widgets import Button
import datetime
import numpy as np
from llm_nsga2 import LLM_NSGA2_Humanitarian
from nsga2 import NSGA2_Humanitarian
from convergence_metrics import ConvergenceMetrics
import time
import traceback

# Instance مشترک از کلاس Main برای استفاده در همه جا
_main_instance = None
def get_main_instance():
    """Lazy initialization of Main instance"""
    global _main_instance
    if _main_instance is None:
        from main import Main
        _main_instance = Main()
    return _main_instance

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
        
        # دریافت متغیرها از instance مشترک Main
        main_instance = get_main_instance()
        distances = main_instance.distance
        homeless = main_instance.homeless
        cost = main_instance.cost
        capacity = main_instance.capacity

        # Initialize LLM NSGA-II
        llm_nsga2 = LLM_NSGA2_Humanitarian(
            max_iter=300,
            pop_size=150,
            p_crossover=0.9,
            p_mutation=0.1,
            verbose=True,
            resume=True,
            using_ollama=False,
            use_llm_init_pop=False,
            llm_iter=5,
            shelter_id=shelter_id,
            distribution_center_id=distribution_center_id,
            damage_points_id=damage_points_id,
            hospital_id=hospital_id,
            temporary_medical_id=temporary_medical_id,
            distances=distances,
            homeless=homeless,
            cost=cost,
            capacity=capacity,
            )

        start_time = time.time()
        result = llm_nsga2.run(problem)
        end_time = time.time()

        # Extract metrics history
        metrics: ConvergenceMetrics = result.get('metrics', {})

        result_data = {
            'algorithm': algorithm_name,
            'hypervolume': metrics.normalized_hypervolume,
            'spacing': metrics.normalized_spacing,
            'spread': metrics.normalized_spread,
            'min_objs': metrics.min_objectives_history,
            'mean_objs':metrics.mean_objectives_history,
            'pareto_count': metrics.n_pareto_history,
            'pareto_front': [ind['cost'] for ind in result.get('pareto_pop', [])],
            'pareto_history': [[ind['cost'] for ind in gen] for gen in result.get('pareto_history', [])],
            'generations': len(metrics.normalized_hypervolume),
            'execution_time': end_time - start_time,
            # 'final_hypervolume': float(metrics.normalized_hypervolume[-1])
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
            # 'final_hypervolume': 0
        }
        result_queue.put(error_data)
        print(f"[ERROR] in {__file__} {algorithm_name} failed:")
        traceback.print_exc()

def run_standard_nsga2_RandomPop(problem, result_queue, algorithm_name, RUN_NUM):
    """Run standard NSGA-II algorithm with random pop"""
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
            max_iter=300,
            pop_size=150,
            p_crossover=0.9,
            p_mutation=0.1,
            verbose=True,  # Disable verbose output for parallel execution
            resume = True
        )

        start_time = time.time()
        result = nsga2.run(problem, RUN_NUM, INIT_POP_TYPE="random")
        end_time = time.time()
        
        # Extract metrics history
        metrics: ConvergenceMetrics = result.get('metrics', {})

        result_data = {
            'algorithm': algorithm_name,
            'hypervolume': metrics.normalized_hypervolume,
            'spacing': metrics.normalized_spacing,
            'spread': metrics.normalized_spread,
            'min_objs': metrics.min_objectives_history,
            'mean_objs':metrics.mean_objectives_history,
            'pareto_count': metrics.n_pareto_history,
            'pareto_front': [ind['cost'] for ind in result.get('pareto_pop', [])],
            'pareto_history': [[ind['cost'] for ind in gen] for gen in result.get('pareto_history', [])],
            'generations': len(metrics.normalized_hypervolume),
            'execution_time': end_time - start_time,
            # 'final_hypervolume': float(metrics.normalized_hypervolume[-1])
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
            # 'final_hypervolume': 0
        }
        result_queue.put(error_data)
        print(f"[ERROR] in {__file__} {algorithm_name} failed:")
        traceback.print_exc()

def run_standard_nsga2_LLMPop(problem, result_queue, algorithm_name, RUN_NUM):
    """Run standard NSGA-II algorithm with LLM pop"""
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
            max_iter=300,
            pop_size=150,
            p_crossover=0.9,
            p_mutation=0.1,
            verbose=True,  # Disable verbose output for parallel execution
            resume = True
        )

        start_time = time.time()
        result = nsga2.run(problem, RUN_NUM, INIT_POP_TYPE="llm")
        end_time = time.time()
        
        # Extract metrics history
        metrics: ConvergenceMetrics = result.get('metrics', {})

        result_data = {
            'algorithm': algorithm_name,
            'hypervolume': metrics.normalized_hypervolume,
            'spacing': metrics.normalized_spacing,
            'spread': metrics.normalized_spread,
            'min_objs': metrics.min_objectives_history,
            'mean_objs':metrics.mean_objectives_history,
            'pareto_count': metrics.n_pareto_history,
            'pareto_front': [ind['cost'] for ind in result.get('pareto_pop', [])],
            'pareto_history': [[ind['cost'] for ind in gen] for gen in result.get('pareto_history', [])],
            'generations': len(metrics.normalized_hypervolume),
            'execution_time': end_time - start_time,
            # 'final_hypervolume': float(metrics.normalized_hypervolume[-1])
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
            # 'final_hypervolume': 0
        }
        result_queue.put(error_data)
        print(f"[ERROR] in {__file__} {algorithm_name} failed:")
        traceback.print_exc()

class RealCostFunction:
    """Real humanitarian cost function for multiprocessing"""
    def __init__(self):
        # استفاده از instance مشترک Main
        self.main_instance = get_main_instance()

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
    }

def plot_comparison(results, std_pareto_fronts_list, llm_pareto_fronts_list, reference_point, idx):
    """Plot comparison of metrics between the two algorithms"""
    llm_result = None
    standard_result = None
    for result in results:
        if 'LLM' in result['algorithm']:
            llm_result = result
        else:
            standard_result = result
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('NSGA-II Algorithms Comparison: LLM-Enhanced vs Standard', fontsize=14)

    # Hypervolume comparison
    # ax1 = axes[0, 0]
    # if llm_result['hypervolume'] and standard_result['hypervolume']:
    #     min_len = min(len(llm_result['hypervolume']), len(standard_result['hypervolume']))
    #     ax1.plot(range(min_len), llm_result['hypervolume'][:min_len],
    #             label=f"LLM-Enhanced (Final: {llm_result['final_hypervolume']:.4f})",
    #             color='blue', linewidth=2)
    #     ax1.plot(range(min_len), standard_result['hypervolume'][:min_len],
    #             label=f"Standard (Final: {standard_result['final_hypervolume']:.4f})",
    #             color='red', linewidth=2)
    #     ax1.set_xlabel('Generation')
    #     ax1.set_ylabel('Hypervolume')
    #     ax1.set_title('Hypervolume Comparison')
    #     ax1.set_ylim(0, 1)
    #     ax1.legend()
    #     ax1.grid(True, alpha=0.3)
    #     # Add data point markers
    #     ax1.scatter(range(min_len), llm_result['hypervolume'][:min_len],
    #                color='blue', s=20, alpha=0.7)
    #     ax1.scatter(range(min_len), standard_result['hypervolume'][:min_len],
    #                color='red', s=20, alpha=0.7)
    
    from convergence_metrics import ConvergenceMetrics
    metrics = ConvergenceMetrics()
    
    llm_hypervolume_history = metrics.hypervolume(llm_pareto_fronts_list, reference_point)
    standard_hypervolume_history = metrics.hypervolume(std_pareto_fronts_list, reference_point)
    
    ax1 = axes[0, 0]
    if llm_hypervolume_history and standard_hypervolume_history:
        min_len = min(len(llm_hypervolume_history), len(standard_hypervolume_history))
        ax1.plot(range(min_len),llm_hypervolume_history[:min_len],
                label=f"LLM_NSGA-II (Final: {llm_hypervolume_history[-1]:.4f})",
                color='blue', linewidth=1, marker='o', markersize=2)
        ax1.plot(range(min_len), standard_hypervolume_history[:min_len],
                label=f"NSGA-II (Final: {standard_hypervolume_history[-1]:.4f})",
                color='red', linewidth=1, marker='o', markersize=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Hypervolume')
        ax1.set_title('Hypervolume Comparison')
        # ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add data point markers
        # ax1.scatter(range(min_len), llm_hypervolume_history[:min_len],
        #            color='blue', s=20, alpha=0.7)
        # ax1.scatter(range(min_len), standard_hypervolume_history[:min_len],
        #            color='red', s=20, alpha=0.7)

    # Spacing comparison
    ax2 = axes[0, 1]
    if llm_result['spacing'] and standard_result['spacing']:
        min_len = min(len(llm_result['spacing']), len(standard_result['spacing']))
        ax2.plot(range(min_len), llm_result['spacing'][:min_len],  'b-', linewidth=1, marker='o', markersize=2,
                label=f"LLM_NSGA-II (Final: {llm_result['spacing'][-1]:.4f})")
        ax2.plot(range(min_len), standard_result['spacing'][:min_len],  'r-', linewidth=1, marker='o', markersize=2,
                label=f"LLM_NSGA-II (Final: {standard_result['spacing'][-1]:.4f})")
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Spacing')
        ax2.set_title('Spacing Metric Comparison')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add data point markers
        # ax2.scatter(range(min_len), llm_result['spacing'][:min_len],
        #            color='blue', s=20, alpha=0.7)
        # ax2.scatter(range(min_len), standard_result['spacing'][:min_len],
        #            color='red', s=20, alpha=0.7)

    # Spread comparison
    ax2 = axes[1, 0]
    if llm_result['spread'] and standard_result['spread']:
        min_len = min(len(llm_result['spread']), len(standard_result['spread']))
        ax2.plot(range(min_len), llm_result['spread'][:min_len], 'b-', linewidth=1, marker='o', markersize=2,
                label=f"LLM_NSGA-II (Final: {llm_result['spread'][-1]:.4f})",
                )
        ax2.plot(range(min_len), standard_result['spread'][:min_len], 'r-', linewidth=1, marker='o', markersize=2,
                label=f"NSGA-II (Final: {standard_result['spread'][-1]:.4f})",
                )
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Spread')
        ax2.set_title('Spread Metric Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add data point markers
        # ax2.scatter(range(min_len), llm_result['spread'][:min_len],
        #            color='blue', s=20, alpha=0.7)
        # ax2.scatter(range(min_len), standard_result['spread'][:min_len],
        #            color='red', s=20, alpha=0.7)
        
    # Pareto front size comparison
    ax4 = axes[1, 1]
    if llm_result['pareto_count'] and standard_result['pareto_count']:
        min_len = min(len(llm_result['pareto_count']), len(standard_result['pareto_count']))
        ax4.plot(range(min_len), llm_result['pareto_count'][:min_len], 'b-', linewidth=1, marker='o', markersize=2,
                label='LLM-Enhanced')
        ax4.plot(range(min_len), standard_result['pareto_count'][:min_len], 'r-', linewidth=1, marker='o', markersize=2,
                label='Standard')
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Pareto Front Size')
        ax4.set_title('Pareto Front Size Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Add data point markers
        # ax4.scatter(range(min_len), llm_result['pareto_count'][:min_len],
        #            color='blue', alpha=0.7)
        # ax4.scatter(range(min_len), standard_result['pareto_count'][:min_len],
        #            color='red', alpha=0.7)
        
    plt.tight_layout()

    # Save the plot to file
    # timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'nsga2_comparison_Run{idx+1}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved as: {filename}")

    # plt.show(block=True)
    # Do not block; close the figure to free resources
    # plt.close(fig)

    # Print summary
    # print("\n" + "="*60)
    # print("NSGA-II ALGORITHMS COMPARISON SUMMARY")
    # print("="*60)
    # print(f"LLM-Enhanced NSGA-II Execution Time: {llm_result['execution_time']:.2f} seconds")
    # print(f"Standard NSGA-II Execution Time: {standard_result['execution_time']:.2f} seconds")
    # print(f"LLM-Enhanced Final Hypervolume: {llm_result['final_hypervolume']:.4f}")
    # print(f"Standard Final Hypervolume: {standard_result['final_hypervolume']:.4f}")

    # if llm_result['final_hypervolume'] > standard_result['final_hypervolume']:
    #     print("[WINNER] LLM-Enhanced NSGA-II achieved better final hypervolume!")
    # elif llm_result['final_hypervolume'] < standard_result['final_hypervolume']:
    #     print("[WINNER] Standard NSGA-II achieved better final hypervolume!")
    # else:
    #     print("[TIE] Both algorithms achieved similar final hypervolume!")

def pareto_3dplot_comparison(results, idx):
    llm_result = next((r for r in results if 'LLM' in r['algorithm']), None)
    standard_result = next((r for r in results if 'Standard' in r['algorithm']), None)

    if not llm_result or not standard_result:
        print("Could not find both LLM and Standard results for 3D plotting.")
        return

    # Get Pareto history for animation
    llm_history = llm_result.get('pareto_history', [])
    standard_history = standard_result.get('pareto_history', [])

    if not llm_history or not standard_history:
        print("No Pareto history data available for 3D animation.")
        return

    # Convert to numpy arrays
    llm_costs_history = [np.array([ind for ind in gen if len(ind) >= 3]) for gen in llm_history]
    standard_costs_history = [np.array([ind for ind in gen if len(ind) >= 3]) for gen in standard_history]

    # Filter out empty generations
    llm_costs_history = [gen for gen in llm_costs_history if gen.size > 0]
    standard_costs_history = [gen for gen in standard_costs_history if gen.size > 0]

    if not llm_costs_history or not standard_costs_history:
        print("No valid Pareto history data for 3D animation.")
        return

    # Compute stable axis limits across all generations
    all_llm_costs = np.vstack([gen for gen in llm_costs_history if gen.size > 0])
    all_standard_costs = np.vstack([gen for gen in standard_costs_history if gen.size > 0])

    if all_llm_costs.size == 0 or all_standard_costs.size == 0:
        print("No cost data available for axis limits.")
        return

    all_costs = np.vstack([all_llm_costs, all_standard_costs])
    x_min, x_max = np.min(all_costs[:, 0]), np.max(all_costs[:, 0])
    y_min, y_max = np.min(all_costs[:, 1]), np.max(all_costs[:, 1])
    z_min, z_max = np.min(all_costs[:, 2]), np.max(all_costs[:, 2])

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Initialize scatter plots for both algorithms
    scat_llm = ax.scatter([], [], [], c='blue', s=50, alpha=0.7, edgecolors='black', label='LLM-Enhanced')
    scat_standard = ax.scatter([], [], [], c='red', s=50, alpha=0.7, edgecolors='black', label='Standard')

    ax.set_xlabel('F1', fontsize=12)
    ax.set_ylabel('F2', fontsize=12)
    ax.set_zlabel('F3', fontsize=12)
    ax.set_title('3D Pareto Front Evolution: LLM-Enhanced vs Standard', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.legend()

    # Animation control state
    is_paused = {'value': False}

    def init():
        scat_llm._offsets3d = ([], [], [])
        scat_standard._offsets3d = ([], [], [])
        return (scat_llm, scat_standard)

    def update(frame_idx):
        # Update LLM scatter
        if frame_idx < len(llm_costs_history) and llm_costs_history[frame_idx].size > 0:
            frame_costs = llm_costs_history[frame_idx]
            xs, ys, zs = frame_costs[:, 0], frame_costs[:, 1], frame_costs[:, 2]
            scat_llm._offsets3d = (xs, ys, zs)
        else:
            scat_llm._offsets3d = ([], [], [])

        # Update Standard scatter
        if frame_idx < len(standard_costs_history) and standard_costs_history[frame_idx].size > 0:
            frame_costs = standard_costs_history[frame_idx]
            xs, ys, zs = frame_costs[:, 0], frame_costs[:, 1], frame_costs[:, 2]
            scat_standard._offsets3d = (xs, ys, zs)
        else:
            scat_standard._offsets3d = ([], [], [])

        ax.set_title(f'3D Pareto Front Evolution (Generation {frame_idx + 1})')
        return (scat_llm, scat_standard)

    max_frames = max(len(llm_costs_history), len(standard_costs_history))
    anim = FuncAnimation(fig, update, init_func=init, frames=max_frames, interval=400, blit=False, repeat=True)
    
    # # Play/Pause button
    # btn_ax = fig.add_axes([0.8, 0.02, 0.1, 0.05])
    # btn_playpause = Button(btn_ax, 'Play/Pause')

    # def on_playpause_clicked(event):
    #     if is_paused['value']:
    #         anim.event_source.start()
    #         is_paused['value'] = False
    #     else:
    #         anim.event_source.stop()
    #         is_paused['value'] = True

    # btn_playpause.on_clicked(on_playpause_clicked)

    # # Save button (tries GIF then MP4)
    # btn_save_ax = fig.add_axes([0.67, 0.02, 0.1, 0.05])
    # btn_save = Button(btn_save_ax, 'Save')

    # def on_save_clicked(event):
    #     timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    #     # Try GIF with Pillow
    #     try:
    #         gif_path = f'pareto_3d_comparison_run{idx+1}.gif'
    #         writer = PillowWriter(fps=max(1, int(1000/anim.event_source.interval)))
    #         anim.save(gif_path, writer=writer)
    #         print(f'Saved GIF: {gif_path}')
    #         return
    #     except Exception as e:
    #         print(f'GIF save failed: {e}')
    #     # Try MP4 with ffmpeg
    #     try:
    #         mp4_path = f'pareto_3d_comparison_{timestamp}.mp4'
    #         writer = FFMpegWriter(fps=max(1, int(1000/anim.event_source.interval)))
    #         anim.save(mp4_path, writer=writer)
    #         print(f'Saved MP4: {mp4_path}')
    #     except Exception as e:
    #         print(f'MP4 save failed: {e}')

    # btn_save.on_clicked(on_save_clicked)

    plt.tight_layout()
    try:
        gif_path = f'pareto_3d_comparison_run{idx+1}.gif'
        writer = PillowWriter(fps=max(1, int(1000/anim.event_source.interval)))
        anim.save(gif_path, writer=writer)
        print(f'Saved GIF: {gif_path}')
        return
    except Exception as e:
        print(f'GIF save failed: {e}')
    plt.show(block=True)

def plot_objective_trends(results, idx):
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

        ax.plot(range(min_len), llm_min_obj[:min_len], label=f"LLM_NSGA-II (Final: {llm_min_obj[-1]:.4f})", color='blue', linestyle='-')
        ax.plot(range(min_len), std_min_obj[:min_len], label=f"NSGA-II (Final: {std_min_obj[-1]:.4f})", color='red', linestyle='-')
        ax.set_title(f'Min Objective {i+1}')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.4)
        ax.legend()

        # Add final values as text annotations
        # final_llm_min = llm_min_obj[-1] if llm_min_obj else 0
        # final_std_min = std_min_obj[-1] if std_min_obj else 0
        # ax.text(0.02, 0.98, f'Final LLM: {final_llm_min:.4f}\nFinal Std: {final_std_min:.4f}',
        #         transform=ax.transAxes, fontsize=9, verticalalignment='top',
        #         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    # Plot Mean Objectives
    for i in range(num_objectives):
        ax = axes[1, i]
        llm_mean_obj = [gen[i] for gen in llm_result['mean_objs']]
        std_mean_obj = [gen[i] for gen in standard_result['mean_objs']]
        min_len = min(len(llm_mean_obj), len(std_mean_obj))

        ax.plot(range(min_len), llm_mean_obj[:min_len], label=f"LLM_NSGA-II (Final: {llm_mean_obj[-1]:.4f})", color='cyan', linestyle='--')
        ax.plot(range(min_len), std_mean_obj[:min_len], label=f"NSGA-II (Final: {std_mean_obj[-1]:.4f})", color='magenta', linestyle='--')
        ax.set_title(f'Mean Objective {i+1}')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.4)
        ax.legend()

    # timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'nsga2_objective_trends_Run{idx+1}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"objective_trends plot saved as: {filename}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()


def main():
    """Main function to run parallel comparison"""
    print("Starting Parallel NSGA-II Comparison")
    print("="*50)

    # Create problem definition
    # NOTE: Using real humanitarian logistics problem
    problem = create_real_problem()
    NUM_OF_RUN = 10
    run_results = [] # [[LLM_NSGA_II_result, NSGA_II_result], ...]
    for r in range(NUM_OF_RUN):
        print(f"\nRun {r+1}")
        # Create a queue to collect results
        result_queue = mp.Queue()

        # Create processes for both algorithms
        llm_process = mp.Process(
            target=run_llm_nsga2,
            args=(problem, result_queue, "LLM-Enhanced NSGA-II")
        )

        standard_process = mp.Process(
            target=run_standard_nsga2_RandomPop,
            args=(problem, result_queue, "Standard NSGA-II", r+1)
        )
        
        standard_llmpop_process = mp.Process(
            target=run_standard_nsga2_LLMPop,
            args=(problem, result_queue, "Standard_LLMPop NSGA-II", r+1)
        )

        # Start both processes
        print("Starting parallel execution...")
        start_time = time.time()

        llm_process.start()
        standard_process.start()
        standard_llmpop_process.start()

        # Collect results first, then join. This is a more robust pattern.
        results = []
        for _ in range(3): # We expect two results
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
        standard_llmpop_process.join(timeout=60)

        # If any process is still alive after collecting results, terminate it
        if llm_process.is_alive():
            print("[WARN] LLM-Enhanced process did not terminate after collecting results. Forcing termination...")
            llm_process.terminate()
            llm_process.join()
        if standard_process.is_alive():
            print("[WARN] Standard process did not terminate after collecting results. Forcing termination...")
            standard_process.terminate()
            standard_process.join()
        if standard_llmpop_process.is_alive():
            print("[WARN] Standard_llmpop process did not terminate after collecting results. Forcing termination...")
            standard_llmpop_process.terminate()
            standard_llmpop_process.join()
        run_results.append(results)
    std_max_costs = []
    llm_max_costs = []
    stdllmpop_max_costs = []
    std_pareto_fronts_lists = [] # لیست پرتوفرانت‌های همه اجراهای NSGA-II
    llm_pareto_fronts_lists = [] # لیست پرتوفرانت‌های همه اجراهای LLM_NSGA-II
    stdllmpop_pareto_fronts_lists = [] # لیست پرتوفرانت‌های همه اجراهای LLM_NSGA-II
    for rr in run_results: # برای یافتن رفرنس پوینت بین تمامی ران‌ها جهت محاسبه هایپرولوم
        llm_result = None
        standard_result = None
        standard_llmpop_result = None
        for result in rr:
            if 'LLM-Enhanced NSGA-II' in result['algorithm']:
                llm_result = result
            elif 'Standard NSGA-II' in result['algorithm']:
                standard_result = result
            elif 'Standard_LLMPop NSGA-II' in result['algorithm']:
                standard_llmpop_result = result    
        max_cost = []
        std_pareto_fronts_list = [] # پرتوفرانت‌های یک اجرای NSGA-II
        for pp in standard_result.get('pareto_history'):
            pareto_front_list = pp
            std_pareto_fronts_list.append(pareto_front_list)
            max_cost.append(np.max(pareto_front_list, axis=0))
        std_pareto_fronts_lists.append(std_pareto_fronts_list)
        std_max_costs.append(np.max(max_cost, axis=0))
        max_cost = []
        llm_pareto_fronts_list = [] # پرتوفرانت‌های یک اجرای LLM_NSGA-II
        for pp in llm_result.get('pareto_history'):
            pareto_front_list = pp
            llm_pareto_fronts_list.append(pareto_front_list)
            max_cost.append(np.max(pareto_front_list, axis=0))
        llm_pareto_fronts_lists.append(llm_pareto_fronts_list)
        llm_max_costs.append(np.max(max_cost, axis=0))
        max_cost = []
        stdllmpop_pareto_fronts_list = []
        for pp in standard_llmpop_result.get('pareto_history'):
            pareto_front_list = pp
            stdllmpop_pareto_fronts_list.append(pareto_front_list)
            max_cost.append(np.max(pareto_front_list, axis=0))
        stdllmpop_pareto_fronts_lists.append(stdllmpop_pareto_fronts_list)
        stdllmpop_max_costs.append(np.max(max_cost, axis=0))
    std_max_cost = np.max(std_max_costs, axis=0)    
    llm_max_cost = np.max(llm_max_costs, axis=0)
    stdllmpop_max_cost = np.max(stdllmpop_max_costs, axis=0)
    reference_point = np.max([std_max_cost, llm_max_cost, stdllmpop_max_cost], axis=0) *1.1
    for idx, rr in enumerate(run_results):
        llm_result = None
        standard_result = None
        stdllmpop_result = None
        for result in rr:
            if 'LLM-Enhanced NSGA-II' in result['algorithm']:
                llm_result = result
            elif 'Standard NSGA-II' in result['algorithm']:
                standard_result = result
            elif 'Standard_LLMPop NSGA-II' in result['algorithm']:
                stdllmpop_result = result
        from convergence_metrics import ConvergenceMetrics
        metrics = ConvergenceMetrics()
        llm_hypervolume_history = metrics.hypervolume(llm_pareto_fronts_lists[idx], reference_point)
        standard_hypervolume_history = metrics.hypervolume(std_pareto_fronts_lists[idx], reference_point)
        stdllmpop_hypervolume_history = metrics.hypervolume(stdllmpop_pareto_fronts_lists[idx], reference_point)
        text_output = (
            f"llm HV: {llm_hypervolume_history[-1]:.4f}\n"
            f"std HV: {standard_hypervolume_history[-1]:.4f}\n"
            f"stdllmpop HV: {stdllmpop_hypervolume_history[-1]:.4f}\n\n"
            f"llm Spread: {llm_result['spread'][-1]:.4f}\n"
            f"std Spread: {standard_result['spread'][-1]:.4f}\n"
            f"stdllmpop Spread: {stdllmpop_result['spread'][-1]:.4f}\n\n"
            f"llm Spacing: {llm_result['spacing'][-1]:.4f}\n"
            f"std Spacing: {standard_result['spacing'][-1]:.4f}\n"
            f"stdllmpop Spacing: {stdllmpop_result['spacing'][-1]:.4f}\n\n")
        plt.figure(figsize=(10, 4))
        plt.text(
            0.01, 0.5,
            text_output,
            fontsize=12,
            verticalalignment='center',
            horizontalalignment='left',
            family='monospace'
        )
        plt.axis('off')
        plt.savefig(f"comparison_binary_indicators_Run{idx+1}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.set_start_method('spawn', force=True)
    main()