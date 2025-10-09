import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform
from hyper_volume import calculate_hypervolume_3d

class ConvergenceMetrics:
    """
    کلاس محاسبه شاخص‌های همگرایی برای الگوریتم‌های چندهدفه
    """
    
    def __init__(self):
        self.hypervolume_history = []
        self.spacing_history = []
        self.spread_history = []
        self.gd_history = []  # Generational Distance
        self.igd_history = []  # Inverted Generational Distance
        self.n_pareto_history = []
        self.mean_objectives_history = []
        self.std_objectives_history = []
        
    # def hypervolume(self, pareto_front, reference_point=None):
    #     """
    #     محاسبه Hypervolume Indicator
    #     حجم فضای اهداف که توسط جواب‌های پارتو تحت پوشش قرار می‌گیرد
        
    #     Parameters:
    #     -----------
    #     pareto_front: np.array of shape (n_solutions, n_objectives)
    #     reference_point: np.array یا None (اگر None باشد، بدترین مقادیر + 10% استفاده می‌شود)
    #     """
    #     if len(pareto_front) == 0:
    #         return 0.0
            
    #     if reference_point is None:
    #         reference_point = np.max(pareto_front, axis=0) * 1.1
            
    #     # برای سادگی از روش WFG استفاده می‌کنیم (برای 2-3 هدف)
    #     # نرمال‌سازی نسبت به reference point
    #     normalized_pf = pareto_front / reference_point
        
    #     # مرتب‌سازی بر اساس اولین هدف
    #     sorted_indices = np.argsort(normalized_pf[:, 0])
    #     sorted_pf = normalized_pf[sorted_indices]
        
    #     hv = 0.0
    #     n_obj = pareto_front.shape[1]
        
    #     if n_obj == 2:
    #         # محاسبه مستقیم برای 2 هدف
    #         for i in range(len(sorted_pf)):
    #             if i == 0:
    #                 width = sorted_pf[i, 0]
    #             else:
    #                 width = sorted_pf[i, 0] - sorted_pf[i-1, 0]
    #             height = 1.0 - sorted_pf[i, 1]
    #             hv += width * height
    #     else:
    #         # تقریب ساده برای 3 هدف
    #         for point in sorted_pf:
    #             volume = np.prod(1.0 - point)
    #             hv += volume
    #         hv /= len(sorted_pf)
            
    #     return hv
    def hypervolume(self, pareto_front, reference_point=None):
        """
        محاسبه Hypervolume با الگوریتم WFG
        """
        if len(pareto_front) == 0:
            return 0.0
        
        # استفاده از کلاس حرفه‌ای
        hv = calculate_hypervolume_3d(
            pareto_front, 
            reference_point=reference_point,
            method='wfg'  # دقیق‌ترین روش
        )
        
        return hv
        
    def spacing(self, pareto_front):
        """
        محاسبه Spacing Metric
        یکنواختی توزیع جواب‌ها در پارتو فرانت
        مقدار کمتر = توزیع یکنواخت‌تر
        
        Parameters:
        -----------
        pareto_front: np.array of shape (n_solutions, n_objectives)
        """
        if len(pareto_front) <= 1:
            return 0.0
        
        # محاسبه فاصله هر جواب تا نزدیک‌ترین همسایه‌اش
        distances = cdist(pareto_front, pareto_front, metric='euclidean')
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        
        # محاسبه spacing
        d_mean = np.mean(min_distances)
        spacing_metric = np.sqrt(np.sum((min_distances - d_mean)**2) / (len(min_distances) -1))
        
        return spacing_metric
    
    # def spread(self, pareto_front):
    #     """
    #     محاسبه Spread (Delta) Metric
    #     سنجش میزان پوشش و تنوع جواب‌ها
    #     مقدار کمتر = پوشش بهتر
        
    #     Parameters:
    #     -----------
    #     pareto_front: np.array of shape (n_solutions, n_objectives)
    #     """
    #     if len(pareto_front) <= 2:
    #         return 1.0
        
    #     n_obj = pareto_front.shape[1]
        
    #     # پیدا کردن نقاط انتهایی (extreme points)
    #     extreme_points = []
    #     for i in range(n_obj):
    #         extreme_points.append(pareto_front[np.argmin(pareto_front[:, i])])
    #     extreme_points = np.array(extreme_points)
        
    #     # محاسبه فاصله‌ها
    #     distances = cdist(pareto_front, pareto_front, metric='euclidean')
    #     np.fill_diagonal(distances, np.inf)
    #     min_distances = np.min(distances, axis=1)
    #     d_mean = np.mean(min_distances)
        
    #     # فاصله تا نقاط انتهایی
    #     d_f = 0
    #     for ext_point in extreme_points:
    #         dist_to_ext = np.linalg.norm(pareto_front - ext_point, axis=1)
    #         d_f += np.min(dist_to_ext)
        
    #     # محاسبه spread
    #     numerator = d_f + np.sum(np.abs(min_distances - d_mean))
    #     denominator = d_f + len(pareto_front) * d_mean
        
    #     if denominator == 0:
    #         return 1.0
            
    #     spread_metric = numerator / denominator
        
    #     return spread_metric
    def spread(self, pareto_front):
        """
        Spread (Δ) metric for multi-objective fronts
        Smaller = better diversity and coverage
        """
        n = len(pareto_front)
        if n <= 2:
            return 1.0

        # فاصله بین تمام نقاط
        D = cdist(pareto_front, pareto_front)
        np.fill_diagonal(D, np.inf)

        # فاصله‌های مینیمم بین همسایه‌ها
        d_i = np.min(D, axis=1)
        d_mean = np.mean(d_i)

        # دو نقطه‌ی انتهایی (دورترین از هم)
        D_full = squareform(pdist(pareto_front))
        i, j = np.unravel_index(np.argmax(D_full), D_full.shape)
        d_f = np.linalg.norm(pareto_front[i] - pareto_front[j])

        # Spread metric
        numerator = d_f + np.sum(np.abs(d_i - d_mean))
        denominator = d_f + (n - 1) * d_mean

        return numerator / denominator
    
    def generational_distance(self, pareto_front, true_pareto_front):
        """
        محاسبه Generational Distance (GD)
        میانگین فاصله جواب‌های پیدا شده تا true Pareto front
        مقدار کمتر = نزدیک‌تر به پارتو واقعی
        
        Parameters:
        -----------
        pareto_front: np.array جواب‌های پیدا شده
        true_pareto_front: np.array پارتو فرانت واقعی (اگر در دسترس باشد)
        """
        if len(pareto_front) == 0 or len(true_pareto_front) == 0:
            return np.inf
        
        distances = cdist(pareto_front, true_pareto_front, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        gd = np.mean(min_distances)
        
        return gd
    
    def inverted_generational_distance(self, pareto_front, true_pareto_front):
        """
        محاسبه Inverted Generational Distance (IGD)
        میانگین فاصله true Pareto front تا جواب‌های پیدا شده
        مقدار کمتر = پوشش بهتر
        
        Parameters:
        -----------
        pareto_front: np.array جواب‌های پیدا شده
        true_pareto_front: np.array پارتو فرانت واقعی
        """
        if len(pareto_front) == 0 or len(true_pareto_front) == 0:
            return np.inf
        
        distances = cdist(true_pareto_front, pareto_front, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        igd = np.mean(min_distances)
        
        return igd
    
    def update_metrics(self, pareto_pop, iteration, true_pareto_front=None):
        """
        به‌روزرسانی تمام شاخص‌ها در هر تکرار
        
        Parameters:
        -----------
        pareto_pop: list of individuals در پارتو فرانت فعلی
        iteration: شماره تکرار فعلی
        true_pareto_front: پارتو فرانت واقعی (اختیاری)
        """
        if len(pareto_pop) == 0:
            return
        
        # استخراج مقادیر اهداف
        pareto_front = np.array([ind['cost'] for ind in pareto_pop])
        
        # محاسبه شاخص‌ها
        # hv = self.hypervolume(pareto_front)
        sp = self.spacing(pareto_front)
        spr = self.spread(pareto_front)
        
        # self.hypervolume_history.append(hv)
        self.spacing_history.append(sp)
        self.spread_history.append(spr)
        self.n_pareto_history.append(len(pareto_pop))
        
        # میانگین و انحراف معیار اهداف
        mean_obj = np.mean(pareto_front, axis=0)
        std_obj = np.std(pareto_front, axis=0)
        self.mean_objectives_history.append(mean_obj)
        self.std_objectives_history.append(std_obj)
        
        # اگر true Pareto front در دسترس باشد
        if true_pareto_front is not None:
            gd = self.generational_distance(pareto_front, true_pareto_front)
            igd = self.inverted_generational_distance(pareto_front, true_pareto_front)
            self.gd_history.append(gd)
            self.igd_history.append(igd)
    
    def plot_convergence(self, save_path=None):
        """
        رسم نمودارهای همگرایی
        """
        n_metrics = 6  # تعداد شاخص‌های اصلی
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Convergence Metrics Over Iterations', fontsize=16, fontweight='bold')
        
        iterations = range(1, len(self.spacing_history) + 1)
        
        # 1. Hypervolume
        # ax = axes[0, 0]
        # ax.plot(iterations, self.hypervolume_history, 'b-', linewidth=2, marker='o', markersize=4)
        # ax.set_xlabel('Iteration', fontsize=11)
        # ax.set_ylabel('Hypervolume', fontsize=11)
        # ax.set_title('Hypervolume Indicator\n(بیشتر = بهتر)', fontsize=12, fontweight='bold')
        # ax.grid(True, alpha=0.3)
        
        # 2. Spacing
        ax = axes[0, 1]
        ax.plot(iterations, self.spacing_history, 'r-', linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Spacing', fontsize=11)
        ax.set_title('Spacing Metric\n(کمتر = یکنواخت‌تر)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 3. Spread
        ax = axes[1, 0]
        ax.plot(iterations, self.spread_history, 'g-', linewidth=2, marker='^', markersize=4)
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Spread (Delta)', fontsize=11)
        ax.set_title('Spread Metric\n(کمتر = پوشش بهتر)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 4. Number of Pareto Solutions
        ax = axes[1, 1]
        ax.plot(iterations, self.n_pareto_history, 'm-', linewidth=2, marker='d', markersize=4)
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Number of Solutions', fontsize=11)
        ax.set_title('Pareto Front Size\n(تعداد جواب‌های پارتو)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 5. Mean Objectives
        ax = axes[2, 0]
        mean_objectives = np.array(self.mean_objectives_history)
        for i in range(mean_objectives.shape[1]):
            ax.plot(iterations, mean_objectives[:, i], linewidth=2, 
                   marker='o', markersize=3, label=f'F{i+1}')
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Mean Objective Value', fontsize=11)
        ax.set_title('Mean of Objectives\n(میانگین توابع هدف)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Std Objectives
        ax = axes[2, 1]
        std_objectives = np.array(self.std_objectives_history)
        for i in range(std_objectives.shape[1]):
            ax.plot(iterations, std_objectives[:, i], linewidth=2, 
                   marker='s', markersize=3, label=f'F{i+1}')
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Std of Objective Value', fontsize=11)
        ax.set_title('Standard Deviation of Objectives\n(انحراف معیار توابع هدف)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # اگر GD و IGD محاسبه شده باشند، نمودار جداگانه
        if len(self.gd_history) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            iterations = range(1, len(self.gd_history) + 1)
            
            axes[0].plot(iterations, self.gd_history, 'b-', linewidth=2, marker='o')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('GD')
            axes[0].set_title('Generational Distance\n(کمتر = بهتر)')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(iterations, self.igd_history, 'r-', linewidth=2, marker='s')
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('IGD')
            axes[1].set_title('Inverted Generational Distance\n(کمتر = بهتر)')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path.replace('.png', '_distance.png'), dpi=300, bbox_inches='tight')
            plt.show()
    
    def print_summary(self):
        """
        چاپ خلاصه نتایج
        """
        print("\n" + "="*60)
        print("CONVERGENCE METRICS SUMMARY")
        print("="*60)
        
        if len(self.hypervolume_history) > 0:
            print(f"\n📊 Hypervolume:")
            print(f"   Initial: {self.hypervolume_history[0]:.6f}")
            print(f"   Final:   {self.hypervolume_history[-1]:.6f}")
            print(f"   Improvement: {((self.hypervolume_history[-1] - self.hypervolume_history[0]) / self.hypervolume_history[0] * 100):.2f}%")
        
        if len(self.spacing_history) > 0:
            print(f"\n📏 Spacing:")
            print(f"   Initial: {self.spacing_history[0]:.6f}")
            print(f"   Final:   {self.spacing_history[-1]:.6f}")
            print(f"   Improvement: {((self.spacing_history[0] - self.spacing_history[-1]) / self.spacing_history[0] * 100):.2f}%")
        
        if len(self.spread_history) > 0:
            print(f"\n📐 Spread:")
            print(f"   Initial: {self.spread_history[0]:.6f}")
            print(f"   Final:   {self.spread_history[-1]:.6f}")
            print(f"   Improvement: {((self.spread_history[0] - self.spread_history[-1]) / self.spread_history[0] * 100):.2f}%")
        
        if len(self.n_pareto_history) > 0:
            print(f"\n🎯 Pareto Front Size:")
            print(f"   Initial: {self.n_pareto_history[0]}")
            print(f"   Final:   {self.n_pareto_history[-1]}")
            print(f"   Change:  {self.n_pareto_history[-1] - self.n_pareto_history[0]:+d}")
        
        print("\n" + "="*60)