import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.cm as cm

class TradeoffAnalysis:
    """کلاس برای تحلیل trade-off بین توابع هدف در بهینه‌سازی چندهدفه"""
    
    def __init__(self, pareto_front):
        """
        Parameters:
        -----------
        pareto_front: numpy array با شکل (n_solutions, n_objectives)
        """
        self.pareto_front = np.array(pareto_front)
        self.n_solutions = self.pareto_front.shape[0]
        self.n_objectives = self.pareto_front.shape[1]
        self.normalized_front = self._normalize_objectives()
        self.objective_names = {
            0: 'F1: Distance',
            1: 'F2: Unmet Demand',
            2: 'F3: Death Probability'
        }
        
    def _normalize_objectives(self):
        """نرمال‌سازی توابع هدف به بازه [0, 1]"""
        scaler = MinMaxScaler()
        return scaler.fit_transform(self.pareto_front)

    def local_tradeoff_rate_knn(self, i, j, k, n_neighbors=6, bandwidth=None):
        """
        خروجی: نرخ مبادله‌ی شرطی dFj/dFi | Fk (به‌صورت موضعی برای هر نقطه)
        """
        from sklearn.neighbors import NearestNeighbors
        import numpy as np

        X = self.pareto_front[:, [i, k]].astype(float)
        y = self.pareto_front[:, j].astype(float)

        # نرمال‌سازی ستونی (اختیاری ولی توصیه می‌شود)
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-12)
        y = (y - y.min()) / (y.max() - y.min() + 1e-12)

        n_max_neighbors = len(self.pareto_front) - 1
        k_to_use = min(n_neighbors, n_max_neighbors)
        nbrs = NearestNeighbors(n_neighbors = k_to_use).fit(np.c_[X, y])
        idxs = nbrs.kneighbors(n_neighbors = k_to_use, return_distance=False)

        betas = np.full(len(self.pareto_front), np.nan)
        for t, neigh in enumerate(idxs):
            Xt = np.c_[np.ones(len(neigh)), X[neigh]]   # [1, Fi, Fk]
            yt = y[neigh]
            # وزن‌دهی اختیاری بر اساس فاصله در فضای (Fi,Fj,Fk)
            # اینجا ساده: OLS
            try:
                beta = np.linalg.lstsq(Xt, yt, rcond=None)[0]  # [b0, b_i, b_k]
                betas[t] = beta[1]  # dFj/dFi | Fk
            except np.linalg.LinAlgError:
                pass
        return betas  # آرایه‌ای از نرخ‌های موضعی

    def calculate_global_tradeoff_metrics(self):
        """محاسبه معیارهای کلی trade-off برای تمام جفت توابع هدف"""
        metrics = {}
        
        for i in range(self.n_objectives):
            for j in range(i+1, self.n_objectives):
                rates = self.local_tradeoff_rate_knn(i, j, k=(3 - i - j))
                rates_finite = rates[rates != np.inf]
                
                if len(rates_finite) > 0:
                    metrics[f'F{i+1}_F{j+1}'] = {
                        'mean_tradeoff': np.mean(rates_finite),
                        'std_tradeoff': np.std(rates_finite),
                        'max_tradeoff': np.max(rates_finite),
                        'min_tradeoff': np.min(rates_finite),
                        'median_tradeoff': np.median(rates_finite)
                    }
                    
        return metrics
    
    def find_knee_points(self, threshold=1.5):
        """
        یافتن نقاط زانو (knee points) که بهترین trade-off را دارند
        
        Parameters:
        -----------
        threshold: آستانه برای تشخیص نقاط زانو بر اساس انحراف معیار
        """
        knee_points = []
        
        for i in range(self.n_objectives):
            for j in range(i+1, self.n_objectives):
                rates = self.local_tradeoff_rate_knn(i, j, k=(3 - i - j))
                if len(rates) > 2:
                    mean_rate = np.mean(rates[rates != np.inf])
                    std_rate = np.std(rates[rates != np.inf])
                    
                    for idx, rate in enumerate(rates):
                        if rate != np.inf and abs(rate - mean_rate) > threshold * std_rate:
                            knee_points.append({
                                'solution_idx': idx,
                                'objectives': (i, j),
                                'tradeoff_rate': rate,
                                # 'point': sorted_front[idx]
                            })
                            
        return knee_points
    
    def calculate_hypervolume_contribution(self, reference_point=None):
        """محاسبه سهم هر نقطه در hypervolume"""
        if reference_point is None:
            reference_point = np.max(self.pareto_front, axis=0) * 1.1
            
        contributions = []
        
        for i in range(self.n_solutions):
            # حذف نقطه i و محاسبه hypervolume
            temp_front = np.delete(self.pareto_front, i, axis=0)
            hv_without = self._calculate_hypervolume(temp_front, reference_point)
            hv_with = self._calculate_hypervolume(self.pareto_front, reference_point)
            contribution = hv_with - hv_without
            contributions.append(contribution)
            
        return np.array(contributions)
    
    def _calculate_hypervolume(self, points, reference_point):
        """محاسبه ساده hypervolume برای 2D و 3D"""
        if self.n_objectives == 2:
            # مرتب‌سازی نقاط
            sorted_points = points[np.argsort(points[:, 0])]
            hv = 0
            prev_x = 0
            
            for point in sorted_points:
                hv += (point[0] - prev_x) * (reference_point[1] - point[1])
                prev_x = point[0]
                
            return hv
        elif self.n_objectives == 3:
            # تقریب ساده برای 3D
            return np.sum([np.prod(reference_point - p) for p in points])
        else:
            return 0
    
    def plot_tradeoff_analysis(self, save_path=None):
        """
        رسم نمودارهای انتخاب شده برای تحلیل trade-off سه‌هدفه:
        1. 3D Pareto Front
        2. Trade-off Rates Distribution (Box Plot)
        3. Parallel Coordinates
        """

        if self.n_objectives != 3:
            print("این تابع فقط برای تحلیل مسائل سه‌هدفه (n_objectives=3) طراحی شده است.")
            return

        fig = plt.figure(figsize=(18, 6)) # کاهش اندازه کلی figure

        # --- ۱. 3D Pareto Front
        ax1 = fig.add_subplot(131, projection='3d') # یک ردیف، سه ستون، اولین نمودار
        scatter = ax1.scatter(self.pareto_front[:, 0], self.pareto_front[:, 1], 
                              self.pareto_front[:, 2], 
                              c=self.pareto_front[:, 0], cmap='viridis', s=50, alpha=0.6)
        ax1.set_xlabel(self.objective_names[0])
        ax1.set_ylabel(self.objective_names[1])
        ax1.set_zlabel(self.objective_names[2])
        ax1.set_title('1. 3D Pareto Front (Color by F1)')
        fig.colorbar(scatter, ax=ax1, label='F1 Value') # اضافه کردن Colorbar

        # --- ۲. توزیع نرخ‌های مبادله (Box Plot)
        ax5 = fig.add_subplot(132) # یک ردیف، سه ستون، دومین نمودار
        obj_pairs = [(0, 1), (0, 2), (1, 2)]
        all_rates = []
        labels = []
        
        for i, j in obj_pairs:
            rates = self.local_tradeoff_rate_knn(i, j, k=(3 - i - j))
            finite_rates = rates[rates != np.inf]
            all_rates.append(finite_rates)
            labels.append(f'ΔF{j+1}/ΔF{i+1}')
        
        ax5.boxplot(all_rates, labels=labels, patch_artist=True)
        ax5.set_ylabel('Trade-off Rate (Absolute Value)')
        ax5.set_title('2. Distribution of local_tradeoff_rate_knn')
        ax5.grid(axis='y', alpha=0.3)
        
        # --- ۳. نمودار مختصات موازی (Parallel Coordinates)
        ax6 = fig.add_subplot(133) # یک ردیف، سه ستون، سومین نمودار
        normalized = self.normalized_front
        
        for i in range(len(normalized)):
            color_val = normalized[i, 0] # رنگ‌بندی بر اساس F1
            ax6.plot([0, 1, 2], normalized[i], color=cm.viridis(color_val), alpha=0.5)
        
        ax6.set_xticks([0, 1, 2])
        ax6.set_xticklabels(['F1', 'F2', 'F3'])
        ax6.set_ylabel('Normalized Value (0 to 1)')
        ax6.set_title('3. Parallel Coordinates Plot')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(-0.1, 1.1)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_knee_points(self):
        """نمایش نقاط زانو روی Pareto Front"""
        knee_points = self.find_knee_points()
        
        if self.n_objectives == 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(self.pareto_front[:, 0], self.pareto_front[:, 1],
                       c='blue', s=50, alpha=0.6, label='Pareto Solutions')
            
            for knee in knee_points:
                plt.scatter(knee['point'][0], knee['point'][1],
                          c='red', s=100, marker='*', label='Knee Point')
            
            plt.xlabel('F1: Total Distance')
            plt.ylabel('F2: Unmet Demand')
            plt.title('Knee Points on Pareto Front')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
    
    def print_tradeoff_summary(self):
        """چاپ خلاصه آماری trade-off"""
        print("\n" + "="*60)
        print("Trade-off Analysis Summary")
        print("="*60)
        
        metrics = self.calculate_global_tradeoff_metrics()
        
        for pair, values in metrics.items():
            print(f"\n{pair} Trade-off:")
            print(f"  Mean: {values['mean_tradeoff']:.4f}")
            print(f"  Std: {values['std_tradeoff']:.4f}")
            print(f"  Median: {values['median_tradeoff']:.4f}")
            print(f"  Range: [{values['min_tradeoff']:.4f}, {values['max_tradeoff']:.4f}]")
        
        knee_points = self.find_knee_points()
        if knee_points:
            print(f"\nNumber of Knee Points Found: {len(knee_points)}")
            for i, knee in enumerate(knee_points[:5]):  # نمایش 5 نقطه اول
                print(f"  Knee {i+1}: F{knee['objectives'][0]+1}-F{knee['objectives'][1]+1}, "
                      f"Rate: {knee['tradeoff_rate']:.4f}")


# استفاده در کد اصلی شما
def integrate_tradeoff_analysis(main_instance):
    """
    تابعی برای اضافه کردن تحلیل trade-off به کد اصلی
    
    این تابع را در انتهای متد main() کلاس Main اضافه کنید
    """
    # دریافت Pareto Front
    pf_costs = np.array([ind['cost'] for ind in main_instance.results['pareto_pop']])
    
    # ایجاد شیء تحلیل trade-off
    tradeoff_analyzer = TradeoffAnalysis(pf_costs)
    
    # نمایش نمودارهای trade-off
    tradeoff_analyzer.plot_tradeoff_analysis(save_path='tradeoff_analysis.png')
    
    # نمایش نقاط زانو
    tradeoff_analyzer.plot_knee_points()
    
    # چاپ خلاصه آماری
    tradeoff_analyzer.print_tradeoff_summary()
    
    return tradeoff_analyzer
