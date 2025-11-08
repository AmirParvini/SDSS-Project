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
    
    def find_knee_points(self,
                        robust_z_threshold=2.5,
                        min_finite_points=8,
                        require_pairs=2):
        # حدس تعداد جواب‌ها
        F = self.pareto_front
        F = np.asarray(F)
        n = F.shape[0]
        if not hasattr(self, "n_objectives"):
            self.n_objectives = F.shape[1]
        votes = np.zeros(n, dtype=int)
        scores = np.zeros(n, dtype=float)
        per_idx_details = {i: [] for i in range(n)}  # ذ
        for i in range(self.n_objectives):
            for j in range(i+1, self.n_objectives):
                rates = self.local_tradeoff_rate_knn(i, j, k=(3 - i - j))
                if rates.shape[0] != n:
                    raise ValueError(f"rates length mismatch for pair ({i},{j}): got {len(rates)} vs {n}")

                # فقط مقادیر متناهی
                finite_mask = np.isfinite(rates)
                if finite_mask.sum() < min_finite_points:
                    continue

                r = rates[finite_mask]

                # آمار مقاوم: median و MAD
                med = np.median(r)
                mad = np.median(np.abs(r - med))

                # اگر MAD=0 بود، به std (با مراقبت) برگردیم
                if mad == 0:
                    std = np.std(r)
                    if std == 0:
                        # هیچ تغییری در این جفت؛ زانویی تعریف‌پذیر نیست
                        continue
                    robust_z = (rates - np.mean(r)) / (std + 1e-12)
                else:
                    # ضریب 0.6745 برای برآورد سازگار با انحراف معیار گاوسی
                    robust_z = 0.6745 * (rates - med) / (mad + 1e-12)

                # کَندیداهای این جفت
                cand_mask = np.isfinite(robust_z) & (np.abs(robust_z) >= robust_z_threshold)
                cand_idx = np.nonzero(cand_mask)[0]

                # رأی‌گیری و امتیازدهی (امتیاز = |z| مقاوم)
                for idx in cand_idx:
                    votes[idx] += 1
                    # جمع امتیازها روی جفت‌های مختلف
                    scores[idx] += float(abs(robust_z[idx]))
                    per_idx_details[idx].append({
                        "objectives": (i, j),
                        "rate": float(rates[idx]),
                        "robust_z": float(robust_z[idx])
                    })
        # اجماع بین جفت‌ها
        knee_idx = np.where(votes >= require_pairs)[0]

        # رتبه‌بندی بر اساس امتیاز کل (جمع |z| مقاوم در جفت‌های مختلف)
        order = np.argsort(-scores[knee_idx])  # نزولی
        ranked_idx = knee_idx[order]

        result = []
        for idx in ranked_idx:
            result.append({
                "solution_idx": int(idx),
                "votes": int(votes[idx]),
                "knee_score": float(scores[idx]),
                "details": per_idx_details[idx]
            })                    
        return result
    
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
    
    def plot_knee_points(self,
                     normalize=False,
                     robust_z_threshold=2.5,
                     min_finite_points=8,
                     require_pairs=2,
                     show_3d=False,
                     annotate=True,
                     figsize=(12, 4)):
        """
        رسم نقاط زانو روی نمودارهای دو به دو (و در صورت نیاز سه‌بعدی).
        این تابع خودش find_knee_points را اجرا می‌کند.

        Parameters
        ----------
        normalize : bool
            اگر True باشد، اهداف را min-max نرمال‌سازی می‌کند (صرفاً برای رسم).
        show_3d : bool
            اگر True باشد، یک نمودار سه‌بعدی هم رسم می‌شود.
        annotate : bool
            اگر True باشد، ایندکس نقاط زانو بالای نقاط نوشته می‌شود.
        """
        # گرفتن ماتریس اهداف
        F = self.pareto_front

        # نرمال‌سازی برای نمایش (اختیاری)
        F_plot = F.copy()
        if normalize:
            mins = F_plot.min(axis=0)
            maxs = F_plot.max(axis=0)
            span = np.maximum(maxs - mins, 1e-12)
            F_plot = (F_plot - mins) / span

        # گرفتن نقاط زانو
        knees = self.find_knee_points(robust_z_threshold=robust_z_threshold,
                                      min_finite_points=min_finite_points,
                                      require_pairs=require_pairs)
        knee_indices = [d["solution_idx"] for d in knees]
        knee_set = set(knee_indices)

        # رسم نمودارهای دو به دو
        pairs = [(i, j) for i in range(self.n_objectives) for j in range(i+1, self.n_objectives)]
        fig, axes = plt.subplots(1, len(pairs), figsize=(figsize[0]*len(pairs), figsize[1])) if len(pairs) > 1 \
            else (plt.figure(figsize=figsize), [plt.gca()])

        if len(pairs) == 1:
            axes = [axes]  # سازگاری

        for ax, (i, j) in zip(axes, pairs):
            # همه نقاط
            ax.scatter(F_plot[:, i], F_plot[:, j], s=20, alpha=0.6, label="Solutions")

            # نقاط زانو
            if knee_indices:
                ax.scatter(F_plot[knee_indices, i], F_plot[knee_indices, j],
                        s=80, marker='o', facecolors='none', edgecolors='r', linewidths=1.8,
                        label='Knee')

            # برچسب‌گذاری
            if annotate and knee_indices:
                for idx in knee_indices:
                    ax.annotate(str(idx), (F_plot[idx, i], F_plot[idx, j]),
                                xytext=(5, 5), textcoords='offset points', fontsize=9)

            ax.set_xlabel(f'Objective {i}')
            ax.set_ylabel(f'Objective {j}')
            ax.set_title(f'Pair ({i}, {j})')
            ax.grid(True, alpha=0.25)
            ax.legend(loc='best')

        plt.tight_layout()
        plt.show()

        # نمودار سه‌بعدی اختیاری
        if show_3d and self.n_objectives >= 3:
            fig3d = plt.figure(figsize=(6, 5))
            ax3d = fig3d.add_subplot(111, projection='3d')

            ax3d.scatter(F_plot[:, 0], F_plot[:, 1], F_plot[:, 2], s=18, alpha=0.6)

            if knee_indices:
                ax3d.scatter(F_plot[knee_indices, 0], F_plot[knee_indices, 1], F_plot[knee_indices, 2],
                            s=80, marker='o', facecolors='none', edgecolors='r', linewidths=1.6)

                if annotate:
                    for idx in knee_indices:
                        ax3d.text(F_plot[idx, 0], F_plot[idx, 1], F_plot[idx, 2],
                                str(idx), fontsize=9)

            ax3d.set_xlabel('Objective 0')
            ax3d.set_ylabel('Objective 1')
            ax3d.set_zlabel('Objective 2')
            ax3d.set_title('3D view (optional)')
            plt.tight_layout()
            plt.show()

        return knees  # خروجی لیست نقاط زانو برای لاگ/استفاده‌ی بعدی
    
    def print_tradeoff_summary(self,
                           robust_z_threshold=2.5,
                           min_finite_points=8,
                           require_pairs=2,
                           top_knees=5,
                           top_details=2):
        """
        چاپ خلاصه‌ی تحلیلی trade-off و نقاط زانو (با نسخه‌ی جدید find_knee_points)

        Parameters
        ----------
        robust_z_threshold : float
            آستانه‌ی |z| مقاوم برای علامت‌گذاری کَندیداها در هر جفت هدف.
        min_finite_points : int
            حداقل تعداد نرخ‌های متناهی لازم تا یک جفت هدف بررسی شود.
        require_pairs : int
            حداقل تعداد جفت‌هایی که یک جواب باید در آن‌ها کَندیدا شود تا «زانو» محسوب گردد.
        top_knees : int
            چند زانوی اول را چاپ کنیم.
        top_details : int
            برای هر زانو، چند جفت هدفِ مؤثرتر (برحسب |robust_z|) را نمایش دهیم.
        """
        print("\n" + "="*60)
        print("Trade-off Analysis Summary")
        print("="*60)

        # 1) خلاصه‌ی آماری نرخ‌های مبادله (همان کد قبلی شما)
        metrics = self.calculate_global_tradeoff_metrics()
        for pair, values in metrics.items():
            print(f"\n{pair} Trade-off:")
            print(f"  Mean:   {values['mean_tradeoff']:.4f}")
            print(f"  Std:    {values['std_tradeoff']:.4f}")
            print(f"  Median: {values['median_tradeoff']:.4f}")
            print(f"  Range:  [{values['min_tradeoff']:.4f}, {values['max_tradeoff']:.4f}]")

        # 2) نقاط زانو براساس نسخه‌ی جدید find_knee_points
        knees = self.find_knee_points(
            robust_z_threshold=robust_z_threshold,
            min_finite_points=min_finite_points,
            require_pairs=require_pairs
        )

        if not knees:
            print("\nNo knee points found with current settings "
                f"(threshold={robust_z_threshold}, require_pairs={require_pairs}).")
            return

        print(f"\nNumber of Knee Points Found: {len(knees)}")
        print(f"(showing top {min(top_knees, len(knees))} by knee_score)\n")

        # مرتب‌سازی knees همین حالا در find_knee_points انجام شده؛
        # صرفاً همان ترتیب را چاپ می‌کنیم (یا دوباره sort بر اساس knee_score)
        for r, knee in enumerate(knees[:top_knees], start=1):
            idx = knee["solution_idx"]
            votes = knee.get("votes", 0)
            score = knee.get("knee_score", 0.0)

            print(f"Knee {r}: solution #{idx} | votes={votes} | knee_score={score:.3f}")

            # جزئیات per-pair را بر اساس |robust_z| مرتب و تا top_details چاپ کن
            details = knee.get("details", [])
            if details:
                # sort by |robust_z| descending
                details_sorted = sorted(details, key=lambda d: abs(d.get("robust_z", 0.0)), reverse=True)
                for d_i, d in enumerate(details_sorted[:top_details], start=1):
                    i, j = d["objectives"]
                    rate = d.get("rate", float("nan"))
                    rz = d.get("robust_z", float("nan"))
                    print(f"  └─ [{d_i}] F{i+1}-F{j+1}: rate={rate:.4f}, |robust_z|={abs(rz):.3f}")
            else:
                print("  └─ (no per-pair details)")

        # نکته: اگر بخواهید ایندکس‌ها را با مقادیر اهداف چاپ کنید:
        # F = getattr(self, 'F', getattr(self, 'front', getattr(self, 'objectives', None)))
        # اگر F موجود بود، می‌توانید F[idx] را هم چاپ کنید.

    # استفاده در کد اصلی شما
    def integrate_tradeoff_analysis(self):
        """
        تابعی برای اضافه کردن تحلیل trade-off به کد اصلی
        
        این تابع را در انتهای متد main() کلاس Main اضافه کنید
        """
        # ایجاد شیء تحلیل trade-off
        tradeoff_analyzer = TradeoffAnalysis(self.pareto_front)
        
        # نمایش نمودارهای trade-off
        tradeoff_analyzer.plot_tradeoff_analysis(save_path='tradeoff_analysis.png')
        
        # نمایش نقاط زانو
        knees = tradeoff_analyzer.plot_knee_points(normalize=True, robust_z_threshold=2.5, 
                                                min_finite_points=8, require_pairs=2,
                                                show_3d=True, annotate=True)
        print('knee: ', knees)
        # چاپ خلاصه آماری
        tradeoff_analyzer.print_tradeoff_summary()
        
        return tradeoff_analyzer
