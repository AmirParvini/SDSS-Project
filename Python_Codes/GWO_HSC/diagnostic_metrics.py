import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from collections import defaultdict

class DiagnosticMetrics:
    """
    شاخص‌های تشخیصی برای یافتن منشأ مشکلات الگوریتم
    """
    
    def __init__(self):
        # شاخص‌های قبلی
        self.hypervolume_history = []
        self.spacing_history = []
        self.spread_history = []
        self.n_pareto_history = []
        
        # 🔍 شاخص‌های تشخیصی جدید
        
        # 1. برای تشخیص مشکل عملگرها
        self.diversity_history = []  # تنوع ژنتیکی
        self.mutation_success_rate = []  # نرخ موفقیت جهش
        self.crossover_success_rate = []  # نرخ موفقیت تقاطع
        
        # 2. برای تشخیص مشکل selection
        self.selection_pressure = []  # فشار انتخاب
        self.elite_dominance = []  # تسلط نخبگان
        
        # 3. برای تشخیص مشکل convergence
        self.convergence_rate = []  # سرعت همگرایی
        self.stagnation_counter = []  # شمارش stagnation
        
        # 4. برای تشخیص مشکل مدلسازی
        self.constraint_violation_rate = []  # نرخ نقض محدودیت
        self.feasible_solutions_ratio = []  # نسبت جواب‌های feasible
        self.objective_correlation = []  # همبستگی بین اهداف
        
        # 5. برای تشخیص exploration vs exploitation
        self.exploration_score = []  # امتیاز اکتشاف
        self.exploitation_score = []  # امتیاز بهره‌برداری
        
        # ذخیره موقت برای محاسبات
        self.prev_pareto_front = None
        self.offspring_improvements = []
        self.mutation_improvements = []
        
    def calculate_diversity(self, population):
        """
        محاسبه تنوع ژنتیکی جمعیت
        تنوع بالا = exploration خوب
        تنوع پایین = احتمال premature convergence
        """
        if len(population) < 2:
            return 0.0
        
        costs = np.array([ind['cost'] for ind in population])
        
        # محاسبه واریانس نرمال شده
        normalized_costs = (costs - costs.min(axis=0)) / (costs.max(axis=0) - costs.min(axis=0) + 1e-10)
        diversity = np.mean(np.std(normalized_costs, axis=0))
        
        return diversity
    
    def calculate_mutation_success_rate(self, mutation_improvements):
        """
        نرخ موفقیت جهش
        پایین بودن = جهش ضعیف یا نرخ جهش نامناسب
        """
        if len(mutation_improvements) == 0:
            return 0.0
        
        success_rate = sum(mutation_improvements) / len(mutation_improvements)
        return success_rate
    
    def calculate_crossover_success_rate(self, offspring_improvements):
        """
        نرخ موفقیت تقاطع
        پایین بودن = عملگر crossover ضعیف
        """
        if len(offspring_improvements) == 0:
            return 0.0
        
        success_rate = sum(offspring_improvements) / len(offspring_improvements)
        return success_rate
    
    def calculate_selection_pressure(self, population):
        """
        فشار انتخاب
        بالا بودن = loss of diversity سریع
        پایین بودن = همگرایی کند
        """
        if len(population) < 2:
            return 0.0
        
        ranks = [ind['rank'] for ind in population]
        rank_diversity = len(set(ranks)) / len(population)
        
        # فشار انتخاب معکوس تنوع rank است
        selection_pressure = 1.0 - rank_diversity
        
        return selection_pressure
    
    def calculate_elite_dominance(self, population, top_percent=0.1):
        """
        میزان تسلط نخبگان
        بالا بودن = احتمال premature convergence
        """
        if len(population) < 10:
            return 0.0
        
        n_elite = max(1, int(len(population) * top_percent))
        
        # مقایسه hypervolume نخبگان با کل جمعیت
        costs = np.array([ind['cost'] for ind in population])
        elite_costs = costs[:n_elite]
        
        elite_range = np.max(elite_costs, axis=0) - np.min(elite_costs, axis=0)
        total_range = np.max(costs, axis=0) - np.min(costs, axis=0) + 1e-10
        
        dominance = np.mean(elite_range / total_range)
        
        return dominance
    
    def calculate_convergence_rate(self):
        """
        سرعت همگرایی بر اساس تغییرات hypervolume
        کند بودن = مشکل در exploration یا عملگرها
        """
        if len(self.hypervolume_history) < 2:
            return 0.0
        
        recent_change = self.hypervolume_history[-1] - self.hypervolume_history[-2]
        convergence_rate = recent_change / (abs(self.hypervolume_history[-2]) + 1e-10)
        
        return convergence_rate
    
    def calculate_stagnation(self, window=5):
        """
        تشخیص stagnation
        بالا بودن = الگوریتم stuck شده
        """
        if len(self.hypervolume_history) < window:
            return 0
        
        recent_hvs = self.hypervolume_history[-window:]
        variance = np.var(recent_hvs)
        
        # اگر واریانس خیلی کم باشد، stagnation داریم
        is_stagnant = 1 if variance < 1e-6 else 0
        
        return is_stagnant
    
    def calculate_constraint_violation_rate(self, population):
        """
        نرخ نقض محدودیت‌ها
        بالا بودن = مشکل در مدلسازی یا repair mechanism
        
        این تابع باید با تابع هدف شما سفارشی شود
        """
        # فرض: اگر cost خیلی بالا باشد، احتمالاً constraint نقض شده
        # شما باید این را بر اساس مدل خودتان تغییر دهید
        
        violation_count = 0
        for ind in population:
            # مثال: اگر هر یک از اهداف از حد threshold بیشتر باشد
            if np.any(ind['cost'] > 1e8):  # threshold را تنظیم کنید
                violation_count += 1
        
        violation_rate = violation_count / len(population) if len(population) > 0 else 0
        
        return violation_rate
    
    def calculate_objective_correlation(self, pareto_front):
        """
        محاسبه همبستگی بین توابع هدف
        همبستگی بالا = اهداف مستقل نیستند (مشکل در مدلسازی)
        همبستگی صفر یا منفی = trade-off صحیح
        """
        if len(pareto_front) < 3:
            return np.zeros((3, 3))
        
        correlation_matrix = np.corrcoef(pareto_front.T)
        
        return correlation_matrix
    
    def calculate_exploration_exploitation(self, pareto_front):
        """
        تشخیص exploration vs exploitation
        """
        if self.prev_pareto_front is None or len(pareto_front) < 2:
            exploration = 0.5
            exploitation = 0.5
        else:
            # Exploration: تعداد جواب‌های جدید در نواحی جدید
            distances = cdist(pareto_front, self.prev_pareto_front, metric='euclidean')
            min_distances = np.min(distances, axis=1)
            exploration = np.mean(min_distances > np.median(min_distances))
            
            # Exploitation: بهبود جواب‌های موجود
            if len(self.hypervolume_history) >= 2:
                improvement = self.hypervolume_history[-1] - self.hypervolume_history[-2]
                exploitation = max(0, min(1, improvement * 100))
            else:
                exploitation = 0.5
        
        return exploration, exploitation
    
    def update_all_metrics(self, population, pareto_pop, offspring_pop=None, mutation_pop=None):
        """
        به‌روزرسانی تمام شاخص‌های تشخیصی
        
        Parameters:
        -----------
        population: کل جمعیت فعلی
        pareto_pop: جواب‌های پارتو فعلی
        offspring_pop: فرزندان تولید شده از crossover
        mutation_pop: جواب‌های تولید شده از mutation
        """
        # شاخص‌های پایه
        pareto_front = np.array([ind['cost'] for ind in pareto_pop])
        
        # 1. Diversity
        diversity = self.calculate_diversity(population)
        self.diversity_history.append(diversity)
        
        # 2. Selection metrics
        sel_pressure = self.calculate_selection_pressure(population)
        self.selection_pressure.append(sel_pressure)
        
        elite_dom = self.calculate_elite_dominance(population)
        self.elite_dominance.append(elite_dom)
        
        # 3. Convergence metrics
        conv_rate = self.calculate_convergence_rate()
        self.convergence_rate.append(conv_rate)
        
        stag = self.calculate_stagnation()
        self.stagnation_counter.append(stag)
        
        # 4. Model metrics
        constr_viol = self.calculate_constraint_violation_rate(population)
        self.constraint_violation_rate.append(constr_viol)
        
        feasible_ratio = 1.0 - constr_viol
        self.feasible_solutions_ratio.append(feasible_ratio)
        
        corr_matrix = self.calculate_objective_correlation(pareto_front)
        self.objective_correlation.append(corr_matrix)
        
        # 5. Exploration vs Exploitation
        exploration, exploitation = self.calculate_exploration_exploitation(pareto_front)
        self.exploration_score.append(exploration)
        self.exploitation_score.append(exploitation)
        
        # 6. Operator success rates (اگر داده شده باشند)
        if offspring_pop is not None:
            # بررسی کنید چند فرزند به pareto front اضافه شدند
            offspring_success = self.check_improvement(offspring_pop, pareto_pop)
            self.offspring_improvements.append(offspring_success)
            
            crossover_success = self.calculate_crossover_success_rate(self.offspring_improvements[-10:])
            self.crossover_success_rate.append(crossover_success)
        
        if mutation_pop is not None:
            mutation_success = self.check_improvement(mutation_pop, pareto_pop)
            self.mutation_improvements.append(mutation_success)
            
            mut_success = self.calculate_mutation_success_rate(self.mutation_improvements[-10:])
            self.mutation_success_rate.append(mut_success)
        
        # ذخیره برای تکرار بعدی
        self.prev_pareto_front = pareto_front.copy()
    
    def check_improvement(self, new_pop, pareto_pop):
        """
        بررسی کنید که آیا جواب‌های جدید بهتر از pareto front هستند
        """
        if len(new_pop) == 0:
            return 0
        
        improved_count = 0
        pareto_costs = np.array([ind['cost'] for ind in pareto_pop])
        
        for ind in new_pop:
            # بررسی اینکه آیا این جواب dominate می‌شود یا نه
            is_dominated = False
            for p_cost in pareto_costs:
                if self.dominates_cost(p_cost, ind['cost']):
                    is_dominated = True
                    break
            
            if not is_dominated:
                improved_count += 1
        
        improvement_rate = improved_count / len(new_pop)
        return improvement_rate
    
    def dominates_cost(self, cost1, cost2):
        """بررسی domination بین دو cost"""
        return all(cost1 <= cost2) and any(cost1 < cost2)
    
    def plot_diagnostic_metrics(self, save_path=None):
        """
        رسم نمودارهای تشخیصی
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        iterations = range(1, len(self.diversity_history) + 1)
        
        # Row 1: Operator Performance
        ax1 = fig.add_subplot(gs[0, 0])
        if len(self.crossover_success_rate) > 0:
            ax1.plot(iterations[:len(self.crossover_success_rate)], 
                    self.crossover_success_rate, 'b-', linewidth=2, label='Crossover')
        if len(self.mutation_success_rate) > 0:
            ax1.plot(iterations[:len(self.mutation_success_rate)], 
                    self.mutation_success_rate, 'r-', linewidth=2, label='Mutation')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('🔧 Operator Success Rates\n(پایین = عملگرها ضعیف)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Low threshold')
        
        # Row 1: Diversity
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(iterations, self.diversity_history, 'g-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Diversity')
        ax2.set_title('🌈 Genetic Diversity\n(پایین = Premature Convergence)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Critical')
        
        # Row 1: Exploration vs Exploitation
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(iterations, self.exploration_score, 'b-', linewidth=2, label='Exploration')
        ax3.plot(iterations, self.exploitation_score, 'r-', linewidth=2, label='Exploitation')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Score')
        ax3.set_title('⚖️ Exploration vs Exploitation\n(باید متعادل باشد)', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Row 2: Selection Pressure
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(iterations, self.selection_pressure, 'm-', linewidth=2)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Selection Pressure')
        ax4.set_title('🎯 Selection Pressure\n(بالا = Diversity از دست می‌رود)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='High')
        
        # Row 2: Elite Dominance
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(iterations, self.elite_dominance, 'c-', linewidth=2)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Elite Dominance')
        ax5.set_title('👑 Elite Dominance\n(بالا = نخبگان تسلط دارند)', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Row 2: Convergence Rate
        ax6 = fig.add_subplot(gs[1, 2])
        if len(self.convergence_rate) > 0:
            ax6.plot(range(1, len(self.convergence_rate) + 1), 
                    self.convergence_rate, 'orange', linewidth=2)
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Convergence Rate')
        ax6.set_title('📈 Convergence Rate\n(نزدیک صفر = Stagnation)', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Row 3: Constraint Violation
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(iterations, self.constraint_violation_rate, 'r-', linewidth=2)
        ax7.set_xlabel('Iteration')
        ax7.set_ylabel('Violation Rate')
        ax7.set_title('⚠️ Constraint Violation Rate\n(بالا = مشکل در مدل)', fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Warning')
        
        # Row 3: Feasible Solutions
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(iterations, self.feasible_solutions_ratio, 'g-', linewidth=2)
        ax8.set_xlabel('Iteration')
        ax8.set_ylabel('Feasible Ratio')
        ax8.set_title('✅ Feasible Solutions Ratio\n(پایین = مدل سخت یا محدودیت‌ها)', fontweight='bold')
        ax8.grid(True, alpha=0.3)
        ax8.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5)
        
        # Row 3: Stagnation
        ax9 = fig.add_subplot(gs[2, 2])
        cumulative_stagnation = np.cumsum(self.stagnation_counter)
        ax9.plot(iterations, cumulative_stagnation, 'brown', linewidth=2)
        ax9.set_xlabel('Iteration')
        ax9.set_ylabel('Cumulative Stagnation')
        ax9.set_title('🛑 Stagnation Counter\n(بالا = الگوریتم Stuck شده)', fontweight='bold')
        ax9.grid(True, alpha=0.3)
        
        # Row 4: Objective Correlations (heatmap for last iteration)
        if len(self.objective_correlation) > 0:
            ax10 = fig.add_subplot(gs[3, :])
            last_corr = self.objective_correlation[-1]
            
            im = ax10.imshow(last_corr, cmap='RdYlGn_r', vmin=-1, vmax=1, aspect='auto')
            ax10.set_xticks(range(len(last_corr)))
            ax10.set_yticks(range(len(last_corr)))
            ax10.set_xticklabels([f'F{i+1}' for i in range(len(last_corr))])
            ax10.set_yticklabels([f'F{i+1}' for i in range(len(last_corr))])
            
            # اضافه کردن مقادیر
            for i in range(len(last_corr)):
                for j in range(len(last_corr)):
                    text = ax10.text(j, i, f'{last_corr[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=12)
            
            plt.colorbar(im, ax=ax10)
            ax10.set_title('🔗 Objective Correlations (Last Iteration)\n' + 
                          '(مثبت بالا = اهداف مستقل نیستند، مشکل در مدل)', 
                          fontweight='bold')
        
        fig.suptitle('Diagnostic Metrics for Algorithm Performance', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def diagnose_problems(self):
        """
        تشخیص خودکار مشکلات و ارائه پیشنهادات
        """
        print("\n" + "="*80)
        print("🔍 AUTOMATIC PROBLEM DIAGNOSIS")
        print("="*80)
        
        problems_found = []
        recommendations = []
        
        # بررسی operator success rate
        if len(self.crossover_success_rate) > 5:
            avg_crossover = np.mean(self.crossover_success_rate[-10:])
            if avg_crossover < 0.1:
                problems_found.append("❌ Crossover success rate خیلی پایین است")
                recommendations.append("   → عملگر Crossover را تغییر دهید یا p_crossover را افزایش دهید")
        
        if len(self.mutation_success_rate) > 5:
            avg_mutation = np.mean(self.mutation_success_rate[-10:])
            if avg_mutation < 0.05:
                problems_found.append("❌ Mutation success rate خیلی پایین است")
                recommendations.append("   → نرخ mutation را افزایش دهید یا عملگر mutation را بهبود دهید")
        
        # بررسی diversity
        if len(self.diversity_history) > 10:
            recent_diversity = np.mean(self.diversity_history[-10:])
            if recent_diversity < 0.1:
                problems_found.append("❌ Diversity خیلی پایین - Premature Convergence")
                recommendations.append("   → اندازه جمعیت را افزایش دهید")
                recommendations.append("   → نرخ mutation را افزایش دهید")
                recommendations.append("   → Selection pressure را کاهش دهید")
        
        # بررسی selection pressure
        if len(self.selection_pressure) > 5:
            avg_pressure = np.mean(self.selection_pressure[-10:])
            if avg_pressure > 0.7:
                problems_found.append("❌ Selection pressure خیلی بالا است")
                recommendations.append("   → tournament size را کاهش دهید")
                recommendations.append("   → از roulette wheel selection استفاده کنید")
        
        # بررسی stagnation
        if len(self.stagnation_counter) > 10:
            recent_stagnation = sum(self.stagnation_counter[-10:])
            if recent_stagnation > 5:
                problems_found.append("❌ Stagnation شدید - الگوریتم stuck شده")
                recommendations.append("   → تعداد iteration را افزایش دهید")
                recommendations.append("   → Restart mechanism اضافه کنید")
                recommendations.append("   → Diversity را افزایش دهید")
        
        # بررسی constraint violation
        if len(self.constraint_violation_rate) > 5:
            avg_violation = np.mean(self.constraint_violation_rate[-10:])
            if avg_violation > 0.3:
                problems_found.append("❌ نرخ نقض محدودیت بالا است")
                recommendations.append("   → مدل را بررسی کنید - محدودیت‌ها خیلی سخت هستند")
                recommendations.append("   → Repair mechanism اضافه کنید")
                recommendations.append("   → Penalty function را تنظیم کنید")
        
        # بررسی objective correlation
        if len(self.objective_correlation) > 0:
            last_corr = self.objective_correlation[-1]
            # بررسی همبستگی بالا (غیر از قطر اصلی)
            off_diagonal = last_corr[np.triu_indices(len(last_corr), k=1)]
            if np.any(off_diagonal > 0.7):
                problems_found.append("⚠️ همبستگی بالا بین برخی اهداف")
                recommendations.append("   → بررسی کنید آیا اهداف مستقل هستند")
                recommendations.append("   → ممکن است نیازی به optimization چندهدفه نباشد")
        
        # بررسی exploration vs exploitation
        if len(self.exploration_score) > 10 and len(self.exploitation_score) > 10:
            recent_exploration = np.mean(self.exploration_score[-10:])
            recent_exploitation = np.mean(self.exploitation_score[-10:])
            
            if recent_exploration < 0.2:
                problems_found.append("❌ Exploration خیلی کم - الگوریتم تنها exploit می‌کند")
                recommendations.append("   → Mutation rate را افزایش دهید")
                recommendations.append("   → اندازه جمعیت را افزایش دهید")
            
            if recent_exploitation < 0.1:
                problems_found.append("❌ Exploitation ضعیف - جواب‌ها بهبود نمی‌یابند")
                recommendations.append("   → Crossover rate را افزایش دهید")
                recommendations.append("   → Selection pressure را افزایش دهید")
        
        # نمایش نتایج
        if len(problems_found) == 0:
            print("\n✅ هیچ مشکل جدی شناسایی نشد!")
            print("   الگوریتم به خوبی کار می‌کند.")
        else:
            print(f"\n🔴 {len(problems_found)} مشکل شناسایی شد:\n")
            for problem in problems_found:
                print(problem)
            
            print("\n" + "="*80)
            print("💡 RECOMMENDATIONS")
            print("="*80)
            for rec in recommendations:
                print(rec)
        
        print("\n" + "="*80)