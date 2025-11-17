import numpy as np
from typing import List, Tuple, Optional, Dict
import pickle
import os
from copy import deepcopy

class HypervolumeManager:
    """
    مدیریت محاسبه هایپرولیوم با نقطه مرجع ثابت
    
    این کلاس مشکل تغییر نقطه مرجع در طول اجرای الگوریتم را حل می‌کند
    """
    
    def __init__(self, estimation_method='conservative'):
        """
        Parameters:
        -----------
        estimation_method: str
            روش تخمین نقطه مرجع اولیه
            - 'conservative': تخمین محافظه‌کارانه بر اساس نسل‌های اولیه
            - 'adaptive': تطبیقی بر اساس روند بهبود
            - 'fixed_margin': حاشیه ثابت از بدترین مقادیر
        """
        self.estimation_method = estimation_method
        self.all_pareto_fronts = []  # ذخیره تمام فرانت‌های پارتو
        self.generation_data = {}    # داده‌های هر نسل
        self.reference_point = None  # نقطه مرجع نهایی
        self.estimated_reference_point = None  # نقطه مرجع تخمینی
        self.hypervolume_history = []  # تاریخچه هایپرولیوم با نقطه مرجع ثابت
        self.preliminary_hv_history = []  # تاریخچه هایپرولیوم موقت
        self.random_points = None # نقاط تصادفی برای روش Monte Carlo
        
    def store_generation_data(self, generation: int, pareto_front: np.ndarray):
        """
        ذخیره داده‌های هر نسل
        
        Parameters:
        -----------
        generation: int - شماره نسل
        pareto_front: np.ndarray - فرانت پارتو نسل فعلی
        """
        front_copy = pareto_front.copy()
        self.all_pareto_fronts.append(front_copy)
        self.generation_data[generation] = {
            'front': front_copy,
            'nadir': np.max(front_copy, axis=0),
            'ideal': np.min(front_copy, axis=0)
        }
        
        # تخمین نقطه مرجع در نسل‌های اولیه
        if generation <= 10:  # در 10 نسل اول
            self._update_reference_estimation(generation)
    
    def _update_reference_estimation(self, generation: int):
        """
        به‌روزرسانی تخمین نقطه مرجع بر اساس نسل‌های اولیه
        """
        if self.estimation_method == 'conservative':
            # تخمین محافظه‌کارانه: بدترین مقادیر + 50% حاشیه
            all_nadirs = [data['nadir'] for data in self.generation_data.values()]
            worst_nadir = np.max(all_nadirs, axis=0)
            self.estimated_reference_point = worst_nadir * 1.5
            
        elif self.estimation_method == 'adaptive':
            # تخمین تطبیقی بر اساس روند تغییرات
            if generation >= 5:
                recent_nadirs = [self.generation_data[g]['nadir'] 
                               for g in range(max(0, generation-4), generation+1)]
                trend = np.mean(np.diff(recent_nadirs, axis=0), axis=0)
                current_nadir = self.generation_data[generation]['nadir']
                # پیش‌بینی بدترین حالت با در نظر گیری روند
                predicted_worst = current_nadir + trend * 20  # پیش‌بینی 20 نسل آینده
                self.estimated_reference_point = predicted_worst * 1.2
            else:
                # در نسل‌های اولیه از روش محافظه‌کارانه استفاده کن
                all_nadirs = [data['nadir'] for data in self.generation_data.values()]
                worst_nadir = np.max(all_nadirs, axis=0)
                self.estimated_reference_point = worst_nadir * 1.5
                
        elif self.estimation_method == 'fixed_margin':
            # حاشیه ثابت 20%
            all_nadirs = [data['nadir'] for data in self.generation_data.values()]
            worst_nadir = np.max(all_nadirs, axis=0)
            self.estimated_reference_point = worst_nadir * 1.2
    
    def calculate_preliminary_hypervolume(self, generation: int) -> float:
        """
        محاسبه هایپرولیوم موقت با نقطه مرجع تخمینی
        
        Parameters:
        -----------
        generation: int - شماره نسل
        
        Returns:
        --------
        float: مقدار هایپرولیوم موقت
        """
        if generation not in self.generation_data:
            return 0.0
            
        if self.estimated_reference_point is None:
            return 0.0
            
        pareto_front = self.generation_data[generation]['front']
        hv = self._compute_hypervolume(pareto_front, self.estimated_reference_point)
        self.preliminary_hv_history.append(hv)
        
        return hv
    
    def finalize_reference_point(self, margin_factor: float = 1.1):
        """
        محاسبه نقطه مرجع نهایی بر اساس تمام نسل‌ها
        
        Parameters:
        -----------
        margin_factor: float - ضریب حاشیه (پیش‌فرض 1.1)
        """
        if not self.all_pareto_fronts:
            raise ValueError("هیچ داده‌ای برای محاسبه نقطه مرجع وجود ندارد")
        
        # ترکیب تمام فرانت‌های پارتو
        all_points = np.vstack(self.all_pareto_fronts)
        
        # محاسبه نقطه nadir (بدترین مقادیر در هر هدف)
        nadir_point = np.max(all_points, axis=0)
        
        # محاسبه نقطه مرجع نهایی
        self.reference_point = nadir_point * margin_factor
        
        return self.reference_point
    
    def calculate_final_hypervolumes(self) -> List[float]:
        """
        محاسبه هایپرولیوم نهایی برای تمام نسل‌ها با نقطه مرجع ثابت
        
        Returns:
        --------
        List[float]: لیست مقادیر هایپرولیوم برای هر نسل
        """
        if self.reference_point is None:
            self.finalize_reference_point()
        
        self.hypervolume_history = []
        
        for generation in sorted(self.generation_data.keys()):
            pareto_front = self.generation_data[generation]['front']
            hv = self._compute_hypervolume(pareto_front, self.reference_point)
            self.hypervolume_history.append(hv)
        
        return self.hypervolume_history
    
    def _compute_hypervolume(self, pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
        """
        محاسبه هایپرولیوم با استفاده از الگوریتم WFG
        
        Parameters:
        -----------
        pareto_front: np.ndarray - فرانت پارتو
        reference_point: np.ndarray - نقطه مرجع
        
        Returns:
        --------
        float: مقدار هایپرولیوم
        """
        if len(pareto_front) == 0:
            return 0.0
        
        n_objectives = pareto_front.shape[1]
        
        if n_objectives == 2:
            return self._hypervolume_2d(pareto_front, reference_point)
        else:
            # برای ابعاد بالاتر از روش Monte Carlo استفاده می‌کنیم
            return self._hypervolume_monte_carlo(pareto_front, reference_point)
    
    def _hypervolume_2d(self, pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
        """محاسبه هایپرولیوم برای 2 هدف"""
        # مرتب‌سازی بر اساس هدف اول
        sorted_indices = np.argsort(pareto_front[:, 0])
        sorted_front = pareto_front[sorted_indices]
        
        hypervolume = 0.0
        prev_x = reference_point[0]
        
        for point in sorted_front:
            if point[0] < reference_point[0] and point[1] < reference_point[1]:
                width = prev_x - point[0]
                height = reference_point[1] - point[1]
                hypervolume += width * height
                prev_x = point[0]
        
        return hypervolume
    
    def _hypervolume_3d(self, pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
        """محاسبه هایپرولیوم برای 3 هدف با الگوریتم WFG"""
        # پیاده‌سازی ساده‌شده الگوریتم WFG برای 3 بعد
        hypervolume = 0.0
        
        # فیلتر کردن نقاط dominated
        dominated_points = []
        for point in pareto_front:
            if all(point < reference_point):
                dominated_points.append(point)
        
        if not dominated_points:
            return 0.0
        
        dominated_points = np.array(dominated_points)
        
        # محاسبه تقریبی با تقسیم فضا
        for point in dominated_points:
            volume = np.prod(reference_point - point)
            hypervolume += volume
        
        # تصحیح overlap (تقریبی)
        hypervolume /= len(dominated_points) ** 0.5
        
        return hypervolume
    
    def _hypervolume_monte_carlo(self, pareto_front: np.ndarray, reference_point: np.ndarray, 
                                n_samples: int = 100000) -> float:
        """محاسبه هایپرولیوم با روش Monte Carlo برای ابعاد بالا"""
        if len(pareto_front) == 0:
            return 0.0
        
        # تولید نقاط تصادفی در فضای مرجع
        n_objectives = len(reference_point)
        
        ideal_point = np.array([0.0, 0.0, 0.0])
        
        if self.random_points is None:
            # تولید نقاط تصادفی بین ideal و reference
            self.random_points = np.random.uniform(0, 1, size=(n_samples, n_objectives))
            
        span = reference_point - ideal_point
        if len(pareto_front) == 0:
            hv = 0
        # 3) normalize PF into [0,1]^3
        pf_norm = (pareto_front - ideal_point) / span
        pf_norm = np.clip(pf_norm, 0.0, 1.0)

        # 4) dominance test (vectorized):
        # a rnd point x is dominated if exists s in PF with s <= x (componentwise)
        # shape tricks: compare all rnd to all pf
        # pf_norm[None, :, :] -> (1, N, M); rnd[:, None, :] -> (S, 1, M)
        dominated_by_any = np.all(pf_norm[None, :, :] <= self.random_points[:, None, :], axis=2).any(axis=1)
        hv = dominated_by_any.mean()  # in [0,1]
        return hv
    
    def get_metrics_summary(self) -> Dict:
        """
        خلاصه‌ای از شاخص‌های محاسبه شده
        
        Returns:
        --------
        Dict: خلاصه شاخص‌ها
        """
        summary = {
            'total_generations': len(self.generation_data),
            'reference_point': self.reference_point.tolist() if self.reference_point is not None else None,
            'estimated_reference_point': self.estimated_reference_point.tolist() if self.estimated_reference_point is not None else None,
            'final_hypervolume': self.hypervolume_history[-1] if self.hypervolume_history else None,
            'hypervolume_improvement': None,
            'estimation_method': self.estimation_method
        }
        
        if len(self.hypervolume_history) > 1:
            initial_hv = self.hypervolume_history[0]
            final_hv = self.hypervolume_history[-1]
            if initial_hv > 0:
                improvement = ((final_hv - initial_hv) / initial_hv) * 100
                summary['hypervolume_improvement'] = improvement
        
        return summary
    
    def save_data(self, filepath: str):
        """ذخیره داده‌ها در فایل"""
        data = {
            'all_pareto_fronts': self.all_pareto_fronts,
            'generation_data': self.generation_data,
            'reference_point': self.reference_point,
            'estimated_reference_point': self.estimated_reference_point,
            'hypervolume_history': self.hypervolume_history,
            'preliminary_hv_history': self.preliminary_hv_history,
            'estimation_method': self.estimation_method
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_data(self, filepath: str):
        """بارگذاری داده‌ها از فایل"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.all_pareto_fronts = data['all_pareto_fronts']
        self.generation_data = data['generation_data']
        self.reference_point = data['reference_point']
        self.estimated_reference_point = data['estimated_reference_point']
        self.hypervolume_history = data['hypervolume_history']
        self.preliminary_hv_history = data['preliminary_hv_history']
        self.estimation_method = data['estimation_method']