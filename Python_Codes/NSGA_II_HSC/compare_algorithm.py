import numpy as np

# ============================================================
#                 1) COVERAGE INDICATOR
# ============================================================

def dominates(a, b):
    """
    چک می‌کند که نقطه a، نقطه b را weakly dominate می‌کند یا نه.
    در مسائل کمینه‌سازی:
        a dominates b  ⇔  a_i <= b_i برای تمام i  و حداقل یک ≤ strict باشد.
    """
    return np.all(a <= b) and np.any(a < b)


def coverage_indicator(A, B):
    """
    محاسبه C(A,B):
    سهم نقاط B که توسط حداقل یک نقطه در A weakly dominated می‌شوند.
    مقدار خروجی بین 0 و 1 است.

    اگر C(A,B) = 1  →  تمام نقاط B توسط A پوشش داده شده‌اند.
    اگر C(A,B) = 0  →  هیچ نقطه‌ای از B توسط A پوشش داده نشده.
    """
    dominated_count = 0

    for b in B:
        # اگر b توسط حداقل یک نقطه از A dominated باشد، شمارش بالا می‌رود
        if any(dominates(a, b) for a in A):
            dominated_count += 1

    return dominated_count / len(B)


# ============================================================
#            2) MULTIPLICATIVE EPSILON INDICATOR
# ============================================================

def epsilon_indicator(A, B):
    """
    محاسبه ε(A,B):
    کمترین ε که باعث شود A، مجموعه B را ε-dominant کند.
    
    تعریف ε-dominance:
        A ε-dominates B ⇔ برای هر b ∈ B یک a ∈ A وجود دارد که:
            a_i * ε <= b_i   برای تمام i

    این شاخص اندازه می‌گیرد A چند برابر بدتر باید شود تا بتواند B را بپوشاند.
    اگر ε < 1 باشد → A بهتر از B است.
    """
    eps_values = []

    for b in B:
        # برای هر نقطه در B، باید بهترین نقطه در A را پیدا کنیم
        eps_for_b = np.inf
        safe_b = np.where(b == 0, 1e-10, b)
        for a in A:
            # ε لازم برای اینکه a_i * ε <= b_i 
            eps_i = a / safe_b
            eps = np.max(eps_i)   # حداکثر بعد تعیین‌کننده ε است
            eps_for_b = min(eps_for_b, eps)

        eps_values.append(eps_for_b)

    return max(eps_values)   # بدترین حالت بین همه b ها


# ============================================================
#               3) BINARY HYPERVOLUME DIFFERENCE
# ============================================================

def dominated_hypervolume(A, ref, samples=100000):
    rng = np.random.default_rng(seed=42)
    rand_points = rng.uniform(0, ref, size=(samples, 3))
    def dominated_by_set(p, A):
        return any(np.all(a <= p) for a in A)
    count = sum(dominated_by_set(p, A) for p in rand_points)
    volume_box = np.prod(ref)
    hv_estimate = volume_box * (count / samples)
    return hv_estimate


def binary_hypervolume_indicator(A, B, ref):
    
    # ideal = np.minimum(np.min(A, axis=0), np.min(B, axis=0))
    
    hvA = dominated_hypervolume(A, ref)
    hvB = dominated_hypervolume(B, ref)

    # محاسبه حجم کل جعبه مرجع برای نرمال‌سازی
    ref_volume = np.prod(ref)


    # خروجی نرمال‌شده بین -1 و +1 (حداکثر اختلاف ممکن)
    normalized_diff = (hvA - hvB) / ref_volume if ref_volume != 0 else 0.0
    return normalized_diff