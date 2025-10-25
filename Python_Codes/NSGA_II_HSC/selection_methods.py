import math
import random
from typing import List, Dict, Tuple, Optional

class SelectionMethods:
    
    def __init__(self) -> None:
        pass
    
    Individual = Dict[str, object]

    # --------- Utilities ---------

    def is_feasible(ind: Individual) -> bool:
        return float(ind.get("constraint_violation", 0.0)) <= 0.0

    def better_by_rank_crowding(a: Individual, b: Individual) -> Individual:
        """NSGA-II criterion: lower rank wins; tie-break by higher crowding."""
        ra, rb = int(a["rank"]), int(b["rank"])
        if ra != rb:
            return a if ra < rb else b
        ca, cb = float(a["crowding_distance"]), float(b["crowding_distance"])
        if ca != cb:
            return a if ca > cb else b
        # final random tie-break
        return a if random.random() < 0.5 else b

    def feasibility_rule(self, a: Individual, b: Individual) -> Optional[Individual]:
        """Return winner by feasibility rule if distinguishable, else None."""
        fa, fb = self.is_feasible(a), self.is_feasible(b)
        if fa and not fb:
            return a
        if fb and not fa:
            return b
        if not fa and not fb:
            va, vb = float(a["constraint_violation"]), float(b["constraint_violation"])
            if va != vb:
                return a if va < vb else b
        return None  # need other tie-breaker

    def euclidean(x: List[float], y: List[float]) -> float:
        return math.sqrt(sum((xi - yi) ** 2 for xi, yi in zip(x, y)))

    def distance_to_refs(self, cost: List[float], refs: List[List[float]]) -> float:
        """Distance to nearest reference point (smaller = closer به مرجع)."""
        return min(self.euclidean(cost, r) for r in refs) if refs else 0.0

    def dominates_epsilon(a: Individual, b: Individual, eps: float = 0.0) -> bool:
        """ε-dominance (minimization). a ε-dominates b?"""
        A = a["cost"]
        B = b["cost"]
        assert isinstance(A, (list, tuple)) and isinstance(B, (list, tuple))
        # a is no worse than b + eps in all, and strictly better by >= eps in at least one
        no_worse = all(Ai <= Bi + eps for Ai, Bi in zip(A, B)) 
        strictly_better = any(Ai < Bi - eps for Ai, Bi in zip(A, B))
        return no_worse and strictly_better

    # --------- 1) Crowded Binary Tournament (CBT) ---------
    def select_cbt(self, pop: List[Individual]) -> Individual:
        """Standard NSGA-II crowded binary tournament (rank -> crowding)."""
        a, b = random.sample(pop, 2)
        # (بدون توجه به قیود؛ اگر می‌خواهید قید را دخیل کنید از select_ff یا select_cbt_feas استفاده کنید)
        return self.better_by_rank_crowding(a, b)

    def select_cbt_feas(self, pop: List[Individual]) -> Individual:
        """CBT با قاعده امکان‌پذیری به‌عنوان پیش‌فیلتر."""
        a, b = random.sample(pop, 2)
        by_feas = self.feasibility_rule(a, b)
        if by_feas is not None:
            return by_feas
        return self.better_by_rank_crowding(a, b)

    # --------- 2) Adaptive k-Tournament (A-kT) ---------
    def select_k_tournament(self, pop: List[Individual], k: int = 2, use_feas: bool = False) -> Individual:
        """Tournament of size k; comparator = NSGA-II; optionally feasibility-first."""
        contenders = random.sample(pop, k)
        best = contenders[0]
        for c in contenders[1:]:
            if use_feas:
                by_feas = self.feasibility_rule(best, c)
                if by_feas is not None:
                    best = by_feas
                    continue
            best = self.better_by_rank_crowding(best, c)
        return best

    # --------- 3) Feasibility-First Tournament (FF) ---------
    def select_ff(self, pop: List[Individual], k: int = 2) -> Individual:
        """Feasible-first; among ties use NSGA-II criterion."""
        contenders = random.sample(pop, k)
        # Partition
        feas = [x for x in contenders if self.is_feasible(x)]
        if feas:
            # choose best among feasible
            best = feas[0]
            for c in feas[1:]:
                best = self.better_by_rank_crowding(best, c)
            return best
        # no feasible: pick with smallest violation, tie-break NSGA-II
        best = contenders[0]
        for c in contenders[1:]:
            va, vb = float(best["constraint_violation"]), float(c["constraint_violation"])
            if vb < va:
                best = c
            elif vb == va:
                best = self.better_by_rank_crowding(best, c)
        return best

    # --------- 4) Rank-Based Stochastic (roulette + SUS) ---------

    def rank_weights(pop: List[Individual], power: float = 1.0) -> List[float]:
        """
        Weight ~ 1 / (rank^power). rank=1 → وزن بزرگتر.
        برای جلوگیری از بی‌نهایت: از max(rank,1) استفاده می‌شود.
        """
        ws = [1.0 / (max(1, int(ind["rank"])) ** power) for ind in pop]
        s = sum(ws)
        if s == 0:
            return [1.0 / len(pop)] * len(pop)
        return [w / s for w in ws]

    def select_rank_roulette(self, pop: List[Individual], power: float = 1.0) -> Individual:
        """یک والد با وزن‌دهی مبتنی بر rank انتخاب می‌کند (roulette)."""
        ps = self.rank_weights(pop, power=power)
        r = random.random()
        cum = 0.0
        for ind, p in zip(pop, ps):
            cum += p
            if r <= cum:
                return ind
        return pop[-1]

    def sus_pick_indices(prob: List[float], n: int) -> List[int]:
        """Stochastic Universal Sampling روی توزیع prob."""
        N = len(prob)
        cum = [0.0]
        s = 0.0
        for p in prob:
            s += p
            cum.append(s)
        step = s / n
        start = random.uniform(0, step)
        picks = []
        i, acc = 0, start
        for _ in range(n):
            while acc > cum[i+1]:
                i += 1
            picks.append(i)
            acc += step
        return picks
    
    def select_rank_sus(self, pop: List[Individual], n: int, power: float = 1.0) -> List[Individual]:
        """برمی‌گرداند n والد با SUS (برای ساخت mating pool دسته‌ای)."""
        ps = self.rank_weights(pop, power=power)
        idxs = self.sus_pick_indices(ps, n)
        return [pop[i] for i in idxs]

    # --------- 5) Reference-Biased Tournament (RBT) ---------
    def select_rbt(self, pop: List[Individual],
                refs: List[List[float]],
                k: int = 2,
                tau: float = 1.0) -> Individual:
        """
        CBT با بایاس به سمت نقاط مرجع (extremes).
        معیار: rank بهتر → اگر برابر، فاصله تا مرجع کوچکتر (به وزن exp(-d/tau)) → سپس crowding.
        """
        contenders = random.sample(pop, k)
        # Precompute distances and weights
        scored = []
        for ind in contenders:
            d = self.distance_to_refs(ind["cost"], refs)
            w = math.exp(-(d / max(1e-12, tau)))
            scored.append((ind, d, w))
        # Sort-like comparison:
        best = scored[0][0]
        best_d = scored[0][1]
        best_w = scored[0][2]
        for ind, d, w in scored[1:]:
            # First: rank
            r_best, r_c = int(best["rank"]), int(ind["rank"])
            if r_c < r_best:
                best, best_d, best_w = ind, d, w
                continue
            if r_c > r_best:
                continue
            # Second: closer to reference (bigger w)
            if w > best_w:
                best, best_d, best_w = ind, d, w
                continue
            if w < best_w:
                continue
            # Third: crowding
            best = self.better_by_rank_crowding(best, ind)
        return best

    # --------- 6) Age-Diversity Tournament (ADT) ---------
    def select_adt(self, pop: List[Individual], k: int = 2, prefer_younger: bool = True) -> Individual:
        """
        CBT با پاداش سن کم/زیاد. نیاز به کلید اختیاری 'age' (نسل از تولد).
        اگر نباشد، age=0 فرض می‌شود.
        """
        contenders = random.sample(pop, k)
        def get_age(ind): return int(ind.get("age", 0))
        best = contenders[0]
        for c in contenders[1:]:
            # rank first
            rb, rc = int(best["rank"]), int(c["rank"])
            if rc < rb:
                best = c
                continue
            if rc > rb:
                continue
            # age preference
            ab, ac = get_age(best), get_age(c)
            if prefer_younger:
                if ac < ab:
                    best = c
                    continue
                if ac > ab:
                    continue
            else:
                if ac > ab:
                    best = c
                    continue
                if ac < ab:
                    continue
            # crowding as final tie-break
            best = self.better_by_rank_crowding(best, c)
        return best

    # --------- 7) ε-Dominance Tournament (ε-DT) ---------
    def select_epsilon_dt(self, pop: List[Individual], k: int = 2, eps: float = 0.0) -> Individual:
        """
        Tournament با قضاوت ε-dominance. اگر هیچ‌کس دیگری را ε-dominate نکند،
        می‌افتد روی NSGA-II (rank→crowding).
        """
        contenders = random.sample(pop, k)
        # Try to find someone who ε-dominates all others
        winner = contenders[0]
        for c in contenders[1:]:
            # Try dominance both ways
            if self.dominates_epsilon(c, winner, eps):
                winner = c
            elif not self.dominates_epsilon(winner, c, eps):
                # neither dominates; defer to NSGA-II between them
                winner = self.better_by_rank_crowding(winner, c)
        return winner

    # --------- Master: build mating pool ---------

    def select_one(self, pop: List[Individual],
                method: str,
                **kwargs) -> Individual:
        method = method.lower()
        if method == "cbt":
            return self.select_cbt(pop)
        if method == "cbt_feas":
            return self.select_cbt_feas(pop)
        if method in ("kt", "a-kt", "k_tournament"):
            k = int(kwargs.get("k", 2))
            use_feas = bool(kwargs.get("use_feas", False))
            return self.select_k_tournament(pop, k=k, use_feas=use_feas)
        if method in ("ff", "feasibility_first"):
            k = int(kwargs.get("k", 2))
            return self.select_ff(pop, k=k)
        if method in ("rbs", "rank_roulette"):
            power = float(kwargs.get("power", 1.0))
            return self.select_rank_roulette(pop, power=power)
        if method in ("rbt", "reference_biased"):
            refs = kwargs.get("refs", [])
            k = int(kwargs.get("k", 2))
            tau = float(kwargs.get("tau", 1.0))
            return self.select_rbt(pop, refs=refs, k=k, tau=tau)
        if method in ("adt", "age_diversity"):
            k = int(kwargs.get("k", 2))
            prefer_younger = bool(kwargs.get("prefer_younger", True))
            return self.select_adt(pop, k=k, prefer_younger=prefer_younger)
        if method in ("edt", "epsilon_dt", "epsilon-dt"):
            k = int(kwargs.get("k", 2))
            eps = float(kwargs.get("eps", 0.0))
            return self.select_epsilon_dt(pop, k=k, eps=eps)
        raise ValueError(f"Unknown selection method: {method}")

    def select_mating_pool(self, pop: List[Individual],
                        n_parents: int,
                        method: str,
                        **kwargs) -> List[Individual]:
        """
        اگر method == 'sus' یا 'rbs_sus' از SUS برای n والد استفاده می‌شود.
        در غیر این صورت n بار select_one صدا زده می‌شود.
        """
        method_l = method.lower()
        if method_l in ("sus", "rbs_sus", "rank_sus"):
            power = float(kwargs.get("power", 1.0))
            return self.select_rank_sus(pop, n=n_parents, power=power)
        return [self.select_one(pop, method=method, **kwargs) for _ in range(n_parents)]
