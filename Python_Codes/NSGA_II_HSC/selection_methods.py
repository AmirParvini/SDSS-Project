import math
import random
from typing import List, Dict, Optional
import numpy as np


class SelectionMethods:
    def __init__(self) -> None:
        pass

    Individual = Dict[str, object]

    # --------- Utilities ---------

    def is_feasible(self, ind: Individual) -> bool:
        return float(ind.get("constraint_violation", 0.0)) <= 0.0

    def better_by_rank_crowding(self, a: Individual, b: Individual) -> Individual:
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

    def euclidean(self, x: List[float], y: List[float]) -> float:
        return math.sqrt(sum((xi - yi) ** 2 for xi, yi in zip(x, y)))

    def distance_to_refs(self, cost: List[float], refs: List[List[float]]) -> float:
        """Distance to nearest reference point (smaller = closer به مرجع)."""
        return min(self.euclidean(cost, r) for r in refs) if refs else 0.0

    def dominates_epsilon(self, a: Individual, b: Individual, eps: float = 0.0) -> bool:
        """ε-dominance (minimization). a ε-dominates b?"""
        A = a["cost"]
        B = b["cost"]
        assert isinstance(A, (np.ndarray)) and isinstance(B, (np.ndarray))
        # a is no worse than b + eps in all, and strictly better by >= eps in at least one
        no_worse = all(Ai <= Bi + eps for Ai, Bi in zip(A, B))
        strictly_better = any(Ai < Bi - eps for Ai, Bi in zip(A, B))
        return no_worse and strictly_better

    # --------- 1) Crowded Binary Tournament (CBT) ---------
    def crowded_binary_tournament(self, pop: List[Individual], **selection_args) -> Individual:
        """Standard NSGA-II crowded binary tournament (rank -> crowding)."""
        k = int(selection_args.get("k", 2))
        a, b = random.sample(pop, k)
        by_feas = self.feasibility_rule(a, b)
        if by_feas is not None:
            return by_feas
        return self.better_by_rank_crowding(a, b)

    def crowded_binary_tournament_feasible(self, pop: List[Individual]) -> Individual:
        """CBT با قاعده امکان‌پذیری به‌عنوان پیش‌فیلتر."""
        a, b = random.sample(pop, 2)
        by_feas = self.feasibility_rule(a, b)
        if by_feas is not None:
            return by_feas
        return self.better_by_rank_crowding(a, b)

    # --------- 2) Adaptive k-Tournament (A-kT) ---------
    def adaptive_k_tournament(
        self, pop: List[Individual], **selection_args
    ) -> Individual:
        """Tournament of size k; comparator = NSGA-II."""
        k = int(selection_args.get("k", 2))
        contenders = random.sample(pop, k)
        best = contenders[0]
        for c in contenders[1:]:
            by_feas = self.feasibility_rule(best, c)
            if by_feas is not None:
                best = by_feas
                continue
            best = self.better_by_rank_crowding(best, c)
        return best

    # --------- 3) Feasibility-First Tournament (FF) ---------
    def feasibility_first_tournament(
        self, pop: List[Individual], **selection_args
    ) -> Individual:
        """Feasible-first; among ties use NSGA-II criterion."""
        k = int(selection_args.get("k", 2))
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
            va, vb = (
                float(best["constraint_violation"]),
                float(c["constraint_violation"]),
            )
            if vb < va:
                best = c
            elif vb == va:
                best = self.better_by_rank_crowding(best, c)
        return best

    # --------- 4) Rank-Based Stochastic (roulette + SUS) ---------

    def rank_weights(self, pop: List[Individual], power: float = 1.0) -> List[float]:
        """
        Weight ~ 1 / (rank^power). rank=1 → وزن بزرگتر.
        برای جلوگیری از بی‌نهایت: از max(rank,1) استفاده می‌شود.
        """
        ws = [1.0 / (max(1, int(ind["rank"])) ** power) for ind in pop]
        s = sum(ws)
        if s == 0:
            return [1.0 / len(pop)] * len(pop)
        return [w / s for w in ws]

    def rank_based_roulette(
        self, pop: List[Individual], **selection_args
    ) -> Individual:
        """یک والد با وزن‌دهی مبتنی بر rank انتخاب می‌کند (roulette)."""
        power = float(selection_args.get("power", 1.0))
        # First, try to select from feasible individuals if any exist
        feasible = [ind for ind in pop if self.is_feasible(ind)]
        if feasible:
            ps = self.rank_weights(feasible, power=power)
            r = random.random()
            cum = 0.0
            for ind, p in zip(feasible, ps):
                cum += p
                if r <= cum:
                    return ind
            return feasible[-1]
        # If no feasible, use feasibility_rule for comparison
        # Select two candidates and use feasibility_rule
        candidates = random.sample(pop, min(2, len(pop)))
        if len(candidates) == 2:
            by_feas = self.feasibility_rule(candidates[0], candidates[1])
            if by_feas is not None:
                return by_feas
        # Fallback to original roulette on all population
        ps = self.rank_weights(pop, power=power)
        r = random.random()
        cum = 0.0
        for ind, p in zip(pop, ps):
            cum += p
            if r <= cum:
                return ind
        return pop[-1]

    def sus_pick_indices(self, prob: List[float], n: int) -> List[int]:
        """Stochastic Universal Sampling روی توزیع prob."""
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
            while acc > cum[i + 1]:
                i += 1
            picks.append(i)
            acc += step
        return picks

    def rank_based_sus(
        self, pop: List[Individual], n: int, **selection_args
    ) -> List[Individual]:
        """برمی‌گرداند n والد با SUS (برای ساخت mating pool دسته‌ای)."""
        power = float(selection_args.get("power", 1.0))
        # Prioritize feasible individuals
        feasible = [ind for ind in pop if self.is_feasible(ind)]
        if feasible and len(feasible) >= n:
            # Use only feasible individuals
            ps = self.rank_weights(feasible, power=power)
            idxs = self.sus_pick_indices(ps, n)
            return [feasible[i] for i in idxs]
        # If not enough feasible, use feasibility_rule for selection
        # For each selection, use feasibility_rule when comparing
        selected = []
        for _ in range(n):
            if len(pop) == 1:
                selected.append(pop[0])
            else:
                candidates = random.sample(pop, min(2, len(pop)))
                if len(candidates) == 2:
                    by_feas = self.feasibility_rule(candidates[0], candidates[1])
                    if by_feas is not None:
                        selected.append(by_feas)
                    else:
                        # Use rank-based selection between the two
                        ps = self.rank_weights(candidates, power=power)
                        idxs = self.sus_pick_indices(ps, 1)
                        selected.append(candidates[idxs[0]])
                else:
                    selected.append(candidates[0])
        return selected

    # --------- 5) Reference-Biased Tournament (RBT) ---------
    def reference_biased_tournament(
        self, pop: List[Individual], **selection_args
    ) -> Individual:
        """
        CBT با بایاس به سمت نقاط مرجع (extremes).
        معیار: feasibility → rank بهتر → اگر برابر، فاصله تا مرجع کوچکتر (به وزن exp(-d/tau)) → سپس crowding.
        """
        k = int(selection_args.get("k", 2))
        refs = selection_args.get("refs", [])
        tau = float(selection_args.get("tau", 1.0))
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
            # First: feasibility rule
            by_feas = self.feasibility_rule(best, ind)
            if by_feas is not None:
                if by_feas == ind:
                    best, best_d, best_w = ind, d, w
                continue
            # Second: rank
            r_best, r_c = int(best["rank"]), int(ind["rank"])
            if r_c < r_best:
                best, best_d, best_w = ind, d, w
                continue
            if r_c > r_best:
                continue
            # Third: closer to reference (bigger w)
            if w > best_w:
                best, best_d, best_w = ind, d, w
                continue
            if w < best_w:
                continue
            # Fourth: crowding
            best = self.better_by_rank_crowding(best, ind)
        return best

    # --------- 6) Age-Diversity Tournament (ADT) ---------
    def age_diversity_tournament(
        self, pop: List[Individual], **selection_args
    ) -> Individual:
        """
        CBT با پاداش سن کم/زیاد. نیاز به کلید اختیاری 'age' (نسل از تولد).
        اگر نباشد، age=0 فرض می‌شود.
        """
        k = int(selection_args.get("k", 2))
        prefer_younger = bool(selection_args.get("prefer_younger", True))
        contenders = random.sample(pop, k)

        def get_age(ind):
            return int(ind.get("age", 0))

        best = contenders[0]
        for c in contenders[1:]:
            # First: feasibility rule
            by_feas = self.feasibility_rule(best, c)
            if by_feas is not None:
                best = by_feas
                continue
            # Second: rank
            rb, rc = int(best["rank"]), int(c["rank"])
            if rc < rb:
                best = c
                continue
            if rc > rb:
                continue
            # Third: age preference
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
            # Fourth: crowding as final tie-break
            best = self.better_by_rank_crowding(best, c)
        return best

    # --------- 7) ε-Dominance Tournament (ε-DT) ---------
    def epsilon_dominance_tournament(
        self, pop: List[Individual], **selection_args
    ) -> Individual:
        """
        Tournament با قضاوت ε-dominance. اگر هیچ‌کس دیگری را ε-dominate نکند،
        می‌افتد روی NSGA-II (rank→crowding).
        """
        k = int(selection_args.get("k", 2))
        eps = float(selection_args.get("eps", 0.0))
        contenders = random.sample(pop, k)
        # Try to find someone who ε-dominates all others
        winner = contenders[0]
        for c in contenders[1:]:
            # First: feasibility rule
            by_feas = self.feasibility_rule(winner, c)
            if by_feas is not None:
                winner = by_feas
                continue
            # Second: Try dominance both ways
            if self.dominates_epsilon(c, winner, eps):
                winner = c
            elif not self.dominates_epsilon(winner, c, eps):
                # neither dominates; defer to NSGA-II between them
                winner = self.better_by_rank_crowding(winner, c)
        return winner

    # --------- Master: build mating pool ---------

    def select_one(self, pop: List[Individual], method: str, **kwargs) -> Individual:
        method = method.lower()
        if method in ("cbt", "crowded_binary_tournament"):
            return self.crowded_binary_tournament(pop, **kwargs)
        if method in ("cbt_feas", "crowded_binary_tournament_feasible"):
            return self.crowded_binary_tournament_feasible(pop)
        if method in ("kt", "a-kt", "k_tournament", "adaptive_k_tournament"):
            return self.adaptive_k_tournament(pop, **kwargs)
        if method in ("ff", "feasibility_first", "feasibility_first_tournament"):
            return self.feasibility_first_tournament(pop, **kwargs)
        if method in ("rbs", "rank_roulette", "rank_based_roulette"):
            return self.rank_based_roulette(pop, **kwargs)
        if method in ("rbt", "reference_biased", "reference_biased_tournament"):
            return self.reference_biased_tournament(pop, **kwargs)
        if method in ("adt", "age_diversity", "age_diversity_tournament"):
            return self.age_diversity_tournament(pop, **kwargs)
        if method in (
            "edt",
            "epsilon_dt",
            "epsilon-dt",
            "epsilon_dominance_tournament",
        ):
            return self.epsilon_dominance_tournament(pop, **kwargs)
        raise ValueError(f"Unknown selection method: {method}")

    def select_mating_pool(
        self, pop: List[Individual], n_parents: int, method: str, **kwargs
    ) -> List[Individual]:
        """
        اگر method == 'sus' یا 'rbs_sus' از SUS برای n والد استفاده می‌شود.
        در غیر این صورت n بار select_one صدا زده می‌شود.
        """
        method_l = method.lower()
        if method_l in ("sus", "rbs_sus", "rank_sus", "rank_based_sus"):
            power = float(kwargs.get("power", 1.0))
            return self.rank_based_sus(pop, n=n_parents, power=power)
        return [self.select_one(pop, method=method, **kwargs) for _ in range(n_parents)]
