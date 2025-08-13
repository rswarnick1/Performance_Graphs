
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable
import math

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

@dataclass
class TaxonomyNode:
    name: str
    parent: Optional["TaxonomyNode"] = None
    children: List["TaxonomyNode"] = field(default_factory=list)

    # DAG influence weights
    pos_weight: float = 0.2   # enabling parents (+1)
    neg_weight: float = 0.2   # suppressing parents (-1)

    # Node prevalence P(C=Positive) used in PPV (precision) computation
    prevalence: float = 0.30

    # External DAG parents by name mapped to +1 (enable) / -1 (suppress)
    in_edges: Dict[str, int] = field(default_factory=dict)

    # Threshold lambda in [0,1] controlling the node's own ROC tradeoff
    lam: float = 0.5

    # --- structure methods ---
    def add_child(self, child: "TaxonomyNode") -> None:
        child.parent = self
        self.children.append(child)

    def walk(self) -> Iterable["TaxonomyNode"]:
        yield self
        for c in self.children:
            yield from c.walk()

    # --- Local base (independent) sensitivity & specificity from own threshold ---
    # Use a smooth, concave tradeoff curve: sensitivity ~ sqrt(1 - lam), specificity ~ sqrt(lam)
    def base_sensitivity(self, lam: float) -> float:
        return math.sqrt(clamp(1.0 - lam))

    def base_specificity(self, lam: float) -> float:
        return math.sqrt(clamp(lam))

    # --- Effective sensitivity/specificity with parental influence ---
    def effective_s_and_t(self, lam: float, neighbor_s: Dict[str, float], neighbor_t: Dict[str, float]) -> Tuple[float, float]:
        s0 = self.base_sensitivity(lam)
        t0 = self.base_specificity(lam)

        if not self.in_edges:
            return s0, t0

        pos_par_s = [neighbor_s[nm] for nm, sgn in self.in_edges.items() if sgn > 0 and nm in neighbor_s]
        pos_par_t = [neighbor_t[nm] for nm, sgn in self.in_edges.items() if sgn > 0 and nm in neighbor_t]
        neg_par_s = [neighbor_s[nm] for nm, sgn in self.in_edges.items() if sgn < 0 and nm in neighbor_s]
        neg_par_t = [neighbor_t[nm] for nm, sgn in self.in_edges.items() if sgn < 0 and nm in neighbor_t]

        # Means centered around 0.5 to create monotone shifts consistent with Theorem 2 / Prop 1
        def centered_mean(xs):
            if not xs: return 0.0
            return sum(x - 0.5 for x in xs) / len(xs)

        s = s0 + self.pos_weight * centered_mean(pos_par_s) - self.neg_weight * centered_mean(neg_par_s)
        t = t0 + self.pos_weight * centered_mean(pos_par_t) - self.neg_weight * centered_mean(neg_par_t)

        return clamp(s), clamp(t)

    # --- Derived performance metrics ---
    def precision_from_st(self, s: float, t: float) -> float:
        # PPV given sensitivity s, specificity t, prevalence pi
        pi = clamp(self.prevalence)
        denom = s*pi + (1.0 - t)*(1.0 - pi)
        if denom <= 0.0:
            return 0.0
        return (s*pi) / denom

    def recall_from_st(self, s: float, t: float) -> float:
        # Recall is sensitivity
        return s

    # Convenience wrappers used by the network
    def precision(self, lam: float, neighbor_s: Dict[str, float], neighbor_t: Dict[str, float]) -> float:
        s, t = self.effective_s_and_t(lam, neighbor_s, neighbor_t)
        return self.precision_from_st(s, t)

    def recall(self, lam: float, neighbor_s: Dict[str, float], neighbor_t: Dict[str, float]) -> float:
        s, _t = self.effective_s_and_t(lam, neighbor_s, neighbor_t)
        return self.recall_from_st(s, _t)
