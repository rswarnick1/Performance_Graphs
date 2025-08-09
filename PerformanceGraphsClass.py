from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random

try:
    import networkx as nx
except Exception:
    nx = None  # We'll handle a minimal fallback graph below

from TaxonomyNodeClass import TaxonomyNode


# ---- Minimal Graph fallback if networkx is unavailable ----
class _MiniDiGraph:
    def __init__(self):
        self._nodes = set()
        self._edges = set()
        self._attrs = {}

    def add_node(self, n, **attrs):
        self._nodes.add(n)
        self._attrs.setdefault(n, {}).update(attrs)

    def add_edge(self, u, v, **attrs):
        self._edges.add((u, v))
        # store attrs on (u,v) if needed
        self._attrs.setdefault((u, v), {}).update(attrs)

    @property
    def nodes(self):
        return list(self._nodes)

    @property
    def edges(self):
        return list(self._edges)

    def successors(self, n):
        return [v for (u, v) in self._edges if u == n]

    def predecessors(self, n):
        return [u for (u, v) in self._edges if v == n]


GraphType = _MiniDiGraph if nx is None else nx.DiGraph


@dataclass
class TaxonomyBayesianNetwork:
    """
    Very lightweight "Bayesian network"-like wrapper around a taxonomy tree.

    - Builds a directed graph that follows taxonomy edges (parent -> child).
    - Maintains a threshold per taxonomy node.
    - Provides simple precision/recall calculators via TaxonomyNode models.
    - Includes a coordinate-descent "equilibrium" search that minimizes the
      sum over nodes of (Precision - Recall)^2 (cf. paper's balancing idea).
    """
    taxonomy_root: TaxonomyNode
    graph: GraphType = field(default_factory=GraphType)
    thresholds: Dict[str, float] = field(default_factory=dict)  # node name -> λ in [0,1]
    node_lookup: Dict[str, TaxonomyNode] = field(default_factory=dict)

    def __post_init__(self):
        self._map_taxonomy_to_nodes(self.taxonomy_root)

    # ---- construction ----
    def _map_taxonomy_to_nodes(self, root: TaxonomyNode) -> None:
        def dfs(node: TaxonomyNode):
            name = "/".join(node.path())
            self.node_lookup[name] = node
            if hasattr(self.graph, "add_node"):
                self.graph.add_node(name)
            self.thresholds.setdefault(name, 0.5)
            for child in node.children:
                cname = "/".join(child.path())
                if hasattr(self.graph, "add_node"):
                    self.graph.add_node(cname)
                if hasattr(self.graph, "add_edge"):
                    self.graph.add_edge(name, cname)
                dfs(child)
        dfs(root)

    # ---- metrics ----
    def node_precision_recall(self, name: str, lam: Optional[float] = None) -> Tuple[float, float]:
        node = self.node_lookup[name]
        lam = self.thresholds[name] if lam is None else lam
        prec = node.compute_precision(lam)
        rec = node.compute_recall(lam)
        return prec, rec

    def run_performance_analysis(self, thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
        """
        Returns {node_name: {"precision": p, "recall": r}}
        """
        if thresholds:
            # copy into self.thresholds but don't mutate original permanently
            tmp = self.thresholds.copy()
            try:
                for k, v in thresholds.items():
                    if k in tmp:
                        tmp[k] = float(max(0.0, min(1.0, v)))
                results = {}
                for name in self.node_lookup:
                    lam = tmp[name]
                    p, r = self.node_precision_recall(name, lam)
                    results[name] = {"precision": p, "recall": r}
                return results
            finally:
                pass

        # use current thresholds
        results = {}
        for name in self.node_lookup:
            p, r = self.node_precision_recall(name, self.thresholds[name])
            results[name] = {"precision": p, "recall": r}
        return results

    # ---- equilibrium search ----
    def _node_loss(self, name: str, lam: float) -> float:
        p, r = self.node_precision_recall(name, lam)
        return (p - r) ** 2

    def _best_lambda_for_node(self, name: str, *, steps: int = 50) -> float:
        """
        Brute-force line search over [0,1] to minimize (precision - recall)^2 for this node,
        holding all other nodes fixed (they don't interact in our toy model).
        """
        best_lam, best_loss = None, float("inf")
        for i in range(steps + 1):
            lam = i / steps
            loss = self._node_loss(name, lam)
            if loss < best_loss:
                best_loss, best_lam = loss, lam
        return float(best_lam)

    def equilibrium_search(self, max_iters: int = 50, tol: float = 1e-4, steps: int = 100, verbose: bool = False) -> Dict[str, float]:
        """
        Simple coordinate descent: repeatedly optimize each node's λ to equalize precision/recall.
        """
        thresholds = self.thresholds.copy()
        for it in range(max_iters):
            delta = 0.0
            for name in self.node_lookup:
                cur = thresholds[name]
                new = self._best_lambda_for_node(name, steps=steps)
                thresholds[name] = new
                delta = max(delta, abs(new - cur))
            if verbose:
                print(f"[iter {it}] max Δλ = {delta:.4g}")
            if delta < tol:
                break
        # commit
        self.thresholds.update(thresholds)
        return thresholds


# ---- Convenient taxonomy construction for __main__ ----
def _make_demo_taxonomy() -> TaxonomyNode:
    root = TaxonomyNode("ROOT", detection_params={"sensitivity": 0.65, "specificity": 0.55, "scale": 7.0})
    a = TaxonomyNode("A", detection_params={"sensitivity": 0.6, "specificity": 0.7, "scale": 8.0})
    b = TaxonomyNode("B", detection_params={"sensitivity": 0.7, "specificity": 0.55, "scale": 8.0})
    c = TaxonomyNode("C", detection_params={"sensitivity": 0.55, "specificity": 0.65, "scale": 9.0})
    d = TaxonomyNode("D", detection_params={"sensitivity": 0.7, "specificity": 0.7, "scale": 10.0})
    e = TaxonomyNode("E", detection_params={"sensitivity": 0.5, "specificity": 0.8, "scale": 9.0})

    root.add_child(a)
    root.add_child(b)
    a.add_child(c)
    b.add_child(d)
    c.add_child(e)
    return root


def _pretty_print_perf(perf: Dict[str, Dict[str, float]]) -> None:
    names = sorted(perf.keys())
    width = max(len(n) for n in names)
    print("Node".ljust(width), " | Precision  | Recall")
    print("-" * (width + 26))
    for n in names:
        p = perf[n]["precision"]
        r = perf[n]["recall"]
        print(n.ljust(width), f"| {p:9.3f} | {r:6.3f}")


if __name__ == "__main__":
    # Build a small demo taxonomy and run the analysis
    root = _make_demo_taxonomy()
    tbn = TaxonomyBayesianNetwork(root)

    print("Running performance analysis (initial thresholds):")
    perf = tbn.run_performance_analysis()
    _pretty_print_perf(perf)

    print("\nSearching for equilibrium thresholds (coordinate descent)...")
    res_thresholds = tbn.equilibrium_search(verbose=True)
    print("\nResulting thresholds:")
    for k in sorted(res_thresholds):
        print(f"  {k}: λ = {res_thresholds[k]:.3f}")

    print("\nPerformance at equilibrium:")
    perf2 = tbn.run_performance_analysis(res_thresholds)
    _pretty_print_perf(perf2)
