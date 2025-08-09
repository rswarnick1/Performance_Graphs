
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable, Union
import math
import random

try:
    import networkx as nx  # optional
except Exception:
    nx = None

import numpy as np
import matplotlib.pyplot as plt

from TaxonomyNodeClass import TaxonomyNode


# ---- Minimal Graph fallback if networkx is unavailable ----
class _MiniDiGraph:
    def __init__(self) -> None:
        self.nodes: List[str] = []
        self.edges: List[Tuple[str, str]] = []

    def add_node(self, n: str) -> None:
        if n not in self.nodes:
            self.nodes.append(n)

    def add_edge(self, u: str, v: str) -> None:
        if u not in self.nodes:
            self.nodes.append(u)
        if v not in self.nodes:
            self.nodes.append(v)
        self.edges.append((u, v))

GraphType = nx.DiGraph if nx is not None else _MiniDiGraph


# ====== Base class for detection networks ======
@dataclass
class DetectionNetworkBase:
    thresholds: Dict[str, float] = field(default_factory=dict)
    node_lookup: Dict[str, TaxonomyNode] = field(default_factory=dict)

    # Abstract in spirit: subclass must implement this
    def _evaluate_performance(self, use_dag: bool = True, N: int = 40000) -> Dict[str, Dict[str, float]]:
        raise NotImplementedError

    def _node_loss(self, name: str, lam: float, *, use_dag: bool = True, N: int = 40000) -> float:
        """Quadratic loss (Precision + Recall) (maximization) for node 'name' at lambda=lam, holding others fixed."""
        orig = self.thresholds.get(name, 0.5)
        try:
            self.thresholds[name] = lam
            perf = self._evaluate_performance(use_dag=use_dag, N=N)
            stats = perf.get(name, {"precision": 0.0, "recall": 0.0})
            p = float(stats["precision"]); r = float(stats["recall"])
            return -(p + r) ** 2
        finally:
            self.thresholds[name] = orig

    def _best_lambda_for_node(self, name: str, *, steps: int = 50, use_dag: bool = True, N: int = 40000) -> float:
        best_lam, best_loss = 0.5, float("inf")
        for i in range(steps + 1):
            lam = i / steps
            loss = self._node_loss(name, lam, use_dag=use_dag, N=N)
            if loss < best_loss:
                best_loss, best_lam = loss, lam
        return float(best_lam)

    def equilibrium_search(
        self,
        max_iters: int = 50,
        tol: float = 1e-4,
        steps: int = 100,
        verbose: bool = False,
        use_dag: bool = True,
        N: int = 40000,
    ) -> Dict[str, float]:
        """Coordinate descent over nodes using the selected evaluator."""
        thresholds = self.thresholds.copy()
        for it in range(max_iters):
            delta = 0.0
            for name in list(self.node_lookup.keys()):
                cur = thresholds[name]
                # work with current thresholds
                self.thresholds = thresholds.copy()
                new = self._best_lambda_for_node(name, steps=steps, use_dag=use_dag, N=N)
                thresholds[name] = new
                delta = max(delta, abs(new - cur))
            if verbose:
                print(f"[iter {it}] max Δλ = {delta:.4g}")
            if delta < tol:
                break
        self.thresholds.update(thresholds)
        return thresholds


# ====== Taxonomy + DAG network ======
Edge2 = Tuple[str, str]
Edge3 = Tuple[str, str, str]  # sign in {'+','-'}

@dataclass
class TaxonomyBayesianNetwork(DetectionNetworkBase):
    """
    A lightweight network built on a taxonomy tree, with an overlaid signed DAG:
      - Positive edge (+): child may fire only where parent fired.
      - Negative edge (-): child may fire only where parent did NOT fire.
    """
    taxonomy_root: TaxonomyNode = field(default=None)
    graph: GraphType = field(default_factory=GraphType)
    dag_edges: List[Edge3] = field(default_factory=list)  # (parent, child, sign)

    def __post_init__(self) -> None:
        if self.taxonomy_root is None:
            raise ValueError("taxonomy_root must be provided")
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

    # ---- DAG support ----
    def set_dag_edges(self, edges: Iterable[Union[Edge2, Edge3]]) -> None:
        """Accepts (u,v) for positive edges or (u,v,sign) with sign in {'+','-','pos','neg','positive','negative'}."""
        norm: List[Edge3] = []
        for e in edges:
            if len(e) == 2:
                u, v = e  # type: ignore[misc]
                s = '+'
            else:
                u, v, s = e  # type: ignore[misc]
                s = {
                    '+': '+', 'pos': '+', 'positive': '+',
                    '-': '-', 'neg': '-', 'negative': '-'
                }.get(str(s).lower(), '+')
            norm.append((u, v, s))
        self.dag_edges = norm

    def _parents_map(self) -> Dict[str, Dict[str, List[str]]]:
        pm: Dict[str, Dict[str, List[str]]] = {name: {'+': [], '-': []} for name in self.node_lookup}
        for u, v, s in self.dag_edges:
            if v in pm:
                pm[v][s].append(u)
        return pm

    # ---- analytic per-node evaluation (ignores DAG) ----
    def run_performance_analysis(self, thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
        if thresholds is not None:
            bak = self.thresholds.copy()
            try:
                self.thresholds.update(thresholds)
                results = {
                    name: {"precision": self.node_lookup[name].compute_precision(self.thresholds[name]),
                           "recall":    self.node_lookup[name].compute_recall(self.thresholds[name])}
                    for name in self.node_lookup
                }
            finally:
                self.thresholds = bak
            return results
        else:
            return {
                name: {"precision": self.node_lookup[name].compute_precision(self.thresholds[name]),
                       "recall":    self.node_lookup[name].compute_recall(self.thresholds[name])}
                for name in self.node_lookup
            }

    # ---- DAG-aware Monte Carlo evaluation ----
    def run_performance_analysis_dag(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        N: int = 40000,
        seed: int = 1337,
    ) -> Dict[str, Dict[str, float]]:
        rng = np.random.default_rng(seed)
        if thresholds is not None:
            bak = self.thresholds.copy()
            self.thresholds.update(thresholds)

        try:
            names = list(self.node_lookup.keys())
            parents = self._parents_map()

            # independent base prevalences (demo)
            prevalence = {n: 0.25 for n in names}
            C = {n: (rng.random(N) < prevalence[n]) for n in names}

            # topological order from dag_edges
            indeg = {n: 0 for n in names}
            children = {n: [] for n in names}
            for u, v, _s in self.dag_edges:
                if v in indeg:
                    indeg[v] += 1
                if u in children:
                    children[u].append(v)
            from collections import deque
            q = deque([n for n in names if indeg[n] == 0])
            topo: List[str] = []
            while q:
                x = q.popleft()
                topo.append(x)
                for y in children.get(x, []):
                    indeg[y] -= 1
                    if indeg[y] == 0:
                        q.append(y)
            if len(topo) < len(names):
                topo = names  # disconnected or empty DAG -> arbitrary order

            # simulate conditioned detections
            D: Dict[str, np.ndarray] = {}
            parmap = parents
            for n in topo:
                node = self.node_lookup[n]
                lam = float(self.thresholds.get(n, 0.5))
                P = node.compute_precision(lam)
                R = node.compute_recall(lam)

                Dn = np.zeros(N, dtype=bool)
                pos_mask = C[n]
                Dn[pos_mask] = rng.random(pos_mask.sum()) < R
                neg_mask = ~pos_mask
                Dn[neg_mask] |= rng.random(neg_mask.sum()) < (1 - P)

                par = parmap.get(n, {'+': [], '-': []})
                pos_par, neg_par = par.get('+', []), par.get('-', [])
                if pos_par or neg_par:
                    mask = np.ones(N, dtype=bool)
                    for p in pos_par:
                        mask &= D[p]
                    for p in neg_par:
                        mask &= ~D[p]
                    Dn = Dn & mask

                D[n] = Dn

            # aggregate stats
            results: Dict[str, Dict[str, float]] = {}
            for n in names:
                TP = np.logical_and(D[n], C[n]).sum()
                FP = np.logical_and(D[n], ~C[n]).sum()
                FN = np.logical_and(~D[n], C[n]).sum()
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                results[n] = {"precision": float(precision), "recall": float(recall)}
            return results
        finally:
            if thresholds is not None:
                self.thresholds = bak

    # unified for optimizer
    def _evaluate_performance(self, use_dag: bool = True, N: int = 40000) -> Dict[str, Dict[str, float]]:
        if use_dag and self.dag_edges:
            return self.run_performance_analysis_dag(N=N)
        return self.run_performance_analysis()

    # ---- Visualization: signed-edge DAG ----
    def render_detection_dag(
        self,
        edges: Optional[Iterable[Union[Edge2, Edge3]]] = None,
        save_path: Optional[str] = None,
        title: str = "Detection DAG (signed)",
    ) -> None:
        """Positive edges solid; negative edges dashed. Simple layered layout by path depth."""
        # normalize edges
        if edges is None:
            edges_in: List[Union[Edge2, Edge3]] = self.dag_edges if self.dag_edges else list(self.taxonomy_root.edges())
        else:
            edges_in = list(edges)

        edges3: List[Edge3] = []
        for e in edges_in:
            if len(e) == 2:
                u, v = e  # type: ignore[misc]
                s = '+'
            else:
                u, v, s = e  # type: ignore[misc]
                s = {
                    '+': '+', 'pos': '+', 'positive': '+',
                    '-': '-', 'neg': '-', 'negative': '-'
                }.get(str(s).lower(), '+')
            edges3.append((u, v, s))

        nodes: List[str] = sorted(list({u for u, _, _ in edges3} | {v for _, v, _ in edges3}))

        # positions by taxonomy depth
        def depth(name: str) -> int:
            return max(0, name.count("/"))
        by_layer: Dict[int, List[str]] = {}
        for n in nodes:
            d = depth(n)
            by_layer.setdefault(d, []).append(n)
        max_layer = max(by_layer) if by_layer else 0
        pos: Dict[str, Tuple[float, float]] = {}
        for d in range(max_layer + 1):
            row = sorted(by_layer.get(d, []))
            k = len(row) or 1
            for i, n in enumerate(row):
                x = (i + 1) / (k + 1)
                y = 1.0 - (0.9 * d / (max_layer + 1 if max_layer + 1 else 1))
                pos[n] = (x, y)

        plt.figure(figsize=(10, 6))
        has_pos = has_neg = False
        for u, v, s in edges3:
            x1, y1 = pos[u]; x2, y2 = pos[v]
            if s == '+':
                has_pos = True
                plt.annotate("", xy=(x2, y2), xytext=(x1, y1),
                             arrowprops=dict(arrowstyle="->", lw=1.4))
            else:
                has_neg = True
                plt.annotate("", xy=(x2, y2), xytext=(x1, y1),
                             arrowprops=dict(arrowstyle="->", lw=1.4, linestyle="dashed"))

        for n in nodes:
            x, y = pos[n]
            plt.scatter([x], [y], s=600, edgecolors="#335", linewidths=1.0, c=["#DDEEFF"], zorder=3)
            plt.text(x, y, n, ha="center", va="center", fontsize=9)

        if has_pos or has_neg:
            from matplotlib.lines import Line2D
            handles = []
            if has_pos:
                handles.append(Line2D([0],[0], linestyle="-", marker="", label="positive"))
            if has_neg:
                handles.append(Line2D([0],[0], linestyle="--", marker="", label="negative"))
            plt.legend(handles=handles, loc="best")

        plt.axis("off")
        if title:
            plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


# ---- Convenient taxonomy construction for __main__ ----
def _make_demo_taxonomy() -> TaxonomyNode:
    root = TaxonomyNode("ROOT", detection_params={"sensitivity": 0.65, "specificity": 0.55, "scale": 7.0})
    a = TaxonomyNode("A", detection_params={"sensitivity": 0.6, "specificity": 0.7, "scale": 8.0})
    b = TaxonomyNode("B", detection_params={"sensitivity": 0.7, "specificity": 0.55, "scale": 8.0})
    c = TaxonomyNode("C", detection_params={"sensitivity": 0.55, "specificity": 0.65, "scale": 9.0})
    d = TaxonomyNode("D", detection_params={"sensitivity": 0.7, "specificity": 0.7, "scale": 10.0})
    e = TaxonomyNode("E", detection_params={"sensitivity": 0.5, "specificity": 0.8, "scale": 9.0})
    f = TaxonomyNode("F", detection_params={"sensitivity": 0.35, "specificity": 0.30, "scale": 2.0})
    g = TaxonomyNode("G", detection_params={"sensitivity": 0.275, "specificity": 0.35, "scale": 3.0})
    h = TaxonomyNode("H", detection_params={"sensitivity": 0.325, "specificity": 0.275, "scale": 4.0})
    i = TaxonomyNode("I", detection_params={"sensitivity": 0.35, "specificity": 0.35, "scale": 5.0})
    j = TaxonomyNode("J", detection_params={"sensitivity": 0.40, "specificity": 0.175, "scale": 6.0})

    root.add_child(a)
    root.add_child(b)
    a.add_child(c)
    a.add_child(f)
    b.add_child(d)
    c.add_child(e)
    c.add_child(g)
    root.add_child(h)
    root.add_child(i)
    h.add_child(j)
    i.add_child(j)

    return root


def _make_demo_dag_edges(root: TaxonomyNode) -> List[Edge3]:
    """Return signed DAG edges using the paper example plus one illustrative negative edge if nodes exist."""
    # Build short->full mapping
    name_to_full: Dict[str, str] = {}
    def walk(n: TaxonomyNode) -> None:
        full = "/".join(n.path())
        name_to_full[n.name] = full
        for c in n.children:
            walk(c)
    walk(root)

    pos_edges: List[Tuple[str, str]] = [("A","C"), ("A","E"), ("C","E"), ("B","D"), ("D","E")]
    edges_full: List[Edge3] = []
    for u, v in pos_edges:
        if u in name_to_full and v in name_to_full:
            edges_full.append((name_to_full[u], name_to_full[v], '+'))
    # Add one negative edge if both exist
    if "H" in name_to_full and "E" in name_to_full:
        edges_full.append((name_to_full["H"], name_to_full["E"], '-'))
    return edges_full


def _pretty_print_perf(perf: Dict[str, Dict[str, float]]) -> None:
    names = sorted(perf.keys())
    width = max(len(n) for n in names) if names else 4
    print("Node".ljust(width), " | Precision  | Recall")
    print("-" * (width + 26))
    for n in names:
        p = perf[n]["precision"]; r = perf[n]["recall"]
        print(n.ljust(width), f"| {p:9.3f} | {r:6.3f}")


if __name__ == "__main__":
    # Build demo taxonomy and DAG
    root = _make_demo_taxonomy()
    tbn = TaxonomyBayesianNetwork(taxonomy_root=root)
    dag_edges = _make_demo_dag_edges(root)
    tbn.set_dag_edges(dag_edges)

    print("DAG edges (full names):")
    for u, v, s in dag_edges:
        arrow = "->" if s == "+" else "-|>"
        print(f"  {u} {arrow} {v}  (sign={s})")

    # Initial performance (DAG-aware)
    print("\nInitial DAG-aware performance (using default thresholds):")
    perf0 = tbn.run_performance_analysis_dag(N=30000)
    _pretty_print_perf(perf0)

    # Per-node best λ via DAG-aware loss
    print("\nComputing per-node best λ (DAG-aware)...")
    best_thresholds: Dict[str, float] = {}
    for name in tbn.node_lookup:
        lam = tbn._best_lambda_for_node(name, steps=80, use_dag=True, N=30000)
        best_thresholds[name] = lam

    print("\nPer-node best λ:")
    for k in sorted(best_thresholds):
        print(f"  {k}: λ = {best_thresholds[k]:.3f}")

    print("\nPerformance at per-node best λ (DAG-aware):")
    perf_best = tbn.run_performance_analysis_dag(best_thresholds, N=40000)
    _pretty_print_perf(perf_best)

    # Nash-equilibrium-style search (coordinate descent) with DAG-aware loss
    print("\nSearching for equilibrium thresholds (coordinate descent, DAG-aware)...")
    tbn.thresholds.update(best_thresholds)  # start from per-node bests
    res_thresholds = tbn.equilibrium_search(verbose=True, steps=80, use_dag=True, N=30000)
    print("\nResulting thresholds (equilibrium):")
    for k in sorted(res_thresholds):
        print(f"  {k}: λ = {res_thresholds[k]:.3f}")

    print("\nPerformance at equilibrium (DAG-aware):")
    perf_eq = tbn.run_performance_analysis_dag(res_thresholds, N=40000)
    _pretty_print_perf(perf_eq)

    # Save a signed-edge DAG visualization
    try:
        tbn.render_detection_dag(edges=dag_edges, save_path="dag_signed.png", title="Detection DAG (signed edges)")
        print('\nSaved DAG figure with signed edges to dag_signed.png')
    except Exception as e:
        print("Failed to render DAG figure:", e)
