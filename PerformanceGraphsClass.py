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
    f = TaxonomyNode("F", detection_params={"sensitivity": .35, "specificity": 0.3, "scale": 2.0})
    g = TaxonomyNode("G", detection_params={"sensitivity": 0.275, "specificity": 0.35, "scale": 3.0})
    h = TaxonomyNode("H", detection_params={"sensitivity": 0.325, "specificity": 0.275, "scale": 4.0})
    i = TaxonomyNode("I", detection_params={"sensitivity": 0.35, "specificity": 0.35, "scale": 5.0})
    j = TaxonomyNode("J", detection_params={"sensitivity": 0.4, "specificity": 0.175, "scale": 6.0})
    k = TaxonomyNode("K", detection_params={"sensitivity": 0.89, "specificity": 0.11, "scale": 5.0})

    root.add_child(a)
    root.add_child(b)
    a.add_child(c)
    a.add_child(f)
    b.add_child(d)
    b.add_child(e)
    c.add_child(k)
    c.add_child(g)
    root.add_child(h)
    h.add_child(i)
    h.add_child(j)


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


# ===== Rendering utilities (dependency-light) =====
from typing import Iterable, Optional

try:
    import matplotlib.pyplot as plt
except Exception as _e:
    plt = None  # Headless environments may lack display; saving to file still works if backends are present.

def _hierarchy_pos(edges: Iterable[tuple]) -> dict:
    """Simple layered layout for a tree/DAG without networkx layouts."""
    children = {}
    indeg = {}
    nodes = set()
    for u, v in edges:
        nodes.add(u); nodes.add(v)
        children.setdefault(u, []).append(v)
        indeg[v] = indeg.get(v, 0) + 1
        indeg.setdefault(u, indeg.get(u, 0))

    roots = [n for n in nodes if indeg.get(n, 0) == 0] or (sorted(nodes)[:1] if nodes else [])
    from collections import deque, defaultdict
    depth = {}
    q = deque()
    for r in roots:
        depth[r] = 0
        q.append(r)
    while q:
        x = q.popleft()
        for y in children.get(x, []):
            if y not in depth:
                depth[y] = depth[x] + 1
                q.append(y)
    by_layer = {}
    for n in nodes:
        by_layer.setdefault(depth.get(n, 0), []).append(n)
    pos = {}
    max_layer = max(by_layer) if by_layer else 0
    for d in range(0, max_layer + 1):
        row = sorted(by_layer.get(d, []))
        k = len(row) or 1
        for i, n in enumerate(row):
            x = (i + 1) / (k + 1)
            y = 1.0 - (0.9 * d / (max_layer + 1 if max_layer + 1 else 1))
            pos[n] = (x, y)
    return pos

def _maybe_graphviz_layout(G):
    if nx is None:
        return None
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        return graphviz_layout(G, prog="dot")
    except Exception:
        try:
            from networkx.drawing.nx_pydot import graphviz_layout
            return graphviz_layout(G, prog="dot")
        except Exception:
            return None

def _draw_graph(edges, node_labels=None, node_colors=None, figsize=(11, 7), title=None, save_path=None):
    edges = list(edges)
    nodes = set([u for u, _ in edges] + [v for _, v in edges])

    # Build positions
    if nx is not None:
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        pos = _maybe_graphviz_layout(G) or _hierarchy_pos(edges)
    else:
        pos = _hierarchy_pos(edges)

    if plt is None:
        raise RuntimeError("matplotlib is required to render graphs")

    import matplotlib.pyplot as _plt  # ensure backend is initialized
    _plt.figure(figsize=figsize)
    for (u, v) in edges:
        x1, y1 = pos[u]; x2, y2 = pos[v]
        _plt.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2))
    for n in nodes:
        x, y = pos[n]
        color = (node_colors or {}).get(n, "#DDEEFF")
        _plt.scatter([x], [y], s=600, edgecolors="#335", linewidths=1.0, c=[color], zorder=3)
        label = (node_labels or {}).get(n, n)
        _plt.text(x, y, label, ha="center", va="center", fontsize=10, zorder=4)
    _plt.axis("off")
    if title:
        _plt.title(title)
    _plt.tight_layout()
    if save_path:
        _plt.savefig(save_path, dpi=150, bbox_inches="tight")
        _plt.close()
    else:
        _plt.show()

# ----- Instance methods (bound after definition) -----
def _tbn_render_taxonomy_tree(self, save_path: Optional[str] = None, annotate_thresholds: bool = True, figsize=(11,7)):
    root = getattr(self, "taxonomy_root", None)
    if root is None:
        raise ValueError("taxonomy_root is not set on this instance.")
    # Collect edges and nodes
    try:
        edges = list(root.edges())
        names = [n.name for n in root.walk()]
    except Exception:
        # Fallback: rebuild from node_lookup paths
        names = list(self.node_lookup.keys())
        edges = []
        for name in names:
            parts = name.split("/")
            if len(parts) > 1:
                parent = "/".join(parts[:-1])
                edges.append((parent, name))
    node_labels = {}
    thresholds = getattr(self, "thresholds", None) if annotate_thresholds else None
    for name in names:
        if thresholds and name in thresholds:
            node_labels[name] = f"{name}\nλ={thresholds[name]:.2f}"
        else:
            node_labels[name] = name
    _draw_graph(edges=edges, node_labels=node_labels, figsize=figsize, title="Taxonomy Tree", save_path=save_path)

def _tbn_render_detection_dag(self, edges=None, save_path: Optional[str] = None, figsize=(11,7), title: str = "Detection Dependency DAG"):
    if edges is None:
        # default to taxonomy edges
        try:
            edges = list(self.taxonomy_root.edges())
        except Exception:
            # rebuild from node_lookup path relations
            edges = []
            for name in self.node_lookup.keys():
                parts = name.split("/")
                if len(parts) > 1:
                    parent = "/".join(parts[:-1])
                    edges.append((parent, name))
    node_colors = {}
    names_in_model = set(getattr(self, "node_lookup", {}).keys() or [])
    for (u, v) in edges:
        if u in names_in_model:
            node_colors[u] = "#EAF7E6"
        if v in names_in_model:
            node_colors[v] = "#EAF7E6"
    _draw_graph(edges=edges, node_labels=None, node_colors=node_colors, figsize=figsize, title=title, save_path=save_path)

# bind
TaxonomyBayesianNetwork.render_taxonomy_tree = _tbn_render_taxonomy_tree
TaxonomyBayesianNetwork.render_detection_dag = _tbn_render_detection_dag
