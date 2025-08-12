
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable, Set
import math
import random
from pathlib import Path
import collections

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from TaxonomyNodeClass import TaxonomyNode, clamp

@dataclass
class TaxonomyBayesianNetwork:
    def _dag_children_map(self) -> Dict[str, List[str]]:
        children = {}
        names = [n.name for n in self.nodes()]
        for nm in names:
            children[nm] = []
        for child, parents in self.dag_parents.items():
            for p in parents.keys():
                if p in children:
                    children[p].append(child)
        return children

    root: TaxonomyNode
    dag_parents: Dict[str, Dict[str, int]]  # child -> {parent_name: +1/-1}
    a: float = 1.0
    b: float = 1.0
    gamma: float = 0.95

    def nodes(self) -> List[TaxonomyNode]:
        return list(self.root.walk())

    def name_to_node(self) -> Dict[str, TaxonomyNode]:
        return {n.name: n for n in self.nodes()}

    # Neighbor maps for (sensitivity, specificity) at current lambdas
    def neighbor_st(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        s_map, t_map = {}, {}
        for n in self.nodes():
            s, t = n.effective_s_and_t(n.lam, s_map, t_map)  # note: order doesn't matter if we use current stored maps?
            # To avoid order effects, compute using current neighbor values from a snapshot computed in two passes.
        # So compute in two steps: first base s/t, then apply parents:
        base_s = {n.name: n.base_sensitivity(n.lam) for n in self.nodes()}
        base_t = {n.name: n.base_specificity(n.lam) for n in self.nodes()}
        s_map, t_map = {}, {}
        for n in self.nodes():
            s, t = n.effective_s_and_t(n.lam, base_s, base_t)
            s_map[n.name] = s
            t_map[n.name] = t
        return s_map, t_map

    def objective(self, node: TaxonomyNode, lam: float) -> float:
        # Snapshot base s/t for current lambdas
        base_s = {n.name: n.base_sensitivity(n.lam) for n in self.nodes()}
        base_t = {n.name: n.base_specificity(n.lam) for n in self.nodes()}
        # Node's own metrics under candidate lam
        p = node.precision(lam, base_s, base_t)
        r = node.recall(lam, base_s, base_t)
        # Bottom-up: children contribution evaluated with parent's base replaced by candidate lam
        children_map = self._dag_children_map()
        child_bonus = 0.0
        ch_names = children_map.get(node.name, [])
        if ch_names:
            base_s2 = dict(base_s); base_t2 = dict(base_t)
            base_s2[node.name] = node.base_sensitivity(lam)
            base_t2[node.name] = node.base_specificity(lam)
            vals = []
            for chn in ch_names:
                ch = self.name_to_node()[chn]
                sc, tc = ch.effective_s_and_t(ch.lam, base_s2, base_t2)
                vals.append(0.5*sc + 0.5*tc)
            child_bonus = float(sum(vals)/len(vals))
        return self.a * p + self.b * r + self.gamma * child_bonus

    # single-node maximization over lam in [0,1] for quasi-concave function
    def best_response(self, node: TaxonomyNode, grid: int = 401) -> float:
        best_lam = node.lam
        best_val = -1.0
        for i in range(grid):
            lam = i/(grid-1)
            val = self.objective(node, lam)
            if val > best_val:
                best_val = val
                best_lam = lam
        return best_lam


    def best_response_against_maps(self, node: TaxonomyNode, base_s: Dict[str, float], base_t: Dict[str, float], grid: int = 401) -> float:
        """Maximize a*P+b*R for this node against FROZEN neighbor sensitivity/specificity maps."""
        best_lam = node.lam
        best_val = -1.0
        for i in range(grid):
            lam = i/(grid-1)
            val = self.a*node.precision(lam, base_s, base_t) + self.b*node.recall(lam, base_s, base_t)
            if val > best_val:
                best_val = val
                best_lam = lam
        return best_lam

    def coordinate_ascent(self, max_iter: int = 200, tol: float = 1e-4) -> None:
        nodes = self.nodes()
        for n in nodes:
            n.lam = random.uniform(0.2, 0.8)
        last_obj = -1.0
        for _ in range(max_iter):
            improved = False
            for n in nodes:
                new_lam = self.best_response(n)
                if abs(new_lam - n.lam) > tol:
                    n.lam = new_lam
                    improved = True
            cur = self.total_objective()
            if not improved or abs(cur - last_obj) <= tol:
                break
            last_obj = cur

    
    # ---------- DAG validation ----------
    def _validate_and_toposort(self) -> List[List[str]]:
        """Validate dag_parents is a directed acyclic graph and return topo levels."""
        name_set: Set[str] = set(n.name for n in self.nodes())

        import collections
        parents = {child: set(pmap.keys()) for child, pmap in self.dag_parents.items()}
        children = collections.defaultdict(set)
        for child, ps in parents.items():
            if child not in name_set:
                raise ValueError(f"DAG child '{child}' not in taxonomy")
            for p in ps:
                if p not in name_set:
                    raise ValueError(f"DAG parent '{p}' not in taxonomy")
                if p == child:
                    raise ValueError(f"Self-loop at '{p}' not allowed in DAG")
                children[p].add(child)
        for n in name_set:
            parents.setdefault(n, set())
            children.setdefault(n, set())

        indeg = {n: len(parents[n]) for n in name_set}
        frontier = [n for n in name_set if indeg[n] == 0]
        levels: List[List[str]] = []
        visited = 0
        while frontier:
            level = sorted(frontier)
            levels.append(level)
            next_frontier = []
            for u in level:
                visited += 1
                for v in children[u]:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        next_frontier.append(v)
            frontier = next_frontier
        if visited != len(name_set):
            raise ValueError("DAG has at least one cycle; topological sort failed")
        return levels

    def staged_local_optimization_dag(self) -> None:
        """Single topological sweep: optimize parents first, then children exactly once."""
        for n in self.nodes():
            n.lam = random.uniform(0.2, 0.8)
        levels_names = self._validate_and_toposort()
        name_map = self.name_to_node()
        for lvl_names in levels_names:
            base_s = {n.name: n.base_sensitivity(n.lam) for n in self.nodes()}
            base_t = {n.name: n.base_specificity(n.lam) for n in self.nodes()}
            for nm in lvl_names:
                node = name_map[nm]
                node.lam = self.best_response_against_maps(node, base_s, base_t)
    def forward_backward_until_converged_dag(self, max_iter: int =1000, tol: float = 1e-6) -> None:
        """Forward-then-backward Gauss–Seidel sweeps through the DAG until convergence.
        Parents are NOT frozen; each update uses the latest neighbor values.
        """
        # initialize like the other routines for fairness
        for n in self.nodes():
            n.lam = random.uniform(0.2, 0.8)
        levels_names = self._validate_and_toposort()
        name_map = self.name_to_node()
        last_obj = -1.0
        for _ in range(max_iter):
            improved = False
            # forward sweep
            for lvl_names in levels_names:
                for nm in lvl_names:
                    node = name_map[nm]
                    new_lam = self.best_response(node)
                    if abs(new_lam - node.lam) > tol:
                        node.lam = new_lam
                        improved = True
            # backward sweep
            for lvl_names in reversed(levels_names):
                for nm in reversed(lvl_names):
                    node = name_map[nm]
                    new_lam = self.best_response(node)
                    if abs(new_lam - node.lam) > tol:
                        node.lam = new_lam
                        improved = True
            cur = self.total_objective()
            if not improved or abs(cur - last_obj) <= tol:
                break
            last_obj = cur
    
    def total_objective(self) -> float:
        # Compute a consistent snapshot-based total objective including child bonuses.
        base_s = {n.name: n.base_sensitivity(n.lam) for n in self.nodes()}
        base_t = {n.name: n.base_specificity(n.lam) for n in self.nodes()}
        children_map = self._dag_children_map()
        tot = 0.0
        for n in self.nodes():
            p = n.precision(n.lam, base_s, base_t)
            r = n.recall(n.lam, base_s, base_t)
            ch_names = children_map.get(n.name, [])
            child_bonus = 0.0
            if ch_names:
                base_s2 = dict(base_s); base_t2 = dict(base_t)
                base_s2[n.name] = n.base_sensitivity(n.lam)
                base_t2[n.name] = n.base_specificity(n.lam)
                vals = []
                for chn in ch_names:
                    ch = self.name_to_node()[chn]
                    sc, tc = ch.effective_s_and_t(ch.lam, base_s2, base_t2)
                    vals.append(0.5*sc + 0.5*tc)
                child_bonus = float(sum(vals)/len(vals))
            tot += self.a*p + self.b*r + self.gamma*child_bonus
        return tot
    


def build_demo_taxonomy() -> TaxonomyNode:
    # Build 11 nodes (root + 10)
    root = TaxonomyNode("ROOT")

    # First layer (3)
    a = TaxonomyNode("A")
    b = TaxonomyNode("B")
    c = TaxonomyNode("C")
    root.add_child(a); root.add_child(b); root.add_child(c)

    # Second layer (3 under A,B,C)
    d = TaxonomyNode("D"); e = TaxonomyNode("E")
    f = TaxonomyNode("F")
    a.add_child(d); a.add_child(e)
    b.add_child(f)

    # Third layer (4) to reach 11 total nodes
    g = TaxonomyNode("G"); h = TaxonomyNode("H"); i = TaxonomyNode("I"); j = TaxonomyNode("J")
    c.add_child(g)
    d.add_child(h)
    e.add_child(i)
    f.add_child(j)

    return root

def build_demo_dag() -> Dict[str, Dict[str, int]]:
    """
    Return signed DAG with exactly 10 edges across the 11 taxonomy nodes.
    Format: child -> {parent_name: +1/-1}
    """
    dag = {
        # Expanded connections (maintains acyclicity)
        "D": {"B": +1, "A": +1},    # B and A enable D
        "E": {"C": -1},             # C suppresses E
        "F": {"A": +1, "B": -1},    # A enables F; B suppresses F
        "G": {"A": +1},             # A enables G
        "H": {"D": +1},             # D enables H
        "I": {"D": +1},             # D enables I
        "J": {"C": +1, "F": +1},    # C and F enable J
    }
    # Validate count is 10 edges
    assert sum(len(pars) for pars in dag.values()) == 10, "DAG must have exactly 10 edges"
    return dag

def register_dag_on_nodes(root: TaxonomyNode, dag: Dict[str, Dict[str, int]]) -> None:
    name_map = {n.name: n for n in root.walk()}
    for child_name, parents in dag.items():
        if child_name in name_map:
            name_map[child_name].in_edges = dict(parents)

# ---------- Visualization ----------

def _assign_tree_positions(root: TaxonomyNode) -> Dict[str, Tuple[float, float]]:
    levels = {}
    def dfs(node: TaxonomyNode, depth: int):
        levels.setdefault(depth, []).append(node)
        for c in node.children:
            dfs(c, depth+1)
    dfs(root, 0)
    pos = {}
    max_depth = max(levels.keys())
    for d in range(max_depth+1):
        nodes = levels[d]
        n = len(nodes)
        for i, node in enumerate(nodes):
            x = (i+1)/(n+1)
            y = 1.0 - (d/(max_depth if max_depth>0 else 1))
            pos[node.name] = (x, y)
    return pos

def draw_taxonomy_and_dag(root: TaxonomyNode, dag: Dict[str, Dict[str, int]], savepath: str) -> None:
    pos = _assign_tree_positions(root)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.set_title("Taxonomy (solid) with DAG overlays (arrows; dashed '+', dotted '-')")

    # taxonomy edges
    for node in root.walk():
        for child in node.children:
            x0, y0 = pos[node.name]
            x1, y1 = pos[child.name]
            ax.plot([x0, x1], [y0, y1], linestyle='solid')
    # nodes
    for name, (x, y) in pos.items():
        ax.plot([x], [y], marker='o')
        ax.text(x, y+0.03, name, ha='center', va='bottom', fontsize=9)

    # dag overlays with arrows
    for child, parents in dag.items():
        for parent, sign in parents.items():
            if parent in pos and child in pos:
                x0, y0 = pos[parent]
                x1, y1 = pos[child]
                ls = '--' if sign>0 else 'dotted'
                ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle='->', linestyle=ls))
                xm, ym = (x0+x1)/2, (y0+y1)/2 + 0.05
                ax.text(xm, ym, '+' if sign>0 else '-', ha='center', va='bottom', fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(savepath, dpi=160)
    plt.close(fig)

# ---------- Tables ----------

def rows_staged_dag(net: TaxonomyBayesianNetwork) -> List[Tuple[str, float, float, float]]:
    net.staged_local_optimization_dag()
    base_s = {n.name: n.base_sensitivity(n.lam) for n in net.nodes()}
    base_t = {n.name: n.base_specificity(n.lam) for n in net.nodes()}
    rows = []
    for n in net.nodes():
        p = n.precision(n.lam, base_s, base_t)
        r = n.recall(n.lam, base_s, base_t)
        rows.append((n.name, n.lam, p, r))
    return rows

def equilibrium_rows(net: TaxonomyBayesianNetwork) -> List[Tuple[str, float, float, float]]:
    net.coordinate_ascent()
    base_s = {n.name: n.base_sensitivity(n.lam) for n in net.nodes()}
    base_t = {n.name: n.base_specificity(n.lam) for n in net.nodes()}
    rows = []
    for n in net.nodes():
        p = n.precision(n.lam, base_s, base_t)
        r = n.recall(n.lam, base_s, base_t)
        rows.append((n.name, n.lam, p, r))
    return rows

def forward_backward_rows(net: TaxonomyBayesianNetwork) -> List[Tuple[str, float, float, float]]:
    net.forward_backward_until_converged_dag()
    base_s = {n.name: n.base_sensitivity(n.lam) for n in net.nodes()}
    base_t = {n.name: n.base_specificity(n.lam) for n in net.nodes()}
    rows = []
    for n in net.nodes():
        p = n.precision(n.lam, base_s, base_t)
        r = n.recall(n.lam, base_s, base_t)
        rows.append((n.name, n.lam, p, r))
    return rows


def format_rows(rows: List[Tuple[str,float,float,float]], title: str) -> str:
    name_w = max(len(r[0]) for r in rows) if rows else 4
    hdr = f"{title}\n{'Name'.ljust(name_w)}  {'λ':>6}  {'Precision':>10}  {'Recall':>8}  {'aP+bR':>8}"
    sep = "-" * len(hdr)
    out = [hdr, sep]
    for name, lam, p, r in rows:
        val = 0.5*p + 0.5*r
        out.append(f"{name.ljust(name_w)}  {lam:6.3f}  {p:10.3f}  {r:8.3f}  {val:8.3f}")
    return "\n".join(out)

def main():
    num_iters = 30
    precision_diff = np.zeros((num_iters,11))
    recall_diff = np.zeros((num_iters,11))
    utility_diff = np.zeros((num_iters,11))
    precision_diff_fb = np.zeros((num_iters,11))
    recall_diff_fb = np.zeros((num_iters,11))
    utility_diff_fb = np.zeros((num_iters,11))
    for i in range(num_iters):
        random.seed(i)
        np.random.seed(i)
        # Build and register
        root = build_demo_taxonomy()
        dag = build_demo_dag()
        register_dag_on_nodes(root, dag)
        net = TaxonomyBayesianNetwork(root, dag, a=0.4, b=0.6, gamma=2*stats.beta.rvs(1,1,1))

        # Validate DAG (directed, acyclic) and get topo levels
        levels = net._validate_and_toposort()
        print("Topological levels (parents -> children):", levels)
        print(f"Taxonomy nodes: {sum(1 for _ in root.walk())} (expected 11)")
        print(f"DAG edges: {sum(len(pars) for pars in dag.values())} (expected 10)")

        # --- Staged single-sweep with FROZEN parents ---
        net.staged_local_optimization_dag()
        base_s = {n.name: n.base_sensitivity(n.lam) for n in net.nodes()}
        base_t = {n.name: n.base_specificity(n.lam) for n in net.nodes()}
        staged_rows = []
        for n in net.nodes():
            p = n.precision(n.lam, base_s, base_t)
            r = n.recall(n.lam, base_s, base_t)
            staged_rows.append((n.name, n.lam, p, r))
        print() 
        print(format_rows(staged_rows, "STAGED (Single Sweep; Parents Frozen per Level)"))
        staged_obj = net.total_objective()
        print(f"\nTotal utility (staged snapshot): {staged_obj:.4f}")

        # --- Reference equilibrium (full coordinate ascent) ---
        net.coordinate_ascent(max_iter=400, tol=1e-6)
        base_s = {n.name: n.base_sensitivity(n.lam) for n in net.nodes()}
        base_t = {n.name: n.base_specificity(n.lam) for n in net.nodes()}
        eq_rows = []
        for n in net.nodes():
            p = n.precision(n.lam, base_s, base_t)
            r = n.recall(n.lam, base_s, base_t)
            eq_rows.append((n.name, n.lam, p, r))
        print()
        print(format_rows(eq_rows, "NASH-STYLE EQUILIBRIUM (Reference)"))
        eq_obj = net.total_objective()
        print(f"\nTotal utility (equilibrium): {eq_obj:.4f}\nDelta vs staged: {eq_obj - staged_obj:+.4f}")

        precision_diff[i, :] = np.array(eq_rows)[:,1].astype(float) - np.array(staged_rows)[:,1].astype(float)
        recall_diff[i, :] = np.array(eq_rows)[:,2].astype(float) - np.array(staged_rows)[:,2].astype(float)

        utility_diff[i, :] = np.array(eq_rows)[:,3].astype(float) - np.array(staged_rows)[:,3].astype(float)
        # --- Forward-Backward non-frozen sweeps until convergence ---
        fb_rows = forward_backward_rows(net)
        precision_diff_fb[i, :] = np.array(eq_rows)[:,1].astype(float)-np.array(fb_rows)[:,1].astype(float) 
        recall_diff_fb[i, :] = np.array(eq_rows)[:,2].astype(float) - np.array(fb_rows)[:,2].astype(float) 
        utility_diff_fb[i, :] =  np.array(eq_rows)[:,3].astype(float) - np.array(fb_rows)[:,3].astype(float)
    # Figure
    fig_path = str(Path(__file__).with_name("taxonomy_dag.png"))
    draw_taxonomy_and_dag(root, dag, fig_path)
    print(f"\nSaved taxonomy/DAG figure to: {fig_path}")

    # plot a boxplot showing the distribution of the differences
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].boxplot(precision_diff)
    axs[0].set_title("Precision Difference (Equilibrium - Staged)")
    axs[0].set_xlabel("Node Index")
    axs[0].set_ylabel("Difference")
    axs[1].boxplot(recall_diff)
    axs[1].set_title("Recall Difference (Equilibrium - Staged)")
    axs[1].set_xlabel("Node Index")
    axs[1].set_ylabel("Difference")
    axs[2].boxplot(utility_diff)
    axs[2].set_title("Utility Difference (Equilibrium - Staged)")
    axs[2].set_xlabel("Node Index")
    axs[2].set_ylabel("Difference")
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name("taxonomy_dag_differences.png"), dpi=160)
    # Additional boxplots: Forward-Backward vs Equilibrium
    fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))
    axs2[0].boxplot(precision_diff_fb)
    axs2[0].set_title("Precision Difference (Equilibrium - Fwd-Bwd)")
    axs2[0].set_xlabel("Node Index"); axs2[0].set_ylabel("Difference")
    axs2[1].boxplot(recall_diff_fb)
    axs2[1].set_title("Recall Difference (Equilibrium - Fwd-Bwd)")
    axs2[1].set_xlabel("Node Index"); axs2[1].set_ylabel("Difference")
    axs2[2].boxplot(utility_diff_fb)
    axs2[2].set_title("Utility Difference (Equilibrium - Fwd-Bwd)")
    axs2[2].set_xlabel("Node Index"); axs2[2].set_ylabel("Difference")
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name("taxonomy_dag_differences_forward_backward.png"), dpi=160)
    print(f"Saved forward-backward differences boxplot to: {Path(__file__).with_name('taxonomy_dag_differences_forward_backward.png')}")
    
    print(f"\nSaved differences boxplot to: {Path(__file__).with_name('taxonomy_dag_differences.png')}")
    print("\nDifferences summary:")
    print(f"Precision: {precision_diff.mean(axis=0)}")
    print(f"Recall: {recall_diff.mean(axis=0)}")
    print(f"Utility: {utility_diff.mean(axis=0)}")
    print("\nDifferences stddev:")
    print(f"Precision: {precision_diff.std(axis=0)}")
    print(f"Recall: {recall_diff.std(axis=0)}")
    print(f"Utility: {utility_diff.std(axis=0)}")
    print("\nDifferences min/max:")
    print(f"Precision: {precision_diff.min(axis=0)} / {precision_diff.max(axis=0)}")
    print(f"Recall: {recall_diff.min(axis=0)} / {recall_diff.max(axis=0)}")
    print(f"Utility: {utility_diff.min(axis=0)} / {utility_diff.max(axis=0)}")
    print("\nDifferences median:")
    print(f"Precision: {np.median(precision_diff, axis=0)}")
    print(f"Recall: {np.median(recall_diff, axis=0)}")
    print(f"Utility: {np.median(utility_diff, axis=0)}")
    print("\nDifferences 25th/75th percentiles:")
    print(f"Precision: {np.percentile(precision_diff, 25, axis=0)} / {np.percentile(precision_diff, 75, axis=0)}")
    print(f"Recall: {np.percentile(recall_diff, 25, axis=0)} / {np.percentile(recall_diff, 75, axis=0)}")
    print(f"Utility: {np.percentile(utility_diff, 25, axis=0)} / {np.percentile(utility_diff, 75, axis=0)}")
    print("\nDifferences 5th/95th percentiles:")
    print(f"Precision: {np.percentile(precision_diff, 5, axis=0)} / {np.percentile(precision_diff, 95, axis=0)}")
    print(f"Recall: {np.percentile(recall_diff, 5, axis=0)} / {np.percentile(recall_diff, 95, axis=0)}")
    print(f"Utility: {np.percentile(utility_diff, 5, axis=0)} / {np.percentile(utility_diff, 95, axis=0)}")

if __name__ == "__main__":
    main()