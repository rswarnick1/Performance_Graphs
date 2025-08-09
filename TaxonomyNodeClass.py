
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math

@dataclass
class TaxonomyNode:
    """
    Lightweight taxonomy node with simple precision/recall models.

    We avoid external dependencies (e.g., anytree) to keep runtime friction low.
    """
    name: str
    parent: Optional["TaxonomyNode"] = None
    children: List["TaxonomyNode"] = field(default_factory=list)
    detection_params: Dict[str, float] = field(default_factory=lambda: {
        # sensitivity ~= recall at low threshold; specificity influences precision at high threshold
        "sensitivity": 0.6,
        "specificity": 0.6,
        "scale": 8.0,  # steepness of logistic curves
    })

    def add_child(self, child: "TaxonomyNode") -> None:
        child.parent = self
        self.children.append(child)

    # ---- Simple analytic models ----
    def compute_recall(self, lam: float) -> float:
        """
        Recall(λ): decreasing logistic with λ — stricter thresholds catch fewer positives.
        """
        sens = float(self.detection_params.get("sensitivity", 0.5))
        scale = float(self.detection_params.get("scale", 8.0))
        return 1.0 / (1.0 + math.exp(scale * (lam - sens)))

    def compute_precision(self, lam: float) -> float:
        """
        Precision(λ): increasing logistic with λ — stricter thresholds reduce false positives.
        """
        spec = float(self.detection_params.get("specificity", 0.5))
        scale = float(self.detection_params.get("scale", 8.0))
        # center around (1 - specificity) so higher specificity shifts curve left (higher precision sooner)
        return 1.0 / (1.0 + math.exp(-scale * (lam - (1.0 - spec))))

    # ---- helpers ----
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def path(self) -> List[str]:
        out = []
        cur = self
        while cur is not None:
            out.append(cur.name)
            cur = cur.parent
        return list(reversed(out))

    def __repr__(self) -> str:
        return f"TaxonomyNode(name={self.name!r}, children={[c.name for c in self.children]!r})"
