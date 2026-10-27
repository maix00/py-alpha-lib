"""Category membership leaves for the public factor API."""

from __future__ import annotations

import pandas as pd

from .core import FactorExpr


class CategoryBoolRef(FactorExpr):
    """Boolean panel: product belongs to ``category[label]``.

    Memberships are effective-time snapshots (including removals), frozen for
    the Plan product universe; all execution entries share the Rust cursor.
    """

    def __init__(self, category=None, label: str | None = None):
        self.label = label or getattr(category, "alias", None)
        if self.label is None:
            raise ValueError("CategoryBoolRef requires a label")
        if category is not None and self.label not in getattr(category, "categories", ()):
            raise ValueError(f"unknown category label {self.label!r}")
        self.category = category

    def _structural_key(self):
        return (
            "alpha.factor.CategoryBoolRef",
            getattr(self.category, "alias", None),
            getattr(self.category, "_definition_key", None),
            self.label,
        )

    # Evaluation inherits the common Rust Plan entry.


def category_node(expression, products, categories):
    """Freeze latest-snapshot changes once; runtime membership is Rust-owned."""
    changes=[]
    for index,product in enumerate(products):
        if categories is None:
            if expression.category is None:
                raise ValueError('CategoryBoolRef needs categories or a category object')
            changes.append([index,-(2**63),int(expression.category.is_in_category(expression.label,product))])
            continue
        entry=categories.get(product,categories.get(str(product)))
        entries=entry if isinstance(entry,(tuple,list)) and all(hasattr(x,"labels") for x in entry) else (entry,)
        snapshots={}
        for item in entries:
            if item is None:continue
            labels=getattr(item,'labels',item)
            effective=getattr(item,'effective_at_ns',None)
            at=-(2**63) if effective is None else int(effective)
            member=int(expression.label in labels)
            if at in snapshots and snapshots[at]!=member:
                raise ValueError('conflicting category snapshots at the same effective time')
            snapshots[at]=member
        previous=None
        for at,member in sorted(snapshots.items()):
            if member!=previous:changes.append([index,at,member]);previous=member
    return {'kind':'category','memberships':changes}
