"""Bar-search operators for the public factor API (Rust execution)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import FactorExpr, _to_expr
from .scope import BarCountScope


class BarSinceExpr(FactorExpr):
    def __init__(self, condition: FactorExpr, scope: BarCountScope | None, *,
                 lookback: int | None = None,
                 select: str = "nearest", default: float | None = None,
                 include_current: bool = True):
        if select not in {"nearest","farthest"}:raise ValueError("select must be nearest or farthest")
        self.condition = condition
        if scope is None and lookback is None:raise ValueError("search needs partition scope or lookback")
        self.scope = scope
        self.lookback = lookback
        self.select = select
        self.default = default
        self.include_current = include_current

    def _structural_key(self):
        return ("alpha.factor.BarSinceExpr:v2", None if self.scope is None else self.scope.count,self.lookback, self.select,
                self.include_current, self.default,
                self.condition._structural_key())



class BarDistanceExpr(BarSinceExpr):
    """Distance to the nearest/farthest matching bar in a bar-count scope."""

    def _structural_key(self):
        return ("alpha.factor.BarDistanceExpr:v2", None if self.scope is None else self.scope.count,self.lookback, self.select,
                self.include_current, self.default,
                self.condition._structural_key())


def bar_distance(condition, *, scope: BarCountScope | None = None, lookback: int | None = None, select: str = "nearest",
                 default: float | None = None, include_current: bool = True) -> BarDistanceExpr:
    return BarDistanceExpr(
        _to_expr(condition), scope, lookback=lookback,select=select, default=default,
        include_current=include_current,
    )


def bar_since(condition, *, scope: BarCountScope | None = None, lookback: int | None = None, select: str = "nearest",
              default: float | None = None, include_current: bool = True) -> BarSinceExpr:
    return BarSinceExpr(
        _to_expr(condition), scope, lookback=lookback,select=select, default=default,
        include_current=include_current,
    )
