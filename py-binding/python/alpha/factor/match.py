"""CURRENT/CANDIDATE bar-search expressions for the public API."""

from __future__ import annotations


from .core import CompositeExpr, ConstExpr, FactorExpr, _to_expr
from .scope import BarCountScope

_BINARY = {"add", "sub", "mul", "div", "gt", "lt", "ge", "le", "eq", "ne"}
_UNARY = {"abs", "neg"}


class MatchValueRef(FactorExpr):
    def __init__(self, kind: str):
        self.kind = kind

    def _structural_key(self):
        return ("alpha.factor.MatchValueRef", self.kind)


CURRENT = MatchValueRef("current")
CANDIDATE = MatchValueRef("candidate")


def _predicate(expr) -> dict:
    if isinstance(expr, MatchValueRef):
        return {"op": expr.kind}
    if isinstance(expr, ConstExpr):
        return {"op": "const", "value": float(expr.value)}
    if isinstance(expr, CompositeExpr):
        if expr.op == 'neg':
            return {'op':'sub','a':{'op':'const','value':0.0},'b':_predicate(expr.operands[0])}
        if expr.op in _UNARY:
            return {"op": expr.op, "a": _predicate(expr.operands[0])}
        if expr.op in _BINARY:
            return {"op": expr.op, "a": _predicate(expr.operands[0]), "b": _predicate(expr.operands[1])}
    raise TypeError(f"unsupported match predicate node {type(expr).__name__}")


class MatchBarDistanceExpr(FactorExpr):
    def __init__(self, value, condition, scope: BarCountScope, *,
                 select: str = "nearest", default: float = 0.0,
                 include_current: bool = True):
        if select not in {"nearest", "farthest"}:
            raise ValueError("select must be nearest or farthest")
        self.value = _to_expr(value)
        self.condition = _to_expr(condition)
        self.scope = scope
        self.select = select
        self.default = float(default)
        self.include_current = include_current

    def _structural_key(self):
        return ("alpha.factor.MatchBarDistanceExpr", self.scope.count, self.select,
                self.default, self.include_current,
                self.value._structural_key(), self.condition._structural_key())

    # Inherit the shared matrix Plan; no Python predicate/window evaluator.


def bar_distance(value, condition=None, *, scope: BarCountScope, select: str = "nearest",
                 default: float = 0.0, include_current: bool = True) -> MatchBarDistanceExpr:
    if condition is None:
        from .bar_search import BarDistanceExpr
        return BarDistanceExpr(value,scope,select=select,default=default,include_current=include_current)
    return MatchBarDistanceExpr(value, condition, scope, select=select,
                                default=default, include_current=include_current)
