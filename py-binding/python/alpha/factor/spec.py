"""Serialize an alpha.factor graph into the bridge's Rust plan JSON."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .core import ColumnRef, CompositeExpr, ConstExpr, RollingOp, ShiftOp, WhereOp
from .cross_sectional import CrossSectionalExpr
from .scope import GroupByScopeOp
from .bar_search import BarDistanceExpr, BarSinceExpr, BarSinceExpr as BarSince
from .match import MatchBarDistanceExpr, _predicate
from .signal import SignalAlign
from .term import TermStructureExpr
from .scopes import SessionScope, TradingDayScope
from .category import CategoryBoolRef

_BINARY = {"add", "sub", "mul", "div", "gt", "lt", "ge", "le", "eq", "ne"}
_UNARY = {"abs", "sign", "sqrt", "log", "tanh"}


def to_plan_spec(expressions: list[Any], *, products=None, categories=None) -> dict[str, Any] | None:
    columns: list[str] = []
    nodes: list[dict[str, Any]] = []
    keys: list[str] = []
    memo: dict[tuple, int] = {}

    def build(expr) -> int | None:
        key = expr._structural_key()
        if key in memo:
            return memo[key]
        if isinstance(expr, TermStructureExpr): 
            node = {"kind": "term", "op": expr.op, "near_rank": expr.near_rank,
                    "far_rank": expr.far_rank, "depth": expr.depth}
        elif isinstance(expr, CategoryBoolRef):
            if products is None:
                return None
            from .category import category_node
            node=category_node(expr,products,categories)
        elif isinstance(expr, ColumnRef):
            if expr.column not in columns:
                columns.append(expr.column)
            node = {"kind": "column", "index": columns.index(expr.column)}
        elif isinstance(expr, ConstExpr):
            node = {"kind": "const", "value": float(expr.value)}
        elif isinstance(expr, RollingOp):
            if not isinstance(expr.window, (int, np.integer)):
                raise ValueError("Bind rolling window parameters before compiling a factor plan")
            child = build(expr.data)
            if child is None:
                return None
            op, _, q = expr.op.partition(":")
            node = {
                "kind": "rolling", "child": child, "window": expr.window,
                "op": op, "min_periods": expr.semantics.min_periods_for(op),
                "propagate_nan": expr.semantics.nan_policy.value == "propagate",
            }
            if q:
                node["quantile"] = float(q)
        elif isinstance(expr, GroupByScopeOp):
            if expr.op.startswith("quantile"):
                return None
            child = build(expr.owner.data)
            if child is None:
                return None
            scope = expr.owner.scope
            if isinstance(scope, SessionScope):
                sessions = [[int(item.start_ns), int(item.end_ns), int(item.start_ns)] for item in scope.sessions]
                count = 0
            else:
                sessions = []
                count = getattr(scope, "count", 0)
            node = {
                "kind": "scope", "child": child, "count": count,
                "op": expr.op, "min_periods": expr.owner.semantics.effective_min_periods,
                "propagate_nan": expr.owner.semantics.nan_policy.value == "propagate",
                "sessions": sessions,
            }
        elif isinstance(expr, CrossSectionalExpr):
            child = build(expr.operand)
            if child is None:
                return None
            node = {"kind": "cross", "child": child, "op": expr.op.removeprefix("cs_")}
        elif isinstance(expr, CompositeExpr):
            children = [build(item) for item in expr.operands]
            if any(child is None for child in children):
                return None
            if expr.op == "neg":
                node = {"kind": "neg", "child": children[0]}
            elif expr.op in _UNARY:
                node = {"kind": "unary", "op": expr.op, "child": children[0]}
            elif expr.op in _BINARY:
                node = {"kind": "binary", "op": expr.op, "left": children[0], "right": children[1]}
            else:
                return None
        elif isinstance(expr, ShiftOp):
            child = build(expr.data)
            if child is None:
                return None
            node = {"kind": "shift", "child": child, "periods": expr.periods}
        elif isinstance(expr, WhereOp):
            children = [build(expr.condition), build(expr.when_true), build(expr.when_false)]
            if any(child is None for child in children):
                return None
            node = {"kind": "where", "condition": children[0],
                    "when_true": children[1], "when_false": children[2]}
        elif isinstance(expr, SignalAlign):
            calendar_scope = isinstance(expr.scope, SessionScope) or (
                isinstance(expr.scope, TradingDayScope) and bool(expr.scope.sessions)
            )
            if expr.scope_anchor != "start" and not calendar_scope:
                return None
            child = build(expr.child)
            if child is None:
                return None
            node = {
                "kind": "signal", "child": child,
                "signal_ns": int(pd.Timedelta(expr.signal_freq).value),
                "basepoint": expr.basepoint, "scope_anchor": expr.scope_anchor,
                "include_current": expr.include_current, "partial": expr.partial,
                "sessions": [],
            }
            if isinstance(expr.scope, (SessionScope, TradingDayScope)) and expr.scope.sessions:
                signal_ns = node["signal_ns"]
                sessions = []
                for session in expr.scope.sessions:
                    start, end = int(session.start_ns), int(session.end_ns)
                    if expr.scope_anchor == "end":
                        first = end - ((end - start) // signal_ns) * signal_ns
                    elif expr.scope_anchor != "start":
                        clock = pd.Timestamp(expr.scope_anchor).time()
                        first = int((pd.Timestamp(start, unit="ns").normalize()
                                     + pd.Timedelta(hours=clock.hour, minutes=clock.minute,
                                                    seconds=clock.second)).value)
                        if first < start:
                            first += 86_400_000_000_000
                    else:
                        first = -(-start // signal_ns) * signal_ns
                    sessions.append([start, end, first])
                node["sessions"] = sessions
            elif expr.scope is not None:
                return None
        elif isinstance(expr, MatchBarDistanceExpr):
            child = build(expr.value)
            if child is None:
                return None
            node = {
                "kind": "match_distance", "child": child,
                "predicate": _predicate(expr.condition),
                "count": expr.scope.count, "nearest": expr.select == "nearest",
                "default": None if np.isnan(expr.default) else expr.default, "include_current": expr.include_current,
            }
        elif isinstance(expr, BarSince):
            child = build(expr.condition)
            if child is None:
                return None
            node = {"kind": "distance" if isinstance(expr, BarDistanceExpr) else "since",
                    "condition": child, "count": expr.scope.count,
                    "include_current": bool(expr.include_current),
                    "nearest": expr.select == "nearest",
                    "default_scope": expr.default is None,
                    "default": 0.0 if expr.default is None else float(expr.default)}
        else:
            return None
        memo[key] = len(nodes)
        keys.append(repr(key))
        nodes.append(node)
        return len(nodes) - 1

    outputs = [build(expression) for expression in expressions]
    if any(output is None for output in outputs):
        return None
    return {"columns": columns, "keys": keys, "nodes": nodes, "outputs": outputs}


def plan_blockers(expression: Any, *, execution="batch", products=None, categories=None) -> tuple[str, ...]:
    """Report the requested execution entry, not a stale global node blacklist."""
    if execution not in {"batch","incremental","bridge"}:
        raise ValueError('unknown factor execution entry')
    if execution != 'batch':
        try:
            return () if to_plan_spec([expression],products=products,categories=categories) is not None else (type(expression).__name__,)
        except (TypeError,ValueError,AttributeError) as error:
            return (f'{type(expression).__name__}: {error}',)
    from .annotations import children
    supported=(ColumnRef,ConstExpr,RollingOp,ShiftOp,WhereOp,CompositeExpr,
               GroupByScopeOp,CrossSectionalExpr,BarSinceExpr,MatchBarDistanceExpr,
               CategoryBoolRef,SignalAlign,TermStructureExpr)
    blockers=[];seen=set()
    def visit(node):
        if id(node) in seen:return
        seen.add(id(node))
        if not isinstance(node,supported):
            blockers.append(type(node).__name__);return
        if isinstance(node,MatchBarDistanceExpr):
            try:_predicate(node.condition)
            except TypeError as error:blockers.append(str(error));return
            visit(node.value);return
        for child in children(node):
            if type(child).__name__ not in {'GroupByScopeExpr'}:visit(child)
    visit(expression)
    return tuple(dict.fromkeys(blockers))


NODE_CHILD_FIELDS = ('child','left','right','condition','when_true','when_false',
                     'boundary','trunc_start_child','trunc_end_child')


def scope_plan_outputs(spec, requests, products):
    """Share product-local ancestors; specialize cross barriers by frozen universe.

    Requests contain (old output slot, active product indices). Static DAG rewrite
    only: existing Rust product/where/cross nodes own all per-frame computation.
    """
    import json
    nodes=[];by_key={};memo={}
    def emit(node):
        key=json.dumps(node,sort_keys=True,separators=(',',':'),allow_nan=False)
        if key not in by_key:
            by_key[key]=len(nodes);nodes.append(node)
        return by_key[key]
    def visit(old,active):
        key=(old,active)
        if key in memo:return memo[key]
        node=dict(spec['nodes'][old])
        for field in NODE_CHILD_FIELDS:
            if node.get(field) is not None:node[field]=visit(node[field],active)
        if node['kind']=='cross' and len(active)!=products:
            mask=emit(dict(kind='product',values=[float(i in active) for i in range(products)]))
            zero=emit(dict(kind='const',value=0.0))
            missing=emit(dict(kind='binary',op='div',left=zero,right=zero))
            node['child']=emit(dict(kind='where',condition=mask,when_true=node['child'],when_false=missing))
        value=emit(node);memo[key]=value;return value
    outputs=[];slots=[]
    for slot,scope in requests:
        active=tuple(sorted(scope))
        if not active or len(set(active))!=len(active) or active[0]<0 or active[-1]>=products:
            raise ValueError('invalid factor product universe')
        value=visit(spec['outputs'][slot],active)
        if value not in outputs:outputs.append(value)
        slots.append(outputs.index(value))
    result={**spec,'nodes':nodes,'outputs':outputs,'keys':[json.dumps(x,sort_keys=True) for x in nodes]}
    result.pop('intermediates',None)
    return result,slots
