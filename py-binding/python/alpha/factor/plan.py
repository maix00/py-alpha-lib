"""Compile the public alpha.factor graph to the Rust matrix BatchPlan."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .timeline import timestamps_ns as _times


@dataclass
class _Grid:
    id: int
    index: pd.Index
    positions: np.ndarray
    times: np.ndarray
    observed: np.ndarray
    days: np.ndarray
    signal_ns: int | None = None
    partitions: list | None = None


@dataclass
class _Ref:
    id: int
    grid: _Grid
    columns: tuple
    scalar: bool = False


def _walk(expressions):
    seen = set()

    def visit(expr):
        key = expr._structural_key()
        if key in seen:
            return
        seen.add(key)
        yield expr
        children = list(getattr(expr, "operands", ()))
        for attribute in ("data", "child", "operand", "owner", "condition", "when_true", "when_false"):
            child = getattr(expr, attribute, None)
            if child is not None and all(child is not x for x in children):
                children.append(child)
        for child in children:
            if hasattr(child, "_structural_key"):
                yield from visit(child)

    for expression in expressions:
        yield from visit(expression)


class _Builder:
    def __init__(self, expressions, ctx):
        from alpha import BatchPlan

        self.ctx = ctx
        self.plan = BatchPlan()
        self.inputs: list[np.ndarray] = []
        self.memo: dict[tuple, _Ref] = {}
        self.products = tuple(ctx.products)
        self.frames = {}
        self.aligned = {}
        self.term_inputs = {}
        self.term_slots = {}
        for node in _walk(expressions):
            if type(node).__name__=="TermStructureExpr":
                self.term_slots[node.column]=max(self.term_slots.get(node.column,0),node.near_rank+1,node.far_rank+1,node.depth)
        for expr in _walk(expressions):
            if type(expr).__name__ == "ColumnRef":
                self.frames[expr._structural_key()] = expr._evaluate(ctx)
        if ctx.timeline is None:
            from .timeline import build_timeline
            object.__setattr__(ctx, "timeline", build_timeline(self.products, ctx.preloaded, ctx.trading_days, frequency=getattr(ctx.freq, "name", ctx.freq)))
        timeline = ctx.timeline
        self.index = timeline.index
        self.observed = timeline.observed
        self.days = timeline.days
        self.base = _Grid(
            self.plan.grid(
                timeline.times_ns, self.days,
                np.ascontiguousarray(self.observed).reshape(-1),
            ),
            self.index, np.arange(len(self.index), dtype=np.int64), timeline.times_ns,
            self.observed,self.days,
        )

    def input(self, values):
        self.inputs.append(np.ascontiguousarray(values, dtype=float).reshape(-1))
        return len(self.inputs) - 1

    def add(self, op, children=(), params=(), indices=(), *, grid=None, scalar=False, columns=None):
        grid = self.base if grid is None else grid
        node = self.plan.add(
            op, [child.id for child in children], list(params), list(indices), grid.id,
        )
        return _Ref(node, grid, self.products if columns is None else tuple(columns), scalar)

    def align(self, ref, grid):
        if ref.scalar or ref.grid is grid:
            return ref
        key=(ref.id,grid.id)
        if key not in self.aligned:
            mapping=(self.plan.asof_partition_indices(ref.grid.times,grid.times,ref.grid.partitions)
                     if ref.grid.partitions is not None else
                     self.plan.asof_indices(ref.grid.times,grid.times) if ref.grid.signal_ns is not None
                     else ref.grid.index.get_indexer(grid.index))
            self.aligned[key]=self.add('reindex',[ref],indices=mapping,grid=grid,columns=ref.columns)
        return self.aligned[key]

    def point(self, op, children):
        if any(x is None for x in children):
            return None
        active=[x for x in children if not x.scalar]
        raw=[x for x in active if x.grid.signal_ns is None]
        grid=(raw[0].grid if raw else min(active,key=lambda x:x.grid.signal_ns).grid if active else self.base)
        return self.add(op,[self.align(x,grid) for x in children],grid=grid,
                        scalar=not active,columns=active[0].columns if active else ())

    def signal(self, expr):
        from .signal import signal_partitions
        child=self.build(expr.child)
        if child is None:
            return None
        interval=int(pd.Timedelta(expr.signal_freq).value)
        partitions=signal_partitions(expr,self.base.index,self.ctx.timeline.trading_days)
        targets,mapping=self.plan.signal_grid(child.grid.times,
            np.ascontiguousarray(child.grid.observed).reshape(-1),len(self.products),
            partitions,interval,
            expr.basepoint=='first',expr.include_current,expr.partial=='emit')
        times=np.asarray(targets,dtype=np.int64)
        index=pd.DatetimeIndex(times.view("datetime64[ns]"),tz="UTC")
        index=index.tz_localize(None) if self.base.index.tz is None else index.tz_convert(self.base.index.tz)
        positions=np.asarray(self.plan.asof_indices(self.base.times,times),dtype=np.int64)
        observed=(np.asarray(mapping).reshape(len(self.products),len(times))>=0).astype(np.uint8)
        days=self.base.days[np.maximum(positions,0)] if len(times) else np.empty(0,dtype=np.int64)
        grid=_Grid(self.plan.grid(times,days,observed.reshape(-1)),index,positions,times,observed,days,interval,partitions if expr.scope is not None else None)
        return self.add('reindex',[child],indices=mapping,grid=grid,columns=child.columns)

    def term(self, expr):
        from .term import DAYS,RANK,TERM_OPS
        key=expr.column
        if key not in self.term_inputs:
            prepared=[];maximum=0
            for product in self.products:
                source=(self.ctx.term_curves or {}).get(product)
                snapshots=[] if source is None else source if isinstance(source,(list,tuple)) else [source]
                frames=[]
                for snapshot in snapshots:
                    frame=pd.DataFrame(snapshot.rows() if hasattr(snapshot,'rows') else snapshot)
                    if RANK in frame:frame=frame.sort_values(RANK,kind='stable')
                    at=getattr(snapshot,'available_at_ns',None)
                    frames.append((-(1<<63) if at is None else int(at),frame))
                    maximum=max(maximum,len(frame))
                prepared.append(sorted(frames,key=lambda item:item[0]))
            # Out-of-range ranks/depth never allocate imaginary curve slots.
            slots=max(1,min(self.term_slots[key],maximum))
            values,days,lengths=[],[],[]
            mapping=np.full((len(self.products),len(self.index)),-1.0)
            for g,snapshots in enumerate(prepared):
                if not snapshots:continue
                available=[];ids=[]
                for at,frame in snapshots:
                    n=min(len(frame),slots) if expr.column in frame else 0
                    raw=np.full(slots,np.nan);maturity=np.full(slots,np.nan)
                    if n:
                        raw[:n]=frame[expr.column].to_numpy(dtype=float)[:n]
                        if DAYS in frame:maturity[:n]=frame[DAYS].to_numpy(dtype=float)[:n]
                    available.append(at)
                    ids.append(len(lengths));lengths.append(n);values.append(raw);days.append(maturity)
                selected=np.asarray(self.plan.asof_indices(np.asarray(available,dtype=np.int64),self.base.times),dtype=np.int64)
                valid=(selected>=0)&self.observed[g].astype(bool)
                mapping[g,valid]=np.asarray(ids,dtype=np.int64)[selected[valid]]
            self.term_inputs[key]=[self.input(values),self.input(days),self.input(lengths),self.input(mapping),slots]
        return self.add('term',params=[expr.near_rank,expr.far_rank,expr.depth,TERM_OPS.index(expr.op)],
                        indices=self.term_inputs[key])

    def kernel(self, node, child=None):
        import json
        grid=self.base if child is None else child.grid
        nodes=[] if child is None else [dict(kind='column',index=0)]
        nodes.append(node)
        payload=json.dumps(dict(nodes=nodes,outputs=[len(nodes)-1]),allow_nan=False)
        identifier=self.plan.add_kernel(payload,[] if child is None else [child.id],grid.id,len(self.products))
        return _Ref(identifier,grid,self.products if child is None else child.columns)

    def build(self, expr) -> _Ref:
        key = expr._structural_key()
        if key in self.memo:
            return self.memo[key]
        from .core import ColumnRef, CompositeExpr, ConstExpr, RollingOp, ShiftOp, WhereOp
        from .cross_sectional import CrossSectionalExpr
        from .scope import GroupByScopeOp
        from .scopes import TradingDayScope, SessionScope
        from .bar_search import BarSinceExpr
        from .match import MatchBarDistanceExpr, _predicate
        from .category import CategoryBoolRef,category_node

        from .signal import SignalAlign
        from .term import TermStructureExpr

        if isinstance(expr,CategoryBoolRef):
            ref=self.kernel(category_node(expr,self.products,self.ctx.categories))
        elif isinstance(expr,MatchBarDistanceExpr):
            child=self.build(expr.value)
            ref=self.kernel(dict(kind='match_distance',child=0,predicate=_predicate(expr.condition),
                count=expr.scope.count,nearest=expr.select=='nearest',default=None if np.isnan(expr.default) else expr.default,
                include_current=expr.include_current),child)
        elif isinstance(expr,SignalAlign):
            ref=self.signal(expr)
        elif isinstance(expr,TermStructureExpr):
            ref=self.term(expr)
        elif isinstance(expr, ColumnRef):
            frame = self.frames.get(key)
            if frame is None:
                frame = expr._evaluate(self.ctx)
                self.frames[key] = frame
            frame = frame.reindex(index=self.index, columns=self.products)
            ref = self.add("column", params=[self.input(frame.to_numpy(dtype=float).T)])
        elif isinstance(expr, ConstExpr):
            ref = self.add("const", params=[expr.value], scalar=True, columns=())
        elif isinstance(expr, RollingOp):
            child = self.build(expr.data)
            op, _, q = expr.op.partition(":")
            ref = self.add(
                f"rolling:{op}", [child],
                [0, expr.window, 0, expr.semantics.min_periods_for(op),
                 float(expr.semantics.nan_policy.value == "propagate"),
                 float(q) if q else 0.5, -1.0, -1.0],
                columns=child.columns, grid=child.grid,
            )
        elif isinstance(expr, GroupByScopeOp):
            child = self.build(expr.owner.data)
            op, _, q = expr.op.partition(":")
            if isinstance(expr.owner.scope, TradingDayScope):
                scope_params = [2, 0, 0]
            elif isinstance(expr.owner.scope, SessionScope):
                sessions = expr.owner.scope.sessions
                ref = self.add(
                    f"session_rolling:{op}", [child],
                    [1, expr.owner.semantics.effective_min_periods,
                     float(expr.owner.semantics.nan_policy.value == "propagate"), float(q) if q else 0.5],
                    [len(sessions), *[value for session in sessions for value in (session.start_ns, session.end_ns, 0)]],
                    columns=child.columns,
                )
                self.memo[key] = ref
                return ref
            else:
                scope_params = [1, expr.owner.scope.count, 0]
            ref = self.add(
                f"scope:{op}", [child],
                [*scope_params,
                 expr.owner.semantics.effective_min_periods,
                 float(expr.owner.semantics.nan_policy.value == "propagate"),
                 float(q) if q else 0.5, -1.0, -1.0],
                columns=child.columns, grid=child.grid,
            )
        elif isinstance(expr, CrossSectionalExpr):
            child = self.build(expr.operand)
            priority = [i for i, _ in enumerate(self.products)]
            ref = self.add(
                "cross:" + expr.op.removeprefix("cs_"), [child], [], priority,
                columns=self.products,
            )
        elif isinstance(expr, BarSinceExpr):
            child = self.build(expr.condition)
            ref=self.kernel(dict(kind='since',condition=0,count=expr.scope.count,
                include_current=expr.include_current,nearest=expr.select=='nearest',
                default_scope=expr.default is None,default=0.0 if expr.default is None else float(expr.default)),child)
        elif isinstance(expr, CompositeExpr):
            children = [self.build(item) for item in expr.operands]
            ref = self.point(expr.op,children)
        elif isinstance(expr, WhereOp):
            ref = self.point("where", [self.build(expr.condition), self.build(expr.when_true),self.build(expr.when_false)])
        elif isinstance(expr, ShiftOp):
            child = self.build(expr.data)
            ref = self.add("shift", [child], [expr.periods], columns=child.columns,grid=child.grid)
        else:
            return None
        self.memo[key] = ref
        return ref


def compile_plan(expressions, ctx):
    builder = _Builder(expressions, ctx)
    outputs = [builder.build(expr) for expr in expressions]
    if any(output is None for output in outputs):
        return None
    arrays = builder.plan.execute_panel(
        builder.inputs, [output.id for output in outputs], len(builder.products),
    )
    result = []
    for values, output in zip(arrays, outputs, strict=True):
        width = len(output.columns) or len(builder.products)
        frame = np.asarray(values, dtype=float).reshape(width, len(output.grid.index)).T
        result.append(pd.DataFrame(frame, index=output.grid.index, columns=output.columns))
    return result
