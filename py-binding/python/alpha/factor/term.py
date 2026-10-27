"""Futures term-structure operators for the public factor API."""

from __future__ import annotations

from numbers import Integral
import pandas as pd

from .core import FactorExpr

DAYS = "DAYS_TO_MATURITY"
RANK = "TERM_RANK"
PAIR = {
    "term_spread", "term_ratio", "term_log_ratio", "term_contango",
    "term_carry_annualized", "term_slope_segment",
}
DEPTH = {"term_slope", "term_curvature"}


TERM_OPS=['term_spread','term_ratio','term_log_ratio','term_contango',
          'term_carry_annualized','term_slope_segment','term_rank_value','term_slope','term_curvature']


def evaluate_term_metric(op, curve, *, near_rank, far_rank, depth, column):
    """Compatibility adapter: scalar query uses the SAME Rust term Plan."""
    from .core import EvalContext
    index=pd.RangeIndex(1)
    context=EvalContext(products=('curve',),preloaded={'curve':pd.DataFrame(index=index)},
                        term_curves={'curve':curve})
    return float(TermStructureExpr(op,near_rank=near_rank,far_rank=far_rank,depth=depth,column=column).evaluate(context).iloc[0,0])


class TermStructureExpr(FactorExpr):
    def __init__(self, op: str, *, near_rank: int = 0, far_rank: int = 1,
                 depth: int = 4, column: str = "CLOSE"):
        if op not in PAIR | DEPTH | {"term_rank_value"}:
            raise ValueError(f"unknown term operation {op}")
        if any(isinstance(x,bool) or not isinstance(x,Integral) or x<0 for x in (near_rank,far_rank,depth)):
            raise ValueError("term ranks and depth must be nonnegative integers")
        self.op = op
        self.near_rank = near_rank
        self.far_rank = far_rank
        self.depth = depth
        self.column = column

    def _structural_key(self):
        return ("alpha.factor.TermStructureExpr", self.op, self.near_rank,
                self.far_rank, self.depth, self.column)

    # Evaluation is inherited from the shared Rust matrix Plan.


def _make(op):
    return lambda *, near_rank=0, far_rank=1, column="CLOSE", **kwargs: TermStructureExpr(
        op, near_rank=near_rank, far_rank=far_rank, column=column, **kwargs,
    )


term_spread = _make("term_spread")
term_ratio = _make("term_ratio")
term_log_ratio = _make("term_log_ratio")
term_contango = _make("term_contango")
term_carry_annualized = _make("term_carry_annualized")
term_slope_segment = _make("term_slope_segment")
term_slope = lambda *, depth=4, column="CLOSE": TermStructureExpr("term_slope", depth=depth, column=column)
term_curvature = lambda *, depth=4, column="CLOSE": TermStructureExpr("term_curvature", depth=depth, column=column)
term_rank_value = lambda *, rank=0, column="CLOSE": TermStructureExpr("term_rank_value", near_rank=rank, column=column)
