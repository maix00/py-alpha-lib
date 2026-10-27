"""Causal signal alignment for nested factors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import FactorExpr
from .scopes import SessionScope, TradingDayScope


class SignalAlign(FactorExpr):
    """Resample a factor to a lower signal frequency without look-ahead.

    The signal at target time ``t`` uses a source value at or before ``t``;
    "before the anchor" is inclusive unless ``include_current=False``.
    No future value is ever used.
    """

    def __init__(
        self,
        child: FactorExpr,
        signal_freq: str,
        scope=None,
        basepoint: str = "last",
        scope_anchor: str = "start",
        daily_basepoint: str | None = None,
        end_session_skip: bool = False,
        end_session_gap: str | None = None,
        include_current: bool = True,
        partial: str = "drop",
    ):
        if basepoint not in {"last", "first"}:
            raise ValueError("basepoint must be 'last' or 'first'")
        if daily_basepoint is not None and not str(daily_basepoint).count(":"):
            raise ValueError("daily_basepoint must be a time of day")
        if daily_basepoint is not None:
            scope_anchor = str(daily_basepoint)
        if scope_anchor not in {"start", "end"} and not str(scope_anchor).count(":"):
            raise ValueError("scope_anchor must be 'start', 'end', or a time of day")
        if partial not in {"drop", "emit"}:
            raise ValueError("partial must be 'drop' or 'emit'")
        if end_session_skip or end_session_gap is not None:
            raise ValueError("session boundaries must come from a calendar-backed SessionScope")
        self.child = child
        self.signal_freq = str(signal_freq)
        self.scope = scope
        self.scope_anchor = str(scope_anchor)
        self.partial = partial
        self.basepoint = basepoint
        self.daily_basepoint = daily_basepoint
        self.end_session_skip = end_session_skip
        self.end_session_gap = end_session_gap
        self.include_current = include_current

    def _structural_key(self):
        if (
            self.scope is None
            and self.scope_anchor == "start"
            and self.partial == "drop"
            and self.basepoint == "last"
            and self.daily_basepoint is None
            and not self.end_session_skip
            and self.end_session_gap is None
            and self.include_current
        ):
            # Preserve the frozen factor:v2 fingerprint for legacy calls.
            return ("alpha.factor.SignalAlign", self.signal_freq, self.basepoint,
                    self.child._structural_key())
        return ("alpha.factor.SignalAlign:v2", self.signal_freq,
                (type(self.scope).__name__,tuple((s.start_ns,s.end_ns,str(s.trading_day)) for s in getattr(self.scope,"sessions",()))) if self.scope is not None else None,
                self.scope_anchor, self.partial, self.basepoint,
                self.daily_basepoint, self.end_session_skip, self.end_session_gap,
                self.include_current, self.child._structural_key())

    # Evaluation is inherited: the shared Rust matrix Plan owns sampling.


def signal_partitions(expression, index, trading_days=None):
    """Freeze calendar metadata only; Rust generates/samples all target slots."""
    if not isinstance(index,pd.DatetimeIndex):
        raise TypeError('SignalAlign requires a DatetimeIndex source')
    if index.empty:
        return []
    interval=int(pd.Timedelta(expression.signal_freq).value)
    if interval<=0:
        raise ValueError('signal frequency must be positive')
    first,last=int(index[0].value),int(index[-1].value)
    scope=expression.scope
    if isinstance(scope,SessionScope) or isinstance(scope,TradingDayScope) and scope.sessions:
        groups=[(int(x.start_ns),int(x.end_ns),True) for x in scope.sessions]
    elif isinstance(scope,TradingDayScope):
        labels=np.asarray(index.normalize() if trading_days is None else trading_days)
        starts=np.r_[0,np.flatnonzero(labels[1:]!=labels[:-1])+1]
        ends=np.r_[starts[1:]-1,len(index)-1]
        groups=[(int(index[a].value),int(index[b].value),True) for a,b in zip(starts,ends)]
    elif scope is None:
        groups=[(first,last,True)]
    else:
        raise TypeError(f'unsupported SignalAlign scope {type(scope).__name__}')
    result=[]
    for start,end,closed in groups:
        if end<first or start>last:
            continue
        anchor=expression.scope_anchor
        if anchor=='end':
            # A scope-less legacy grid begins at the first observation. A frozen
            # calendar end fixes the phase independently of the loaded prefix.
            target=start if scope is None else end-((end-start)//interval)*interval
        elif anchor=='start':
            target=-(-start//interval)*interval
        else:
            clock=pd.Timestamp(anchor).time()
            stamp=pd.Timestamp(start,unit='ns',tz=index.tz)
            target=int((stamp.normalize()+pd.Timedelta(hours=clock.hour,minutes=clock.minute,seconds=clock.second)).value)
            if target<start:
                target+=86400000000000
        result.append([start,min(end,last),target,int(closed and last>=end)])
    return result


def _method(self, signal_freq: str, scope=None, basepoint: str = "last", **kwargs):
    # Older callers passed basepoint positionally as the third argument.
    if isinstance(scope, str) and scope in {"last", "first"}:
        basepoint, scope = scope, None
    return SignalAlign(self, signal_freq, scope, basepoint, **kwargs)


FactorExpr.signal_align = _method
