"""Data-free history requirements for the shared factor DAG.

Amounts remain in their native units. Summaries are conservative full-window
bounds, not minimal dates or a substitute for numerical missing-value checks.
Trading-day windows must be resolved against frozen source calendars.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class HistoryRequirement:
    source_bars: int = 0
    trading_days: int = 0
    elapsed_ns: int = 0
    full_partitions: int = 0
    blockers: tuple[str, ...] = ()

    def manifest(self):
        return {**asdict(self), 'blockers':list(self.blockers),
                'bound':'conservative_full_window','exact_minimum':False,
                'requires_calendar':bool(self.trading_days or self.full_partitions),
                'requires_observation_index':bool(self.source_bars or self.elapsed_ns)}


def _maximum(rows):
    rows=tuple(rows)
    return HistoryRequirement(*(max((getattr(r,k) for r in rows),default=0)
        for k in ('source_bars','trading_days','elapsed_ns','full_partitions')),
        tuple(sorted({b for row in rows for b in row.blockers})))


def warmup_requirements(expression_or_plan):
    """Return a serializable requirement for every selected output, sharing CSE.

    A resolved public expression or a shared Rust Plan is accepted. Unresolved
    operators are reported as blockers; they never become zero warmup.
    Full windows are required even when an operator permits partial min_periods.
    All Where/dependency branches participate, including a search's threshold.
    """
    if isinstance(expression_or_plan,Mapping):plan=expression_or_plan
    else:
        from .spec import to_plan_spec
        plan=to_plan_spec(list(expression_or_plan) if isinstance(expression_or_plan,(list,tuple)) else [expression_or_plan])
        if plan is None:raise ValueError('warmup requires a resolved Rust factor plan')
    requirements=[]
    temporal=[]
    for index,node in enumerate(plan['nodes']):
        dependencies=[node[k] for k in ('child','left','right','condition','when_true','when_false','boundary','trunc_start_child','trunc_end_child') if node.get(k) is not None]
        dependencies+=list(node.get('dependencies',()))
        if any(not isinstance(i,int) or not 0<=i<index for i in dependencies):raise ValueError('warmup plan is not topologically ordered')
        base=_maximum(requirements[i] for i in dependencies);values=asdict(base);kind=node['kind']
        if kind=='column':pass  # current Bar is not historical warmup
        elif kind=='shift':
            periods=int(node['periods'])
            if periods<0:values['blockers']+=('future_shift',)
            else:values['source_bars']+=periods
        elif kind in {'rolling','rolling_dynamic','ema','linreg'}:
            width=int(node.get('window',node.get('span',max(node.get('windows',[1])))))
            values['source_bars']+=max(0,width-1)
            if kind=='ema':values['blockers']+=('ema_requires_convergence_policy',)
            temporal.append(index)
        elif kind=='session_rolling':
            values['trading_days']+=int(node['lookback_sessions']);temporal.append(index)
        elif kind in {'scope','session_reset'}:
            if kind=='scope' and not node.get('sessions'):
                if int(node['count'])<=0:values['blockers']+=('scope_requires_frozen_calendar',)
                else:values['source_bars']+=int(node['count'])-1
            else:values['full_partitions']+=1
            temporal.append(index)
        elif kind=='signal':values['elapsed_ns']+=int(node['signal_ns']);temporal.append(index)
        elif kind in {'search','match_distance','since','distance'}:
            width=node.get('lookback') or node.get('partition') or node.get('count') or 0
            values['source_bars']+=max(0,int(width)-(0 if node.get('lookback') is not None else 1))
            if node.get('boundary') is not None or node.get('gap_ns'):values['full_partitions']+=1
            temporal.append(index)
        elif kind=='bar_count':values['blockers']+=('fixed_interval_requires_source_coverage',)
        elif kind in {'const','product','category','boundary','unary','binary','where','neg','cross','term'}:pass
        else:values['blockers']+=(f'unsupported_node:{kind}',)
        if node.get('trunc_start_child') is not None or node.get('trunc_end_child') is not None:
            values['blockers']+=('dynamic_truncation_requires_bound',)
        requirements.append(HistoryRequirement(**values))
    outputs=[requirements[i] for i in plan['outputs']]
    return {'version':1,'outputs':[x.manifest() for x in outputs],
            'combined':_maximum(outputs).manifest(),'temporal_nodes':temporal,
            'node_count':len(requirements),'policy':'full_windows_and_all_dependencies','source_bars_exclude_current':True}


def required_history_start(requirement, *, trading_start_ns:int, sessions):
    """Resolve a safe read boundary using (start,end,observed-Bar-count) sessions.

    Whole source partitions are retained when metadata lacks individual Bar
    positions. This intentionally over-reads rather than inventing timestamps.
    Session rows must belong to one source clock/product, not a merged count.
    """
    r=requirement.get('combined',requirement)
    if r.get('blockers'):raise ValueError('unresolved warmup: '+', '.join(r['blockers']))
    ordered=[tuple(map(int,row)) for row in sessions]
    if any(a>b or n<=0 for a,b,n in ordered) or any(ordered[i-1][1]>=row[0] for i,row in enumerate(ordered) if i):
        raise ValueError('warmup needs ordered nonoverlapping source sessions')
    anchor=int(trading_start_ns)-int(r.get('elapsed_ns',0))
    prior=[row for row in ordered if row[1]<anchor]
    days=int(r.get('trading_days',0))+int(r.get('full_partitions',0))
    bars=int(r.get('source_bars',0))
    if not days and not bars:return anchor
    if len(prior)<days:raise ValueError(f'insufficient warmup: need {days} complete trading sessions before trading; have {len(prior)}')
    cursor=len(prior)-days
    # Source history upstream of day windows is additional, not max(day,bars).
    while bars>0:
        cursor-=1
        if cursor<0:raise ValueError('insufficient warmup source Bars before the required trading-day history')
        bars-=prior[cursor][2]
    if cursor>=len(prior):return anchor
    return prior[cursor][0]


class WarmupTimeline:
    """Reusable, borrowed frozen observation index for exact structural bounds.

    One product/source clock per instance. Arrays are not copied; callers must
    keep their underlying frozen-version buffers immutable. Derived signal maps
    are cached and no price/history panel is retained here.
    """
    def __init__(self,observation_times,trading_days, *, product_index=None):
        import numpy as np
        self.times=np.asarray(observation_times,dtype=np.int64).view()
        self.days=np.asarray(trading_days).view()
        if self.times.ndim!=1 or len(self.times)!=len(self.days) or not len(self.times):raise ValueError('warmup timeline shape mismatch')
        if np.any(self.times[1:]<=self.times[:-1]):raise ValueError('warmup observation times must be strictly increasing')
        if not self.times.flags.c_contiguous:raise ValueError('warmup index must be contiguous; freeze it once at the source')
        self.times.flags.writeable=False;self.days.flags.writeable=False
        self.starts=np.r_[0,np.flatnonzero(self.days[1:]!=self.days[:-1])+1]
        self.observed=np.ones(len(self.times),dtype=np.uint8)
        self.signal_maps={}
        self.product_index=product_index

    def required_start(self,plan, *, trading_start_ns:int,boundary_indices=None):
        """Exact structural bound for full windows on this frozen source index.

        This is not a data-value-dependent short-circuit/NaN repair oracle.
        Canonical source ordinals are returned so periodic scopes retain their
        anchor if the caller replays only a suffix. The caller must bind them.
        """
        import numpy as np
        from alpha import BatchPlan
        t=self.times;nodes=plan['nodes'];memo={};periodic=False
        position=int(np.searchsorted(t,int(trading_start_ns),side='left'))
        if position>=len(t):raise ValueError('formal start is outside the observation index')
        def day_start(i,count=1):
            day=int(np.searchsorted(self.starts,i,side='right'))-int(count)
            if day<0:raise ValueError('insufficient warmup trading-day history')
            return int(self.starts[day])
        def visit(n,i):
            nonlocal periodic
            if i<0:raise ValueError('insufficient warmup source Bar history')
            key=(n,i)
            if key in memo:return memo[key]
            node=nodes[n];kind=node['kind'];children=[]
            if kind in {'column','const','product','category','term','boundary'}:result=i
            elif kind=='shift':
                shift=int(node['periods'])
                if shift<0:raise ValueError('future shift cannot have causal warmup')
                result=visit(node['child'],i-shift)
            elif kind in {'rolling','rolling_dynamic','linreg'}:
                if kind=='rolling_dynamic':
                    windows=node['windows']
                    if self.product_index is None and len(windows)>1:raise ValueError('exact dynamic-window warmup requires a product_index')
                    width=int(windows[0 if self.product_index is None else self.product_index])
                else:width=int(node['window'])
                result=visit(node['child'],i-width+1)
            elif kind=='session_rolling':result=visit(node['child'],day_start(i,node['lookback_sessions']))
            elif kind in {'scope','session_reset'}:
                if kind=='scope' and not node.get('sessions'):
                    count=int(node['count'])
                    if count<=0:raise ValueError('exact scope warmup requires a frozen calendar')
                    left=i//count*count;periodic=True
                elif kind=='scope':
                    sessions=node['sessions'];stamp=t[i]
                    scope=next((s for s in sessions if s[0]<=stamp<=s[1]),None)
                    if scope is None:raise ValueError('observation outside frozen scope')
                    left=int(np.searchsorted(t,scope[0],side='left'))
                else:
                    channel=nodes[node['boundary']]['index']
                    name=(plan.get('boundaries') or [None]*(channel+1))[channel]
                    if name=='trading_day_start':left=day_start(i)
                    else:
                        offsets=(boundary_indices or {}).get(channel)
                        if offsets is None:raise ValueError('exact warmup needs frozen boundary indices')
                        at=int(np.searchsorted(offsets,i,side='right'))-1
                        if at<0:raise ValueError('missing scope boundary')
                        left=int(offsets[at])
                result=visit(node['child'],left)
            elif kind=='signal':
                interval=int(node['signal_ns']);sessions=tuple(tuple(s) for s in node.get('sessions',()))
                signature=(interval,node.get('basepoint','last'),node.get('include_current',True),node.get('partial','drop'),sessions)
                if signature not in self.signal_maps:
                    partitions=[[s[0],s[1],s[2],1] for s in sessions] if sessions else [[int(t[0]),int(t[-1]),-(-int(t[0])//interval)*interval,1]]
                    targets,mapping=BatchPlan.signal_grid(t,self.observed,1,partitions,interval,
                        signature[1]=='first',signature[2],signature[3]=='emit')
                    self.signal_maps[signature]=(np.asarray(targets),np.asarray(mapping))
                targets,mapping=self.signal_maps[signature]
                at=int(np.searchsorted(targets,t[i],side='right'))-1
                if at<0 or mapping[at]<0:raise ValueError('insufficient warmup signal history')
                sample=int(mapping[at])
                if sample>i:raise ValueError('signal is not available at the formal start')
                result=visit(node['child'],sample)
            elif kind=='search':
                left=i
                if node.get('lookback') is not None:left=max(0,i-int(node['lookback']))
                if node.get('partition') is not None:
                    left=max(left if node.get('lookback') is not None else 0,i//int(node['partition'])*int(node['partition']));periodic=True
                if node.get('boundary') is not None:
                    channel=nodes[node['boundary']]['index'];name=plan['boundaries'][channel]
                    if name!='trading_day_start':raise ValueError('exact search warmup needs frozen boundary indices')
                    left=max(left if node.get('lookback') is not None else 0,day_start(i))
                if node.get('gap_ns'):raise ValueError('exact search warmup needs frozen gap partitions')
                children=[visit(node['child'],left),*(visit(c,i) for c in node.get('dependencies',()))]
                result=min(children)
            elif kind in {'unary','binary','where','neg','cross'}:
                children=[visit(node[k],i) for k in ('child','left','right','condition','when_true','when_false') if node.get(k) is not None]
                result=min(children,default=i)
            else:raise ValueError(f'exact warmup is unresolved for {kind}')
            if node.get('trunc_start_child') is not None or node.get('trunc_end_child') is not None:raise ValueError('exact warmup needs dynamic truncation bounds')
            memo[key]=result;return result
        earliest=min(visit(n,position) for n in plan['outputs'])
        return dict(required_start_ns=int(t[earliest]),source_ordinal=earliest,
            historical_bars=position-earliest,exact_structural_boundary=True,
            numeric_readiness_verified=False,requires_partition_offset=periodic,
            analyzed_states=len(memo))
