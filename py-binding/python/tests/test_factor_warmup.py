import pytest
from alpha.factor import required_history_start,warmup_requirements


def test_nested_history_is_composed_and_shared_not_maximum_window_only():
    plan={'nodes':[{'kind':'column','index':0},{'kind':'shift','child':0,'periods':1},
        {'kind':'session_rolling','child':1,'lookback_sessions':250},
        {'kind':'signal','child':2,'signal_ns':30},
        {'kind':'session_rolling','child':3,'lookback_sessions':20},
        {'kind':'where','condition':0,'when_true':4,'when_false':4}], 'outputs':[5,5]}
    result=warmup_requirements(plan)
    assert result['combined']['trading_days']==270
    assert result['combined']['source_bars']==1
    assert result['combined']['elapsed_ns']==30
    assert result['node_count']==6 and result['outputs'][0]==result['outputs'][1]
    sessions=[(100*i,100*i+90,10) for i in range(300)]
    assert required_history_start(result,trading_start_ns=30000,sessions=sessions)==2800
    with pytest.raises(ValueError,match='insufficient warmup'):
        required_history_start(result,trading_start_ns=18400,sessions=sessions[:184])


def test_unknown_and_future_dependencies_never_mean_zero_warmup():
    result=warmup_requirements({'nodes':[{'kind':'column','index':0},{'kind':'shift','child':0,'periods':-1}], 'outputs':[1]})
    assert result['combined']['blockers']==['future_shift']
    with pytest.raises(ValueError,match='unresolved'):
        required_history_start(result,trading_start_ns=10,sessions=[])


def test_public_expression_and_partial_windows_still_report_full_history():
    from alpha.factor import ColumnRef
    expr=ColumnRef('CLOSE').rolling(20).semantics(min_periods=1).mean()
    assert warmup_requirements(expr)['combined']['source_bars']==19


def test_exact_structural_boundary_uses_frozen_observations_and_reuses_signal_map():
    import numpy as np
    from alpha.factor import WarmupTimeline
    times=np.array([60,120,180,300,360,420,480],dtype=np.int64)
    index=WarmupTimeline(times,[0,0,0,1,1,1,1])
    assert np.shares_memory(index.times,times)
    plan={'nodes':[{'kind':'column','index':0},{'kind':'shift','child':0,'periods':1},
        {'kind':'rolling','child':1,'window':2}], 'outputs':[2]}
    result=index.required_start(plan,trading_start_ns=360)
    assert result['required_start_ns']==180 and result['historical_bars']==2
    signal={'nodes':[{'kind':'column','index':0},{'kind':'signal','child':0,'signal_ns':180}], 'outputs':[1]}
    index.required_start(signal,trading_start_ns=420)
    cached=next(iter(index.signal_maps.values()))
    index.required_start(signal,trading_start_ns=480)
    assert next(iter(index.signal_maps.values())) is cached


def test_conservative_suffix_does_not_change_formal_values_or_leave_warmup_nans():
    import json,numpy as np
    from alpha import FactorPlanState
    from alpha.factor import required_history_start,warmup_requirements
    minute=60000000000;day=86400000000000
    times=np.array([d*day+(571+i)*minute for d in range(10) for i in range(10)],dtype=np.int64)
    sessions=[[int(times[d*10])-minute,int(times[d*10+9]),10] for d in range(10)]
    spec={'nodes':[{'kind':'column','index':0},{'kind':'shift','child':0,'periods':1},
        {'kind':'session_rolling','child':1,'lookback_sessions':3,'sessions':sessions,'op':'mean','min_periods':1,'propagate_nan':True},
        {'kind':'session_rolling','child':2,'lookback_sessions':2,'sessions':sessions,'op':'mean','min_periods':1,'propagate_nan':True}], 'outputs':[3]}
    formal=70;bound=required_history_start(warmup_requirements(spec),trading_start_ns=int(times[formal]),sessions=sessions)
    first=int(np.searchsorted(times,bound));values=np.arange(100,dtype=float)
    def replay(start):
        state=FactorPlanState(json.dumps(spec),1)
        return np.array([state.update_at(int(t),np.array([v]))[0] for t,v in zip(times[start:],values[start:])])
    full=replay(0)[formal:];suffix=replay(first)[formal-first:]
    assert np.isfinite(suffix).all()
    np.testing.assert_allclose(suffix,full,rtol=0,atol=1e-12)


def test_public_scope_quantile_uses_the_shared_kernel_and_keeps_quantile():
    import json,numpy as np
    from alpha import FactorPlanState
    from alpha.factor import ColumnRef,scope_bars,to_plan_spec
    expr=ColumnRef('CLOSE').groupby_scope(scope_bars(3)).quantile(.8)
    spec=to_plan_spec([expr]);assert spec['nodes'][-1]['quantile']==.8
    assert warmup_requirements(expr)['combined']['source_bars']==2
    state=FactorPlanState(json.dumps(spec),1)
    actual=[state.update(np.array([v]))[0] for v in [1.,2.,3.,10.,20.,30.]]
    np.testing.assert_allclose(actual,[1,1.8,2.6,10,18,26])
