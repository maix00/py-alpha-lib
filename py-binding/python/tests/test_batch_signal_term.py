"""Causal public batch contracts through the real Rust matrix core."""
import math
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from alpha.factor import ColumnRef, EvalContext, IncrementalFactorPlan, SignalAlign, term_spread, term_slope, term_curvature
from alpha.factor.inputs import TermCurve, TermPoint
from alpha.factor.plan import compile_plan


def context(times,values):
    return EvalContext(products=('A',),preloaded={'A':pd.DataFrame({'CLOSE':values},index=times)})


@pytest.mark.parametrize('inclusive',[True,False])
def test_signal_gap_each_prefix_and_streaming(inclusive):
    # 11 reveals missed targets 5/10; it cannot be used at either earlier target.
    times=pd.to_datetime([1,4,11,12,15,16],unit='s')
    values=[10.,40.,110.,120.,150.,160.]
    expression=SignalAlign(ColumnRef('CLOSE'),'5s',include_current=inclusive)
    for n in range(1,len(times)+1):
        result=expression.evaluate(context(times[:n],values[:n]))
        for target,actual in result['A'].items():
            eligible=[v for t,v in zip(times[:n],values[:n]) if t<target or inclusive and t==target]
            expected=eligible[-1] if eligible else math.nan
            np.testing.assert_equal(actual,expected)
        plan=IncrementalFactorPlan({'signal':expression},('A',))
        for t,v in zip(times[:n],values[:n]): latest=plan.update_at(t.value,[v])['signal']['A']
        if len(result):np.testing.assert_equal(latest,result.iloc[-1,0])


def test_nested_pointwise_values_forward_fill_only_from_inner_availability():
    times=pd.date_range('2026-01-05 09:30',periods=12,freq='min')
    close=ColumnRef('CLOSE')
    expression=(close.signal_align('2min')+close).signal_align('4min')
    for n in range(1,len(times)+1):
        frame=expression.evaluate(context(times[:n],list(range(n))))
        for t,actual in frame['A'].items():
            r=int((t-times[0]).total_seconds()/60)
            inner=r-r%2
            assert actual==inner+r


def test_term_snapshots_are_mapped_by_availability_not_future_history():
    times=pd.to_datetime([1,2,3,4,5],unit='s')
    early=TermCurve('A',(TermPoint({'CLOSE':10},10,rank=0),TermPoint({'CLOSE':8},20,rank=1)),times[1].value)
    late=TermCurve('A',(TermPoint({'CLOSE':100},10,rank=0),TermPoint({'CLOSE':10},20,rank=1)),times[3].value)
    expression=term_spread()
    for n in range(1,len(times)+1):
        ctx=replace(context(times[:n],range(n)),term_curves={'A':(late,early)})
        frame=expression.evaluate(ctx)
        np.testing.assert_equal(frame['A'].to_numpy(),[np.nan,2,2,90,90][:n])


def test_curve_metrics_use_rust_once_and_share_packed_input():
    times=pd.date_range('2026-01-01',periods=4,freq='min')
    ctx=context(times,range(4))
    ctx=replace(ctx,term_curves={'A':pd.DataFrame({'CLOSE':[1.,4.,9.],'DAYS_TO_MATURITY':[1.,2.,3.]})})
    spread=term_spread();slope=term_slope(depth=3);curve=term_curvature(depth=3)
    frames=compile_plan([spread,slope,curve],ctx)
    for frame,expected in zip(frames,[-3,4,1]):np.testing.assert_allclose(frame,expected,atol=1e-12)


def test_deleted_python_math_cannot_be_used(monkeypatch):
    monkeypatch.setattr(np,'polyfit',lambda *a,**k:pytest.fail('Python polynomial fit'))
    times=pd.date_range('2026-01-01',periods=3,freq='min')
    ctx=replace(context(times,[1,2,3]),term_curves={'A':pd.DataFrame({'CLOSE':[1,4,9],'DAYS_TO_MATURITY':[1,2,3]})})
    assert term_curvature(depth=3).evaluate(ctx).iloc[0,0]==pytest.approx(1)
    assert not hasattr(SignalAlign,'_target_times')


@pytest.mark.parametrize('inclusive',[True,False])
def test_first_sampling_does_not_reuse_an_older_empty_bucket(inclusive):
    times=pd.to_datetime([1,4,11,12,15,16],unit='s');values=[10.,40.,110.,120.,150.,160.]
    expression=SignalAlign(ColumnRef('CLOSE'),'5s',basepoint='first',include_current=inclusive)
    result=expression.evaluate(context(times,values))
    previous=times[0]-pd.Timedelta('1ns')
    for t,actual in result['A'].items():
        eligible=[v for source,v in zip(times,values) if previous<source and (source<t or inclusive and source==t)]
        np.testing.assert_equal(actual,eligible[0] if eligible else np.nan);previous=t
    stream=IncrementalFactorPlan({'s':expression},('A',))
    for t,v in zip(times,values):
        value=stream.update_at(t.value,[v])['s']['A']
        targets=result.loc[:t]
        if len(targets):np.testing.assert_equal(value,targets.iloc[-1,0])


def test_sparse_product_is_sampled_from_its_observed_rows_only():
    times=pd.to_datetime([0,1,2,3,4],unit='s')
    ctx=EvalContext(products=('A','B'),preloaded={
        'A':pd.DataFrame({'CLOSE':[10,20,30,40,50]},index=times),
        'B':pd.DataFrame({'CLOSE':[5,9]},index=times[[1,3]])})
    result=ColumnRef('CLOSE').signal_align('2s').evaluate(ctx)
    np.testing.assert_equal(result['B'].to_numpy(),[np.nan,5,9])


def test_frozen_end_anchor_has_same_batch_and_streaming_phase():
    from alpha.factor import TradingSession,scope_trading_day
    times=pd.to_datetime([0,100,200,250],unit='ns')
    expression=ColumnRef('CLOSE').signal_align('100ns',scope=scope_trading_day((TradingSession(0,250),)),scope_anchor='end')
    for n in range(1,5):
        result=expression.evaluate(context(times[:n],list(range(n))))
        state=IncrementalFactorPlan({'signal':expression},('A',))
        for t,v in zip(times[:n],range(n)):last=state.update_at(t.value,[v])['signal']['A']
        if len(result):np.testing.assert_equal(last,result.iloc[-1,0])
    np.testing.assert_equal(result.index.as_unit('ns').asi8,[50,150,250])
    np.testing.assert_equal(result['A'].to_numpy(),[0,1,3])


def test_multi_universe_shares_pre_cross_state_but_not_cross_results():
    import json
    from alpha import FactorPlanState
    from alpha.factor.spec import scope_plan_outputs
    mean=ColumnRef('CLOSE').rolling(2).semantics(min_periods=1).mean()
    expression=mean.cs_rank()
    original=expression.to_plan_spec(products=('A','B','C'))
    spec,slots=scope_plan_outputs(original,[(0,(0,1)),(0,(1,2)),(0,(0,1))],3)
    assert slots==[0,1,0]
    assert sum(n['kind']=='column' for n in spec['nodes'])==1
    assert sum(n['kind']=='rolling' for n in spec['nodes'])==1
    assert sum(n['kind']=='cross' for n in spec['nodes'])==2
    state=FactorPlanState(json.dumps(spec),3)
    np.testing.assert_equal(np.asarray(state.update_at(1,np.array([1.,2.,9.]))).reshape(2,3),
                            [[0,.5,np.nan],[np.nan,0,.5]])
    assert len(original['nodes'])==3


def test_term_invalid_parameters_and_huge_out_of_range_rank_do_not_allocate(monkeypatch):
    from alpha.factor import term_rank_value
    times=pd.date_range('2026-01-01',periods=2,freq='s')
    ctx=replace(context(times,[1,2]),term_curves={'A':pd.DataFrame({'CLOSE':[1,2]})})
    assert np.isnan(term_rank_value(rank=10**12).evaluate(ctx).iloc[0,0])
    with pytest.raises(ValueError,match='nonnegative'):term_rank_value(rank=-1)
