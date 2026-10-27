"""Public matrix and incremental entries must execute the same Rust state."""
import numpy as np
import pandas as pd
import pytest
from alpha.factor import (CategoryBoolRef,CategoryMembership,ColumnRef,EvalContext,
    IncrementalFactorPlan,CURRENT,CANDIDATE,bar_distance,scope_bars,plan_blockers)
from alpha.factor.bar_search import BarDistanceExpr


def test_category_latest_snapshot_removes_labels_and_never_uses_future():
    times=pd.to_datetime([-10,0,10,20],unit='ns')
    expr=CategoryBoolRef(label='metal')
    snapshots={'A':(CategoryMembership('A',frozenset({'metal'})),
                    CategoryMembership('A',frozenset(),10),
                    CategoryMembership('A',frozenset({'metal'}),30)),
               'B':frozenset()}
    for end in range(1,5):
        ctx=EvalContext(('A','B'),preloaded={p:pd.DataFrame({'CLOSE':range(end)},index=times[:end]) for p in ('A','B')},categories=snapshots)
        batch=expr.evaluate(ctx)
        np.testing.assert_equal(batch['A'].to_numpy(),[1,1,0,0][:end])
        state=IncrementalFactorPlan({'c':expr},('A','B'),categories=snapshots)
        for t,row in zip(times[:end],batch.to_numpy()):
            actual=state.update_at(t.value,[0,0])['c']
            np.testing.assert_equal(list(actual.values()),row)
    assert plan_blockers(expr)==()


def test_category_conflicting_same_time_snapshot_is_not_order_dependent():
    expr=CategoryBoolRef(label='metal')
    categories={'A':(CategoryMembership('A',frozenset(),0),CategoryMembership('A',frozenset({'metal'}),0))}
    with pytest.raises(ValueError,match='conflicting'):
        expr.to_plan_spec(products=('A',),categories=categories)


@pytest.mark.parametrize('nearest',[True,False])
@pytest.mark.parametrize('inclusive',[True,False])
def test_match_distance_every_prefix_has_scalar_oracle(nearest,inclusive):
    times=pd.date_range('2026-01-01',periods=8,freq='s')
    values=[3.,1.,4.,2.,5.,5.,0.,6.]
    expression=bar_distance(ColumnRef('CLOSE'),CANDIDATE<=CURRENT,scope=scope_bars(3),
        select='nearest' if nearest else 'farthest',default=9,include_current=inclusive)
    expected=[]
    for i,current in enumerate(values):
        candidates=[i-j for j in range(max(0,i-3),i+int(inclusive)) if values[j]<=current]
        expected.append((min(candidates) if nearest else max(candidates)) if candidates else 9)
    for end in range(1,len(times)+1):
        frame=expression.evaluate(EvalContext(('A',),preloaded={'A':pd.DataFrame({'CLOSE':values[:end]},index=times[:end])}))
        np.testing.assert_equal(frame['A'].to_numpy(),expected[:end])
        state=IncrementalFactorPlan({'d':expression},('A',))
        for t,value,want in zip(times[:end],values[:end],expected):
            assert state.update_at(t.value,[value])['d']['A']==want
    assert plan_blockers(expression)==()


def test_legacy_boolean_distance_is_explicit_event_age_not_candidate_search():
    times=pd.date_range('2026-01-01',periods=5,freq='s');values=[1.,0.,0.,1.,0.]
    expression=BarDistanceExpr(ColumnRef('CLOSE'),scope_bars(10))
    result=expression.evaluate(EvalContext(('A',),preloaded={'A':pd.DataFrame({'CLOSE':values},index=times)}))
    np.testing.assert_equal(result['A'].to_numpy(),[0,1,2,0,1])
    state=IncrementalFactorPlan({'d':expression},('A',))
    for value,want in zip(values,[0,1,2,0,1]):assert state.update([value])['d']['A']==want
    assert plan_blockers(expression)==()


def test_sparse_match_counts_actual_product_bars_not_union_padding():
    times=pd.date_range('2026-01-01',periods=5,freq='s')
    expr=bar_distance(ColumnRef('CLOSE'),CANDIDATE<CURRENT,scope=scope_bars(2),default=8,include_current=False)
    ctx=EvalContext(('A','B'),preloaded={'A':pd.DataFrame({'CLOSE':range(5)},index=times),
        'B':pd.DataFrame({'CLOSE':[1,4]},index=times[[0,4]])})
    result=expr.evaluate(ctx)
    np.testing.assert_equal(result['B'].to_numpy(),[8,np.nan,np.nan,np.nan,1])


def test_nan_default_and_negated_match_predicate_share_the_kernel():
    times=pd.date_range('2026-01-01',periods=3,freq='s');values=[2.,1.,3.]
    expr=bar_distance(ColumnRef('CLOSE'),-CANDIDATE < -CURRENT,scope=scope_bars(2),default=float('nan'))
    batch=expr.evaluate(EvalContext(('A',),preloaded={'A':pd.DataFrame({'CLOSE':values},index=times)}))
    np.testing.assert_equal(batch['A'].to_numpy(),[np.nan,1,np.nan])
    state=IncrementalFactorPlan({'d':expr},('A',))
    for value,want in zip(values,[np.nan,1,np.nan]):np.testing.assert_equal(state.update([value])['d']['A'],want)
