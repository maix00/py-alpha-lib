"""Public formula/annotation/query contracts; no TradeResearch imports."""
import gc
import weakref
from types import SimpleNamespace
import numpy as np
import pytest
from alpha.factor import (ColumnRef, Parameter, ParamRef, FactorFamily, FactorParam,
                          IncrementalFactorPlan, IntermediateIndex, IntermediateResults,
                          compile_factor_outputs)


def test_annotation_does_not_change_plan_or_identity():
    expr=ColumnRef('CLOSE').rolling(3).mean()
    before=(expr.semantic_fingerprint(),expr.to_plan_spec())
    expr.as_intermediate('均值')
    assert before==(expr.semantic_fingerprint(),expr.to_plan_spec())
    assert r'\text{均值}' in expr.to_latex()


def test_nested_symbolic_and_bound_formulas_have_lexical_parameters():
    n=Parameter('N',3)
    class Inner(FactorFamily):
        alias='Inner'
        @staticmethod
        def factor_expr():
            return ColumnRef('CLOSE').rolling(n).mean().as_intermediate('inside')
    original=Inner.factor_expr()
    class SharedInner(Inner):
        @staticmethod
        def factor_expr():return original
    inner=SharedInner.get_factor(values={'N':2})
    parameter=FactorParam('P',inner)
    class Outer(FactorFamily):
        alias='Outer'
        @staticmethod
        def factor_expr():
            return (ParamRef(parameter)+ParamRef(n)).as_intermediate('outside')
    factor=Outer.get_factor(values={'N':7})
    formula=factor.formulas()
    assert 'Inner.N' in formula.symbolic
    assert r'\textcolor{red}{N}' in formula.symbolic
    assert r'_{2}' in formula.bound and '7.0' in formula.bound
    assert r'\textcolor{red}' not in formula.bound
    assert [layer.name for layer in formula.layers]==['P.Inner.inside','outside']
    assert all(layer.bound is not None for layer in formula.layers)
    assert isinstance(original.window,ParamRef)
    assert n.default_value==3
    assert SharedInner.get_factor(values={'N':4}).expression.window==4
    plan=IncrementalFactorPlan({'outer':factor.expression},('A',),intermediates='all')
    assert plan.update_at(1,[10])['outer']['A']==17
    np.testing.assert_equal(plan.get_intermediate('P.Inner.inside'),[10])
    assert factor.identity.semantic_fingerprint==factor.expression.semantic_fingerprint()


def test_latest_lookup_reuses_computation_and_releases_prior_frames():
    inner=ColumnRef('CLOSE').rolling(2).semantics(min_periods=1).mean().as_intermediate('mean')
    outer=inner+inner
    default=IncrementalFactorPlan({'root':outer},('A',))
    selected=IncrementalFactorPlan({'root':outer},('A',),intermediates=('mean',))
    assert default.node_count==selected.node_count==3
    assert default.get_intermediate('mean') is None
    selected.update_at(1,[2])
    previous=selected.get_intermediate('mean')
    ref=weakref.ref(previous)
    assert selected.get_intermediate('mean') is previous
    del previous
    for t in range(2,1002):
        result=selected.update_at(t,[t])
    gc.collect()
    assert ref() is None
    assert len(selected._intermediate_results.values)==1
    assert selected._intermediate_results.timestamp_ns==1001
    assert result['root']['A']==2001
    assert selected.get_intermediate('mean').nbytes==8
    assert selected.get_intermediate('missing') is None
    spec,_,_=compile_factor_outputs({'root':outer},('A',),intermediates=('mean',))
    assert len(spec['nodes'])==3 and spec['intermediates']=={'mean':1}


def test_batch_lookup_is_explicit_borrowed_and_never_evaluates():
    first=ColumnRef('CLOSE').as_intermediate('a')
    second=ColumnRef('OPEN').as_intermediate('b')
    index=IntermediateIndex(SimpleNamespace(operands=(first,second)))
    a=np.zeros((200,3));b=np.ones((200,3))
    cache={first._structural_key():a,second._structural_key():b}
    empty=IntermediateResults(index)
    empty.replace(cache)
    assert empty.values=={}
    selected=IntermediateResults(index,('a',))
    selected.replace(cache)
    assert selected.get('a') is a
    assert selected.get('b') is None
    selected.clear()
    assert selected.get('a') is None


def test_conflicting_names_fail_without_evaluating():
    a=ColumnRef('CLOSE').as_intermediate('same')
    b=ColumnRef('OPEN').as_intermediate('same')
    root=a+b
    with pytest.raises(ValueError,match='冲突'):
        root.to_latex()
    with pytest.raises(ValueError,match='冲突'):
        IntermediateIndex(root)


def test_parameter_value_export_does_not_mutate_template():
    n=Parameter('n',2)
    expr=(ColumnRef('CLOSE')+ParamRef(n)).as_intermediate('X_1')
    original=expr._structural_key()
    assert r'\textcolor{red}{n}' in expr.to_latex()
    assert '9' in expr.to_latex(values={'n':9})
    assert n.default_value==2 and original==expr._structural_key()
    assert r'X\_1' in expr.to_latex()


def test_annotation_origin_does_not_retain_a_factor_or_its_data():
    class Origin:
        alias='origin'
        def __init__(self):
            self.alias='origin'
            self.factor_ref='factor:v2:example'
            self.history=np.zeros((1000,20))
    origin=Origin(); reference=weakref.ref(origin)
    expr=ColumnRef('CLOSE').as_intermediate('value',factor=origin)
    del origin
    gc.collect()
    assert reference() is None
    assert expr._node_annotation.source_alias=='origin'
    assert expr._node_annotation.source_ref=='factor:v2:example'
    assert expr._intermediate_factor is None


def test_public_operator_formulas_cover_nested_non_arithmetic_nodes():
    from alpha.factor import (SignalAlign, scope_bars, term_spread, CURRENT, CANDIDATE,
                              bar_distance, CategoryBoolRef)
    column=ColumnRef('CLOSE')
    expressions=[column.shift(2),column.where(column>1,0),SignalAlign(column,'5min'),
                 column.groupby_scope(scope_bars(3)).mean(),term_spread(),
                 bar_distance(column,CANDIDATE>CURRENT,scope=scope_bars(3))]
    for expression in expressions:
        assert expression.to_latex()


def test_two_instances_of_one_nested_family_have_distinct_binding_scopes():
    n=Parameter('N',2)
    class Inner(FactorFamily):
        alias='Inner'
        @staticmethod
        def factor_expr():return ColumnRef('CLOSE').rolling(n).mean().as_intermediate('mean')
    a=FactorParam('left',Inner.get_factor(values={'N':2}))
    b=FactorParam('right',Inner.get_factor(values={'N':4}))
    class Outer(FactorFamily):
        @staticmethod
        def factor_expr():return ParamRef(a)+ParamRef(b)
    factor=Outer.get_factor()
    formula=factor.formulas()
    assert 'left.Inner.N' in formula.symbolic and 'right.Inner.N' in formula.symbolic
    assert [x.name for x in formula.layers]==['left.Inner.mean','right.Inner.mean']
    assert r'_{2}' in formula.layers[0].bound and r'_{4}' in formula.layers[1].bound
    plan=IncrementalFactorPlan({'root':factor.expression},('A',),intermediates='all')
    assert plan.node_count==4
    assert plan.update_at(1,[10])['root']['A']==20


def test_symbolic_rolling_window_requires_binding_only_for_execution():
    expression=ColumnRef('CLOSE').rolling(Parameter('N',3)).mean()
    assert r'\textcolor{red}{N}' in expression.to_latex()
    with pytest.raises(ValueError,match='Bind rolling window'):
        expression.to_plan_spec()
