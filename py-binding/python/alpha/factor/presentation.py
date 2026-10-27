"""Pure formula export over a neutral expression view; never executes a factor.

Frontends may supply _presentation_fields() to translate their own field types.
All formatting, precedence, labeling and substitution live here in alpha.
"""
from __future__ import annotations
from functools import partial
from types import SimpleNamespace
from . import latex_formatters as fmt
from .annotations import annotated_nodes, annotation_label, children, is_expression, is_kind


class View:
    def __init__(self, source, session, kind, fields):
        self.source, self.session, self.kind, self.fields = source, session, kind, fields
        self._presentation_kind=kind
        self._key = source._structural_key() if hasattr(source, '_structural_key') else ('scope', id(source))
        if session.scope:
            self._key=(self._key, 'formula-scope', session.scope)

    def _structural_key(self):
        return self._key

    def _to_latex(self, subst=None):
        return self.session.render(self, subst)

    def to_latex(self, subst=None):
        return self._to_latex(subst)

    def __getattr__(self, name):
        if name=='_annotation_scope':
            original=getattr(self.source,'_annotation_scope',())
            return (self.session.scope,*original) if self.session.scope else original
        if name in {'_Ops', '_biOps', '_LATEX', '_LATEX_PRECEDENCE'}:
            return getattr(fmt, name)
        if name == '_PAIR_OPS':
            return {'term_spread','term_ratio','term_log_ratio','term_contango',
                    'term_carry_annualized','term_slope_segment'}
        if name == '_operand_latex_arg':
            return fmt._operand_latex_arg
        if name in {'_expr_precedence','_needs_parenthesis','_binary_operand_latex','_parameter_latex'}:
            return partial(getattr(fmt, name), self)
        value = self.fields[name] if name in self.fields else getattr(self.source, name)
        if isinstance(value, (list,tuple)):
            return tuple(self.session.view(x) if is_expression(x) else x for x in value)
        return self.session.view(value) if is_expression(value) else value


def _native_fields(expr):
    """alpha's public node fields. No imports of a host's Product/Parameter."""
    kind = type(expr).__name__
    fields = {}
    if kind == 'ColumnRef':
        fields={'column':SimpleNamespace(value=expr.column,name=expr.column)}
    elif kind == 'ParamRef':
        fields = {'param': expr.parameter, 'indexed_parameter': type(expr.parameter).__name__ == 'FactorParam'}
        default=expr.parameter.default_value
        if hasattr(default,'expression'):
            fields.update(nested_template=getattr(default,'template',None) or default.expression,
                          nested_bound=default.expression,nested_label=default.alias)
    elif kind == 'RollingOp':
        window = expr.window if is_expression(expr.window) else SimpleNamespace(value=expr.window, _structural_key=lambda: ('window',expr.window))
        fields = {'window': window, '_trunc_start':None, '_trunc_end':None,
                  '_n_data':1, '_data_start':0, 'operands':(expr.data,),
                  'op':'rolling_'+expr.op.split(':')[0],
                  'quantile_expr': expr.op.split(':')[1] if ':' in expr.op else None}
        # Constants use the same formatter without becoming computation nodes.
        if not is_expression(expr.window):
            window._presentation_fields=lambda: ('ConstExpr', {'value':expr.window})
    elif kind == 'CompositeExpr':
        fields = {'operands':expr.operands}
    elif kind == 'ShiftOp':
        period=SimpleNamespace(value=expr.periods, _structural_key=lambda: ('period',expr.periods))
        period._presentation_fields=lambda: ('ConstExpr', {'value':expr.periods})
        fields={'operand':expr.data, 'periods':period}
    elif kind == 'WhereOp':
        kind='OperandExpr';fields={'op':'where','operands':(expr.condition,expr.when_true,expr.when_false)}
    elif kind == 'SignalAlign':
        fields={'operands':(expr.child,)}
    elif kind == 'CategoryBoolRef':
        fields={'category':SimpleNamespace(alias=getattr(expr.category,'alias','membership')),
                'category_name':expr.label}
    elif kind == 'CrossSectionalExpr':
        kind='CrossSectionalOp'
    elif kind == 'GroupByScopeOp':
        fields={'scope':expr.owner.scope,'trunc_start':None,'trunc_end':None}
    elif kind == 'BarCountScope':
        count_value = getattr(expr.count, 'value', expr.count)
        count=SimpleNamespace(value=count_value,_structural_key=lambda: ('count',count_value))
        count._presentation_fields=lambda: ('ConstExpr', {'value':count_value})
        fields={'count':count}
    elif kind in {'BarSinceExpr','BarDistanceExpr','MatchBarDistanceExpr'}:
        kind='BarSinceOp' if kind=='BarSinceExpr' else 'BarDistanceOp'
        fields={'scope':expr.scope}
    elif kind == 'MatchValueRef':
        fields={'role':expr.kind}
    elif kind == 'TermStructureExpr':
        def constant(value):
            x=SimpleNamespace(value=value,display_column_name=str(value),_structural_key=lambda: ('term-arg',value))
            x._presentation_fields=lambda: ('ConstExpr',{'value':value,'display_column_name':str(value)})
            return x
        kind='TermStructureOp'
        pair=expr.op in {'term_spread','term_ratio','term_log_ratio','term_contango','term_carry_annualized','term_slope_segment'}
        args=(expr.near_rank,expr.far_rank,expr.column) if pair else (expr.depth if expr.op!='term_rank_value' else expr.near_rank,expr.column)
        fields={'operands':tuple(constant(x) for x in args)}
    elif kind == 'SessionScope':
        fields={'gap':getattr(expr,'gap','frozen-calendar')}
    return kind,fields


class RenderSession:
    def __init__(self, values=None, scope=""):
        self.views = {}
        self.binding = values is not None
        self.values = values or {}
        self.scope = scope

    def view(self, expression):
        if isinstance(expression,View):
            return expression
        if id(expression) in self.views:
            return self.views[id(expression)]
        hook=getattr(expression,'_presentation_fields',None)
        kind, fields=hook() if hook is not None else _native_fields(expression)
        if kind=='ParamRef':
            param=fields.get('param',getattr(expression,'parameter',None))
            lookup=f'{self.scope}.{param.alias}' if self.scope else param.alias
            if self.binding:
                value=self.values.get(lookup,param.default_value)
                if value is None and 'nested_bound' not in fields:
                    raise ValueError(f'parameter {lookup} has no bound value')
                label=None
                if hasattr(value,'expression'):
                    label=f'{param.alias}.{value.alias}'
                    value=value.expression
                elif 'nested_bound' in fields and lookup not in self.values:
                    value=fields['nested_bound']
                if is_expression(value):
                    result=self.view(value)
                    if label is not None:
                        result.fields.update(_is_intermediate=True,_intermediate_name=label)
                    self.views[id(expression)]=result
                    return result
                if 'bound_column' in fields:
                    kind,fields='ColumnRef',{'column':fields['bound_column'](value)}
                else:
                    kind,fields='ConstExpr',{'value':value}
            elif fields.get('nested_template') is not None:
                label='.'.join(x for x in (self.scope,param.alias,fields.get('nested_label',param.alias)) if x)
                nested=RenderSession(scope=label).view(fields['nested_template'])
                nested.fields['_is_intermediate']=True
                nested.fields['_intermediate_name']=getattr(fields['nested_template'],'_intermediate_name',None) or 'result'
                nested.fields['_bound_expression']=fields.get('nested_bound')
                self.views[id(expression)]=nested
                return nested
            elif self.scope:
                fields['param']=SimpleNamespace(alias=lookup)
        result=View(expression,self,kind,fields)
        self.views[id(expression)]=result
        return result

    def operation(self,op,*operands):
        obj=SimpleNamespace(op=op,operands=operands,_structural_key=lambda: ('display-op',op,id(operands)))
        obj._presentation_fields=lambda: ('CompositeExpr',{'op':op,'operands':operands})
        return self.view(obj)

    def render(self, expression, subst=None):
        node=self.view(expression)
        if subst is not None and node._structural_key() in subst:
            return f'{subst[node._structural_key()]}_t'
        renderer=fmt.RENDERERS.get(node.kind)
        if renderer is None:
            raise TypeError(f'no formula adapter for {node.kind}')
        # Scope fields are view objects as well, not a host's calendar class.
        if 'scope' in node.fields:
            node.fields['scope']=self.view(node.fields['scope'])
        return renderer(node,subst)


def render_expression(expression, subst=None):
    return RenderSession().render(expression,subst)


def symbol(name):
    # Escape text syntax but keep a readable Unicode annotation.
    value=fmt.escape_text(name)
    return ('\\text{' if not str(name).isascii() or ' ' in str(name) else '\\mathrm{')+value+'}'


def export_latex(expression, final_name='X', values=None):
    session=RenderSession(values)
    root=session.view(expression)
    nodes=list(annotated_nodes(root))
    if not nodes:
        return session.render(expression)
    substitutions={};used={};lines=[]
    for number,node in enumerate(nodes,1):
        name=annotation_label(node,number)
        key=node._structural_key()
        if name in used and used[name]!=key:
            raise ValueError(f'Intermediate 名称冲突: {name} 被用于不同表达式。')
        used[name]=key
        if key in substitutions:
            continue
        label=symbol(name)
        lines.append(f'{label}_t &:= {session.render(node,substitutions)},')
        substitutions[key]=label
    if root._structural_key() not in substitutions:
        label=final_name
        n=2
        while label in used:
            label=f'{final_name}_{n}';n+=1
        lines.append(f'{label}_t &:= {session.render(expression,substitutions)}.')
    elif lines:
        lines[-1]=lines[-1].removesuffix(',')+'.'
    return '\\begin{aligned}\n'+' \\\\\n'.join(lines)+'\n\\end{aligned}'




from dataclasses import dataclass

@dataclass(frozen=True)
class LayerFormula:
    name: str
    symbolic: str
    bound: str | None


@dataclass(frozen=True)
class FormulaBundle:
    symbolic: str
    bound: str
    # Each layer has its own formula; no result data is attached here.
    layers: tuple[LayerFormula, ...]


def formula_bundle(expression, *, values=None, bound_expression=None, final_name='X'):
    symbolic=export_latex(expression,final_name=final_name)
    bound=(export_latex(bound_expression,final_name=final_name) if bound_expression is not None
           else export_latex(expression,values={} if values is None else values,final_name=final_name))
    root=RenderSession().view(expression)
    bound_root=RenderSession({} if values is None else values).view(expression) if bound_expression is None else RenderSession().view(bound_expression)
    bound_nodes={annotation_label(node,i):node
                 for i,node in enumerate(annotated_nodes(bound_root),1)}
    layers=[]
    for i,node in enumerate(annotated_nodes(root),1):
        name=annotation_label(node,i)
        concrete=bound_nodes.get(name)
        resolved=concrete.session.render(concrete) if concrete is not None else None
        if resolved is None and node.fields.get('_bound_expression') is not None:
            resolved=RenderSession().render(node.fields['_bound_expression'])
        layers.append(LayerFormula(name,node.session.render(node),resolved))
    return FormulaBundle(symbolic,bound,tuple(layers))
