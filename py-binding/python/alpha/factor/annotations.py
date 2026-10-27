"""Non-computational node annotations, shared by expression frontends."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class NodeAnnotation:
    name: str | None = None
    source_alias: str | None = None
    source_ref: str | None = None
    scope: tuple[str, ...] = ()


def is_expression(value):
    return hasattr(value,'_structural_key')


def is_kind(value,names):
    if isinstance(names,str):
        names=(names,)
    return getattr(value,'_presentation_kind',type(value).__name__) in names


def children(expression):
    """Walk expression references, never evaluate or inspect result arrays."""
    seen = set()
    for field in ('_operands', 'operands', 'data', 'child', 'operand', 'condition',
                  'when_true', 'when_false', 'value', 'window', 'periods', 'owner'):
        value = getattr(expression, field, None)
        for item in value if isinstance(value, (tuple, list)) else (value,):
            if is_expression(item) and id(item) not in seen:
                seen.add(id(item))
                yield item


def annotated_nodes(expression):
    seen = set()
    def visit(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        for child in children(node):
            yield from visit(child)
        if getattr(node, '_is_intermediate', False):
            yield node
    return visit(expression)


class ExpressionAnnotations:
    """An annotation never changes semantic identity or schedules computation."""
    _is_intermediate = False
    _intermediate_name = None
    _intermediate_factor = None

    @property
    def _annotation_scope(self):
        return getattr(self,"_node_annotation",NodeAnnotation()).scope

    def as_intermediate(self, name=None, factor=None):
        self._is_intermediate = True
        self._intermediate_name = name
        # An origin is a label/reference, never an owning Factor/Family with
        # runtime caches (nor a self-cycle).
        origin=vars(factor) if factor is not None and hasattr(factor,'__dict__') else {}
        self._node_annotation = NodeAnnotation(name,
            str(origin.get('alias')) if origin.get('alias') is not None else None,
            str(origin.get('factor_ref')) if origin.get('factor_ref') is not None else None)
        vars(self).pop('_intermediate_factor',None)
        return self

    def iter_intermediate_nodes(self):
        return annotated_nodes(self)

    def _to_latex(self, subst=None):
        from .presentation import render_expression
        return render_expression(self, subst=subst)

    def to_latex(self, final_name='X', *, values=None):
        from .presentation import export_latex
        return export_latex(self, final_name=final_name, values=values)


def copy_annotation(source, target):
    if getattr(source, '_is_intermediate', False):
        target.as_intermediate(getattr(source, '_intermediate_name', None))
        target._node_annotation=getattr(source,'_node_annotation',target._node_annotation)
    return target


def annotation_label(node, ordinal):
    name=getattr(node,'_intermediate_name',None) or f'I{ordinal}'
    scope=getattr(node,'_annotation_scope',())
    return '.'.join((*scope,str(name)))


def scoped_annotations(expression, prefix):
    """Copy only DAG objects/metadata to bind a nested definition's namespace.

    Arrays, parameters and immutable payloads are borrowed, never duplicated.
    A memo preserves shared subgraphs; numeric node identities are unchanged.
    """
    from dataclasses import replace
    memo={}
    def clone(node):
        if id(node) in memo:
            return memo[id(node)]
        # Compatibility nodes may require operands in __new__. This is a DAG
        # metadata clone, not construction/evaluation of a new expression.
        result=object.__new__(type(node))
        vars(result).update(vars(node))
        memo[id(node)]=result
        for name,value in vars(node).items():
            if name.startswith('_intermediate') or name=='_node_annotation':
                continue
            if isinstance(value,ExpressionAnnotations):
                setattr(result,name,clone(value))
            elif isinstance(value,tuple):
                setattr(result,name,tuple(clone(x) if isinstance(x,ExpressionAnnotations) else x for x in value))
        if getattr(result,'_is_intermediate',False):
            annotation=getattr(result,'_node_annotation',NodeAnnotation(result._intermediate_name))
            result._node_annotation=replace(annotation,scope=(prefix,*annotation.scope))
        return result
    result=clone(expression)
    if not getattr(result,'_is_intermediate',False):
        result.as_intermediate('result')
        from dataclasses import replace
        result._node_annotation=replace(result._node_annotation,scope=(prefix,))
    return result
