"""Borrowed intermediate-result lookup; no evaluator and no history accumulator."""
from __future__ import annotations
from .annotations import annotated_nodes, annotation_label


class IntermediateIndex:
    def __init__(self, expression):
        self.nodes = {}
        self.aliases = {}
        for ordinal,node in enumerate(annotated_nodes(expression),1):
            key=node._structural_key()
            name=annotation_label(node,ordinal)
            if name in self.aliases and self.aliases[name]!=key:
                raise ValueError(f'Intermediate 名称冲突: {name} 被用于不同表达式。')
            self.aliases[name]=key
            self.nodes.setdefault(key,node)

    def select(self, requested=()):
        if requested=='all':
            return tuple(self.nodes)
        if isinstance(requested,str):
            requested=(requested,)
        keys=[]
        for item in requested:
            key=self.aliases.get(item) if isinstance(item,str) else item
            if key not in self.nodes:
                raise KeyError(f'unknown intermediate {item!r}')
            if key not in keys:
                keys.append(key)
        return tuple(keys)


class IntermediateResults:
    """One batch or one latest frame, only for explicitly selected outputs.

    Replacing a frame releases prior references. Full batch values may be
    supplied explicitly; this object never evaluates or appends observations.
    """
    def __init__(self,index,requested=()):
        self.index=index
        self.selected=index.select(requested)
        self.values={}
        self.timestamp_ns=None

    def replace(self, values, *, timestamp_ns=None):
        self.values={key:values[key] for key in self.selected if key in values}
        self.timestamp_ns=timestamp_ns

    def get(self,key):
        if isinstance(key,str):
            key=self.index.aliases.get(key)
        return self.values.get(key)

    def clear(self):
        self.values.clear()
        self.timestamp_ns=None


def compile_factor_outputs(expressions, products, *, categories=None, intermediates=()):
    """Select extra outputs in ONE shared Plan; selection adds no math nodes."""
    from types import SimpleNamespace
    from .spec import to_plan_spec
    index=IntermediateIndex(SimpleNamespace(operands=tuple(expressions.values())))
    selected=index.select(intermediates)
    nodes=[index.nodes[key] for key in selected]
    spec=to_plan_spec([*expressions.values(),*nodes],products=tuple(products),categories=categories)
    if spec is None:
        raise ValueError('factor outputs cannot be compiled to the Rust plan')
    offsets={key:len(expressions)+i for i,key in enumerate(selected)}
    spec['intermediates']={name:offsets[key] for name,key in index.aliases.items() if key in offsets}
    return spec,index,selected
