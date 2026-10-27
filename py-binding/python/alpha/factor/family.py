"""Factor families and versioned factor identities owned by the public API."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from .core import FactorExpr, ParamRef, _to_expr


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return base64.urlsafe_b64encode(hashlib.sha256(encoded.encode("utf-8")).digest()).decode("ascii")[:43]


@dataclass(frozen=True)
class FactorIdentity:
    owner_ref: str
    family_alias: str
    factor_alias: str
    semantic_fingerprint: str

    @property
    def family_ref(self) -> str:
        return "factor-family:v2:" + _digest({
            "family_alias": self.family_alias,
            "family_formula_fingerprint": self.semantic_fingerprint,
            "owner_ref": self.owner_ref,
        })

    @property
    def factor_ref(self) -> str:
        return "factor:v2:" + _digest({
            "factor_alias": self.factor_alias,
            "family_alias": self.family_alias,
            "family_formula_fingerprint": self.semantic_fingerprint,
            "owner_ref": self.owner_ref,
            "self_formula_fingerprint": self.semantic_fingerprint,
        })


@dataclass(frozen=True)
class FactorDefinition:
    alias: str
    expression: FactorExpr
    identity: FactorIdentity
    template: FactorExpr | None = None
    parameter_values: Mapping[str, Any] | None = None

    @property
    def ref(self) -> str:
        return self.identity.factor_ref

    def formulas(self):
        from .presentation import formula_bundle
        return formula_bundle(self.template if self.template is not None else self.expression,
                              bound_expression=self.expression,final_name=self.alias)

    def evaluate(self, ctx):
        return self.expression.evaluate(ctx)


class FactorFamily:
    """Parameterised expression template; FactorExpr is the public surface."""

    alias: ClassVar[str] = ""
    params: ClassVar[tuple[Any, ...]] = ()

    @staticmethod
    def factor_expr() -> FactorExpr:
        raise NotImplementedError

    @classmethod
    def get_factor(cls, *, owner_ref: str = "public", alias: str | None = None,
                   values: Mapping[str, Any] | None = None) -> FactorDefinition:
        from copy import deepcopy
        template = deepcopy(cls.factor_expr())
        expression = _resolve_params(template, values or {})
        family_alias = cls.alias or cls.__name__
        factor_alias = alias or family_alias
        identity = FactorIdentity(
            owner_ref=owner_ref,
            family_alias=family_alias,
            factor_alias=factor_alias,
            semantic_fingerprint=expression.semantic_fingerprint(),
        )
        return FactorDefinition(factor_alias, expression, identity, template, dict(values or {}))


def _resolve_params(expression: FactorExpr, values: Mapping[str, Any], _memo=None) -> FactorExpr:
    from copy import copy
    from dataclasses import replace
    from .annotations import copy_annotation, scoped_annotations
    memo={} if _memo is None else _memo
    if id(expression) in memo:
        return memo[id(expression)]
    if isinstance(expression, ParamRef):
        value=values.get(expression.parameter.alias,expression.parameter.default_value)
        if isinstance(value,FactorDefinition):
            result=scoped_annotations(value.expression,f'{expression.parameter.alias}.{value.alias}')
        else:
            result=expression.resolve(values)
        result=copy_annotation(expression,result)
        memo[id(expression)]=result
        return result
    result=copy(expression)
    memo[id(expression)]=result
    for attribute,child in vars(expression).items():
        if attribute.startswith('_intermediate') or attribute=='_node_annotation':
            continue
        if isinstance(child,FactorExpr):
            setattr(result,attribute,_resolve_params(child,values,memo))
        elif isinstance(child,tuple):
            setattr(result,attribute,tuple(_resolve_params(x,values,memo) if isinstance(x,FactorExpr) else x for x in child))
    if hasattr(result,'window') and isinstance(result.window,FactorExpr):
        from .core import ConstExpr
        if not isinstance(result.window,ConstExpr):
            raise ValueError('rolling window must bind to a constant')
        result.window=int(result.window.value)
        if hasattr(result,'semantics') and hasattr(result.semantics,'min_periods_for'):
            result.semantics=replace(result.semantics,window=result.window)
    return result
