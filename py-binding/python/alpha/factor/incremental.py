"""Incremental FactorPlan backed by the same Rust kernel as the bridge."""

from __future__ import annotations

import json

import numpy as np


class IncrementalFactorPlan:
    def __init__(self, expressions: dict[str, object], products: tuple[str, ...], *, categories=None, intermediates=()):
        self.aliases = tuple(expressions)
        self.products = tuple(products)
        from .results import compile_factor_outputs, IntermediateResults
        spec,index,selected=compile_factor_outputs(expressions,self.products,
            categories=categories,intermediates=intermediates)
        self._intermediate_index=index
        self._intermediate_results=IntermediateResults(index,selected)
        from alpha import FactorPlanState
        self._state = FactorPlanState(json.dumps(spec), len(self.products))
        self._outputs = tuple(spec["outputs"])
        self.node_count = len(spec['nodes'])
        self._output_count = len(self._outputs)

    def get_intermediate(self, key):
        return self._intermediate_results.get(key)

    def _decode(self, flat, timestamp_ns=None):
        matrix=flat.reshape(self._output_count,len(self.products))
        selected=self._intermediate_results.selected
        offset=len(self.aliases)
        # No second rolling state or expression evaluation; views of one frame.
        if selected:
            self._intermediate_results.replace(
                {key:matrix[offset+i] for i,key in enumerate(selected)}, timestamp_ns=timestamp_ns)
        return {alias:dict(zip(self.products,matrix[i],strict=True))
                for i,alias in enumerate(self.aliases)}

    def update_with_curves(self, timestamp_ns: int, values, curves) -> dict[str, dict[str, float]]:
        array = np.asarray(values, dtype=float)
        flat = np.asarray(self._state.update_with_curves(
            int(timestamp_ns), np.ascontiguousarray(array), curves,
        ), dtype=float)
        return self._decode(flat, int(timestamp_ns))

    def update_at(self, timestamp_ns: int, values) -> dict[str, dict[str, float]]:
        array = np.asarray(values, dtype=float)
        flat = np.asarray(self._state.update_at(int(timestamp_ns), np.ascontiguousarray(array)), dtype=float)
        return self._decode(flat, int(timestamp_ns))

    def update(self, values: list[float] | np.ndarray) -> dict[str, dict[str, float]]:
        array = np.asarray(values, dtype=float)
        if array.shape != (len(self.products),):
            raise ValueError("incremental plan requires one value per product")
        flat = np.asarray(self._state.update(np.ascontiguousarray(array)), dtype=float)
        return self._decode(flat)
