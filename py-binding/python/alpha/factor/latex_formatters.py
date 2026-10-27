"""Single LaTeX operator formatter implementation; no host-domain imports."""
from __future__ import annotations
from typing import Any
import pandas as pd
from .annotations import is_expression, is_kind

_LATEX_PRECEDENCE = {'or': 10, 'and': 20, 'gt': 30, 'lt': 30, 'ge': 30, 'le': 30, 'eq': 30, 'ne': 30, 'add': 40, 'sub': 40, 'mul': 50, 'div': 50, 'pow': 60, 'neg': 70, 'abs': 70, 'not': 70, 'log': 70, 'sign': 70, 'sqrt': 70, 'tanh': 70, 'max': 80, 'min': 80}
_LATEX = {'term_spread': '\\mathrm{TermSpread}', 'term_ratio': '\\mathrm{TermRatio}', 'term_slope': '\\mathrm{TermSlope}', 'term_log_ratio': '\\mathrm{TermLogRatio}', 'term_contango': '\\mathrm{TermContango}', 'term_carry_annualized': '\\mathrm{TermCarryAnn}', 'term_curvature': '\\mathrm{TermCurvature}', 'term_slope_segment': '\\mathrm{TermSlopeSeg}', 'term_rank_value': '\\mathrm{TermRankValue}'}

_LATEX_TIME_UNITS = {'days': '\\mathrm{d}', 'hours': '\\mathrm{h}', 'minutes': '\\mathrm{min}', 'seconds': '\\mathrm{s}', 'milliseconds': '\\mathrm{ms}', 'microseconds': '\\mu\\mathrm{s}', 'nanoseconds': '\\mathrm{ns}'}

def escape_text(value):
    escapes={'\\':r'\textbackslash{}','{':r'\{','}':r'\}',
             '_':r'\_','%':r'\%','#':r'\#','$':r'\$','&':r'\&'}
    return ''.join(escapes.get(char,char) for char in str(value))


def _format_latex_value(value: Any) -> str:
    """Format scalar expression values without leaking Python repr syntax.

    ``str(pd.Timedelta)`` is a human-readable Python value such as
    ``25 days 00:00:00``.  In a math environment that becomes a sequence of
    implicit variables and punctuation.  Keep the canonical duration while
    making every unit an upright LaTeX operator (``25\\,\\mathrm{d}``).
    """
    if isinstance(value, pd.Timedelta):
        sign = '-' if value < pd.Timedelta(0) else ''
        components = abs(value).components
        parts = [f'{int(getattr(components, name))}\\,{unit}' for name, unit in _LATEX_TIME_UNITS.items() if int(getattr(components, name))]
        return sign + '\\,'.join(parts) if parts else '0'
    return str(value)

def render_ColumnRef(self, subst: dict | None=None) -> str:
    sk = self._structural_key()
    if subst is not None and sk in subst:
        return f'{subst[sk]}_t'
    col_to_latex = {'O': 'O_t', 'H': 'H_t', 'L': 'L_t', 'C': 'C_t', 'OA': '\\tilde{O}_t', 'HA': '\\tilde{H}_t', 'LA': '\\tilde{L}_t', 'CA': '\\tilde{C}_t', 'V': 'V_t', 'OI': 'OI_t', 'OPEN': 'O_t', 'HIGH': 'H_t', 'LOW': 'L_t', 'CLOSE': 'C_t', 'VOLUME': 'V_t', 'OPEN_INTEREST': 'OI_t'}
    return col_to_latex.get(self.column.value) or self.column.value

def render_CategoryBoolRef(self, subst: dict | None=None) -> str:
    sk = self._structural_key()
    if subst is not None and sk in subst:
        return f'{subst[sk]}_t'
    return f'\\operatorname{{Cat}}_{{{self.category.alias}:{self.category_name}}}'

def render_ParamRef(self, subst: dict | None=None) -> str:
    """LaTeX 变量名。ParamRef 的参数名作为基础变量，如 'P' → P_t。"""
    sk = self._structural_key()
    if subst is not None and sk in subst:
        return f'{subst[sk]}_t'
    param_latex = self._parameter_latex()
    if self.indexed_parameter:
        return f'{param_latex}_{{t}}'
    return param_latex

def _parameter_latex(self) -> str:
    alias = escape_text(self.param.alias)
    if not str(self.param.alias).isascii():
        alias = r'\text{' + alias + '}'
    return f'\\textcolor{{red}}{{{alias}}}'

def render_ConstExpr(self, subst: dict | None=None) -> str:
    sk = self._structural_key()
    if subst is not None and sk in subst:
        return f'{subst[sk]}_t'
    return _format_latex_value(self.value)

def render_CompositeExpr(self, subst: dict | None=None) -> str:
    sk = self._structural_key()
    if subst is not None and sk in subst:
        return f'{subst[sk]}_t'
    operands_latex = [opnd._to_latex(subst) for opnd in self.operands]
    if self.op in self._biOps:
        op_latex = self._biOps[self.op]['latex']
        return f'{op_latex}\\left({operands_latex[0]}, {operands_latex[1]}\\right)'
    if self.op in self._Ops:
        if self._Ops[self.op]['nop'] == 1:
            operand_latex = operands_latex[0]
            operator = self._Ops[self.op]['latex']
            if self.op == 'neg' and getattr(self.operands[0], 'op', None) == 'SIGNAL_ALIGN':
                operator = '\\textcolor{red}{-}'
            if self.op == 'sqrt':
                return f'\\sqrt{{{operand_latex}}}'
            return f'{operator}\\left({operand_latex}\\right)'
        elif self._Ops[self.op]['nop'] == 2:
            left_latex = self._binary_operand_latex(self.operands[0], operands_latex[0], side='left', subst=subst)
            right_latex = self._binary_operand_latex(self.operands[1], operands_latex[1], side='right', subst=subst)
            if self.op == 'pow':
                return f'{left_latex}^{{{operands_latex[1]}}}'
            if self.op in ('add', 'sub', 'mul', 'gt', 'lt', 'ge', 'le', 'eq', 'ne', 'and', 'or'):
                return f"{left_latex} {self._Ops[self.op]['latex']} {right_latex}"
            elif self.op in 'div':
                return f"{self._Ops[self.op]['latex']}{{{left_latex}}}{{{right_latex}}}"
            else:
                return f"{self._Ops[self.op]['latex']}\\left({left_latex}, {right_latex}\\right)"
        else:
            return f"{self._Ops[self.op]['latex']}\\left({', '.join(operands_latex)}\\right)"
    else:
        return f"\\text{{{self.op.capitalize()}}}\\left({', '.join(operands_latex)}\\right)"

def _expr_precedence(cls, expr: Any) -> int:
    if not is_kind(expr, ('CompositeExpr',)):
        return 10000
    return cls._LATEX_PRECEDENCE.get(expr.op, 0)

def _needs_parenthesis(self, child: Any, side: str, subst: dict | None=None) -> bool:
    if subst is not None and child._structural_key() in subst:
        return False
    if self.op == 'pow' and side == 'left':
        if is_kind(child, ('ConstExpr',)) and child._to_latex().startswith('-'):
            return True
        if is_kind(child, ('CompositeExpr',)) and child.op == 'neg':
            return True
    if not is_kind(child, ('CompositeExpr',)):
        return False
    if self.op == 'div':
        return False
    parent_prec = self._LATEX_PRECEDENCE.get(self.op, 0)
    child_prec = self._expr_precedence(child)
    if child_prec < parent_prec:
        return True
    if child_prec > parent_prec:
        return False
    if self.op == 'sub' and side == 'right':
        return True
    if self.op == 'pow':
        return True
    if self.op == 'mul' and child.op == 'div':
        return True
    if self.op in ('gt', 'lt', 'ge', 'le', 'eq', 'ne', 'and', 'or'):
        return True
    return False

def _binary_operand_latex(self, child: Any, child_latex: str, side: str, subst: dict | None=None) -> str:
    return f'\\left({child_latex}\\right)' if self._needs_parenthesis(child, side, subst=subst) else child_latex

def render_OperandExpr(self, subst: dict | None=None) -> str:
    sk = self._structural_key()
    if subst is not None and sk in subst:
        return f'{subst[sk]}_t'
    parts = [opnd._to_latex(subst) for opnd in self.operands]
    return f"\\text{{{self.op}}}({', '.join(parts)})"

def render_RollingExpr(self, subst: dict | None=None) -> str:
    sk = self._structural_key()
    if subst is not None and sk in subst:
        return f'{subst[sk]}_t'
    w_str = self._window._to_latex(subst) if is_expression(self._window) else str(self._window)
    if self._trunc_start is not None and self._trunc_end is not None:
        ts_str = self._trunc_start._to_latex(subst) if is_expression(self._trunc_start) else str(self._trunc_start)
        te_str = self._trunc_end._to_latex(subst) if is_expression(self._trunc_end) else str(self._trunc_end)
        return f'\\mathrm{{R}}_{{{w_str},\\mathrm{{trunc}}({ts_str},{te_str})}}'
    return f'\\mathrm{{R}}_{{{w_str}}}'

def render_WindowBarsExpr(self, subst: dict | None=None) -> str:
    window_latex = self.window._to_latex(subst)
    return f'\\mathrm{{Bars}}\\left({window_latex}\\right)'

def render_RollingOp(self, subst: dict | None=None) -> str:
    sk = self._structural_key()
    if subst is not None and sk in subst:
        return f'{subst[sk]}_t'
    w_str = self.window._to_latex(subst) if is_expression(self.window) else str(self.window)
    trunc_suffix = ''
    if self._trunc_start is not None and self._trunc_end is not None:
        ts_str = self._trunc_start._to_latex(subst) if is_expression(self._trunc_start) else str(self._trunc_start)
        te_str = self._trunc_end._to_latex(subst) if is_expression(self._trunc_end) else str(self._trunc_end)
        trunc_suffix = f',\\mathrm{{trunc}}({ts_str},{te_str})'
    if self._n_data == 1:
        operand_latex = self.operands[self._data_start]._to_latex(subst)
        _LATEX_MAP = {'rolling_mean': f'\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{Mean}}\\left({operand_latex}\\right)', 'rolling_std': f'\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{Std}}\\left({operand_latex}\\right)', 'rolling_var': f'\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{Var}}\\left({operand_latex}\\right)', 'rolling_min': f'\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{Min}}\\left({operand_latex}\\right)', 'rolling_max': f'\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{Max}}\\left({operand_latex}\\right)', 'rolling_sum': f'\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{Sum}}\\left({operand_latex}\\right)', 'rolling_ema': f'\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{EMA}}\\left({operand_latex}\\right)', 'rolling_skew': f'\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{Skew}}\\left({operand_latex}\\right)', 'rolling_median': f'\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{Median}}\\left({operand_latex}\\right)', 'rolling_mad': f'\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{MAD}}\\left({operand_latex}\\right)', 'rolling_linreg_slope': f'\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{LinRegSlope}}\\left({operand_latex}\\right)', 'rolling_linreg_r2': f'\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{LinRegR2}}\\left({operand_latex}\\right)', 'rolling_linreg_tstat': f'\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{LinRegT}}\\left({operand_latex}\\right)', 'rolling_linreg_resid_std': f'\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{LinRegResidStd}}\\left({operand_latex}\\right)', 'rolling_argmax': f'\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{ArgMax}}\\left({operand_latex}\\right)', 'rolling_argmin': f'\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{ArgMin}}\\left({operand_latex}\\right)', 'rolling_argmax_raw': f'\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{ArgMaxRaw}}\\left({operand_latex}\\right)', 'rolling_argmin_raw': f'\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{ArgMinRaw}}\\left({operand_latex}\\right)'}
        if self.op == 'rolling_quantile':
            q_expr = self.quantile_expr
            q_latex = q_expr._to_latex(subst) if is_expression(q_expr) else str(q_expr)
            return f'\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{Quantile}}_{{{q_latex}}}\\left({operand_latex}\\right)'
        return _LATEX_MAP.get(self.op, f'{self.op}_{{{w_str}}}{operand_latex}')
    else:
        left_latex = self.operands[self._data_start]._to_latex(subst)
        right_latex = self.operands[self._data_start + 1]._to_latex(subst)
        return f"\\text{{R}}_{{{w_str}{trunc_suffix}}}\\text{{{self.op.replace('rolling_', '').capitalize()}}}\\left({left_latex}, {right_latex}\\right)"

def render_IntervalBarsExpr(self, subst: dict | None=None) -> str:
    return f'\\operatorname{{bars}}[{self.start_ns},{self.end_ns}]'

def render_ShiftOp(self, subst: dict | None=None) -> str:
    sk = self._structural_key()
    if subst is not None and sk in subst:
        return f'{subst[sk]}_t'
    operand_latex = self.operand._to_latex(subst)
    if _is_zero_shift_period(self.periods):
        return operand_latex
    p_label = self.periods._to_latex(subst)
    named = subst is not None and self.operand._structural_key() in subst
    indexed_leaf = is_kind(self.operand, ('ColumnRef', 'ParamRef')) or named
    operand_base = _strip_latex_time_subscript(operand_latex)
    if not indexed_leaf or operand_base == operand_latex:
        return f'\\operatorname{{Shift}}_{{{p_label}}}\\left({operand_latex}\\right)'
    subtraction = self.session.operation('sub', self.operand, self.periods)
    p_label = subtraction._binary_operand_latex(self.periods, p_label, side='right', subst=subst)
    if p_label.startswith('-'):
        p_label = f'\\left({p_label}\\right)'
    return f'{operand_base}_{{t - {p_label}}}'

def _is_zero_shift_period(periods: Any) -> bool:
    if not is_kind(periods, ('ConstExpr',)):
        return False
    value = periods.value
    if value == 0:
        return True
    try:
        return pd.Timedelta(value) == pd.Timedelta(0)
    except Exception:
        return False

def _strip_latex_time_subscript(latex: str) -> str:
    """
    Remove a trailing time subscript from a LaTeX fragment.

    Historically some leaf nodes emitted `_t` while others emitted `_{t}`.
    For operators like SHIFT that need to *replace* the time index, we must
    normalize both forms to avoid MathJax "Double subscripts" errors
    (e.g. `\\tilde{O}_t_{t-N}`).
    """
    if latex.endswith('_{t}'):
        return latex[:-len('_{t}')]
    if latex.endswith('_t'):
        return latex[:-len('_t')]
    return latex

def render_SignalAlign(self, subst: dict | None=None) -> str:
    sk = self._structural_key()
    if subst is not None and sk in subst:
        return f'{subst[sk]}_t'
    operand = self.operands[0]
    operand_key = operand._structural_key()
    if subst is not None and operand_key in subst:
        inner = f'{subst[operand_key]}_t'
    else:
        inner = operand._to_latex(subst)
    frequency = str(self.signal_freq if self.signal_freq is not None else '$F')
    frequency = escape_text(frequency)
    return '\\operatorname{Resample}_{\\textcolor{red}{' + frequency + '}}\\left(' + inner + '\\right)'

def render_CrossSectionalOp(self, subst: dict | None=None) -> str:
    sk = self._structural_key()
    if subst is not None and sk in subst:
        return f'{subst[sk]}_t'
    if self.op == 'cs_spearman':
        left_latex = self.left._to_latex(subst)
        right_latex = self.right._to_latex(subst)
        return f'\\rho_s({left_latex}, {right_latex})'
    if self.op == 'cs_corr':
        left_latex = self.left._to_latex(subst)
        right_latex = self.right._to_latex(subst)
        return f'\\rho({left_latex}, {right_latex})'
    operand_latex = self.operand._to_latex(subst)
    _LATEX_MAP = {'cs_zscore': f'Z({operand_latex})', 'cs_rank': f'\\text{{Rank}}({operand_latex})', 'cs_rank_masked': f'\\text{{Rank}}_{{mask}}({operand_latex})', 'cs_ordinal_rank_asc': f'\\text{{OrdinalRank}}_\\uparrow({operand_latex})', 'cs_ordinal_rank_desc': f'\\text{{OrdinalRank}}_\\downarrow({operand_latex})'}
    return _LATEX_MAP.get(self.op, f'\\text{{{self.op}}}({operand_latex})')

def _scope_latex(scope: Any) -> str:
    if is_kind(scope, ('TradingDayScope',)):
        return '\\text{trading\\_day}'
    if is_kind(scope, ('SessionScope',)):
        return '\\text{session}(' + str(scope.gap) + ')'
    if is_kind(scope, ('BarCountScope',)):
        return '\\text{bars}(K)'
    return '\\text{scope}'

def render_GroupByScopeExpr(self, subst: dict | None=None) -> str:
    sk = self._structural_key()
    if subst is not None and sk in subst:
        return f'{subst[sk]}_t'
    scope = _scope_latex(self.scope)
    start = self.trunc_start
    end = self.trunc_end
    if start is not None or end is not None:
        left = start._to_latex(subst) if is_expression(start) else str(start)
        right = end._to_latex(subst) if is_expression(end) else str(end)
        return '\\mathrm{R}^{' + scope + ',\\mathrm{trunc}(' + left + ',' + right + ')}'
    return '\\mathrm{R}^{' + scope + '}'

def render_GroupByScopeOp(self, subst: dict | None=None) -> str:
    sk = self._structural_key()
    if subst is not None and sk in subst:
        return f'{subst[sk]}_t'
    scope = _scope_latex(self.scope)
    start, end = (self.trunc_start, self.trunc_end)
    if start is not None or end is not None:
        left = start._to_latex(subst) if is_expression(start) else str(start)
        right = end._to_latex(subst) if is_expression(end) else str(end)
        return '\\mathrm{R}^{' + scope + ',\\mathrm{trunc}(' + left + ',' + right + ')}'
    return '\\mathrm{R}^{' + scope + '}'

def _search_latex(*, scope: Any, select: str, body: str, subst: dict | None=None) -> str:
    operator = '\\min' if select == 'nearest' else '\\max'
    subscript = f'[t-k,t]\\in{scope.to_latex(subst)}'
    return f'{operator}_{{{subscript}}}\\left\\{{k\\middle|\\begin{{aligned}}{body}\\end{{aligned}}\\right\\}}'

def render_ScopeLengthDefault(self, subst: dict | None=None) -> str:
    return '\\operatorname{ScopeLength}'

def render_BarSinceOp(self, subst: dict | None=None) -> str:
    condition = self.condition._to_latex(subst)
    return _search_latex(scope=self.scope, select=self.select, body=condition, subst=subst)

def render_BarDistanceOp(self, subst: dict | None=None) -> str:
    value = self.value._to_latex(subst)
    condition = self.condition._to_latex(subst)
    return _search_latex(scope=self.scope, select=self.select, body=f'X_t:={value},\\\\{condition}', subst=subst)

def _operand_latex_arg(expr: Any, *, column: bool=False) -> str:
    if column:
        if is_kind(expr, ('ColumnRef',)):
            return expr.column.name
        if is_kind(expr, ('ConstExpr',)):
            return expr.display_column_name
        if is_kind(expr, ('ParamRef',)):
            return expr._parameter_latex()
    if is_kind(expr, ('ConstExpr',)):
        return str(expr.value)
    return expr._to_latex()

def render_TermStructureOp(self, subst: dict | None=None) -> str:
    sk = self._structural_key()
    if subst is not None and sk in subst:
        return f'{subst[sk]}_t'
    if self.op in self._PAIR_OPS:
        near_rank = self._operand_latex_arg(self.operands[0])
        far_rank = self._operand_latex_arg(self.operands[1])
        column = self._operand_latex_arg(self.operands[2], column=True)
        return f'{self._LATEX[self.op]}_{{{near_rank},{far_rank}}}({column})'
    depth = self._operand_latex_arg(self.operands[0])
    column = self._operand_latex_arg(self.operands[1], column=True)
    return f'{self._LATEX[self.op]}_{{{depth}}}({column})'

def render_MatchValueRef(self, subst=None) -> str:
    return 'X_t' if self.role == 'current' else 'X_{t-k}'

RENDERERS = {name.removeprefix("render_"): value for name,value in list(globals().items()) if name.startswith("render_")}

_biOps = {'bimax': {'latex': '\\max', 'nop': 2}, 'bimin': {'latex': '\\min', 'nop': 2}}
_Ops = {'add': {'latex': '+', 'nop': 2}, 'sub': {'latex': '-', 'nop': 2}, 'mul': {'latex': '\\times', 'nop': 2}, 'div': {'latex': '\\frac', 'nop': 2}, 'gt': {'latex': '>', 'nop': 2}, 'lt': {'latex': '<', 'nop': 2}, 'ge': {'latex': '\\ge', 'nop': 2}, 'le': {'latex': '\\le', 'nop': 2}, 'eq': {'latex': '=', 'nop': 2}, 'ne': {'latex': '\\neq', 'nop': 2}, 'and': {'latex': '\\wedge', 'nop': 2}, 'or': {'latex': '\\vee', 'nop': 2}, 'neg': {'latex': '-', 'nop': 1}, 'abs': {'latex': '\\mathrm{abs}', 'nop': 1}, 'not': {'latex': '\\neg', 'nop': 1}, 'log': {'latex': '\\log', 'nop': 1}, 'sign': {'latex': '\\mathrm{sign}', 'nop': 1}, 'sqrt': {'latex': '\\sqrt', 'nop': 1}, 'tanh': {'latex': '\\tanh', 'nop': 1}, 'pow': {'latex': '^', 'nop': 2}, 'max': {'latex': '\\max', 'nop': -1}, 'min': {'latex': '\\min', 'nop': -1}}

def render_LookbackScope(self, subst: dict | None=None) -> str:
    return f'\\operatorname{{Scope{self.kind.title()}}}'

def render_BarCountScope(self, subst: dict | None=None) -> str:
    return f'\\operatorname{{ScopeBars}}\\left({self.count._to_latex(subst)}\\right)'

def render_SessionScope(self, subst: dict | None=None) -> str:
    return f'\\operatorname{{ScopeSession}}\\left(\\mathrm{{{self.gap}}}\\right)'

def render_TradingDayScope(self, subst: dict | None=None) -> str:
    return '\\operatorname{ScopeTradingDay}'

RENDERERS.update({name.removeprefix("render_"): value for name,value in list(globals().items()) if name.startswith("render_")})


def render_Literal(self, subst=None):
    return self.literal

RENDERERS["Literal"] = render_Literal
