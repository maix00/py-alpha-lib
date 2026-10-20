// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause
//
// FactorTester fork additions: batch plan executor (ADR-161 P4).
//
// A compiled plan is a flat operator table. `slots` holds every live buffer as
// `groups * rows` contiguous panels. The executor copies the input slots into
// its own storage, then runs each op in order into its destination slot. Slot
// ids come from the Python compiler, which allocates them from lifetime
// intervals, so peak memory is O(active slots) instead of O(nodes).
//
// Every op has four integer fields [code, out, a, b] and four float fields
// [window, min_count, quantile, periods]. Taking both tables as `f64` keeps the
// generated pyo3 binding simple; codes and slot ids stay exact. Elementwise ops
// follow numpy semantics (NaN propagates; `maximum/minimum` return NaN when
// either side is NaN; `sign(-0.0) == 0`).

use num_traits::{Float, ToPrimitive};

use crate::algo::{
  ta_argmax, ta_argmax_raw, ta_argmin, ta_argmin_raw, ta_corr_nan, ta_cov_nan,
  ta_ema_adjust, ta_linreg_r2, ta_linreg_resid_std, ta_linreg_slope, ta_linreg_tstat, ta_mad,
  ta_max_nan, ta_mean_nan, ta_median, ta_min_nan, ta_quantile_nan, ta_skew_exact, ta_std_nan,
  ta_sum_nan, ta_var_nan, Context, Error,
};

pub const PLAN_SHIFT: usize = 1;

pub const PLAN_NEG: usize = 10;
pub const PLAN_ABS: usize = 11;
pub const PLAN_LOG: usize = 12;
pub const PLAN_SQRT: usize = 13;
pub const PLAN_TANH: usize = 14;
pub const PLAN_EXP: usize = 15;
pub const PLAN_SIGN: usize = 16;

pub const PLAN_ADD: usize = 20;
pub const PLAN_SUB: usize = 21;
pub const PLAN_MUL: usize = 22;
pub const PLAN_DIV: usize = 23;
pub const PLAN_POW: usize = 24;
pub const PLAN_MAX: usize = 25;
pub const PLAN_MIN: usize = 26;

pub const PLAN_MEAN: usize = 30;
pub const PLAN_STD: usize = 31;
pub const PLAN_VAR: usize = 32;
pub const PLAN_ROLL_MIN: usize = 33;
pub const PLAN_ROLL_MAX: usize = 34;
pub const PLAN_SUM: usize = 35;
pub const PLAN_MEDIAN: usize = 36;
pub const PLAN_MAD: usize = 37;
pub const PLAN_SKEW: usize = 38;
pub const PLAN_EMA: usize = 39;
pub const PLAN_QUANTILE: usize = 40;
pub const PLAN_LINREG_SLOPE: usize = 41;
pub const PLAN_LINREG_R2: usize = 42;
pub const PLAN_LINREG_TSTAT: usize = 43;
pub const PLAN_LINREG_RESID_STD: usize = 44;
pub const PLAN_ARGMAX: usize = 45;
pub const PLAN_ARGMAX_RAW: usize = 46;
pub const PLAN_ARGMIN: usize = 47;
pub const PLAN_ARGMIN_RAW: usize = 48;

pub const PLAN_CORR: usize = 60;
pub const PLAN_COV: usize = 61;

/// One operand of an elementwise op: either a slot panel or an immediate.
enum PlanOperand<'a, NumT: Float> {
  Slot(&'a [NumT]),
  Scalar(NumT),
}

impl<NumT: Float> PlanOperand<'_, NumT> {
  #[inline]
  fn at(&self, index: usize) -> NumT {
    match self {
      PlanOperand::Slot(values) => values[index],
      PlanOperand::Scalar(value) => *value,
    }
  }

  fn slot(&self) -> Result<&[NumT], Error> {
    match self {
      PlanOperand::Slot(values) => Ok(values),
      PlanOperand::Scalar(_) => {
        Err(Error::InvalidParameter("kernel operand must be a slot".into()))
      }
    }
  }
}

fn panel_layout(len: usize, groups: usize) -> (usize, usize) {
  let groups = groups.max(1);
  (groups.min(len.max(1)), len / groups.max(1))
}

fn plan_shift<NumT: Float>(
  out: &mut [NumT], input: &[NumT], groups: usize, rows: usize, periods: i64,
) -> Result<(), Error> {
  if out.len() != input.len() {
    return Err(Error::LengthMismatch(out.len(), input.len()));
  }
  if periods == 0 {
    out.copy_from_slice(input);
    return Ok(());
  }
  out.fill(NumT::nan());
  for g in 0..groups {
    for t in 0..rows {
      let source = t as i64 - periods;
      if source >= 0 && (source as usize) < rows {
        out[g * rows + t] = input[g * rows + source as usize];
      }
    }
  }
  Ok(())
}

fn plan_unary<NumT: Float>(
  code: usize, out: &mut [NumT], input: &PlanOperand<'_, NumT>,
) -> Result<(), Error> {
  for i in 0..out.len() {
    let value = input.at(i);
    out[i] = match code {
      PLAN_NEG => -value,
      PLAN_ABS => value.abs(),
      PLAN_LOG => value.ln(),
      PLAN_SQRT => value.sqrt(),
      PLAN_TANH => value.tanh(),
      PLAN_EXP => value.exp(),
      PLAN_SIGN => {
        if value.is_nan() {
          NumT::nan()
        } else if value > NumT::zero() {
          NumT::one()
        } else if value < NumT::zero() {
          -NumT::one()
        } else {
          NumT::zero()
        }
      }
      _ => return Err(Error::InvalidParameter(format!("unknown unary plan op {code}"))),
    };
  }
  Ok(())
}

fn plan_binary<NumT: Float>(
  code: usize,
  out: &mut [NumT],
  a: &PlanOperand<'_, NumT>,
  b: &PlanOperand<'_, NumT>,
) -> Result<(), Error> {
  for i in 0..out.len() {
    let left = a.at(i);
    let right = b.at(i);
    out[i] = match code {
      PLAN_ADD => left + right,
      PLAN_SUB => left - right,
      PLAN_MUL => left * right,
      PLAN_DIV => left / right,
      PLAN_POW => left.powf(right),
      // numpy.maximum/minimum propagate NaN instead of ignoring it.
      PLAN_MAX => {
        if left.is_nan() || right.is_nan() {
          NumT::nan()
        } else {
          left.max(right)
        }
      }
      PLAN_MIN => {
        if left.is_nan() || right.is_nan() {
          NumT::nan()
        } else {
          left.min(right)
        }
      }
      _ => return Err(Error::InvalidParameter(format!("unknown binary plan op {code}"))),
    };
  }
  Ok(())
}

fn rolling_op<NumT: Float + Send + Sync>(
  code: usize,
  ctx: &Context,
  out: &mut [NumT],
  a: &[NumT],
  b: &[NumT],
  window: usize,
  min_count: usize,
  quantile: NumT,
) -> Result<(), Error> {
  match code {
    PLAN_MEAN => ta_mean_nan(ctx, out, a, window, min_count),
    PLAN_STD => ta_std_nan(ctx, out, a, window, min_count),
    PLAN_VAR => ta_var_nan(ctx, out, a, window, min_count),
    PLAN_ROLL_MIN => ta_min_nan(ctx, out, a, window, min_count),
    PLAN_ROLL_MAX => ta_max_nan(ctx, out, a, window, min_count),
    PLAN_SUM => ta_sum_nan(ctx, out, a, window, min_count),
    PLAN_MEDIAN => ta_median(ctx, out, a, window, min_count),
    PLAN_MAD => ta_mad(ctx, out, a, window, min_count),
    PLAN_SKEW => ta_skew_exact(ctx, out, a, window, min_count),
    PLAN_EMA => ta_ema_adjust(ctx, out, a, window, min_count),
    PLAN_QUANTILE => ta_quantile_nan(ctx, out, a, window, min_count, quantile),
    PLAN_LINREG_SLOPE => ta_linreg_slope(ctx, out, a, window, min_count),
    PLAN_LINREG_R2 => ta_linreg_r2(ctx, out, a, window, min_count),
    PLAN_LINREG_TSTAT => ta_linreg_tstat(ctx, out, a, window, min_count),
    PLAN_LINREG_RESID_STD => ta_linreg_resid_std(ctx, out, a, window, min_count),
    PLAN_ARGMAX => ta_argmax(ctx, out, a, window),
    PLAN_ARGMAX_RAW => ta_argmax_raw(ctx, out, a, window),
    PLAN_ARGMIN => ta_argmin(ctx, out, a, window),
    PLAN_ARGMIN_RAW => ta_argmin_raw(ctx, out, a, window),
    PLAN_CORR => ta_corr_nan(ctx, out, a, b, window, min_count),
    PLAN_COV => ta_cov_nan(ctx, out, a, b, window, min_count),
    _ => Err(Error::InvalidParameter(format!("unknown rolling plan op {code}"))),
  }
}

/// Run a compiled plan over `(groups, rows)` slot panels.
///
/// `slots` carries the input panels; the returned buffer starts as a copy of
/// them so the executor can write every intermediate in place. `plan` and
/// `params` are flat tables of four fields per op (see the module comment).
pub fn ta_ft_run_plan<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  slots: &[NumT],
  plan: &[NumT],
  params: &[NumT],
  slot_count: usize,
  op_count: usize,
  leaf_count: usize,
) -> Result<(), Error> {
  if slot_count == 0 || slots.is_empty() || slots.len() % slot_count != 0 {
    return Err(Error::LengthMismatch(slots.len(), slot_count));
  }
  if r.len() != slots.len() {
    return Err(Error::LengthMismatch(r.len(), slots.len()));
  }
  if plan.len() < op_count * 4 || params.len() < op_count * 6 {
    return Err(Error::LengthMismatch(plan.len(), op_count * 6));
  }
  if leaf_count > slot_count {
    return Err(Error::InvalidParameter(format!(
      "leaf_count {leaf_count} exceeds slot_count {slot_count}"
    )));
  }
  let len = slots.len() / slot_count;
  let (groups, rows) = panel_layout(len, ctx.groups());
  // Leaves are allocated first by the compiler; every other slot is written
  // before it is read, so only the input panels need copying.
  r[..leaf_count * len].copy_from_slice(&slots[..leaf_count * len]);
  let pointer = r.as_mut_ptr();
  for index in 0..op_count {
    let base = index * 4;
    let code = plan[base]
      .to_usize()
      .ok_or_else(|| Error::InvalidParameter("plan code is not an integer".into()))?;
    let out_idx = plan[base + 1]
      .to_usize()
      .ok_or_else(|| Error::InvalidParameter("plan out slot is not an integer".into()))?;
    if out_idx >= slot_count {
      return Err(Error::InvalidParameter(format!(
        "plan out slot {out_idx} out of range (slots={slot_count})"
      )));
    }
    let param_base = index * 6;
    // 负的槽位号表示立即数，值放在参数表第 5/6 列。
    let a_raw = plan[base + 2];
    let b_raw = plan[base + 3];
    let a_index = a_raw.to_usize();
    let b_index = b_raw.to_usize();
    if let Some(index_value) = a_index {
      if index_value >= slot_count {
        return Err(Error::InvalidParameter(format!(
          "plan input slot {index_value} out of range (slots={slot_count})"
        )));
      }
    }
    if let Some(index_value) = b_index {
      if index_value >= slot_count {
        return Err(Error::InvalidParameter(format!(
          "plan input slot {index_value} out of range (slots={slot_count})"
        )));
      }
    }
    // The compiler never aliases the destination with an input slot: kernels
    // fill the output before reading, so sharing storage would corrupt values.
    debug_assert!(a_index != Some(out_idx) && b_index != Some(out_idx));
    // SAFETY: slot ranges are disjoint (checked above) and `r` is a valid
    // allocation of `slot_count * len` elements for the whole loop.
    let out = unsafe { std::slice::from_raw_parts_mut(pointer.add(out_idx * len), len) };
    let a = match a_index {
      Some(slot) => PlanOperand::Slot(unsafe {
        std::slice::from_raw_parts(pointer.add(slot * len), len)
      }),
      None => PlanOperand::Scalar(params[param_base + 4]),
    };
    let b = match b_index {
      Some(slot) => PlanOperand::Slot(unsafe {
        std::slice::from_raw_parts(pointer.add(slot * len), len)
      }),
      None => PlanOperand::Scalar(params[param_base + 5]),
    };
    let window = params[param_base].to_usize().unwrap_or(0);
    let min_count = params[param_base + 1].to_usize().unwrap_or(0);
    let quantile = params[param_base + 2];
    let periods = params[param_base + 3].to_i64().unwrap_or(0);
    if code == PLAN_SHIFT {
      plan_shift(out, a.slot()?, groups, rows, periods)?;
    } else if (PLAN_NEG..=PLAN_SIGN).contains(&code) {
      plan_unary(code, out, &a)?;
    } else if (PLAN_ADD..=PLAN_MIN).contains(&code) {
      plan_binary(code, out, &a, &b)?;
    } else {
      rolling_op(code, ctx, out, a.slot()?, b.slot()?, window, min_count, quantile)?;
    }
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn plan_shift_add_and_sign() {
    // 2 products x 4 bars; slot 0 input, slot 1 constant 1.0,
    // op0: shift(input, 1) -> slot 2, op1: add(slot2, const) -> slot 3,
    // op2: sign(slot3) -> slot 4.
    let groups = 2usize;
    let rows = 4usize;
    let input = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let mut slots = vec![0.0; groups * rows * 5];
    slots[..groups * rows].copy_from_slice(&input);
    for value in slots.iter_mut().skip(groups * rows).take(groups * rows) {
      *value = 1.0;
    }
    let plan = [
      PLAN_SHIFT as f64, 2.0, 0.0, 0.0,
      PLAN_ADD as f64, 3.0, 2.0, -1.0,
      PLAN_SIGN as f64, 4.0, 3.0, 3.0,
    ];
    let params = [
      0.0, 0.0, 0.0, 1.0, 0.0, 1.0, // shift(input, 1)
      0.0, 0.0, 0.0, 0.0, 0.0, 1.0, // add, b = 1.0 immediate
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // sign
    ];
    let ctx = Context::new(0, groups as u32, 0);
    let mut out = vec![0.0; slots.len()];
    ta_ft_run_plan(&ctx, &mut out, &slots, &plan, &params, 5, 3, 2).unwrap();
    let shifted = &out[2 * groups * rows..3 * groups * rows];
    let signed = &out[4 * groups * rows..5 * groups * rows];
    assert!(shifted[0].is_nan() && shifted[4].is_nan());
    assert_eq!(shifted[1], 1.0);
    assert_eq!(shifted[5], 10.0);
    assert!(signed[0].is_nan());
    assert_eq!(signed[1], 1.0);
    assert_eq!(signed[2], 1.0);
    assert_eq!(signed[3], 1.0);
  }

  #[test]
  fn plan_rolling_mean_matches_window() {
    let groups = 1usize;
    let rows = 5usize;
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut slots = vec![0.0; rows * 2];
    slots[..rows].copy_from_slice(&input);
    let plan = [PLAN_MEAN as f64, 1.0, 0.0, 0.0];
    let params = [2.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let ctx = Context::new(0, groups as u32, 0);
    let mut out = vec![0.0; slots.len()];
    ta_ft_run_plan(&ctx, &mut out, &slots, &plan, &params, 2, 1, 1).unwrap();
    let mean = &out[rows..2 * rows];
    // min_count = 1, so the first partial window is already eligible.
    assert_eq!(mean[0], 1.0);
    assert_eq!(mean[1], 1.5);
    assert_eq!(mean[2], 2.5);
  }
}
