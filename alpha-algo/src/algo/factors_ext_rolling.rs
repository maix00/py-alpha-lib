// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause
//
// FactorTester fork additions (batch path only): rolling operators that the
// upstream crate does not provide with matching semantics. NaN is treated as
// "not observed": it is skipped in statistics while still consuming a bar.
// Warm-up is controlled by the caller through `min_count` (or by window
// position for arg-extrema, matching the FactorTester contract).

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal};

fn collect_window<NumT: Float>(x: &[NumT], start: usize, periods: usize) -> Vec<NumT> {
  let mut buf = Vec::with_capacity(periods);
  let win_start = start.saturating_sub(periods - 1);
  for k in win_start..=start {
    let value = x[k];
    if is_normal(&value) {
      buf.push(value);
    }
  }
  buf
}

/// Exponential moving average matching pandas `ewm(span=periods, adjust=True, ignore_na=False).mean()`.
///
/// Every bar multiplies the running numerator and denominator by `1 - alpha`
/// (missing bars included). Observed bars then add the value / unit weight;
/// missing bars only decay, and the output repeats the previous EMA value.
/// `min_count` counts observed bars, matching pandas `min_periods`.
pub fn ta_ema_adjust<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
  min_count: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }
  r.fill(NumT::nan());
  if periods == 0 {
    return Ok(());
  }
  let alpha = NumT::from(2).unwrap() / NumT::from(periods + 1).unwrap();
  let decay = NumT::one() - alpha;
  let minimum = min_count.max(1);

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      let mut numerator = NumT::zero();
      let mut denominator = NumT::zero();
      let mut observed = 0usize;
      let mut last = NumT::nan();
      for i in start..end {
        let value = x[i];
        numerator = numerator * decay;
        denominator = denominator * decay;
        if is_normal(&value) {
          numerator = numerator + value;
          denominator = denominator + NumT::one();
          observed += 1;
          last = numerator / denominator;
        }
        if observed >= minimum {
          r[i] = last;
        }
      }
    });
  Ok(())
}

/// Rolling skewness matching pandas `rolling(periods, min_periods=min_count).skew()`.
///
/// Two-pass window computation with the bias-corrected Fisher-Pearson G1
/// formula. Fewer than three valid observations yield NaN; a zero-variance
/// window yields 0 (pandas behaviour).
pub fn ta_skew_exact<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
  min_count: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }
  r.fill(NumT::nan());
  if periods == 0 {
    return Ok(());
  }
  let minimum = min_count.max(1);

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      for i in start..end {
        let buf = collect_window(x, i, periods);
        let n = buf.len();
        if n < minimum || n < 3 {
          continue;
        }
        let count = NumT::from(n).unwrap();
        let mean = buf.iter().fold(NumT::zero(), |acc, v| acc + *v) / count;
        let mut m2 = NumT::zero();
        let mut m3 = NumT::zero();
        for value in &buf {
          let delta = *value - mean;
          let delta2 = delta * delta;
          m2 = m2 + delta2;
          m3 = m3 + delta2 * delta;
        }
        if !(m2 > NumT::zero()) {
          r[i] = NumT::zero();
          continue;
        }
        let sample_var = m2 / (count - NumT::one());
        let std = sample_var.sqrt();
        let nf = count;
        r[i] = nf / ((nf - NumT::one()) * (nf - NumT::from(2).unwrap()))
          * (m3 / (std * std * std));
      }
    });
  Ok(())
}

fn arg_extreme<NumT: Float>(
  window: &[NumT],
  find_max: bool,
  normalized: bool,
) -> NumT {
  let width = window.len();
  if width == 1 {
    return if is_normal(&window[0]) { NumT::zero() } else { NumT::nan() };
  }
  let mut best_position: Option<usize> = None;
  let mut best_value = NumT::nan();
  for (position, value) in window.iter().enumerate() {
    if !is_normal(value) {
      continue;
    }
    let replace = match best_position {
      None => true,
      // Ties resolve to the most recent bar, matching the FactorTester contract.
      Some(_) => {
        if find_max {
          *value >= best_value
        } else {
          *value <= best_value
        }
      }
    };
    if replace {
      best_position = Some(position);
      best_value = *value;
    }
  }
  match best_position {
    // A fully missing window keeps the neutral position 0.
    None => NumT::zero(),
    Some(position) => {
      let value = NumT::from(position).unwrap();
      if normalized {
        value / NumT::from(width - 1).unwrap()
      } else {
        value
      }
    }
  }
}

/// Rolling argmax of the trailing window.
///
/// Ties resolve to the most recent bar; a fully missing window returns the
/// neutral position 0; the first `periods - 1` bars are NaN. When `normalized`
/// is true the output is `position / (periods - 1)`.
pub fn ta_argmax<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  arg_impl(ctx, r, input, periods, true, true)
}

/// Rolling argmax expressed as a window offset in `0..periods-1`.
pub fn ta_argmax_raw<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  arg_impl(ctx, r, input, periods, false, true)
}

/// Rolling argmin of the trailing window; same contract as `ta_argmax`.
pub fn ta_argmin<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  arg_impl(ctx, r, input, periods, true, false)
}

/// Rolling argmin expressed as a window offset in `0..periods-1`.
pub fn ta_argmin_raw<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  arg_impl(ctx, r, input, periods, false, false)
}

fn arg_impl<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
  normalized: bool,
  find_max: bool,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }
  r.fill(NumT::nan());
  if periods == 0 {
    return Ok(());
  }
  let warmup = periods.saturating_sub(1);

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      for i in start..end {
        if i < warmup {
          continue;
        }
        let win_start = i + 1 - periods;
        r[i] = arg_extreme(&x[win_start..=i], find_max, normalized);
      }
    });
  Ok(())
}

/// Rolling quantile with linear interpolation and NaN skipping.
///
/// Matches numpy/pandas `rolling(periods, min_periods=min_count).quantile(q)`:
/// only observed values participate, and the position is `q * (n - 1)` with
/// linear interpolation between neighbours.
pub fn ta_quantile_nan<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
  min_count: usize,
  q: NumT,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }
  r.fill(NumT::nan());
  if periods == 0 || !(q >= NumT::zero() && q <= NumT::one()) {
    return Ok(());
  }
  let minimum = min_count.max(1).min(periods);

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      let end = ctx.end(r.len());
      for i in start..end {
        let mut buf = collect_window(x, i, periods);
        let n = buf.len();
        if n < minimum {
          continue;
        }
        buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if n == 1 {
          r[i] = buf[0];
          continue;
        }
        let position = q * NumT::from(n - 1).unwrap();
        let lower = position.floor().to_usize().unwrap_or(0).min(n - 1);
        let upper = (lower + 1).min(n - 1);
        let frac = position - NumT::from(lower).unwrap();
        r[i] = buf[lower] + (buf[upper] - buf[lower]) * frac;
      }
    });
  Ok(())
}
