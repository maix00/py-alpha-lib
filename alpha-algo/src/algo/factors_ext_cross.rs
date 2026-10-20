// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause
//
// FactorTester fork additions: cross-sectional (per timestamp) operators.
//
// Layout matches the rest of the crate: `groups` contiguous chunks of equal
// length, i.e. flat[g * T + t]. Cross-sectional operators walk all groups at a
// fixed t, which is strided but cheap for the panel sizes FactorTester uses.
// NaN means "not observed" everywhere: it never participates in a statistic
// and is never propagated into another product's value.

use num_traits::Float;

use crate::algo::{Context, Error, is_normal};

fn layout(len: usize, groups: usize) -> (usize, usize) {
  let groups = groups.max(1);
  let rows = len / groups;
  (groups.min(len.max(1)), rows)
}

fn average_ranks<NumT: Float>(values: &[NumT]) -> (Vec<usize>, Vec<NumT>) {
  let mut order: Vec<usize> = (0..values.len()).collect();
  order.sort_by(|a, b| {
    values[*a]
      .partial_cmp(&values[*b])
      .unwrap_or(std::cmp::Ordering::Equal)
  });
  let mut ranks = vec![NumT::nan(); values.len()];
  let mut i = 0usize;
  while i < order.len() {
    let mut j = i + 1;
    while j < order.len() && values[order[j]] == values[order[i]] {
      j += 1;
    }
    // average rank over the tie group, 1-based
    let start = NumT::from(i + 1).unwrap();
    let end = NumT::from(j).unwrap();
    let rank = (start + end) / NumT::from(2).unwrap();
    for k in i..j {
      ranks[order[k]] = rank;
    }
    i = j;
  }
  (order, ranks)
}

/// Cross-sectional percentage rank: `rank / valid_count - 0.5`.
pub fn ta_ft_rank<NumT: Float + Send + Sync>(
  ctx: &Context, r: &mut [NumT], input: &[NumT],
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }
  r.fill(NumT::nan());
  let (groups, rows) = layout(input.len(), ctx.groups());
  for t in 0..rows {
    let mut values = Vec::with_capacity(groups);
    let mut positions = Vec::with_capacity(groups);
    for g in 0..groups {
      let v = input[g * rows + t];
      if is_normal(&v) {
        values.push(v);
        positions.push(g * rows + t);
      }
    }
    if values.is_empty() {
      continue;
    }
    let (_, ranks) = average_ranks(&values);
    let count = NumT::from(values.len()).unwrap();
    for (index, position) in positions.iter().enumerate() {
      r[*position] = ranks[index] / count - NumT::from(0.5).unwrap();
    }
  }
  Ok(())
}

/// Cross-sectional percentage rank restricted to an eligibility mask.
pub fn ta_ft_rank_masked<NumT: Float + Send + Sync>(
  ctx: &Context, r: &mut [NumT], input: &[NumT], mask: &[NumT],
) -> Result<(), Error> {
  if r.len() != input.len() || mask.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }
  r.fill(NumT::nan());
  let (groups, rows) = layout(input.len(), ctx.groups());
  for t in 0..rows {
    let mut values = Vec::with_capacity(groups);
    let mut positions = Vec::with_capacity(groups);
    for g in 0..groups {
      let index = g * rows + t;
      let v = input[index];
      if is_normal(&v) && (mask[index] > NumT::zero()) {
        values.push(v);
        positions.push(index);
      }
    }
    if values.is_empty() {
      continue;
    }
    let (_, ranks) = average_ranks(&values);
    let count = NumT::from(values.len()).unwrap();
    for (index, position) in positions.iter().enumerate() {
      r[*position] = ranks[index] / count - NumT::from(0.5).unwrap();
    }
  }
  Ok(())
}

/// Ordinal rank (1..n) with `tiebreak` deciding equal values; `ascending == 0`
/// ranks descending. Non-eligible entries stay NaN.
pub fn ta_ft_ordinal_rank<NumT: Float + Send + Sync>(
  ctx: &Context, r: &mut [NumT], input: &[NumT], mask: &[NumT], tiebreak: &[NumT], ascending: usize,
) -> Result<(), Error> {
  if r.len() != input.len() || mask.len() != input.len() || tiebreak.len() < input.len() / ctx.groups().max(1) {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }
  r.fill(NumT::nan());
  let (groups, rows) = layout(input.len(), ctx.groups());
  let descending = ascending == 0;
  for t in 0..rows {
    let mut eligible: Vec<usize> = Vec::with_capacity(groups);
    for g in 0..groups {
      let index = g * rows + t;
      if is_normal(&input[index]) && (mask[index] > NumT::zero()) {
        eligible.push(g);
      }
    }
    if eligible.is_empty() {
      continue;
    }
    eligible.sort_by(|a, b| {
      let va = input[*a * rows + t];
      let vb = input[*b * rows + t];
      let base = va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal);
      let base = if descending { base.reverse() } else { base };
      if base == std::cmp::Ordering::Equal {
        tiebreak[*a].partial_cmp(&tiebreak[*b]).unwrap_or(std::cmp::Ordering::Equal)
      } else {
        base
      }
    });
    for (position, g) in eligible.iter().enumerate() {
      r[*g * rows + t] = NumT::from(position + 1).unwrap();
    }
  }
  Ok(())
}

/// Cross-sectional z-score with sample standard deviation (ddof=1).
/// Constant rows produce 0 for observed entries, matching the pandas backend.
pub fn ta_ft_zscore<NumT: Float + Send + Sync>(
  ctx: &Context, r: &mut [NumT], input: &[NumT],
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }
  r.fill(NumT::nan());
  let (groups, rows) = layout(input.len(), ctx.groups());
  for t in 0..rows {
    let mut sum = NumT::zero();
    let mut count = 0usize;
    for g in 0..groups {
      let v = input[g * rows + t];
      if is_normal(&v) {
        sum = sum + v;
        count += 1;
      }
    }
    if count == 0 {
      continue;
    }
    let n = NumT::from(count).unwrap();
    let mean = sum / n;
    if count == 1 {
      for g in 0..groups {
        let index = g * rows + t;
        if is_normal(&input[index]) {
          r[index] = NumT::zero();
        }
      }
      continue;
    }
    let mut m2 = NumT::zero();
    for g in 0..groups {
      let v = input[g * rows + t];
      if is_normal(&v) {
        let d = v - mean;
        m2 = m2 + d * d;
      }
    }
    let std = (m2 / (n - NumT::one())).sqrt();
    for g in 0..groups {
      let index = g * rows + t;
      if is_normal(&input[index]) {
        r[index] = if std > NumT::zero() {
          (input[index] - mean) / std
        } else {
          NumT::zero()
        };
      }
    }
  }
  Ok(())
}

fn pair_rows<NumT: Float>(
  rows: usize,
  groups: usize,
  left: &[NumT],
  right: &[NumT],
  t: usize,
  rank: bool,
) -> (Vec<NumT>, Vec<NumT>, Vec<usize>) {
  let mut xs = Vec::with_capacity(groups);
  let mut ys = Vec::with_capacity(groups);
  let mut positions = Vec::with_capacity(groups);
  for g in 0..groups {
    let index = g * rows + t;
    if is_normal(&left[index]) && is_normal(&right[index]) {
      xs.push(left[index]);
      ys.push(right[index]);
      positions.push(index);
    }
  }
  if rank && !xs.is_empty() {
    let (_, rx) = average_ranks(&xs);
    let (_, ry) = average_ranks(&ys);
    xs = rx;
    ys = ry;
  }
  (xs, ys, positions)
}

fn pair_stat<NumT: Float + Send + Sync>(
  ctx: &Context, r: &mut [NumT], left: &[NumT], right: &[NumT], correlation: bool, rank: bool,
) -> Result<(), Error> {
  if r.len() != left.len() || left.len() != right.len() {
    return Err(Error::LengthMismatch(left.len(), right.len()));
  }
  r.fill(NumT::nan());
  let (groups, rows) = layout(left.len(), ctx.groups());
  for t in 0..rows {
    let (xs, ys, positions) = pair_rows(rows, groups, left, right, t, rank);
    let n = xs.len();
    if n < 2 {
      continue;
    }
    let count = NumT::from(n).unwrap();
    let mut sx = NumT::zero();
    let mut sy = NumT::zero();
    let mut sxx = NumT::zero();
    let mut syy = NumT::zero();
    let mut sxy = NumT::zero();
    for i in 0..n {
      sx = sx + xs[i];
      sy = sy + ys[i];
      sxx = sxx + xs[i] * xs[i];
      syy = syy + ys[i] * ys[i];
      sxy = sxy + xs[i] * ys[i];
    }
    let covariance = (sxy - sx * sy / count) / (count - NumT::one());
    if !correlation {
      for position in &positions {
        r[*position] = covariance;
      }
      continue;
    }
    let var_x = sxx - sx * sx / count;
    let var_y = syy - sy * sy / count;
    if !(var_x > NumT::zero()) || !(var_y > NumT::zero()) {
      continue;
    }
    let value = (sxy - sx * sy / count) / (var_x * var_y).sqrt();
    for position in &positions {
      r[*position] = value;
    }
  }
  Ok(())
}

/// Cross-sectional Pearson correlation per timestamp.
pub fn ta_ft_corr<NumT: Float + Send + Sync>(
  ctx: &Context, r: &mut [NumT], left: &[NumT], right: &[NumT],
) -> Result<(), Error> {
  pair_stat(ctx, r, left, right, true, false)
}

/// Cross-sectional covariance per timestamp (ddof=1).
pub fn ta_ft_cov<NumT: Float + Send + Sync>(
  ctx: &Context, r: &mut [NumT], left: &[NumT], right: &[NumT],
) -> Result<(), Error> {
  pair_stat(ctx, r, left, right, false, false)
}

/// Cross-sectional Spearman correlation per timestamp.
pub fn ta_ft_spearman<NumT: Float + Send + Sync>(
  ctx: &Context, r: &mut [NumT], left: &[NumT], right: &[NumT],
) -> Result<(), Error> {
  pair_stat(ctx, r, left, right, true, true)
}

/// Category-partitioned cross-sectional transform: 0 = rank, 1 = z-score,
/// 2 = demean. `category` holds one category id per group (-1 = unassigned).
pub fn ta_ft_group<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  mask: &[NumT],
  category: &[NumT],
  op: usize,
) -> Result<(), Error> {
  if r.len() != input.len() || mask.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }
  r.fill(NumT::nan());
  let (groups, rows) = layout(input.len(), ctx.groups());
  let max_id = category.iter().map(|v| v.to_usize().unwrap_or(0)).max().unwrap_or(0);
  for t in 0..rows {
    for id in 0..=max_id {
      let mut values = Vec::new();
      let mut positions = Vec::new();
      for g in 0..groups {
        if category[g].to_usize().unwrap_or(usize::MAX) != id {
          continue;
        }
        let index = g * rows + t;
        if is_normal(&input[index]) && (mask[index] > NumT::zero()) {
          values.push(input[index]);
          positions.push(index);
        }
      }
      if values.is_empty() {
        continue;
      }
      match op {
        0 => {
          let (_, ranks) = average_ranks(&values);
          let count = NumT::from(values.len()).unwrap();
          for (i, position) in positions.iter().enumerate() {
            r[*position] = ranks[i] / count - NumT::from(0.5).unwrap();
          }
        }
        1 => {
          if values.len() < 2 {
            for position in &positions {
              r[*position] = NumT::nan();
            }
            continue;
          }
          let count = NumT::from(values.len()).unwrap();
          let mean = values.iter().fold(NumT::zero(), |a, v| a + *v) / count;
          let mut m2 = NumT::zero();
          for value in &values {
            let d = *value - mean;
            m2 = m2 + d * d;
          }
          let std = (m2 / (count - NumT::one())).sqrt();
          for (i, position) in positions.iter().enumerate() {
            r[*position] = if std > NumT::zero() {
              (values[i] - mean) / std
            } else {
              NumT::zero()
            };
          }
        }
        _ => {
          let count = NumT::from(values.len()).unwrap();
          let mean = values.iter().fold(NumT::zero(), |a, v| a + *v) / count;
          for (i, position) in positions.iter().enumerate() {
            r[*position] = values[i] - mean;
          }
        }
      }
    }
  }
  Ok(())
}

/// Cross-sectional OLS residualization on up to three exposures.
pub fn ta_ft_residualize<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  signal: &[NumT],
  exposure1: &[NumT],
  exposure2: &[NumT],
  exposure3: &[NumT],
  mask: &[NumT],
  exposure_count: usize,
) -> Result<(), Error> {
  if r.len() != signal.len() || mask.len() != signal.len() {
    return Err(Error::LengthMismatch(r.len(), signal.len()));
  }
  let k = exposure_count.min(3);
  r.fill(NumT::nan());
  let (groups, rows) = layout(signal.len(), ctx.groups());
  for t in 0..rows {
    let mut ys: Vec<NumT> = Vec::new();
    let mut rows_x: Vec<[NumT; 3]> = Vec::new();
    let mut positions: Vec<usize> = Vec::new();
    for g in 0..groups {
      let index = g * rows + t;
      if !(mask[index] > NumT::zero()) || !is_normal(&signal[index]) {
        continue;
      }
      let x1 = exposure1[index];
      if !is_normal(&x1) {
        continue;
      }
      let mut values = [NumT::zero(); 3];
      values[0] = x1;
      if k >= 2 {
        if !is_normal(&exposure2[index]) {
          continue;
        }
        values[1] = exposure2[index];
      }
      if k >= 3 {
        if !is_normal(&exposure3[index]) {
          continue;
        }
        values[2] = exposure3[index];
      }
      ys.push(signal[index]);
      rows_x.push(values);
      positions.push(index);
    }
    let n = ys.len();
    if n <= k + 1 {
      continue;
    }
    // normal equations for [1, x1..xk]
    let size = k + 1;
    let mut xtx = vec![vec![NumT::zero(); size]; size];
    let mut xty = vec![NumT::zero(); size];
    for (i, y) in ys.iter().enumerate() {
      let mut row = [NumT::zero(); 3];
      row[0] = NumT::one();
      for j in 0..k {
        row[j + 1] = rows_x[i][j];
      }
      for a in 0..size {
        xty[a] = xty[a] + row[a] * *y;
        for b in 0..size {
          xtx[a][b] = xtx[a][b] + row[a] * row[b];
        }
      }
    }
    // gaussian elimination with partial pivoting
    for col in 0..size {
      let mut pivot = col;
      for row in col + 1..size {
        if xtx[row][col].abs() > xtx[pivot][col].abs() {
          pivot = row;
        }
      }
      if !(xtx[pivot][col].abs() > NumT::from(1e-18).unwrap()) {
        continue;
      }
      xtx.swap(col, pivot);
      xty.swap(col, pivot);
      let diag = xtx[col][col];
      for b in col..size {
        xtx[col][b] = xtx[col][b] / diag;
      }
      xty[col] = xty[col] / diag;
      for row in 0..size {
        if row == col {
          continue;
        }
        let factor = xtx[row][col];
        if factor == NumT::zero() {
          continue;
        }
        for b in col..size {
          xtx[row][b] = xtx[row][b] - factor * xtx[col][b];
        }
        xty[row] = xty[row] - factor * xty[col];
      }
    }
    for (i, position) in positions.iter().enumerate() {
      let mut fitted = xty[0];
      for j in 0..k {
        fitted = fitted + xty[j + 1] * rows_x[i][j];
      }
      r[*position] = ys[i] - fitted;
    }
  }
  Ok(())
}
