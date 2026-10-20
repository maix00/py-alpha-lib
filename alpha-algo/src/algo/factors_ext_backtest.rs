// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;

use crate::algo::{Context, Error};

/// Batch searchsorted-left over ragged sorted index arrays.
///
/// ``indices`` is the concatenation of per-product index values (relative to
/// ``query``); ``offsets`` has P+1 entries. For each product the first position
/// with index >= query is returned, or -1 when the product has no such bar.
/// The caller adds the execution delay and constructs timestamps.
pub fn ta_ft_exec_positions<NumT: Float + Send + Sync>(
  _ctx: &Context,
  r: &mut [NumT],
  indices: &[NumT],
  offsets: &[NumT],
  query: NumT,
  _delay: usize,
) -> Result<(), Error> {
  if r.len() != indices.len() {
    return Err(Error::LengthMismatch(r.len(), indices.len()));
  }
  let products = offsets.len().saturating_sub(1);
  for product in 0..products {
    let start = offsets[product].to_usize().unwrap_or(0);
    let end = offsets[product + 1].to_usize().unwrap_or(start);
    if start > end || end > indices.len() {
      continue;
    }
    let mut low = start;
    let mut high = end;
    while low < high {
      let mid = (low + high) / 2;
      if indices[mid] < query {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    let value = if low >= end {
      NumT::from(-1i32).unwrap()
    } else {
      NumT::from(low - start).unwrap()
    };
    for offset in start..end {
      r[offset] = value;
    }
  }
  Ok(())
}

/// Batch execution timestamp resolution for ragged sorted index arrays.
///
/// ``indices``/``offsets`` are relative-ns ragged arrays (query = 0).
/// ``mode`` 0 returns the price timestamp, 1 the visibility timestamp.
pub fn ta_ft_exec_schedule_ts<NumT: Float + Send + Sync>(
  _ctx: &Context,
  r: &mut [NumT],
  indices: &[NumT],
  offsets: &[NumT],
  delay: usize,
  freq: NumT,
  open_delay: NumT,
  end_delay: NumT,
  open_basis: usize,
  mode: usize,
) -> Result<(), Error> {
  if r.len() != indices.len() {
    return Err(Error::LengthMismatch(r.len(), indices.len()));
  }
  let products = offsets.len().saturating_sub(1);
  for product in 0..products {
    let start = offsets[product].to_usize().unwrap_or(0);
    let end = offsets[product + 1].to_usize().unwrap_or(start);
    if start >= end || end > indices.len() {
      continue;
    }
    let mut low = start;
    let mut high = end;
    while low < high {
      let mid = (low + high) / 2;
      if indices[mid] < NumT::zero() { low = mid + 1 } else { high = mid }
    }
    let price_pos = low + delay;
    let value = if price_pos >= end {
      NumT::nan()
    } else if mode == 0 {
      indices[price_pos]
    } else if open_basis != 0 {
      if price_pos == start {
        NumT::nan()
      } else {
        let previous = indices[price_pos - 1] + open_delay;
        let bar_open = if freq > NumT::zero() {
          indices[price_pos] - freq + open_delay
        } else {
          previous
        };
        if previous > bar_open { previous } else { bar_open }
      }
    } else {
      indices[price_pos] + end_delay
    };
    for offset in start..end {
      r[offset] = value;
    }
  }
  Ok(())
}
