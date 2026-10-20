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
