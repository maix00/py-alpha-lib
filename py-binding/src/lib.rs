// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use std::sync::{LazyLock, RwLock};

use alpha_algo::Context;
use pyo3::prelude::*;

static _ALGO_CTX_: LazyLock<RwLock<Context>> = LazyLock::new(|| RwLock::new(Context::default()));

mod algo_impl {
  use log::debug;
  use numpy::{PyReadonlyArray1, PyReadwriteArray1};
  use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};
  use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

  use crate::_ALGO_CTX_;

  use alpha_algo::*;

  fn ctx<'py>(_py: Python<'py>) -> Context {
    let ctx = match _ALGO_CTX_.try_read() {
      Ok(ctx) => ctx.clone(),
      Err(_e) => Context::default(),
    };
    debug!("current {}", ctx);
    ctx
  }

  #[pyfunction]
  #[pyo3(signature = (/, start=None, end=None, groups=None, flags=None))]
  pub fn set_ctx<'py>(
    _py: Python<'py>,
    start: Option<i32>,
    end: Option<i32>,
    groups: Option<u32>,
    flags: Option<u64>,
  ) -> PyResult<()> {
    match _ALGO_CTX_.try_write() {
      Ok(mut ctx) => {
        start.map(|s| ctx._start = s);
        end.map(|e| ctx._end = e);
        groups.map(|g| ctx._groups = g);
        flags.map(|f| ctx._flags = f);
        Ok(())
      }
      Err(_e) => Err(PyValueError::new_err("failed to set context")),
    }
  }

  /// Exponential Moving Average (variant of EMA)
  ///
  /// alpha = 2 / (n + 1)
  ///
  /// https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
  ///
  #[pyfunction]
  pub fn ema<'py>(
    py: Python<'py>,
    r: &'py Bound<'_, PyAny>,
    input: &'py Bound<'_, PyAny>,
    periods: usize,
  ) -> PyResult<()> {
    // 1. get context
    let mut ctx = ctx(py);

    // 2. check input type and do dispatch
    if let Some((mut r, input)) = r
      .extract::<PyReadwriteArray1<'py, f64>>()
      .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, f64>>().ok())
    {
      // input is f64 array
      let mut r = r.as_array_mut();
      let r = r
        .as_slice_mut()
        .ok_or(PyValueError::new_err("failed to get mutable slice"))?;

      let input = input.as_array();
      let input = input
        .as_slice()
        .ok_or(PyValueError::new_err("failed to get slice"))?;
      ta_ema(&ctx, r, input, periods).map_err(|e| e.into())
    } else if let Some((mut r, input)) = r
      .extract::<PyReadwriteArray1<'py, f32>>()
      .ok()
      .zip(input.extract::<PyReadonlyArray1<'py, f32>>().ok())
    {
      // input is f32 array
      let mut r = r.as_array_mut();
      let r = r
        .as_slice_mut()
        .ok_or(PyValueError::new_err("invalid input"))?;

      let input = input.as_array();
      let input = input
        .as_slice()
        .ok_or(PyValueError::new_err("invalid input"))?;
      ta_ema(&ctx, r, input, periods).map_err(|e| e.into())
    } else if let Some((r, input)) = r.cast::<PyList>().ok().zip(input.cast::<PyList>().ok()) {
      // input is list of arrays

      // each array is a group, ensure groups is set to 1
      ctx._groups = 1;

      if r.len() != input.len() {
        return Err(PyValueError::new_err("length mismatch"));
      }

      // check if each array is f64 array
      if let Some((mut r, input)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f64>>>()
        .ok()
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, f64>>>().ok())
      {
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();

        let mut _r = vec![];
        r.into_par_iter()
          .zip(input.into_par_iter())
          .map(|(mut out, input)| {
            let out = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((out, input)) = out.zip(input) {
              ta_ema(&ctx, out, input, periods).map_err(|e| e.into())
            } else {
              Err(PyValueError::new_err("invalid input"))
            }
          })
          .collect_into_vec(&mut _r);

        match _r.into_iter().find(|x| x.is_err()) {
          Some(e) => e,
          None => Ok(()),
        }
      // check if each array is f32 array
      } else if let Some((mut r, input)) = r
        .extract::<Vec<PyReadwriteArray1<'py, f32>>>()
        .ok()
        .zip(input.extract::<Vec<PyReadonlyArray1<'py, f32>>>().ok())
      {
        let r = r.iter_mut().map(|x| x.as_array_mut()).collect::<Vec<_>>();
        let input = input.iter().map(|x| x.as_array()).collect::<Vec<_>>();

        let mut _r = vec![];
        r.into_par_iter()
          .zip(input.into_par_iter())
          .map(|(mut out, input)| {
            let out = out.as_slice_mut();
            let input = input.as_slice();
            if let Some((out, input)) = out.zip(input) {
              ta_ema(&ctx, out, input, periods).map_err(|e| e.into())
            } else {
              Err(PyValueError::new_err("invalid input"))
            }
          })
          .collect_into_vec(&mut _r);

        match _r.into_iter().find(|x| x.is_err()) {
          Some(e) => e,
          None => Ok(()),
        }
      } else {
        // NumT array can only be f64 or f32
        Err(PyValueError::new_err("invalid input"))
      }
    } else {
      // NumT array can only be f64 or f32
      Err(PyValueError::new_err("invalid input"))
    }
  }

  include!("./algo_bindings.rs");
}

#[pyclass(unsendable)]
pub struct EventHeap {
  heap: std::collections::BinaryHeap<std::cmp::Reverse<EventHeapEntry>>,
  counter: u64,
}

struct EventHeapEntry {
  timestamp: i64,
  kind: i32,
  sequence: i32,
  counter: u64,
  payload: Py<PyAny>,
}

impl PartialEq for EventHeapEntry {
  fn eq(&self, other: &Self) -> bool {
    (self.timestamp, self.kind, self.sequence, self.counter)
      == (other.timestamp, other.kind, other.sequence, other.counter)
  }
}
impl Eq for EventHeapEntry {}
impl PartialOrd for EventHeapEntry {
  fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
    Some(self.cmp(other))
  }
}
impl Ord for EventHeapEntry {
  fn cmp(&self, other: &Self) -> std::cmp::Ordering {
    (self.timestamp, self.kind, self.sequence, self.counter)
      .cmp(&(other.timestamp, other.kind, other.sequence, other.counter))
  }
}

#[pymethods]
impl EventHeap {
  #[new]
  fn new() -> Self {
    Self { heap: std::collections::BinaryHeap::new(), counter: 0 }
  }

  fn push(&mut self, timestamp: i64, kind: i32, sequence: i32, payload: Py<PyAny>) {
    let counter = self.counter;
    self.counter += 1;
    self.heap.push(std::cmp::Reverse(EventHeapEntry {
      timestamp, kind, sequence, counter, payload,
    }));
  }

  fn push_with_counter(&mut self, timestamp: i64, kind: i32, sequence: i32, counter: u64, payload: Py<PyAny>) {
    self.heap.push(std::cmp::Reverse(EventHeapEntry {
      timestamp, kind, sequence, counter, payload,
    }));
    if counter >= self.counter {
      self.counter = counter + 1;
    }
  }

  fn pop(&mut self, py: Python<'_>) -> Option<(i64, i32, i32, u64, Py<PyAny>)> {
    self.heap.pop().map(|std::cmp::Reverse(entry)| {
      (entry.timestamp, entry.kind, entry.sequence, entry.counter, entry.payload.clone_ref(py))
    })
  }

  fn pop_batch(&mut self, py: Python<'_>) -> Option<(i32, Vec<Py<PyAny>>)> {
    let first = self.pop(py)?;
    let (first_ts, first_kind, _sequence, _counter, first_payload) = first;
    let mut batch = vec![first_payload];
    while let Some(std::cmp::Reverse(entry)) = self.heap.peek() {
      if entry.timestamp != first_ts || entry.kind != first_kind {
        break;
      }
      let next = self.pop(py)?;
      batch.push(next.4);
    }
    Some((first_kind, batch))
  }

  fn peek_key(&self) -> Option<(i64, i32, i32, u64)> {
    self.heap.peek().map(|std::cmp::Reverse(entry)| {
      (entry.timestamp, entry.kind, entry.sequence, entry.counter)
    })
  }

  #[getter]
  fn len(&self) -> usize {
    self.heap.len()
  }
}

#[pymodule]
fn _algo(m: &Bound<'_, PyModule>) -> PyResult<()> {
  use algo_impl::*;

  pyo3_log::init();

  m.add_function(wrap_pyfunction!(ema, m)?)?;
  m.add_function(wrap_pyfunction!(set_ctx, m)?)?;
  m.add_class::<EventHeap>()?;
  register_functions(m)?;
  Ok(())
}
