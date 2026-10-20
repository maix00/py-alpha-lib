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
pub struct LedgerState {
  accounts: std::collections::HashMap<String, LedgerAccount>,
  margins: std::collections::HashMap<String, MarginAccount>,
}

#[pymethods]
impl LedgerState {
  #[new]
  fn new() -> Self {
    Self {
      accounts: std::collections::HashMap::new(),
      margins: std::collections::HashMap::new(),
    }
  }

  fn ensure(&mut self, ledger_id: &str, cash: f64) {
    self.accounts.entry(ledger_id.to_string()).or_insert_with(|| LedgerAccount::new(cash));
  }

  fn set_cash(&mut self, ledger_id: &str, cash: f64) {
    let account = self.accounts.entry(ledger_id.to_string()).or_insert_with(|| LedgerAccount::new(cash));
    account.cash = cash;
  }

  fn set_position(&mut self, ledger_id: &str, product: &str, quantity: f64, cost: f64) {
    let account = self.accounts.entry(ledger_id.to_string()).or_insert_with(|| LedgerAccount::new(0.0));
    account.positions.insert(product.to_string(), (quantity, cost));
  }

  fn apply_fill(
    &mut self, ledger_id: &str, product: &str,
    quantity: f64, price: f64, multiplier: f64, fee: f64,
    margin_ratio: f64, margin_accounting: bool,
  ) -> f64 {
    let account = self.accounts.get_mut(ledger_id);
    let Some(account) = account else { return 0.0 };
    let realized = account.apply_fill(product, quantity, price, multiplier, fee);
    // Fully-funded cash accounting settles notional, not realized PnL.
    account.cash -= realized;
    if !margin_accounting {
      account.cash -= quantity * price * multiplier;
    }
    if margin_accounting {
      let margin = self.margins.entry(ledger_id.to_string()).or_insert_with(MarginAccount::new);
      let (position, _) = account.position(product);
      margin.update(product, position, price, multiplier, margin_ratio);
    }
    realized
  }

  fn cash(&self, ledger_id: &str) -> f64 {
    self.accounts.get(ledger_id).map(|account| account.cash).unwrap_or(0.0)
  }

  fn position(&self, ledger_id: &str, product: &str) -> (f64, f64) {
    self.accounts.get(ledger_id).map(|account| account.position(product)).unwrap_or((0.0, 0.0))
  }

  fn reserved(&self, ledger_id: &str) -> f64 {
    self.margins.get(ledger_id).map(|account| account.reserved).unwrap_or(0.0)
  }
}

#[pyclass(unsendable)]
pub struct MarginAccount {
  reserved: f64,
  by_product: std::collections::HashMap<String, f64>,
}

#[pymethods]
impl MarginAccount {
  #[new]
  fn new() -> Self { Self { reserved: 0.0, by_product: std::collections::HashMap::new() } }

  fn update(&mut self, product: &str, quantity: f64, price: f64, multiplier: f64, ratio: f64) -> f64 {
    let margin = quantity.abs() * price * multiplier * ratio;
    let previous = self.by_product.insert(product.to_string(), margin).unwrap_or(0.0);
    self.reserved += margin - previous;
    margin
  }

  #[getter]
  fn reserved(&self) -> f64 { self.reserved }

  fn product_margin(&self, product: &str) -> f64 {
    self.by_product.get(product).copied().unwrap_or(0.0)
  }
}

#[pyclass(unsendable)]
pub struct OrderCore {
  quantity: f64,
  filled: f64,
  status: i32,
  average_price: f64,
}

#[pymethods]
impl OrderCore {
  #[new]
  fn new(quantity: f64) -> Self {
    Self { quantity: quantity.abs(), filled: 0.0, status: 0, average_price: 0.0 }
  }

  fn apply_fill(&mut self, quantity: f64, price: f64) -> f64 {
    if self.status != 0 && self.status != 4 {
      return 0.0;
    }
    let remaining = (self.quantity - self.filled).max(0.0);
    let fill = quantity.abs().min(remaining);
    if fill > 0.0 {
      let total = self.average_price * self.filled + price * fill;
      self.filled += fill;
      self.average_price = total / self.filled;
      if self.filled >= self.quantity - 1e-12 {
        self.filled = self.quantity;
        self.status = 1;
      } else {
        self.status = 4;
      }
    }
    fill
  }

  fn cancel(&mut self) { if self.status == 0 { self.status = 2; } }
  fn reject(&mut self) { if self.status == 0 { self.status = 3; } }

  #[getter]
  fn quantity(&self) -> f64 { self.quantity }
  #[getter]
  fn filled(&self) -> f64 { self.filled }
  #[getter]
  fn remaining(&self) -> f64 { (self.quantity - self.filled).max(0.0) }
  #[getter]
  fn status(&self) -> i32 { self.status }
  #[getter]
  fn average_price(&self) -> f64 { self.average_price }
}

#[pyclass(unsendable)]
pub struct LedgerAccount {
  cash: f64,
  positions: std::collections::HashMap<String, (f64, f64)>,
}

#[pymethods]
impl LedgerAccount {
  #[new]
  fn new(cash: f64) -> Self {
    Self { cash, positions: std::collections::HashMap::new() }
  }

  fn apply_fill(
    &mut self, product: &str, quantity: f64, price: f64,
    multiplier: f64, fee: f64,
  ) -> f64 {
    let (prior_quantity, prior_cost) = self.positions
      .get(product).copied().unwrap_or((0.0, 0.0));
    let new_quantity = prior_quantity + quantity;
    let epsilon = 1e-12;
    let mut realized = 0.0;
    let mut new_cost = prior_cost;
    if prior_quantity == 0.0 || (prior_quantity > 0.0) == (quantity > 0.0) {
      new_cost = if new_quantity.abs() <= epsilon {
        0.0
      } else {
        (prior_quantity * prior_cost + quantity * price) / new_quantity
      };
    } else {
      let close_abs = prior_quantity.abs().min(quantity.abs());
      realized = close_abs * (price - prior_cost) * prior_quantity.signum() * multiplier;
      if new_quantity.abs() <= epsilon {
        new_cost = 0.0;
      } else if quantity.abs() > prior_quantity.abs() {
        new_cost = price;
      }
    }
    let quantity = if new_quantity.abs() <= epsilon { 0.0 } else { new_quantity };
    self.positions.insert(product.to_string(), (quantity, new_cost));
    self.cash += realized - fee;
    realized
  }

  fn apply_fill_batch(
    &mut self, product: &str,
    quantities: numpy::PyReadonlyArray1<'_, f64>,
    prices: numpy::PyReadonlyArray1<'_, f64>,
    multipliers: numpy::PyReadonlyArray1<'_, f64>,
    fees: numpy::PyReadonlyArray1<'_, f64>,
  ) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let quantities = quantities.as_slice()?;
    let prices = prices.as_slice()?;
    let multipliers = multipliers.as_slice()?;
    let fees = fees.as_slice()?;
    let length = quantities.len();
    if prices.len() != length || multipliers.len() != length || fees.len() != length {
      return Err(pyo3::exceptions::PyValueError::new_err("fill batch length mismatch"));
    }
    let (mut prior_quantity, mut prior_cost) = self.positions
      .get(product).copied().unwrap_or((0.0, 0.0));
    let epsilon = 1e-12;
    let mut realized = Vec::with_capacity(length);
    let mut ended_quantity = Vec::with_capacity(length);
    let mut ended_cost = Vec::with_capacity(length);
    for index in 0..length {
      let quantity = quantities[index];
      let price = prices[index];
      let new_quantity = prior_quantity + quantity;
      let mut pnl = 0.0;
      let mut new_cost = prior_cost;
      if prior_quantity == 0.0 || (prior_quantity > 0.0) == (quantity > 0.0) {
        new_cost = if new_quantity.abs() <= epsilon {
          0.0
        } else {
          (prior_quantity * prior_cost + quantity * price) / new_quantity
        };
      } else {
        let close_abs = prior_quantity.abs().min(quantity.abs());
        pnl = close_abs * (price - prior_cost) * prior_quantity.signum() * multipliers[index];
        if new_quantity.abs() <= epsilon {
          new_cost = 0.0;
        } else if quantity.abs() > prior_quantity.abs() {
          new_cost = price;
        }
      }
      prior_quantity = if new_quantity.abs() <= epsilon { 0.0 } else { new_quantity };
      prior_cost = new_cost;
      self.cash += pnl - fees[index];
      realized.push(pnl);
      ended_quantity.push(prior_quantity);
      ended_cost.push(prior_cost);
    }
    self.positions.insert(product.to_string(), (prior_quantity, prior_cost));
    Ok((realized, ended_quantity, ended_cost))
  }

  fn position(&self, product: &str) -> (f64, f64) {
    self.positions.get(product).copied().unwrap_or((0.0, 0.0))
  }

  #[getter]
  fn cash(&self) -> f64 { self.cash }
}

#[pyclass(unsendable)]
pub struct MarketDataMatrix {
  timestamps: Vec<i64>,
  values: Vec<f64>,
  rows: usize,
  columns: usize,
  labels: Vec<Py<PyAny>>,
}

#[pymethods]
impl MarketDataMatrix {
  #[new]
  fn new(
    py: Python<'_>,
    timestamps: numpy::PyReadonlyArray1<'_, i64>,
    values: numpy::PyReadonlyArray2<'_, f64>,
    labels: Vec<Py<PyAny>>,
  ) -> PyResult<Self> {
    let timestamps = timestamps.as_slice()?.to_vec();
    let array = values.as_array();
    let (rows, columns) = array.dim();
    if labels.len() != columns {
      return Err(pyo3::exceptions::PyValueError::new_err("label count mismatch"));
    }
    let values = array.iter().copied().collect::<Vec<f64>>();
    Ok(Self { timestamps, values, rows, columns, labels })
  }

  fn row_at(&self, py: Python<'_>, timestamp: i64, asof: bool) -> PyResult<Py<PyAny>> {
    let dict = pyo3::types::PyDict::new(py);
    let position = if asof {
      match self.timestamps.binary_search(&timestamp) {
        Ok(index) => index as isize,
        Err(0) => -1,
        Err(index) => index as isize - 1,
      }
    } else {
      match self.timestamps.binary_search(&timestamp) {
        Ok(index) => index as isize,
        Err(_) => -1,
      }
    };
    if position < 0 || position as usize >= self.rows {
      return Ok(dict.into());
    }
    let row = position as usize;
    for column in 0..self.columns {
      let value = self.values[row * self.columns + column];
      if !value.is_nan() {
        dict.set_item(self.labels[column].clone_ref(py), value)?;
      }
    }
    Ok(dict.into())
  }

  #[getter]
  fn rows(&self) -> usize { self.rows }

  #[getter]
  fn columns(&self) -> usize { self.columns }
}

#[pyclass(unsendable)]
pub struct EventHeap {
  heap: std::cell::RefCell<std::collections::BinaryHeap<std::cmp::Reverse<EventHeapEntry>>>,
  counter: std::cell::Cell<u64>,
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
    Self {
      heap: std::cell::RefCell::new(std::collections::BinaryHeap::new()),
      counter: std::cell::Cell::new(0),
    }
  }

  fn push(&self, timestamp: i64, kind: i32, sequence: i32, payload: Py<PyAny>) {
    let counter = self.counter.get();
    self.counter.set(counter + 1);
    self.heap.borrow_mut().push(std::cmp::Reverse(EventHeapEntry {
      timestamp, kind, sequence, counter, payload,
    }));
  }

  fn push_with_counter(&self, timestamp: i64, kind: i32, sequence: i32, counter: u64, payload: Py<PyAny>) {
    self.heap.borrow_mut().push(std::cmp::Reverse(EventHeapEntry {
      timestamp, kind, sequence, counter, payload,
    }));
    if counter >= self.counter.get() {
      self.counter.set(counter + 1);
    }
  }

  fn pop(&self, py: Python<'_>) -> Option<(i64, i32, i32, u64, Py<PyAny>)> {
    self.heap.borrow_mut().pop().map(|std::cmp::Reverse(entry)| {
      (entry.timestamp, entry.kind, entry.sequence, entry.counter, entry.payload.clone_ref(py))
    })
  }

  fn pop_batch(&self, py: Python<'_>) -> Option<(i32, Vec<Py<PyAny>>)> {
    let first = self.pop(py)?;
    let (first_ts, first_kind, _sequence, _counter, first_payload) = first;
    let mut batch = vec![first_payload];
    loop {
      let same = self.heap.borrow().peek().map(|std::cmp::Reverse(entry)| {
        entry.timestamp == first_ts && entry.kind == first_kind
      }).unwrap_or(false);
      if !same {
        break;
      }
      let next = self.pop(py)?;
      batch.push(next.4);
    }
    Some((first_kind, batch))
  }

  fn peek_key(&self) -> Option<(i64, i32, i32, u64)> {
    self.heap.borrow().peek().map(|std::cmp::Reverse(entry)| {
      (entry.timestamp, entry.kind, entry.sequence, entry.counter)
    })
  }

  #[getter]
  fn len(&self) -> usize {
    self.heap.borrow().len()
  }

  fn run(&self, py: Python<'_>, callback: &Bound<'_, PyAny>) -> PyResult<()> {
    use pyo3::types::PyList;
    loop {
      let entry = {
        let batch = self.pop_batch(py);
        batch
      };
      let Some((kind, items)) = entry else { break };
      let list = PyList::new(py, items)?;
      callback.call1((kind, list))?;
    }
    Ok(())
  }
}

#[pymodule]
fn _algo(m: &Bound<'_, PyModule>) -> PyResult<()> {
  use algo_impl::*;

  pyo3_log::init();

  m.add_function(wrap_pyfunction!(ema, m)?)?;
  m.add_function(wrap_pyfunction!(set_ctx, m)?)?;
  m.add_class::<EventHeap>()?;
  m.add_class::<MarketDataMatrix>()?;
  m.add_class::<LedgerAccount>()?;
  m.add_class::<OrderCore>()?;
  m.add_class::<MarginAccount>()?;
  m.add_class::<LedgerState>()?;
  register_functions(m)?;
  Ok(())
}
