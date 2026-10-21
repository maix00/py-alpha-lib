// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use std::sync::{LazyLock, RwLock};

use alpha_algo::Context;
use pyo3::prelude::*;
use numpy::PyArrayMethods;

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
pub struct LotBook {
  method: i32,
  lots: Vec<(f64, f64, f64, i32)>,
}

#[pymethods]
impl LotBook {
  #[new]
  fn new(method: i32) -> Self { Self { method, lots: Vec::new() } }

  fn apply_fill(&mut self, quantity: f64, price: f64, multiplier: f64, is_today: i32, offset: i32) -> f64 {
    let prior: f64 = self.lots.iter().map(|lot| lot.0).sum();
    if prior == 0.0 || (prior > 0.0) == (quantity > 0.0) {
      self.lots.push((quantity, price, multiplier, is_today));
      return 0.0;
    }
    let close_abs = prior.abs().min(quantity.abs());
    let mut remaining = close_abs;
    let mut realized = 0.0;
    let mut order: Vec<usize> = (0..self.lots.len()).collect();
    if self.method == 1 {
      order.reverse();
    } else if self.method == 2 {
      order.sort_by(|&left, &right| self.lots[right].1.partial_cmp(&self.lots[left].1).unwrap_or(std::cmp::Ordering::Equal));
    }
    for index in order {
      if remaining <= 1e-12 { break; }
      let today = self.lots[index].3;
      if (offset == 1 && today != 1) || (offset == 2 && today != 0) { continue; }
      let take = remaining.min(self.lots[index].0.abs());
      realized += take * (price - self.lots[index].1) * self.lots[index].2;
      self.lots[index].0 -= if self.lots[index].0 > 0.0 { take } else { -take };
      remaining -= take;
    }
    self.lots.retain(|lot| lot.0.abs() > 1e-12);
    let flip = quantity.abs() - close_abs;
    if flip > 1e-12 {
      self.lots.push((if quantity > 0.0 { flip } else { -flip }, price, multiplier, is_today));
    }
    if prior > 0.0 { realized } else { -realized }
  }

  fn quantity(&self) -> f64 { self.lots.iter().map(|lot| lot.0).sum() }

  fn lots_snapshot(&self) -> Vec<(f64, f64, f64, i32)> { self.lots.clone() }
}

#[pyclass(unsendable)]
pub struct LedgerState {
  accounts: std::collections::HashMap<String, LedgerAccount>,
  margins: std::collections::HashMap<String, MarginAccount>,
  cost_methods: std::collections::HashMap<String, i32>,
  lots: std::collections::HashMap<String, std::collections::HashMap<String, LotBook>>,
}

#[pymethods]
impl LedgerState {
  #[new]
  fn new() -> Self {
    Self {
      accounts: std::collections::HashMap::new(),
      margins: std::collections::HashMap::new(),
      cost_methods: std::collections::HashMap::new(),
      lots: std::collections::HashMap::new(),
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

  fn apply_fills_batch(
    &mut self, ledger_id: &str, products: Vec<String>,
    quantities: Vec<f64>, prices: Vec<f64>, multipliers: Vec<f64>,
    fees: Vec<f64>, margin_ratios: Vec<f64>, margin_accounting: bool,
  ) -> PyResult<Vec<(f64, f64, f64)>> {
    let length = products.len();
    if quantities.len() != length || prices.len() != length
      || multipliers.len() != length || fees.len() != length
      || margin_ratios.len() != length
    {
      return Err(pyo3::exceptions::PyValueError::new_err("fill batch length mismatch"));
    }
    let mut result = Vec::with_capacity(length);
    for index in 0..length {
      let Some(account) = self.accounts.get_mut(ledger_id) else {
        result.push((0.0, 0.0, 0.0));
        continue;
      };
      let realized = account.apply_fill(
        &products[index], quantities[index], prices[index],
        multipliers[index], fees[index],
      );
      account.cash -= realized;
      if !margin_accounting {
        account.cash -= quantities[index] * prices[index] * multipliers[index];
      }
      let (position, _) = account.position(&products[index]);
      let cash = account.cash;
      if margin_accounting {
        let margin = self.margins.entry(ledger_id.to_string()).or_insert_with(MarginAccount::new);
        margin.update(&products[index], position, prices[index], multipliers[index], margin_ratios[index]);
      }
      result.push((realized, cash, position));
    }
    Ok(result)
  }

  fn apply_lot_position(
    &mut self, ledger_id: &str, product: &str, method: i32,
    quantity: f64, price: f64, multiplier: f64, is_today: i32, offset: i32,
  ) -> (f64, f64) {
    let method = if method > 0 { method } else { *self.cost_methods.get(ledger_id).unwrap_or(&0) };
    if !self.lots.contains_key(ledger_id) {
      self.lots.insert(ledger_id.to_string(), std::collections::HashMap::new());
    }
    let book = self.lots.get_mut(ledger_id).unwrap()
      .entry(product.to_string()).or_insert_with(|| LotBook::new(method));
    let realized = book.apply_fill(quantity, price, multiplier, is_today, offset);
    let quantity_after = book.quantity();
    if let Some(account) = self.accounts.get_mut(ledger_id) {
      let prior_cost = account.positions.get(product).map(|entry| entry.1).unwrap_or(0.0);
      account.positions.insert(product.to_string(), (quantity_after, prior_cost));
    }
    (realized, quantity_after)
  }

  fn settle_margin_fill(
    &mut self, ledger_id: &str, product: &str, basis_price: f64, ratio: f64,
    fee: f64, realized: f64, quantity_after: f64, multiplier: f64,
  ) -> f64 {
    let Some(account) = self.accounts.get_mut(ledger_id) else { return 0.0 };
    let margin = self.margins.entry(ledger_id.to_string()).or_insert_with(MarginAccount::new);
    let prior_margin = margin.product_margin(product);
    let after_margin = margin.update(product, quantity_after, basis_price, multiplier, ratio);
    account.cash += realized - fee - (after_margin - prior_margin);
    after_margin
  }

  fn apply_margin_fill(
    &mut self, ledger_id: &str, product: &str,
    quantity: f64, price: f64, multiplier: f64, fee: f64,
    basis_price: f64, ratio: f64,
  ) -> f64 {
    let Some(account) = self.accounts.get_mut(ledger_id) else { return 0.0 };
    let (prior_quantity, prior_cost) = account.position(product);
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
    let quantity_after = if new_quantity.abs() <= epsilon { 0.0 } else { new_quantity };
    account.positions.insert(product.to_string(), (quantity_after, new_cost));
    let margin = self.margins.entry(ledger_id.to_string()).or_insert_with(MarginAccount::new);
    let prior_margin = margin.product_margin(product);
    let after_margin = margin.update(product, quantity_after, basis_price, multiplier, ratio);
    account.cash += realized - fee - (after_margin - prior_margin);
    realized
  }

  fn set_cost_method(&mut self, ledger_id: &str, method: i32) {
    self.cost_methods.insert(ledger_id.to_string(), method);
  }

  fn apply_lot_fill(
    &mut self, ledger_id: &str, product: &str, method: i32,
    quantity: f64, price: f64, multiplier: f64, fee: f64,
    is_today: i32, offset: i32,
  ) -> f64 {
    let method = if method > 0 { method } else { *self.cost_methods.get(ledger_id).unwrap_or(&0) };
    if !self.lots.contains_key(ledger_id) {
      self.lots.insert(ledger_id.to_string(), std::collections::HashMap::new());
    }
    let books = self.lots.get_mut(ledger_id).unwrap();
    let book = books.entry(product.to_string()).or_insert_with(|| LotBook::new(method));
    let realized = book.apply_fill(quantity, price, multiplier, is_today, offset);
    let quantity_after = book.quantity();
    if let Some(account) = self.accounts.get_mut(ledger_id) {
      let prior_cost = account.positions.get(product).map(|entry| entry.1).unwrap_or(0.0);
      account.positions.insert(product.to_string(), (quantity_after, prior_cost));
      account.cash -= fee;
      account.cash -= quantity * price * multiplier;
    }
    realized
  }

  fn set_lots(&mut self, ledger_id: &str, product: &str, method: i32, lots: Vec<(f64, f64, f64, i32)>) {
    if !self.lots.contains_key(ledger_id) {
      self.lots.insert(ledger_id.to_string(), std::collections::HashMap::new());
    }
    let book = self.lots.get_mut(ledger_id).unwrap()
      .entry(product.to_string()).or_insert_with(|| LotBook::new(method));
    book.lots = lots;
  }

  fn lots(&self, ledger_id: &str, product: &str) -> Vec<(f64, f64, f64, i32)> {
    self.lots.get(ledger_id)
      .and_then(|books| books.get(product))
      .map(|book| book.lots_snapshot())
      .unwrap_or_default()
  }

  fn cash(&self, ledger_id: &str) -> f64 {
    self.accounts.get(ledger_id).map(|account| account.cash).unwrap_or(0.0)
  }

  fn position(&self, ledger_id: &str, product: &str) -> (f64, f64) {
    self.accounts.get(ledger_id).map(|account| account.position(product)).unwrap_or((0.0, 0.0))
  }

  fn product_margin(&self, ledger_id: &str, product: &str) -> f64 {
    self.margins.get(ledger_id).map(|account| account.product_margin(product)).unwrap_or(0.0)
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

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum TracePlanKind {
  Normal(usize, usize),
  Pending(usize),
  Constraint(usize),
  Rounding,
}

#[derive(Clone)]
struct TraceEntry {
  order_id: u64,
  column: usize,
  delta: f64,
  fee: f64,
  signal_text: String,
  execution_text: String,
  product: String,
  order_text: String,
}

#[pyclass(unsendable)]
pub struct NativeTraceBuilder;

#[pymethods]
impl NativeTraceBuilder {
  #[new]
  fn new() -> Self { Self }

  fn build(
    &self, py: Python<'_>, alias: &str,
    orders: Vec<(u64, usize, f64, f64, i32)>,
    audits: Vec<(usize, u64, f64, f64, f64)>,
    timestamps: Vec<String>, execution_timestamps: Vec<String>,
    products: Vec<String>, initial_cash: f64, base_currency: &str,
    custom: bool, intents: Vec<(u64, f64)>,
    execution_delay: usize,
    pending: Vec<(u64, usize, f64)>,
    constraints: Vec<(usize, u64, usize, f64, f64, f64, f64, f64)>,
  ) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
    let mut audit_by_order = std::collections::HashMap::new();
    let mut audit_rows = std::collections::BTreeSet::new();
    for (row, order_id, _delta, price, _fee) in audits {
      audit_by_order.insert(order_id, (row, price));
      audit_rows.insert(row);
    }
    let mut positions: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    let mut cash = initial_cash;
    let mut order_info: std::collections::HashMap<u64, (f64, f64, &'static str, f64, f64, String, String)> = std::collections::HashMap::new();
    for (order_id, column, delta, fee, _status) in &orders {
      let (execution_row, price) = audit_by_order.get(order_id).copied().unwrap_or((1, 0.0));
      let execution_row = execution_row.min(timestamps.len().saturating_sub(1));
      let signal_row = execution_row.saturating_sub(1);
      let signal_text = timestamps.get(signal_row).cloned().unwrap_or_default();
      let prior = positions.get(column).copied().unwrap_or(0.0);
      let leg = if prior * *delta < 0.0 { "close" } else { "open" };
      let cash_before = cash;
      cash -= *delta * price + *fee;
      positions.insert(*column, prior + *delta);
      let group = format!("{alias}-{signal_text}-group-{order_id}");
      order_info.insert(*order_id, (
        price, *fee, leg, cash_before, cash,
        group.clone(), format!("{group}:intent"),
      ));
    }
    let mut templates: Vec<Py<pyo3::types::PyDict>> = Vec::with_capacity(6);
    for step in 0..6 {
      let (step_name, label) = match step {
        0 => ("construct_order", "构造原子订单"),
        1 => ("resolve_execution_price", "解析成交价"),
        2 => ("slippage", "计算滑点价格"),
        3 => ("fee", "计算手续费"),
        4 => ("ledger_update", "成交落账"),
        _ => ("order_terminal", "订单成交"),
      };
      let template = pyo3::types::PyDict::new(py);
      template.set_item("order_group_id", "")?;
      template.set_item("parent_intent_id", "")?;
      template.set_item("step", step_name)?;
      template.set_item("label", label)?;
      template.set_item("leg_role", "atomic_net")?;
      template.set_item("offset", "open")?;
      template.set_item("revision", 0)?;
      template.set_item("effective_price", py.None())?;
      template.set_item("reject_reason", py.None())?;
      templates.push(template.unbind());
    }
    let fee_detail_template = pyo3::types::PyDict::new(py);
    fee_detail_template.set_item("mode", "zero")?;
    fee_detail_template.set_item("fixed_rate", 0.0)?;
    let empty_detail_template = pyo3::types::PyDict::new(py);
    let mut entries = Vec::with_capacity(orders.len());
    for (order_id, column, delta, fee, _status) in &orders {
      let (execution_row, _price) = audit_by_order.get(order_id).copied().unwrap_or((1, 0.0));
      let execution_row = execution_row.min(timestamps.len().saturating_sub(1));
      let signal_row = execution_row.saturating_sub(1 + execution_delay);
      let signal_text = timestamps.get(signal_row).cloned().unwrap_or_default();
      let execution_text = execution_timestamps.get(signal_row).cloned().unwrap_or_default();
      let product = products.get(*column).cloned().unwrap_or_default();
      let order_text = format!("{alias}-{signal_text}-{order_id}");
      entries.push(TraceEntry {
        order_id: *order_id, column: *column, delta: *delta, fee: *fee,
        signal_text, execution_text, product, order_text,
      });
    }
    let intent_by_order: std::collections::HashMap<u64, f64> = intents.into_iter().collect();
    let mut plan: Vec<(String, i32, i64, usize, TracePlanKind)> = Vec::with_capacity(entries.len() * 6);
    let mut sequence = 0usize;
    for (index, entry) in entries.iter().enumerate() {
      for step in 0..6 {
        let (timestamp, rank, key) = match step {
          0 => (entry.signal_text.clone(), 0, entry.order_id as i64),
          1 => (entry.execution_text.clone(), 1, entry.order_id as i64),
          2 => (entry.execution_text.clone(), 2, entry.order_id as i64),
          3 => (entry.execution_text.clone(), 3, entry.order_id as i64),
          4 => (entry.execution_text.clone(), 5, sequence as i64),
          _ => (entry.execution_text.clone(), 6, entry.order_id as i64),
        };
        plan.push((timestamp, rank, key, sequence, TracePlanKind::Normal(index, step)));
        sequence += 1;
      }
    }
    for (index, (order_id, column, delta)) in pending.iter().enumerate() {
      let timestamp = timestamps.last().cloned().unwrap_or_default();
      plan.push((timestamp, 0, *order_id as i64, sequence, TracePlanKind::Pending(index)));
      let _ = (column, delta);
      sequence += 1;
    }
    for (index, constraint) in constraints.iter().enumerate() {
      let (row, order_id, _column, _before, _after, _scale, _available, _required) = constraint;
      let signal_row = row.saturating_sub(1).min(execution_timestamps.len().saturating_sub(1));
      let timestamp = execution_timestamps.get(signal_row).cloned().unwrap_or_default();
      plan.push((timestamp, 4, *order_id as i64, sequence, TracePlanKind::Constraint(index)));
      sequence += 1;
    }
    if custom {
      for row in &audit_rows {
        if *row < timestamps.len() {
          plan.push((timestamps[*row].clone(), -1, -1, sequence, TracePlanKind::Rounding));
          sequence += 1;
        }
      }
    }
    plan.sort();
    let mut rows = Vec::with_capacity(plan.len());
    for (_timestamp, _rank, _key, seq, plan_kind) in plan {
      if let TracePlanKind::Pending(index) = plan_kind {
        let (pending_id, column, delta) = pending[index];
        let signal_text = timestamps.last().cloned().unwrap_or_default();
        let product = products.get(column).cloned().unwrap_or_default();
        let position = positions.get(&column).copied().unwrap_or(0.0);
        let leg = if position * delta < 0.0 { "close" } else { "open" };
        let order_text = format!("{alias}-{signal_text}-{pending_id}");
        let group_id = format!("{alias}-{signal_text}-group-{pending_id}");
        let dict = templates[0].bind(py).copy()?;
        dict.set_item("order_id", &order_text)?;
        dict.set_item("order_group_id", &group_id)?;
        dict.set_item("parent_intent_id", format!("{group_id}:intent"))?;
        dict.set_item("strategy_id", alias)?;
        dict.set_item("timestamp", &signal_text)?;
        dict.set_item("product", product)?;
        dict.set_item("quantity", delta)?;
        dict.set_item("intent_quantity", delta)?;
        dict.set_item("requested_quantity", delta.abs())?;
        dict.set_item("filled_quantity", 0.0)?;
        dict.set_item("remaining_quantity", delta.abs())?;
        dict.set_item("unfilled_quantity", delta.abs())?;
        dict.set_item("status", "draft")?;
        dict.set_item("leg_role", leg)?;
        dict.set_item("offset", if leg == "close" { "close" } else { "open" })?;
        dict.set_item("execution_effect", "mixed")?;
        dict.set_item("effective_price", py.None())?;
        dict.set_item("fee_cost", py.None())?;
        let detail = pyo3::types::PyDict::new(py);
        detail.set_item("group_policy", "sequential_close_then_open")?;
        detail.set_item("child_order_ids", pyo3::types::PyTuple::new(py, [order_text.as_str()])?)?;
        detail.set_item("paired_execution_policy", py.None())?;
        dict.set_item("details", detail)?;
        rows.push(dict.unbind());
        continue;
      }
      if let TracePlanKind::Rounding = plan_kind {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("order_id", "")?;
        dict.set_item("order_group_id", "")?;
        dict.set_item("parent_intent_id", "")?;
        dict.set_item("strategy_id", alias)?;
        dict.set_item("timestamp", _timestamp)?;
        dict.set_item("step", "quantity_rounding")?;
        dict.set_item("label", "按最小买入手数取整")?;
        dict.set_item("product", py.None())?;
        dict.set_item("quantity", py.None())?;
        dict.set_item("intent_quantity", py.None())?;
        dict.set_item("requested_quantity", py.None())?;
        dict.set_item("filled_quantity", 0.0)?;
        dict.set_item("remaining_quantity", py.None())?;
        dict.set_item("unfilled_quantity", py.None())?;
        dict.set_item("status", "draft")?;
        dict.set_item("leg_role", "")?;
        dict.set_item("offset", "")?;
        dict.set_item("execution_effect", "mixed")?;
        dict.set_item("revision", 0)?;
        dict.set_item("effective_price", py.None())?;
        dict.set_item("fee_cost", py.None())?;
        dict.set_item("reject_reason", py.None())?;
        dict.set_item("details", pyo3::types::PyDict::new(py))?;
        rows.push(dict.unbind());
        continue;
      }
      if let TracePlanKind::Constraint(index) = plan_kind {
        let (row, order_id, column, before, after, scale, available, required) = constraints[index];
        let signal_row = row.saturating_sub(1 + execution_delay).min(execution_timestamps.len().saturating_sub(1));
        let signal_text = timestamps.get(signal_row).cloned().unwrap_or_default();
        let timestamp = execution_timestamps.get(signal_row).cloned().unwrap_or_default();
        let product = products.get(column).cloned().unwrap_or_default();
        let order_text = format!("{alias}-{signal_text}-{order_id}");
        let group_id = format!("{alias}-{signal_text}-group-{order_id}");
        let (price, _fee, _leg, _cash_before, _cash_after, _g, _p) = &order_info[&order_id];
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("order_id", &order_text)?;
        dict.set_item("order_group_id", &group_id)?;
        dict.set_item("parent_intent_id", format!("{group_id}:intent"))?;
        dict.set_item("strategy_id", alias)?;
        dict.set_item("timestamp", timestamp)?;
        dict.set_item("step", "execution_cash_constraint")?;
        dict.set_item("label", "成交增量保证金约束")?;
        dict.set_item("product", product)?;
        dict.set_item("quantity", after)?;
        dict.set_item("intent_quantity", before)?;
        dict.set_item("requested_quantity", before)?;
        dict.set_item("filled_quantity", 0.0)?;
        dict.set_item("remaining_quantity", before)?;
        dict.set_item("unfilled_quantity", before)?;
        dict.set_item("status", "accepted")?;
        dict.set_item("leg_role", "open")?;
        dict.set_item("offset", "open")?;
        dict.set_item("execution_effect", "increase")?;
        dict.set_item("revision", 0)?;
        dict.set_item("effective_price", price)?;
        dict.set_item("fee_cost", 0.0)?;
        dict.set_item("reject_reason", py.None())?;
        let detail = pyo3::types::PyDict::new(py);
        detail.set_item("before_quantity", before)?;
        detail.set_item("after_quantity", after)?;
        detail.set_item("scale", scale)?;
        detail.set_item("available_cash", available)?;
        detail.set_item("required_cash", required)?;
        dict.set_item("details", detail)?;
        rows.push(dict.unbind());
        continue;
      }
      let (entry_index, step) = match plan_kind {
        TracePlanKind::Normal(index, step) => (index, step),
        _ => unreachable!(),
      };
      let entry = &entries[entry_index];
      let order_id = entry.order_id;
      let delta = entry.delta;
      let fee = entry.fee;
      let info = &order_info[&order_id];
      let (price, fee_value, leg, cash_before, cash_after) =
        (info.0, info.1, info.2, info.3, info.4);
      let group_id = &info.5;
      let parent_id = &info.6;
      let filled_quantity_value = delta.abs();
      let intent_quantity = intent_by_order.get(&order_id).copied().map(f64::abs).unwrap_or(filled_quantity_value);
      let sign = if delta < 0.0 { -1.0 } else { 1.0 };
      let quantity = if step < 4 { intent_quantity } else { filled_quantity_value };
      let signed_quantity = if step < 4 { sign * intent_quantity } else { delta };
      let timestamp = if step == 0 { entry.signal_text.as_str() } else { entry.execution_text.as_str() };
      let (status, filled, remaining) = match step {
        0 => ("draft", 0.0, quantity),
        1 | 2 | 3 => ("accepted", 0.0, quantity),
        _ => ("filled", filled_quantity_value, 0.0),
      };
      let dict = templates[step].bind(py).copy()?;
      dict.set_item("order_id", &entry.order_text)?;
      dict.set_item("order_group_id", &group_id)?;
      dict.set_item("parent_intent_id", &parent_id)?;
      dict.set_item("strategy_id", alias)?;
      dict.set_item("timestamp", timestamp)?;
      dict.set_item("product", &entry.product)?;
      dict.set_item("quantity", signed_quantity)?;
      dict.set_item("intent_quantity", signed_quantity)?;
      dict.set_item("requested_quantity", quantity)?;
      dict.set_item("filled_quantity", filled)?;
      dict.set_item("remaining_quantity", remaining)?;
      dict.set_item("unfilled_quantity", remaining)?;
      dict.set_item("status", status)?;
      dict.set_item("leg_role", leg)?;
      dict.set_item("offset", if leg == "close" { "close" } else { "open" })?;
      dict.set_item(
        "execution_effect",
        if step < 2 { "mixed" } else if delta > 0.0 { "increase" } else { "reduce" },
      )?;
      if step == 0 { dict.set_item("effective_price", py.None())?; }
      else { dict.set_item("effective_price", price)?; }
      if step < 3 { dict.set_item("fee_cost", py.None())?; }
      else { dict.set_item("fee_cost", fee)?; }
      let detail_dict = if step == 3 {
        fee_detail_template.copy()?
      } else if step == 5 {
        empty_detail_template.copy()?
      } else {
        pyo3::types::PyDict::new(py)
      };
      match step {
        0 => {
          detail_dict.set_item("group_policy", "sequential_close_then_open")?;
          let children = pyo3::types::PyTuple::new(py, [entry.order_text.as_str()])?;
          detail_dict.set_item("child_order_ids", children)?;
          detail_dict.set_item("paired_execution_policy", py.None())?;
        }
        1 => { detail_dict.set_item("basis", "open")?; detail_dict.set_item("price", price)?; }
        2 => { detail_dict.set_item("mode", "none")?; detail_dict.set_item("slippage_bps", 0.0)?; detail_dict.set_item("base_price", price)?; }
        3 => { detail_dict.set_item("mode", "zero")?; detail_dict.set_item("fixed_rate", 0.0)?; }
        4 => {
          detail_dict.set_item("fill_id", format!("{}:fill:1", entry.order_text))?;
          detail_dict.set_item("ledger_id", format!("private:{alias}"))?;
          detail_dict.set_item("cash_pool_id", format!("private:{alias}"))?;
          detail_dict.set_item("account_currency", base_currency)?;
          detail_dict.set_item("cash_pool_base_currency", base_currency)?;
          detail_dict.set_item("price", price)?;
          detail_dict.set_item("fee_cost", fee)?;
          detail_dict.set_item("cash_before", cash_before)?;
          detail_dict.set_item("cash_after", cash_after)?;
          detail_dict.set_item("margin_before", 0.0)?;
          detail_dict.set_item("margin_after", 0.0)?;
        }
        _ => {}
      }
      let _ = (fee_value, seq);
      dict.set_item("details", detail_dict)?;
      rows.push(dict.unbind());
    }
    Ok(rows)
  }
}

fn update_native_lots(lots: &mut Vec<(f64, f64)>, delta: f64, price: f64, method: i32) -> f64 {
  if delta.abs() <= 1e-15 { return 0.0; }
  let mut remaining = delta;
  let mut realized = 0.0;
  while remaining.abs() > 1e-12 {
    let Some(prior) = lots.first().map(|lot| lot.0) else { break; };
    if prior * remaining > 0.0 { break; }
    let position = lots.iter().map(|lot| lot.0).sum::<f64>();
    if position.abs() <= 1e-15 { break; }
    let select = if method == 1 {
      lots.len() - 1
    } else if method == 2 {
      lots.iter().enumerate().max_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap()).map(|x| x.0).unwrap_or(0)
    } else { 0 };
    let (lot_qty, lot_price) = lots[select];
    let close = remaining.abs().min(lot_qty.abs()) * lot_qty.signum();
    realized += (price - lot_price) * close;
    let left = lot_qty - close;
    if left.abs() <= 1e-12 { lots.remove(select); } else { lots[select].0 = left; }
    remaining -= close;
  }
  if remaining.abs() > 1e-12 {
    if method == 3 {
      if let Some(existing) = lots.first_mut() {
        if existing.0 * remaining > 0.0 {
          let total = existing.0 + remaining;
          existing.1 = (existing.1 * existing.0.abs() + price * remaining.abs()) / total.abs();
          existing.0 = total;
          return realized;
        }
      }
    }
    lots.push((remaining, price));
  }
  realized
}

#[pyclass(unsendable)]
pub struct NativeGroupSession {
  factor: Py<numpy::PyArray2<f64>>,
  prices: Py<numpy::PyArray2<f64>>,
  lot_sizes: Vec<f64>,
  split_count: i64,
  group_index: i64,
  enforce_cash: bool,
  round_to_lot: bool,
  target_scale: f64,
  margin_ratio: f64,
  cost_method: i32,
  daily_mtm: bool,
  fee_rates: Vec<(f64, f64)>,
  volume: Option<Py<numpy::PyArray2<f64>>>,
  volume_rate: f64,
  execution_delay: usize,
}

#[pymethods]
impl NativeGroupSession {
  #[new]
  #[pyo3(signature=(factor, prices, lot_sizes, split_count, group_index, enforce_cash=false, round_to_lot=true, target_scale=1.0, margin_ratio=1.0, cost_method=0, daily_mtm=false, fee_rates=Vec::new(), volume=None, volume_rate=1.0, execution_delay=0))]
  fn new(
    py: Python<'_>, factor: Py<numpy::PyArray2<f64>>,
    prices: Py<numpy::PyArray2<f64>>, lot_sizes: Vec<f64>,
    split_count: i64, group_index: i64, enforce_cash: bool, round_to_lot: bool,
    target_scale: f64, margin_ratio: f64, cost_method: i32, daily_mtm: bool,
    fee_rates: Vec<(f64, f64)>, volume: Option<Py<numpy::PyArray2<f64>>>,
    volume_rate: f64, execution_delay: usize,
  ) -> PyResult<Self> {
    let factor_view = factor.bind(py).readonly();
    let price_view = prices.bind(py).readonly();
    let (rows, columns) = factor_view.as_array().dim();
    if price_view.as_array().dim() != (rows, columns) || lot_sizes.len() != columns {
      return Err(pyo3::exceptions::PyValueError::new_err("native group shape mismatch"));
    }
    if split_count <= 0 || group_index < 0 || group_index >= split_count {
      return Err(pyo3::exceptions::PyValueError::new_err("invalid group selection"));
    }
    Ok(Self { factor, prices, lot_sizes, split_count, group_index, enforce_cash, round_to_lot, target_scale, margin_ratio, cost_method, daily_mtm, fee_rates, volume, volume_rate, execution_delay })
  }

  fn run_event_driven(
    &self, py: Python<'_>, initial_cash: f64, fee_rate: f64,
  ) -> PyResult<(
    Vec<f64>, Vec<(usize, usize, f64, f64, f64, f64)>,
    Vec<(u64, usize, f64, f64, i32)>, Vec<(usize, u64, f64, f64, f64)>,
    Vec<(u64, usize, i32)>, Vec<(u64, usize, f64)>,
    Vec<(usize, u64, usize, f64, f64, f64, f64, f64)>,
    Vec<(usize, u8, u64, usize)>,
  )> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let factor_view = self.factor.bind(py).readonly();
    let price_view = self.prices.bind(py).readonly();
    let factor = factor_view.as_array();
    let prices = price_view.as_array();
    let volume_view = self.volume.as_ref().map(|value| value.bind(py).readonly());
    let volume = volume_view.as_ref().map(|value| value.as_array());
    let (rows, columns) = factor.dim();
    let mut positions = vec![0.0f64; columns];
    let mut cash = initial_cash;
    let mut lots: Vec<Vec<(f64, f64)>> = vec![Vec::new(); columns];
    let mut margins = vec![0.0f64; columns];
    let mut marks = vec![0.0f64; columns];
    let mut equity = vec![initial_cash; rows];
    let mut trades = Vec::new();
    let mut orders = Vec::new();
    let mut audits = Vec::new();
    let mut order_id = 0u64;
    let mut sequence = 0u64;
    let mut queue: BinaryHeap<Reverse<(usize, i32, u64, u64)>> = BinaryHeap::new();
    let mut queued: std::collections::HashMap<u64, (usize, f64)> = std::collections::HashMap::new();
    let mut attempts = Vec::new();
    let mut constraints = Vec::new();
    let mut events = Vec::new();
    for row in 1..rows {
      events.push((row, 0, 0, 0));
      queue.push(Reverse((row, 0, sequence, 0)));
      sequence += 1;
      let signal_offset = row - 1;
      while let Some(Reverse((event_row, kind, _seq, payload))) = queue.peek().copied() {
        if event_row != row { break; }
        queue.pop();
        if kind == 0 {
          let mut ranked: Vec<usize> = (0..columns).collect();
          ranked.sort_by(|&left, &right| {
            factor[[signal_offset, right]].partial_cmp(&factor[[signal_offset, left]])
              .unwrap_or(std::cmp::Ordering::Equal).then_with(|| left.cmp(&right))
          });
          let bucket_size = columns as f64 / self.split_count as f64;
          let start = (self.group_index as f64 * bucket_size).round() as usize;
          let end = (((self.group_index + 1) as f64) * bucket_size).round() as usize;
          let members = &ranked[start..end.min(columns)];
          let mut current_equity = cash;
          if self.margin_ratio < 1.0 {
            for index in 0..columns { current_equity += margins[index]; }
          } else {
            for index in 0..columns { current_equity += positions[index] * prices[[row - 1, index]]; }
          }
          let weight = if members.is_empty() { 0.0 } else { 1.0 / members.len() as f64 };
          let mut deltas = vec![0.0f64; columns];
          for index in 0..columns {
            let price = prices[[row, index]];
            if price <= 0.0 { continue; }
            let lot = if self.lot_sizes[index] > 0.0 { self.lot_sizes[index] } else { 1.0 };
            let target = if members.contains(&index) { weight * current_equity * self.target_scale / prices[[row - 1, index]] } else { 0.0 };
            let target = if self.round_to_lot { (target / lot).floor() * lot } else { target };
            deltas[index] = target - positions[index];
          }
          let mut planned = Vec::new();
          let mut cash_capacity = 0.0;
          let mut cash_required = 0.0;
          let mut cash_scale = 1.0;
          if self.enforce_cash {
            let mut release = 0.0;
            let mut required = 0.0;
            for index in 0..columns {
              let delta = deltas[index];
              let capacity_weight = if self.margin_ratio < 1.0 { self.margin_ratio } else { 1.0 };
              if delta < 0.0 { release += -delta * prices[[row, index]] * capacity_weight; }
              if delta > 0.0 { required += delta * prices[[row, index]] * capacity_weight; }
            }
            cash_capacity = (cash + release).max(0.0);
            cash_required = required;
            cash_scale = if required > cash_capacity + 1e-9 { cash_capacity / required } else { 1.0 };
            for index in 0..columns {
              if deltas[index] < 0.0 { planned.push((index, deltas[index])); }
            }
            for index in 0..columns {
              if deltas[index] > 0.0 { planned.push((index, deltas[index] * cash_scale)); }
            }
          } else if volume.is_some() {
            for index in 0..columns {
              if deltas[index] < 0.0 { planned.push((index, deltas[index])); }
            }
            for index in 0..columns {
              if deltas[index] > 0.0 { planned.push((index, deltas[index])); }
            }
          } else {
            for index in 0..columns {
              planned.push((index, deltas[index]));
            }
          }
          let execution_row = row + self.execution_delay;
          let mut signal_order_ids = vec![0u64; columns];
          for index in 0..columns {
            if deltas[index].abs() <= 1e-12 { continue; }
            order_id += 1;
            signal_order_ids[index] = order_id;
            attempts.push((order_id, row, 0));
            events.push((row, 1, order_id, index));
          }
          for (index, order_delta) in planned {
            if order_delta.abs() <= 1e-12 { continue; }
            let queued_order_id = signal_order_ids[index];
            queued.insert(queued_order_id, (index, order_delta));
            if execution_row < rows {
              if self.enforce_cash && deltas[index] > 0.0 && cash_scale < 1.0 {
                constraints.push((execution_row, queued_order_id, index, deltas[index], order_delta, cash_scale, cash_capacity, cash_required));
              }
              queue.push(Reverse((execution_row, 1, sequence, queued_order_id)));
              sequence += 1;
            }
          }
        } else {
          let order_key = payload;
          let Some((index, intent_delta)) = queued.remove(&order_key) else { continue; };
          let mut delta = intent_delta;
          if let Some(volume_values) = volume {
            let capacity = volume_values[[row, index]] * self.volume_rate;
            if delta.abs() > capacity {
              delta = delta.signum() * capacity.max(0.0);
            }
          }
          if delta.abs() <= 1e-12 { continue; }
          let price = prices[[row, index]];
          let notional = delta * price;
          let effective_fee_rate = if self.fee_rates.is_empty() {
            fee_rate
          } else {
            let (open_rate, close_rate) = self.fee_rates[index];
            if positions[index] * delta < 0.0 { close_rate } else { open_rate }
          };
          let fee = notional.abs() * effective_fee_rate;
          if self.margin_ratio < 1.0 {
            let realized = update_native_lots(&mut lots[index], delta, price, self.cost_method);
            let margin_before = margins[index];
            let new_position = positions[index] + delta;
            let margin_after = new_position.abs() * price * self.margin_ratio;
            cash += margin_before - margin_after + realized - fee;
            margins[index] = margin_after;
            positions[index] = new_position;
            marks[index] = price;
          } else {
            cash -= notional + fee;
            positions[index] += delta;
          }
          let filled_order_id = order_key;
          attempts.push((filled_order_id, row, 1));
          orders.push((filled_order_id, index, intent_delta, fee, 1));
          audits.push((row, filled_order_id, delta, price, fee));
          trades.push((row, index, delta, price, fee, cash));
        }
      }
      if self.margin_ratio < 1.0 {
        if self.daily_mtm {
          for index in 0..columns {
            let position = positions[index];
            if position.abs() <= 1e-15 { continue; }
            let mark = prices[[row, index]];
            let margin_before = margins[index];
            let margin_after = position.abs() * mark * self.margin_ratio;
            cash += (mark - marks[index]) * position + margin_before - margin_after;
            margins[index] = margin_after;
            marks[index] = mark;
          }
        }
        equity[row] = cash + margins.iter().sum::<f64>();
      } else {
        let mut value = cash;
        for index in 0..columns { value += positions[index] * prices[[row, index]]; }
        equity[row] = value;
      }
    }
    let mut pending = Vec::new();
    if rows > 0 {
      events.push((rows - 1, 0, 0, 0));
      let signal_offset = rows - 1;
      let mut ranked: Vec<usize> = (0..columns).collect();
      ranked.sort_by(|&left, &right| {
        factor[[signal_offset, right]].partial_cmp(&factor[[signal_offset, left]])
          .unwrap_or(std::cmp::Ordering::Equal).then_with(|| left.cmp(&right))
      });
      let bucket_size = columns as f64 / self.split_count as f64;
      let start = (self.group_index as f64 * bucket_size).round() as usize;
      let end = (((self.group_index + 1) as f64) * bucket_size).round() as usize;
      let members = &ranked[start..end.min(columns)];
      let mut current_equity = cash;
      if self.margin_ratio < 1.0 {
        for index in 0..columns { current_equity += margins[index]; }
      } else {
        for index in 0..columns { current_equity += positions[index] * prices[[signal_offset, index]]; }
      }
      let weight = if members.is_empty() { 0.0 } else { 1.0 / members.len() as f64 };
      for index in 0..columns {
        let price = prices[[signal_offset, index]];
        if price <= 0.0 { continue; }
        let lot = if self.lot_sizes[index] > 0.0 { self.lot_sizes[index] } else { 1.0 };
        let target = if members.contains(&index) { weight * current_equity * self.target_scale / price } else { 0.0 };
        let target = if self.round_to_lot { (target / lot).floor() * lot } else { target };
        let order_delta = target - positions[index];
        if order_delta.abs() > 1e-12 {
          order_id += 1;
          pending.push((order_id, index, order_delta));
          attempts.push((order_id, signal_offset, 0));
        }
      }
    }
    Ok((equity, trades, orders, audits, attempts, pending, constraints, events))
  }

  fn run(
    &self, py: Python<'_>, initial_cash: f64, fee_rate: f64,
  ) -> PyResult<(
    Vec<f64>, Vec<(usize, usize, f64, f64, f64, f64)>,
    Vec<(u64, usize, f64, f64, i32)>, Vec<(usize, u64, f64, f64, f64)>,
  )> {
    let factor_view = self.factor.bind(py).readonly();
    let price_view = self.prices.bind(py).readonly();
    let factor = factor_view.as_array();
    let prices = price_view.as_array();
    let (rows, columns) = factor.dim();
    let mut positions = vec![0.0f64; columns];
    let mut cash = initial_cash;
    let mut equity = vec![initial_cash; rows];
    let mut trades = Vec::new();
    let mut orders = Vec::new();
    let mut audits = Vec::new();
    let mut order_id = 0u64;
    for row in 1..rows {
      let signal_offset = row - 1;
      let mut ranked: Vec<usize> = (0..columns).collect();
      ranked.sort_by(|&left, &right| {
        factor[[signal_offset, right]].partial_cmp(&factor[[signal_offset, left]])
          .unwrap_or(std::cmp::Ordering::Equal)
          .then_with(|| left.cmp(&right))
      });
      let bucket_size = columns as f64 / self.split_count as f64;
      let start = (self.group_index as f64 * bucket_size).round() as usize;
      let end = (((self.group_index + 1) as f64) * bucket_size).round() as usize;
      let members = &ranked[start..end.min(columns)];
      let mut current_equity = cash;
      for column in 0..columns {
        current_equity += positions[column] * prices[[row - 1, column]];
      }
      let weight = if members.is_empty() { 0.0 } else { 1.0 / members.len() as f64 };
      for column in 0..columns {
        let price = prices[[row, column]];
        if price <= 0.0 { continue; }
        let lot = if self.lot_sizes[column] > 0.0 { self.lot_sizes[column] } else { 1.0 };
        let target = if members.contains(&column) { weight * current_equity * self.target_scale / prices[[row - 1, column]] } else { 0.0 };
        let target = if self.round_to_lot { (target / lot).floor() * lot } else { target };
        let delta = target - positions[column];
        if delta.abs() <= 1e-12 { continue; }
        let notional = delta * price;
        let effective_fee_rate = if self.fee_rates.is_empty() {
          fee_rate
        } else {
          let (open_rate, close_rate) = self.fee_rates[column];
          if positions[column] * delta < 0.0 { close_rate } else { open_rate }
        };
        let fee = notional.abs() * effective_fee_rate;
        cash -= notional + fee;
        positions[column] = target;
        order_id += 1;
        orders.push((order_id, column, delta, fee, 1));
        audits.push((row, order_id, delta, price, fee));
        trades.push((row, column, delta, price, fee, cash));
      }
      let mut value = cash;
      for column in 0..columns {
        value += positions[column] * prices[[row, column]];
      }
      equity[row] = value;
    }
    Ok((equity, trades, orders, audits))
  }
}

#[pyclass(unsendable)]
pub struct NativeSession {
  prices: Py<numpy::PyArray2<f64>>,
  targets: Py<numpy::PyArray2<f64>>,
  lot_sizes: Vec<f64>,
}

#[pymethods]
impl NativeSession {
  #[new]
  fn new(
    py: Python<'_>, prices: Py<numpy::PyArray2<f64>>,
    targets: Py<numpy::PyArray2<f64>>, lot_sizes: Vec<f64>,
  ) -> PyResult<Self> {
    let price_view = prices.bind(py).readonly();
    let target_view = targets.bind(py).readonly();
    let (rows, columns) = price_view.as_array().dim();
    if target_view.as_array().dim() != (rows, columns) || lot_sizes.len() != columns {
      return Err(pyo3::exceptions::PyValueError::new_err("native session shape mismatch"));
    }
    Ok(Self { prices, targets, lot_sizes })
  }

  fn run_with_lifecycle(
    &self, py: Python<'_>, initial_cash: f64, fee_rate: f64,
  ) -> PyResult<(
    Vec<f64>, Vec<(usize, usize, f64, f64, f64, f64)>,
    Vec<(u64, usize, f64, f64, i32)>,
    Vec<(usize, u64, f64, f64, f64)>,
  )> {
    let price_view = self.prices.bind(py).readonly();
    let target_view = self.targets.bind(py).readonly();
    let prices = price_view.as_array();
    let targets = target_view.as_array();
    let (rows, columns) = prices.dim();
    let mut positions = vec![0.0f64; columns];
    let mut cash = initial_cash;
    let mut equity = vec![initial_cash; rows];
    let mut trades = Vec::new();
    let mut orders = Vec::new();
    let mut audits = Vec::new();
    let mut order_id = 0u64;
    for row in 1..rows {
      let mut current_equity = cash;
      for column in 0..columns {
        current_equity += positions[column] * prices[[row - 1, column]];
      }
      for column in 0..columns {
        let price = prices[[row, column]];
        if price <= 0.0 { continue; }
        let lot = if self.lot_sizes[column] > 0.0 { self.lot_sizes[column] } else { 1.0 };
        let target = targets[[row - 1, column]] * current_equity / prices[[row - 1, column]];
        let target = (target / lot).floor() * lot;
        let delta = target - positions[column];
        if delta.abs() <= 1e-12 { continue; }
        let notional = delta * price;
        let fee = notional.abs() * fee_rate;
        cash -= notional + fee;
        positions[column] = target;
        order_id += 1;
        orders.push((order_id, column, delta, fee, 1));
        audits.push((row, order_id, delta, price, fee));
        trades.push((row, column, delta, price, fee, cash));
      }
      let mut value = cash;
      for column in 0..columns {
        value += positions[column] * prices[[row, column]];
      }
      equity[row] = value;
    }
    Ok((equity, trades, orders, audits))
  }

  fn run(
    &self, py: Python<'_>, initial_cash: f64, fee_rate: f64,
  ) -> PyResult<(Vec<f64>, Vec<(usize, usize, f64, f64, f64, f64)>)> {
    let price_view = self.prices.bind(py).readonly();
    let target_view = self.targets.bind(py).readonly();
    let prices = price_view.as_array();
    let targets = target_view.as_array();
    let (rows, columns) = prices.dim();
    let mut positions = vec![0.0f64; columns];
    let mut cash = initial_cash;
    let mut equity = vec![initial_cash; rows];
    let mut trades = Vec::new();
    for row in 1..rows {
      let mut current_equity = cash;
      for column in 0..columns {
        current_equity += positions[column] * prices[[row - 1, column]];
      }
      for column in 0..columns {
        let price = prices[[row, column]];
        if price <= 0.0 { continue; }
        let lot = if self.lot_sizes[column] > 0.0 { self.lot_sizes[column] } else { 1.0 };
        let target = targets[[row - 1, column]] * current_equity / prices[[row - 1, column]];
        let target = (target / lot).floor() * lot;
        let delta = target - positions[column];
        if delta.abs() <= 1e-12 { continue; }
        let notional = delta * price;
        let fee = notional.abs() * fee_rate;
        cash -= notional + fee;
        positions[column] = target;
        trades.push((row, column, delta, price, fee, cash));
      }
      let mut value = cash;
      for column in 0..columns {
        value += positions[column] * prices[[row, column]];
      }
      equity[row] = value;
    }
    Ok((equity, trades))
  }
}

#[pyclass(unsendable)]
pub struct CashConstraintKernel;

#[pymethods]
impl CashConstraintKernel {
  #[new]
  fn new() -> Self { Self }

  fn basic_constraint(
    &self, available: f64, prior: Vec<f64>, order_quantities: Vec<f64>,
    prices: Vec<f64>, multipliers: Vec<f64>,
  ) -> PyResult<(f64, f64, f64, Vec<f64>, Vec<f64>, Vec<bool>)> {
    let count = prior.len();
    if order_quantities.len() != count || prices.len() != count || multipliers.len() != count {
      return Err(pyo3::exceptions::PyValueError::new_err("cash constraint length mismatch"));
    }
    let mut reducing = vec![0.0; count];
    let mut increasing = vec![0.0; count];
    for index in 0..count {
      let order_quantity = order_quantities[index];
      if order_quantity.abs() <= 1e-12 { continue; }
      let prior_quantity = prior[index];
      if prior_quantity.abs() <= 1e-12 || prior_quantity * order_quantity > 0.0 {
        increasing[index] = order_quantity;
      } else {
        let close_abs = prior_quantity.abs().min(order_quantity.abs());
        let sign = if order_quantity > 0.0 { 1.0 } else { -1.0 };
        reducing[index] = sign * close_abs;
        increasing[index] = order_quantity - reducing[index];
      }
    }
    let mut release = 0.0;
    let mut required = 0.0;
    let mut constrained = vec![false; count];
    for index in 0..count {
      let price_multiplier = prices[index] * multipliers[index];
      release += -reducing[index] * price_multiplier;
      let delta = -increasing[index] * price_multiplier;
      if delta < -1e-12 {
        required += -delta;
        constrained[index] = true;
      }
    }
    let capacity = (available + release).max(0.0);
    let scale = if required <= capacity + 1e-9 {
      1.0
    } else if required <= 0.0 {
      0.0
    } else {
      capacity / required
    };
    if scale >= 1.0 {
      constrained.iter_mut().for_each(|value| *value = false);
    }
    Ok((release, required, scale, reducing, increasing, constrained))
  }
}

#[pyclass(unsendable)]
pub struct OrderConstructionKernel;

#[pymethods]
impl OrderConstructionKernel {
  #[new]
  fn new() -> Self { Self }

  fn close_buckets(
    &self, close_quantity: f64, lot_quantities: Vec<f64>,
    lot_today: Vec<i32>, lot_prices: Vec<f64>, method: i32,
  ) -> PyResult<Vec<(i32, i32, f64)>> {
    let count = lot_quantities.len();
    if lot_today.len() != count || lot_prices.len() != count {
      return Err(pyo3::exceptions::PyValueError::new_err("lot bucket length mismatch"));
    }
    if close_quantity <= 1e-12 { return Ok(Vec::new()); }
    if count == 0 { return Ok(vec![(2, 3, close_quantity)]); }
    let mut order: Vec<usize> = (0..count).collect();
    if method == 1 { order.reverse(); }
    else if method == 2 {
      order.sort_by(|&left, &right| lot_prices[right].partial_cmp(&lot_prices[left]).unwrap_or(std::cmp::Ordering::Equal));
    }
    let mut buckets: Vec<(i32, i32, f64)> = Vec::new();
    let mut remaining = close_quantity;
    for index in order {
      if remaining <= 1e-12 { break; }
      let take = remaining.min(lot_quantities[index].abs());
      let (offset, role) = match lot_today[index] {
        1 => (3, 1),
        0 => (4, 2),
        _ => (2, 3),
      };
      if let Some(bucket) = buckets.iter_mut().find(|bucket| bucket.0 == offset && bucket.1 == role) {
        bucket.2 += take;
      } else {
        buckets.push((offset, role, take));
      }
      remaining -= take;
    }
    if remaining > 1e-12 {
      if let Some(bucket) = buckets.iter_mut().find(|bucket| bucket.0 == 2 && bucket.1 == 3) {
        bucket.2 += remaining;
      } else {
        buckets.push((2, 3, remaining));
      }
    }
    Ok(buckets)
  }
}

#[pyclass(unsendable)]
pub struct OrderLifecycleBatch {
  accepted: Vec<f64>,
  filled: Vec<f64>,
}

#[pymethods]
impl OrderLifecycleBatch {
  #[new]
  fn new(accepted: Vec<f64>, filled: Vec<f64>) -> PyResult<Self> {
    if accepted.len() != filled.len() {
      return Err(pyo3::exceptions::PyValueError::new_err("lifecycle batch length mismatch"));
    }
    Ok(Self { accepted, filled })
  }

  fn apply_fills(
    &mut self, indices: Vec<usize>, quantities: Vec<f64>,
  ) -> PyResult<Vec<(usize, f64, bool)>> {
    let length = indices.len();
    if quantities.len() != length {
      return Err(pyo3::exceptions::PyValueError::new_err("fill batch length mismatch"));
    }
    let mut result = Vec::with_capacity(length);
    for position in 0..length {
      let index = indices[position];
      if index >= self.accepted.len() {
        return Err(pyo3::exceptions::PyIndexError::new_err("lifecycle index out of range"));
      }
      let quantity = quantities[position];
      if quantity <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("fill quantity must be positive"));
      }
      let remaining = (self.accepted[index] - self.filled[index]).max(0.0);
      if quantity > remaining + 1e-12 {
        return Err(pyo3::exceptions::PyValueError::new_err("fill exceeds remaining quantity"));
      }
      self.filled[index] += quantity;
      let completed = self.accepted[index] - self.filled[index] <= 1e-12;
      result.push((index, self.filled[index], completed));
    }
    Ok(result)
  }

  fn remaining(&self) -> Vec<f64> {
    (0..self.accepted.len())
      .map(|index| (self.accepted[index] - self.filled[index]).max(0.0))
      .collect()
  }
}

#[pyclass(unsendable)]
pub struct TargetAllocator;

#[pymethods]
impl TargetAllocator {
  #[new]
  fn new() -> Self { Self }

  fn equal_weights(&self, count: usize) -> Vec<f64> {
    if count == 0 { return Vec::new(); }
    vec![1.0 / count as f64; count]
  }

  fn inverse_measure_weights(
    &self, measures: Vec<f64>, present: Vec<bool>, missing_equal_share: bool,
  ) -> PyResult<Vec<f64>> {
    let count = measures.len();
    if present.len() != count {
      return Err(pyo3::exceptions::PyValueError::new_err("measure batch length mismatch"));
    }
    if count == 0 { return Ok(Vec::new()); }
    let mut inverse_total = 0.0;
    let mut inverse = vec![0.0; count];
    for index in 0..count {
      if present[index] && measures[index] > 0.0 {
        inverse[index] = 1.0 / measures[index];
        inverse_total += inverse[index];
      }
    }
    if inverse_total <= 0.0 {
      return Ok(vec![1.0 / count as f64; count]);
    }
    let mut weights = vec![0.0; count];
    if missing_equal_share {
      let missing_count = (0..count).filter(|&index| inverse[index] == 0.0).count();
      let available_share = 1.0 - missing_count as f64 / count as f64;
      let equal_share = 1.0 / count as f64;
      for index in 0..count {
        weights[index] = if inverse[index] > 0.0 {
          available_share * inverse[index] / inverse_total
        } else {
          equal_share
        };
      }
    } else {
      for index in 0..count {
        weights[index] = inverse[index] / inverse_total;
      }
    }
    Ok(weights)
  }

  fn rank_indices(
    &self, values: Vec<f64>, names: Vec<String>, descending: bool,
  ) -> PyResult<Vec<usize>> {
    if names.len() != values.len() {
      return Err(pyo3::exceptions::PyValueError::new_err("rank batch length mismatch"));
    }
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&left, &right| {
      let ordering = values[left].partial_cmp(&values[right]).unwrap_or(std::cmp::Ordering::Equal);
      let ordering = if descending { ordering.reverse() } else { ordering };
      ordering.then_with(|| names[left].cmp(&names[right]))
    });
    Ok(indices)
  }

  fn screen_mask(
    &self, values: Vec<f64>, present: Vec<bool>, rule: i32,
    lower: f64, upper: f64,
  ) -> PyResult<Vec<bool>> {
    let count = values.len();
    if present.len() != count {
      return Err(pyo3::exceptions::PyValueError::new_err("screen batch length mismatch"));
    }
    let mut mask = vec![false; count];
    for index in 0..count {
      let value = values[index];
      if !present[index] || !value.is_finite() { continue; }
      mask[index] = match rule {
        1 => value >= lower,
        2 => value <= upper,
        3 => lower <= value && value <= upper,
        _ => true,
      };
    }
    Ok(mask)
  }

  fn factor_weights(
    &self, values: Vec<f64>, present: Vec<bool>, inverse: bool,
  ) -> PyResult<Vec<f64>> {
    let count = values.len();
    if present.len() != count {
      return Err(pyo3::exceptions::PyValueError::new_err("factor batch length mismatch"));
    }
    let mut raw = vec![0.0; count];
    for index in 0..count {
      let value = values[index];
      if present[index] && value.is_finite() && value > 0.0 {
        raw[index] = if inverse { 1.0 / value } else { value };
      }
    }
    let total: f64 = raw.iter().sum();
    if total <= 0.0 { return Ok(vec![0.0; count]); }
    Ok(raw.into_iter().map(|value| value / total).collect())
  }
}

#[pyclass(unsendable)]
pub struct FeeModel;

#[pymethods]
impl FeeModel {
  #[new]
  fn new() -> Self { Self }

  fn split_open_close(&self, quantity: f64, current_quantity: f64) -> (f64, f64) {
    if quantity == 0.0 { return (0.0, 0.0); }
    if current_quantity == 0.0 || (quantity > 0.0) == (current_quantity > 0.0) {
      return (quantity.abs(), 0.0);
    }
    let close_qty = quantity.abs().min(current_quantity.abs());
    ((quantity.abs() - close_qty).max(0.0), close_qty)
  }

  fn split_close_today_yesterday(
    &self, close_qty: f64, lot_quantities: Vec<f64>, lot_today: Vec<bool>,
    lot_prices: Vec<f64>, policy: i32, method: i32,
  ) -> PyResult<(f64, f64)> {
    let count = lot_quantities.len();
    if lot_today.len() != count || lot_prices.len() != count {
      return Err(pyo3::exceptions::PyValueError::new_err("lot split length mismatch"));
    }
    if close_qty <= 1e-12 { return Ok((0.0, 0.0)); }
    if policy == 1 { return Ok((close_qty, 0.0)); }
    if policy == 2 || policy == 3 { return Ok((0.0, close_qty)); }
    let mut order: Vec<usize> = (0..count).collect();
    if method == 1 { order.reverse(); }
    else if method == 2 {
      order.sort_by(|&left, &right| lot_prices[right].partial_cmp(&lot_prices[left]).unwrap_or(std::cmp::Ordering::Equal));
    }
    let mut remaining = close_qty;
    let mut today = 0.0;
    let mut yesterday = 0.0;
    for index in order {
      if remaining <= 1e-12 { break; }
      let take = remaining.min(lot_quantities[index].abs());
      if lot_today[index] { today += take; } else { yesterday += take; }
      remaining -= take;
    }
    Ok((today, yesterday + remaining.max(0.0)))
  }

  fn market_fee(
    &self, open_quantity: f64, close_yesterday: f64, close_today: f64,
    price: f64, multiplier: f64,
    open_ratio: f64, open_fixed: f64,
    close_ratio: f64, close_fixed: f64,
    close_today_ratio: f64, close_today_fixed: f64,
  ) -> f64 {
    let open_fee = open_quantity.abs() * (price * multiplier * open_ratio + open_fixed);
    let close_fee = close_yesterday.abs() * (price * multiplier * close_ratio + close_fixed);
    let today_fee = close_today.abs() * (price * multiplier * close_today_ratio + close_today_fixed);
    open_fee + close_fee + today_fee
  }

  fn fixed_fee_batch(
    &self, quantities: Vec<f64>, prices: Vec<f64>,
    multipliers: Vec<f64>, rate: f64,
  ) -> PyResult<Vec<f64>> {
    let length = quantities.len();
    if prices.len() != length || multipliers.len() != length {
      return Err(pyo3::exceptions::PyValueError::new_err("fee batch length mismatch"));
    }
    Ok((0..length).map(|index| {
      (quantities[index] * prices[index] * multipliers[index]).abs() * rate
    }).collect())
  }
}

#[pyclass(unsendable)]
pub struct VectorBacktest {
  rows: usize,
  columns: usize,
  prices: Py<numpy::PyArray2<f64>>,
  targets: Py<numpy::PyArray2<f64>>,
}

#[pymethods]
impl VectorBacktest {
  #[new]
  fn new(
    py: Python<'_>, prices: Py<numpy::PyArray2<f64>>,
    targets: Py<numpy::PyArray2<f64>>,
  ) -> PyResult<Self> {
    let price_view = prices.bind(py).readonly();
    let target_view = targets.bind(py).readonly();
    let price_array = price_view.as_array();
    let target_array = target_view.as_array();
    let (rows, columns) = price_array.dim();
    if target_array.dim() != (rows, columns) {
      return Err(pyo3::exceptions::PyValueError::new_err("shape mismatch"));
    }
    Ok(Self { rows, columns, prices, targets })
  }

  fn run(&self, py: Python<'_>, initial_cash: f64, fee_rate: f64, lot_sizes: Vec<f64>) -> Vec<f64> {
    let price_view = self.prices.bind(py).readonly();
    let target_view = self.targets.bind(py).readonly();
    let prices = price_view.as_array();
    let targets = target_view.as_array();
    let mut positions = vec![0.0f64; self.columns];
    let mut cash = initial_cash;
    let mut equity = vec![initial_cash; self.rows];
    for row in 1..self.rows {
      let mut current_equity = cash;
      for column in 0..self.columns {
        current_equity += positions[column] * prices[[row, column]];
      }
      let mut fees = 0.0;
      let mut cash_flow = 0.0;
      for column in 0..self.columns {
        let price = prices[[row, column]];
        if price <= 0.0 { continue; }
        let lot = if lot_sizes[column] > 0.0 { lot_sizes[column] } else { 1.0 };
        let target = targets[[row - 1, column]] * current_equity / price;
        let target = (target / lot).floor() * lot;
        let delta = target - positions[column];
        cash_flow += delta * price;
        fees += (delta * price).abs() * fee_rate;
        positions[column] = target;
      }
      cash -= cash_flow + fees;
      let mut value = cash;
      for column in 0..self.columns {
        value += positions[column] * prices[[row, column]];
      }
      equity[row] = value;
    }
    equity
  }
}

#[pyclass(unsendable)]
pub struct OrderAttemptRecord {
  attempt_id: String,
  order_id: String,
  revision: i32,
  timestamp: Py<PyAny>,
  market_timestamp: Py<PyAny>,
  order: Py<PyAny>,
}

#[pymethods]
impl OrderAttemptRecord {
  #[getter] fn attempt_id(&self) -> &str { &self.attempt_id }
  #[getter] fn order_id(&self) -> &str { &self.order_id }
  #[getter] fn revision(&self) -> i32 { self.revision }
  #[getter] fn timestamp(&self, py: Python<'_>) -> Py<PyAny> { self.timestamp.clone_ref(py) }
  #[getter] fn market_timestamp(&self, py: Python<'_>) -> Py<PyAny> { self.market_timestamp.clone_ref(py) }
  #[getter] fn order(&self, py: Python<'_>) -> Py<PyAny> { self.order.clone_ref(py) }

  fn to_audit_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("type", "OrderAttempt")?;
    dict.set_item("attempt_id", &self.attempt_id)?;
    dict.set_item("order_id", &self.order_id)?;
    dict.set_item("revision", self.revision)?;
    dict.set_item("timestamp", self.timestamp.clone_ref(py))?;
    dict.set_item("market_timestamp", self.market_timestamp.clone_ref(py))?;
    Ok(dict.into())
  }
}

#[pyfunction]
fn create_attempt_records(
  py: Python<'_>, attempt_ids: Vec<String>, order_ids: Vec<String>,
  revisions: Vec<i32>, timestamps: Vec<Py<PyAny>>,
  market_timestamps: Vec<Py<PyAny>>, orders: Vec<Py<PyAny>>,
) -> PyResult<Vec<Py<OrderAttemptRecord>>> {
  let length = attempt_ids.len();
  if order_ids.len() != length || revisions.len() != length
    || timestamps.len() != length || market_timestamps.len() != length
    || orders.len() != length
  {
    return Err(pyo3::exceptions::PyValueError::new_err("attempt batch length mismatch"));
  }
  let mut result = Vec::with_capacity(length);
  for index in 0..length {
    result.push(Py::new(py, OrderAttemptRecord {
      attempt_id: attempt_ids[index].clone(),
      order_id: order_ids[index].clone(),
      revision: revisions[index],
      timestamp: timestamps[index].clone_ref(py),
      market_timestamp: market_timestamps[index].clone_ref(py),
      order: orders[index].clone_ref(py),
    })?);
  }
  Ok(result)
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
  m.add_class::<OrderAttemptRecord>()?;
  m.add_function(wrap_pyfunction!(create_attempt_records, m)?)?;
  m.add_class::<VectorBacktest>()?;
  m.add_class::<FeeModel>()?;
  m.add_class::<TargetAllocator>()?;
  m.add_class::<OrderLifecycleBatch>()?;
  m.add_class::<OrderConstructionKernel>()?;
  m.add_class::<CashConstraintKernel>()?;
  m.add_class::<NativeSession>()?;
  m.add_class::<NativeGroupSession>()?;
  m.add_class::<NativeTraceBuilder>()?;
  m.add_class::<MarginAccount>()?;
  m.add_class::<LedgerState>()?;
  m.add_class::<LotBook>()?;
  register_functions(m)?;
  Ok(())
}
