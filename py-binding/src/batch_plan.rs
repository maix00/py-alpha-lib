//! PyO3 adapter over the pure Rust `alpha_algo::PanelPlan`.
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyclass]
pub(crate) struct BatchPlan {
    inner: alpha_algo::PanelPlan,
}

#[pymethods]
impl BatchPlan {
    #[new]
    fn new() -> Self {
        Self { inner: alpha_algo::PanelPlan::new() }
    }

    fn grid(&mut self, times: PyReadonlyArray1<i64>, days: PyReadonlyArray1<i64>, observed: PyReadonlyArray1<u8>) -> PyResult<usize> {
        Ok(self.inner.grid(times.as_slice()?, days.as_slice()?, observed.as_slice()?)?)
    }

    fn add(&mut self, op: String, children: Vec<usize>, params: Vec<f64>, indices: Vec<i64>, grid: usize) -> PyResult<usize> {
        Ok(self.inner.add(op, children, params, indices, grid)?)
    }

    fn add_kernel(&mut self, json: String, children: Vec<usize>, grid: usize, products: usize) -> PyResult<usize> {
        Ok(self.inner.add_kernel(json,children,grid,products)?)
    }

    #[staticmethod]
    fn asof_indices(source: PyReadonlyArray1<i64>, target: PyReadonlyArray1<i64>) -> PyResult<Vec<i64>> {
        Ok(alpha_algo::PanelPlan::asof_indices(source.as_slice()?, target.as_slice()?)?)
    }

    #[staticmethod]
    fn asof_partition_indices(source: PyReadonlyArray1<i64>, target: PyReadonlyArray1<i64>, partitions: Vec<[i64;4]>) -> PyResult<Vec<i64>> {
        Ok(alpha_algo::PanelPlan::asof_partition_indices(source.as_slice()?,target.as_slice()?,&partitions)?)
    }

    #[staticmethod]
    fn signal_grid(times: PyReadonlyArray1<i64>, observed: PyReadonlyArray1<u8>, products: usize,
        partitions: Vec<[i64;4]>, period: i64, first: bool, inclusive: bool,
        partial: bool) -> PyResult<(Vec<i64>, Vec<i64>)> {
        Ok(alpha_algo::PanelPlan::signal_grid(times.as_slice()?, observed.as_slice()?, products,
            &partitions, period, first, inclusive, partial)?)
    }

    #[staticmethod]
    fn signal_indices(keys: PyReadonlyArray1<i64>, times: PyReadonlyArray1<i64>, multiple: usize, basepoint: &str, mask: Vec<bool>, gap: Option<i64>) -> PyResult<Vec<usize>> {
        Ok(alpha_algo::PanelPlan::signal_indices(keys.as_slice()?, times.as_slice()?, multiple, basepoint, mask, gap)?)
    }

    fn execute_panel(&self, py: Python<'_>, columns: Vec<PyReadonlyArray1<f64>>, outputs: Vec<usize>, products: usize) -> PyResult<Vec<Py<numpy::PyArray1<f64>>>> {
        let columns: Vec<&[f64]> = columns.iter().map(|c| c.as_slice()).collect::<Result<_, _>>()?;
        let result = self.inner.execute_panel(&columns, &outputs, products)?;
        Ok(result.into_iter().map(|values| values.into_pyarray(py).unbind()).collect())
    }
}
