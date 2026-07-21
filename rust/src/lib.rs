mod adapters;
mod core;

use pyo3::prelude::*;

use adapters::PyLMOrchestrator;

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLMOrchestrator>()?;
    Ok(())
}
