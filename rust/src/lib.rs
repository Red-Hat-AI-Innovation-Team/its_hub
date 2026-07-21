mod adapters;
mod core;

use pyo3::prelude::*;

use adapters::RustLMOrchestrator;

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustLMOrchestrator>()?;
    Ok(())
}
