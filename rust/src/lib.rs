use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tokio::sync::Semaphore;

/// Rust implementation of the LMOrchestrator.
///
/// Does not subclass AbstractOrchestrator because PyO3 cannot inherit from
/// Python ABCs (PyO3 issue #991). The Python-side `RustOrchestrator` wrapper
/// (in `its_hub.core.orchestrator`) inherits the ABC and delegates here.
#[pyclass]
pub struct RustLMOrchestrator {
    #[pyo3(get)]
    max_concurrency: i32,
    semaphore: Option<Arc<Semaphore>>,
}

#[pymethods]
impl RustLMOrchestrator {
    #[new]
    #[pyo3(signature = (max_concurrency=32))]
    fn new(max_concurrency: i32) -> PyResult<Self> {
        if max_concurrency < -1 || max_concurrency == 0 {
            return Err(PyValueError::new_err(
                "max_concurrency must be -1 (unlimited concurrency) or a positive integer",
            ));
        }
        let semaphore = if max_concurrency > 0 {
            Some(Arc::new(Semaphore::new(max_concurrency as usize)))
        } else {
            None
        };
        Ok(Self {
            max_concurrency,
            semaphore,
        })
    }

    fn _semaphore_value(&self) -> Option<usize> {
        self.semaphore.as_ref().map(|s| s.available_permits())
    }

    fn _has_semaphore(&self) -> bool {
        self.semaphore.is_some()
    }

    #[pyo3(signature = (
        lm,
        messages_lst,
        stop = None,
        max_tokens = None,
        max_completion_tokens = None,
        temperature = None,
        include_stop_str_in_output = None,
        tools = None,
        tool_choice = None,
        response_format = None,
        usage_accumulator = None,
        **_kwargs
    ))]
    fn agenerate<'py>(
        &self,
        py: Python<'py>,
        lm: Bound<'py, PyAny>,
        messages_lst: Bound<'py, PyList>,
        stop: Option<PyObject>,
        max_tokens: Option<i64>,
        max_completion_tokens: Option<i64>,
        temperature: Option<Bound<'py, PyAny>>,
        include_stop_str_in_output: Option<bool>,
        tools: Option<PyObject>,
        tool_choice: Option<PyObject>,
        response_format: Option<PyObject>,
        usage_accumulator: Option<PyObject>,
        _kwargs: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let n = messages_lst.len();

        if n == 0 {
            let empty = pyo3_async_runtimes::tokio::future_into_py::<_, PyObject>(
                py,
                async move {
                    Python::with_gil(|py| {
                        let list = PyList::empty(py);
                        Ok(list.unbind().into())
                    })
                },
            )?;
            return wrap_future_as_coroutine(py, &empty);
        }

        let resolved_mct =
            resolve_max_completion_tokens(py, max_completion_tokens, max_tokens)?;
        let temperatures = expand_temperatures(py, &temperature, n)?;

        // Build shared kwargs once — only temperature varies per call
        let base_kwargs: Py<PyDict> = {
            let d = PyDict::new(py);
            d.set_item("stop", stop.as_ref())?;
            d.set_item("max_completion_tokens", resolved_mct)?;
            d.set_item("include_stop_str_in_output", include_stop_str_in_output)?;
            d.set_item("tools", tools.as_ref())?;
            d.set_item("tool_choice", tool_choice.as_ref())?;
            d.set_item("response_format", response_format.as_ref())?;
            let loop_obj = py
                .import("asyncio")?
                .call_method0("get_running_loop")
                .map(|l| l.unbind())
                .unwrap_or_else(|_| py.None());
            d.set_item("loop", loop_obj)?;
            d.set_item("usage_accumulator", usage_accumulator.as_ref())?;
            d.unbind()
        };

        let lm = lm.unbind();
        let messages: Vec<PyObject> = messages_lst.iter().map(|m| m.unbind()).collect();
        let sem = self.semaphore.clone();

        let future = pyo3_async_runtimes::tokio::future_into_py::<_, PyObject>(py, async move {
            let futures: Vec<_> = (0..n)
                .map(|i| {
                    let sem = sem.clone();
                    let (lm, msgs, base_kwargs, temp) = Python::with_gil(|py| {
                        (
                            lm.clone_ref(py),
                            messages[i].clone_ref(py),
                            base_kwargs.clone_ref(py),
                            temperatures[i].clone_ref(py),
                        )
                    });

                    async move {
                        let _permit = match &sem {
                            Some(s) => Some(s.acquire().await.expect("semaphore never closed")),
                            None => None,
                        };

                        let future = Python::with_gil(|py| {
                            let kwargs = base_kwargs.bind(py).copy()?;
                            kwargs.set_item("temperature", temp.bind(py))?;

                            let coro = lm.bind(py).call_method(
                                "agenerate_single",
                                (msgs.bind(py),),
                                Some(&kwargs),
                            )?;

                            pyo3_async_runtimes::tokio::into_future(coro)
                        })?;

                        future.await
                    }
                })
                .collect();

            match futures_util::future::try_join_all(futures).await {
                Ok(results) => Python::with_gil(|py| {
                    let list = PyList::new(py, results.iter().map(|r| r.bind(py)))?;
                    Ok(list.unbind().into())
                }),
                Err(e) => {
                    let msg = Python::with_gil(|py| {
                        let type_name = e
                            .value(py)
                            .get_type()
                            .name()
                            .map(|n| n.to_string())
                            .unwrap_or_else(|_| "Unknown".to_string());
                        format!(
                            "LMOrchestrator: 1 error(s), {} cancelled out of {} generation(s) (1x {})",
                            n - 1,
                            n,
                            type_name
                        )
                    });
                    let runtime_err = PyErr::new::<PyRuntimeError, _>(msg);
                    Python::with_gil(|py| {
                        runtime_err
                            .value(py)
                            .setattr("__cause__", e.value(py))?;
                        Ok::<_, PyErr>(())
                    })?;
                    Err(runtime_err)
                }
            }
        })?;

        wrap_future_as_coroutine(py, &future)
    }

    #[pyo3(signature = (lm, messages_lst, stop=None, **kwargs))]
    fn generate(
        self_: &Bound<'_, Self>,
        lm: &Bound<'_, PyAny>,
        messages_lst: &Bound<'_, PyList>,
        stop: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        let py = self_.py();
        let call_kwargs = PyDict::new(py);
        if let Some(s) = stop {
            call_kwargs.set_item("stop", s)?;
        }
        if let Some(kw) = kwargs {
            call_kwargs.update(kw.as_mapping())?;
        }
        // future_into_py (inside agenerate) requires a running event loop.
        // Wrap in a Python async function so asyncio.run() creates the
        // loop before agenerate is called.
        let ns = PyDict::new(py);
        ns.set_item("_orch", self_.as_any())?;
        ns.set_item("_lm", lm)?;
        ns.set_item("_msgs", messages_lst)?;
        ns.set_item("_kw", &call_kwargs)?;
        py.run(
            c"import asyncio as _aio\nasync def _f():\n    return await _orch.agenerate(_lm, _msgs, **_kw)\n_result = _aio.run(_f())",
            Some(&ns),
            Some(&ns),
        )?;
        Ok(ns.get_item("_result")?.expect("result must exist").unbind())
    }
}

fn resolve_max_completion_tokens(
    py: Python<'_>,
    max_completion_tokens: Option<i64>,
    max_tokens: Option<i64>,
) -> PyResult<Option<i64>> {
    if max_completion_tokens.is_some() && max_tokens.is_some() {
        return Err(PyValueError::new_err(
            "Cannot specify both 'max_tokens' and 'max_completion_tokens'. \
             Use 'max_completion_tokens'.",
        ));
    }
    if max_tokens.is_some() {
        let warnings = py.import("warnings")?;
        warnings.call_method1(
            "warn",
            (
                "'max_tokens' is deprecated, use 'max_completion_tokens'.",
                py.get_type::<pyo3::exceptions::PyDeprecationWarning>(),
                3i32,
            ),
        )?;
        return Ok(max_tokens);
    }
    Ok(max_completion_tokens)
}

fn expand_temperatures(
    py: Python<'_>,
    temperature: &Option<Bound<'_, PyAny>>,
    n: usize,
) -> PyResult<Vec<PyObject>> {
    match temperature {
        None => Ok((0..n).map(|_| py.None()).collect()),
        Some(temp) => {
            if let Ok(list) = temp.downcast::<PyList>() {
                Ok(list.iter().map(|item| item.unbind()).collect())
            } else {
                let val = temp.clone().unbind();
                Ok((0..n).map(|_| val.clone_ref(py)).collect())
            }
        }
    }
}

/// Wraps a `future_into_py` result (asyncio.Future) in a native coroutine
/// so it works with `asyncio.create_task()` and `asyncio.run()`.
fn wrap_future_as_coroutine<'py>(
    py: Python<'py>,
    future: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let ns = PyDict::new(py);
    ns.set_item("_f", future)?;
    py.run(c"async def _w():\n    return await _f\n_c = _w()", Some(&ns), Some(&ns))?;
    Ok(ns.get_item("_c")?.expect("coroutine wrapper"))
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustLMOrchestrator>()?;
    Ok(())
}
