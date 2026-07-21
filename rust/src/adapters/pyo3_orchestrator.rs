// Layer 2: PyO3 bridge between Python callers and the pure Rust orchestrator.
//
// Handles Python<->Rust async interop and Python-specific cancellation.
// Not meant to be used directly — Python users go through
// RustLMOrchestrator (Layer 3, in its_hub/core/orchestrator.py), since
// the PyO3 bridge can't inherit the AbstractOrchestrator Python ABC

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyDict, PyList};

use crate::core::Orchestrator;

static COROUTINE_WRAPPER: GILOnceCell<PyObject> = GILOnceCell::new();

fn get_coroutine_wrapper(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    COROUTINE_WRAPPER
        .get_or_try_init(py, || {
            let ns = PyDict::new(py);
            py.run(
                c"async def _wrap(f):\n    return await f\n",
                Some(&ns),
                Some(&ns),
            )?;
            Ok(ns
                .get_item("_wrap")?
                .expect("wrapper function")
                .unbind())
        })
        .map(|obj| obj.bind(py))
}

fn wrap_future_as_coroutine<'py>(
    py: Python<'py>,
    future: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    get_coroutine_wrapper(py)?.call1((future,))
}

#[pyclass(name = "_PyLMOrchestrator")]
pub struct PyLMOrchestrator {
    #[pyo3(get)]
    max_concurrency: i32,
    inner: Arc<Orchestrator>,
}

#[pymethods]
impl PyLMOrchestrator {
    #[new]
    #[pyo3(signature = (max_concurrency=32))]
    fn new(max_concurrency: i32) -> PyResult<Self> {
        let inner = Orchestrator::new(max_concurrency)
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(Self {
            max_concurrency,
            inner: Arc::new(inner),
        })
    }

    fn _semaphore_value(&self) -> Option<usize> {
        self.inner.available_permits()
    }

    fn _has_semaphore(&self) -> bool {
        self.inner.has_semaphore()
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
    ) -> PyResult<Bound<'py, PyAny>> {
        let n = messages_lst.len();

        if n == 0 {
            let fut = pyo3_async_runtimes::tokio::future_into_py::<_, PyObject>(
                py,
                async move {
                    Python::with_gil(|py| Ok(PyList::empty(py).unbind().into()))
                },
            )?;
            return wrap_future_as_coroutine(py, &fut);
        }

        let resolved_mct =
            resolve_max_completion_tokens(py, max_completion_tokens, max_tokens)?;
        let temperatures = expand_temperatures(py, &temperature, n)?;

        let loop_obj = py
            .import("asyncio")?
            .call_method0("get_running_loop")?
            .unbind();

        let base_kwargs: Py<PyDict> = {
            let d = PyDict::new(py);
            d.set_item("stop", stop.as_ref())?;
            d.set_item("max_completion_tokens", resolved_mct)?;
            d.set_item("include_stop_str_in_output", include_stop_str_in_output)?;
            d.set_item("tools", tools.as_ref())?;
            d.set_item("tool_choice", tool_choice.as_ref())?;
            d.set_item("response_format", response_format.as_ref())?;
            d.set_item("loop", loop_obj.bind(py))?;
            d.set_item("usage_accumulator", usage_accumulator.as_ref())?;
            d.unbind()
        };

        let task_refs: Py<PyList> = PyList::empty(py).unbind();
        let lm = lm.unbind();
        let messages: Vec<PyObject> = messages_lst.iter().map(|m| m.unbind()).collect();
        let inner = self.inner.clone();

        let future = pyo3_async_runtimes::tokio::future_into_py::<_, PyObject>(py, async move {
            let task_fns: Vec<_> = Python::with_gil(|py| {
                (0..n)
                    .map(|i| {
                        let lm = lm.clone_ref(py);
                        let msgs = messages[i].clone_ref(py);
                        let base_kwargs = base_kwargs.clone_ref(py);
                        let temp = temperatures[i].clone_ref(py);
                        let loop_obj_i = loop_obj.clone_ref(py);
                        let task_refs_i = task_refs.clone_ref(py);

                        move || async move {
                            let future = Python::with_gil(|py| {
                                let kwargs = base_kwargs.bind(py).copy()?;
                                kwargs.set_item("temperature", temp.bind(py))?;

                                let coro = lm.bind(py).call_method(
                                    "agenerate_single",
                                    (msgs.bind(py),),
                                    Some(&kwargs),
                                )?;

                                let task = loop_obj_i
                                    .bind(py)
                                    .call_method1("create_task", (&coro,))?;
                                task_refs_i.bind(py).append(&task)?;

                                pyo3_async_runtimes::tokio::into_future(task)
                            })?;

                            future.await
                        }
                    })
                    .collect()
            });

            match inner.execute_all(task_fns).await {
                Ok(results) => Python::with_gil(|py| {
                    let list = PyList::new(py, results.iter().map(|r| r.bind(py)))?;
                    Ok(list.unbind().into())
                }),
                Err(e) => {
                    let runtime_err = Python::with_gil(|py| {
                        for task in task_refs.bind(py).iter() {
                            let _ = task.call_method0("cancel");
                        }
                        let type_name = e
                            .value(py)
                            .get_type()
                            .name()
                            .map(|n| n.to_string())
                            .unwrap_or_else(|_| "Unknown".to_string());
                        // note that try_join_all reports only the first error; Python's TaskGroup collects all.
                        let msg = format!(
                            "LMOrchestrator: 1 error(s), {} cancelled out of {} generation(s) (1x {})",
                            n - 1,
                            n,
                            type_name
                        );
                        let err = PyErr::new::<PyRuntimeError, _>(msg);
                        err.value(py)
                            .setattr("__cause__", e.value(py))
                            .ok();
                        err
                    });
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
                if list.len() != n {
                    return Err(PyValueError::new_err(format!(
                        "temperature list length ({}) must match messages_lst length ({})",
                        list.len(),
                        n
                    )));
                }
                Ok(list.iter().map(|item| item.unbind()).collect())
            } else {
                let val = temp.clone().unbind();
                Ok((0..n).map(|_| val.clone_ref(py)).collect())
            }
        }
    }
}
