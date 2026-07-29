// Layer 2: PyO3 bridge between Python callers and the pure Rust orchestrator.
//
// This layer only marshals values across the Python<->Rust boundary and handles
// Python-specific cancellation. The actual concurrency-limited fan-out runs in
// Rust, in the pure Orchestrator (Layer 1).
//
// Don't use this directly. A PyO3 class can't inherit a Python ABC, so Python
// users go through RustLMOrchestrator (Layer 3, in its_hub/core/orchestrator.py),
// which subclasses AbstractOrchestrator and wraps this.
//
// `agenerate` is written as a native `async fn`, which PyO3's experimental-async
// feature exposes to Python as a real coroutine. That means callers can `await`,
// `create_task`, or `gather` it like any other coroutine.

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::core::Orchestrator;

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
        let inner = Orchestrator::new(max_concurrency).map_err(PyValueError::new_err)?;

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
    #[allow(clippy::too_many_arguments)]
    async fn agenerate(
        &self,
        lm: Py<PyAny>,
        messages_lst: Py<PyList>,
        stop: Option<PyObject>,
        max_tokens: Option<i64>,
        max_completion_tokens: Option<i64>,
        temperature: Option<PyObject>,
        include_stop_str_in_output: Option<bool>,
        tools: Option<PyObject>,
        tool_choice: Option<PyObject>,
        response_format: Option<PyObject>,
        usage_accumulator: Option<PyObject>,
    ) -> PyResult<PyObject> {
        let inner = self.inner.clone();

        // Do all the up-front work that needs the GIL in one pass: validate the
        // token args, grab the running event loop, expand per-call temperatures,
        // and build the kwargs dict shared by every generation.
        let (n, base_kwargs, loop_obj, temperatures, task_refs, messages) =
            Python::with_gil(|py| -> PyResult<_> {
                let messages_bound = messages_lst.bind(py);
                let n = messages_bound.len();

                let resolved_mct =
                    resolve_max_completion_tokens(py, max_completion_tokens, max_tokens)?;
                let temperatures =
                    expand_temperatures(py, temperature.as_ref().map(|t| t.bind(py).clone()), n)?;

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
                let messages: Vec<PyObject> = messages_bound.iter().map(Bound::unbind).collect();

                Ok((n, base_kwargs, loop_obj, temperatures, task_refs, messages))
            })?;

        if n == 0 {
            return Python::with_gil(|py| Ok(PyList::empty(py).unbind().into()));
        }

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
                        // Dropping the Rust future returned by `into_future` does not
                        // cancel the Python task it awaits. So instead of awaiting the
                        // coroutine directly, we wrap it in an asyncio.Task and keep a
                        // handle to it — that handle is what lets us cancel siblings on
                        // error further down.
                        let future = Python::with_gil(|py| {
                            let kwargs = base_kwargs.bind(py).copy()?;
                            kwargs.set_item("temperature", temp.bind(py))?;

                            let coro = lm.bind(py).call_method(
                                "agenerate_single",
                                (msgs.bind(py),),
                                Some(&kwargs),
                            )?;

                            let task = loop_obj_i.bind(py).call_method1("create_task", (&coro,))?;
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
                    let mut cancelled = 0usize;
                    for task in task_refs.bind(py).iter() {
                        if task
                            .call_method0("cancel")
                            .and_then(|r| r.is_truthy())
                            .unwrap_or(false)
                        {
                            cancelled += 1;
                        }
                    }
                    let type_name = e
                        .value(py)
                        .get_type()
                        .name()
                        .map_or_else(|_| "Unknown".to_string(), |n| n.to_string());
                    // We always report "1 error(s)" because try_join_all returns as
                    // soon as one task fails, so that's the only error we've seen.
                    // (Python's TaskGroup differs here — it gathers every error into
                    // an ExceptionGroup and can report the true count.)
                    let msg = format!(
                        "LMOrchestrator: 1 error(s), {cancelled} preemptively cancelled out of {n} generation(s) (1x {type_name})"
                    );
                    let err = PyErr::new::<PyRuntimeError, _>(msg);
                    err.value(py).setattr("__cause__", e.value(py)).ok();
                    err
                });
                Err(runtime_err)
            }
        }
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
        // Synchronous entry point: agenerate hands back a coroutine, so we hand it
        // to asyncio.run to drive it to completion on a fresh event loop.
        let coro = self_.call_method("agenerate", (lm, messages_lst), Some(&call_kwargs))?;
        Ok(py.import("asyncio")?.call_method1("run", (coro,))?.unbind())
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
    temperature: Option<Bound<'_, PyAny>>,
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
                Ok(list.iter().map(Bound::unbind).collect())
            } else {
                let val = temp.clone().unbind();
                Ok((0..n).map(|_| val.clone_ref(py)).collect())
            }
        }
    }
}
