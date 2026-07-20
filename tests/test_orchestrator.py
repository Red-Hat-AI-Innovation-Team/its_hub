"""Tests for LMOrchestrator, RustLMOrchestrator, and _ThreadSafeAsyncSemaphore."""

import asyncio
import threading
import time

import pytest

from its_hub.api.types import ChatMessage
from its_hub.core.orchestrator import LMOrchestrator, _ThreadSafeAsyncSemaphore

from its_hub._rust import RustLMOrchestrator


@pytest.fixture(params=[
    pytest.param(LMOrchestrator, id="python"),
    pytest.param(RustLMOrchestrator, id="rust"),
])
def orchestrator_cls(request):
    """Return each orchestrator implementation so shared tests run against both."""
    return request.param


# ---------------------------------------------------------------------------
# Mock LM helpers
# ---------------------------------------------------------------------------

class MockLM:
    """Mock LM that tracks peak concurrency (thread-safe)."""

    def __init__(self, delay: float = 0.05, responses: list[str] | None = None):
        self.delay = delay
        self.responses = responses or ["response"]
        self.call_count = 0
        self.active = 0
        self.peak = 0
        self._lock = threading.Lock()
        self.calls: list[dict] = []

    async def agenerate_single(self, messages, loop=None, **kwargs):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            idx = self.call_count % len(self.responses)
            self.call_count += 1
            self.calls.append({"messages": messages, **kwargs})

        await asyncio.sleep(self.delay)

        with self._lock:
            self.active -= 1

        return {"role": "assistant", "content": self.responses[idx]}

    def reset(self):
        with self._lock:
            self.call_count = 0
            self.active = 0
            self.peak = 0
            self.calls.clear()


class ErrorMockLM:
    """Mock LM that raises on specified call indices (thread-safe)."""

    def __init__(self, error_indices: set[int] | None = None, delay: float = 0.02):
        self.error_indices = error_indices or set()
        self.delay = delay
        self.call_count = 0
        self._lock = threading.Lock()

    async def agenerate_single(self, messages, loop=None, **kwargs):
        with self._lock:
            idx = self.call_count
            self.call_count += 1

        await asyncio.sleep(self.delay)

        if idx in self.error_indices:
            raise RuntimeError(f"Simulated error on call {idx}")

        return {"role": "assistant", "content": f"ok-{idx}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(n: int) -> list[list[ChatMessage]]:
    return [
        [ChatMessage(role="user", content=f"msg-{i}")]
        for i in range(n)
    ]


def _sem_value(orchestrator) -> int:
    """Read the semaphore counter (works for both Python and Rust implementations)."""
    # The Rust orchestrator doesn't expose its tokio::sync::Semaphore as a Python
    # attribute, so it provides a _semaphore_value() method instead.
    if hasattr(orchestrator, "_semaphore_value"):
        val = orchestrator._semaphore_value()
        assert val is not None
        return val
    assert orchestrator._semaphore is not None
    return orchestrator._semaphore._sem._value


def _has_semaphore(orchestrator) -> bool:
    """Check whether the orchestrator has a semaphore."""
    # Similar to _sem_value, we have a _has_semaphore function in the Rust implementation
    if hasattr(orchestrator, "_has_semaphore"):
        return orchestrator._has_semaphore()
    return orchestrator._semaphore is not None


# ===========================================================================
# 1. Construction & Validation
# ===========================================================================

class TestConstruction:
    def test_default_max_concurrency(self, orchestrator_cls):
        orch = orchestrator_cls()
        assert orch.max_concurrency == 32
        assert _has_semaphore(orch)

    def test_custom_max_concurrency(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=10)
        assert orch.max_concurrency == 10
        assert _has_semaphore(orch)

    def test_unlimited_concurrency(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=-1)
        assert orch.max_concurrency == -1
        assert not _has_semaphore(orch)

    def test_min_concurrency(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=1)
        assert orch.max_concurrency == 1

    def test_zero_concurrency_raises(self, orchestrator_cls):
        with pytest.raises(ValueError):
            orchestrator_cls(max_concurrency=0)

    def test_negative_concurrency_raises(self, orchestrator_cls):
        with pytest.raises(ValueError):
            orchestrator_cls(max_concurrency=-2)


# ===========================================================================
# 2. Basic agenerate Behavior
# ===========================================================================

class TestBasicAgenerate:
    @pytest.mark.asyncio
    async def test_empty_batch(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=4)
        lm = MockLM()
        result = await orch.agenerate(lm, [])
        assert result == []
        assert lm.call_count == 0

    @pytest.mark.asyncio
    async def test_single_message(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=4)
        lm = MockLM(responses=["hello"])
        batch = _make_batch(1)
        result = await orch.agenerate(lm, batch)
        assert len(result) == 1
        assert result[0]["content"] == "hello"
        assert lm.call_count == 1

    @pytest.mark.asyncio
    async def test_batch_of_n(self, orchestrator_cls):
        n = 5
        orch = orchestrator_cls(max_concurrency=10)
        lm = MockLM(responses=[f"r{i}" for i in range(n)])
        batch = _make_batch(n)
        result = await orch.agenerate(lm, batch)
        assert len(result) == n
        assert lm.call_count == n

    @pytest.mark.asyncio
    async def test_response_ordering(self, orchestrator_cls):
        """Responses must match input order even with variable latency."""
        n = 6
        orch = orchestrator_cls(max_concurrency=10)

        class VariableDelayLM:
            def __init__(self):
                self._lock = threading.Lock()
                self.call_count = 0

            async def agenerate_single(self, messages, loop=None, **kwargs):
                with self._lock:
                    idx = self.call_count
                    self.call_count += 1
                # Reverse delay so later items finish first
                await asyncio.sleep(0.01 * (n - idx))
                content = messages[0].content if hasattr(messages[0], "content") else messages[0]["content"]
                return {"role": "assistant", "content": f"reply-to-{content}"}

        lm = VariableDelayLM()
        batch = _make_batch(n)
        result = await orch.agenerate(lm, batch)

        for i in range(n):
            assert result[i]["content"] == f"reply-to-msg-{i}"


# ===========================================================================
# 3. Parameter Forwarding
# ===========================================================================

class TestParameterForwarding:
    @pytest.mark.asyncio
    async def test_stop_forwarded(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=4)
        lm = MockLM()
        await orch.agenerate(lm, _make_batch(1), stop="\n")
        assert lm.calls[0]["stop"] == "\n"

    @pytest.mark.asyncio
    async def test_max_completion_tokens_forwarded(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=4)
        lm = MockLM()
        await orch.agenerate(lm, _make_batch(1), max_completion_tokens=100)
        assert lm.calls[0]["max_completion_tokens"] == 100

    @pytest.mark.asyncio
    async def test_max_tokens_deprecated_alias(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=4)
        lm = MockLM()
        with pytest.warns(DeprecationWarning, match="max_tokens.*deprecated"):
            await orch.agenerate(lm, _make_batch(1), max_tokens=100)
        assert lm.calls[0]["max_completion_tokens"] == 100

    @pytest.mark.asyncio
    async def test_scalar_temperature(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=4)
        lm = MockLM()
        await orch.agenerate(lm, _make_batch(3), temperature=0.7)
        for call in lm.calls:
            assert call["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_per_message_temperature(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=4)
        lm = MockLM()
        temps = [0.5, 0.8, 1.0]
        await orch.agenerate(lm, _make_batch(3), temperature=temps)
        # Match by content (run_in_executor does not guarantee execution or completion order)
        temp_by_msg = {
            call["messages"][0].content: call["temperature"]
            for call in lm.calls
        }
        for i, expected_temp in enumerate(temps):
            assert temp_by_msg[f"msg-{i}"] == expected_temp

    @pytest.mark.asyncio
    async def test_include_stop_str_forwarded(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=4)
        lm = MockLM()
        await orch.agenerate(lm, _make_batch(1), include_stop_str_in_output=True)
        assert lm.calls[0]["include_stop_str_in_output"] is True

    @pytest.mark.asyncio
    async def test_tools_forwarded(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=4)
        lm = MockLM()
        tools = [{"type": "function", "function": {"name": "test"}}]
        await orch.agenerate(lm, _make_batch(1), tools=tools)
        assert lm.calls[0]["tools"] == tools

    @pytest.mark.asyncio
    async def test_tool_choice_forwarded(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=4)
        lm = MockLM()
        await orch.agenerate(lm, _make_batch(1), tool_choice="auto")
        assert lm.calls[0]["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_all_none_defaults(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=4)
        lm = MockLM()
        await orch.agenerate(lm, _make_batch(1))
        call = lm.calls[0]
        assert call["stop"] is None
        assert call["max_completion_tokens"] is None
        assert call["temperature"] is None
        assert call["include_stop_str_in_output"] is None
        assert call["tools"] is None
        assert call["tool_choice"] is None


# ===========================================================================
# 4. Sync Wrapper generate()
# ===========================================================================

class TestSyncWrapper:
    def test_generate_returns_same_as_agenerate(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=4)
        lm = MockLM(delay=0.01, responses=["sync-resp"])
        batch = _make_batch(2)
        result = orch.generate(lm, batch)
        assert len(result) == 2
        assert all(r["content"] == "sync-resp" for r in result)

    def test_generate_works_without_running_loop(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=4)
        lm = MockLM(delay=0.01)
        result = orch.generate(lm, _make_batch(1))
        assert len(result) == 1


# ===========================================================================
# 5. Loop Parameter Forwarding
# ===========================================================================

class TestLoopForwarding:
    @pytest.mark.asyncio
    async def test_loop_forwarded_to_agenerate_single(self, orchestrator_cls):
        """Orchestrator passes the current event loop to agenerate_single."""
        orch = orchestrator_cls(max_concurrency=4)
        received_loops = []

        class LoopCaptureLM:
            async def agenerate_single(self, messages, loop=None, **kwargs):
                received_loops.append(loop)
                return {"role": "assistant", "content": "ok"}

        lm = LoopCaptureLM()
        await orch.agenerate(lm, _make_batch(3))

        current_loop = asyncio.get_running_loop()
        assert len(received_loops) == 3
        assert all(l is current_loop for l in received_loops)

    def test_sequential_sync_calls_no_stale_loop(self, orchestrator_cls):
        """Repeated generate() calls must not raise RuntimeError from stale sessions."""
        orch = orchestrator_cls(max_concurrency=4)
        lm = MockLM(delay=0.01, responses=["resp"])

        # Two sequential sync calls — each creates a new event loop via asyncio.run()
        result1 = orch.generate(lm, _make_batch(2))
        result2 = orch.generate(lm, _make_batch(2))

        assert len(result1) == 2
        assert len(result2) == 2
        assert lm.call_count == 4


# ===========================================================================
# 6. Concurrency Limiting (single event loop)
# ===========================================================================

class TestConcurrencyLimiting:
    @pytest.mark.asyncio
    async def test_max_concurrency_respected(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=2)
        lm = MockLM(delay=0.05)
        await orch.agenerate(lm, _make_batch(10))
        assert lm.peak <= 2

    @pytest.mark.asyncio
    async def test_serial_execution(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=1)
        lm = MockLM(delay=0.02)
        await orch.agenerate(lm, _make_batch(5))
        assert lm.peak == 1

    @pytest.mark.asyncio
    async def test_unlimited_concurrency(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=-1)
        lm = MockLM(delay=0.05)
        n = 10
        await orch.agenerate(lm, _make_batch(n))
        assert lm.peak == n

    @pytest.mark.asyncio
    async def test_concurrency_not_artificially_limited(self, orchestrator_cls):
        """When max_concurrency > batch size, all run concurrently."""
        orch = orchestrator_cls(max_concurrency=100)
        lm = MockLM(delay=0.05)
        n = 5
        await orch.agenerate(lm, _make_batch(n))
        assert lm.peak == n

    @pytest.mark.asyncio
    async def test_shared_semaphore_across_sequential_calls(self, orchestrator_cls):
        """Two overlapping agenerate calls share the same semaphore."""
        orch = orchestrator_cls(max_concurrency=2)
        lm = MockLM(delay=0.05)

        async def run_both():
            t1 = asyncio.create_task(orch.agenerate(lm, _make_batch(5)))
            t2 = asyncio.create_task(orch.agenerate(lm, _make_batch(5)))
            await asyncio.gather(t1, t2)

        await run_both()
        assert lm.peak <= 2


# ===========================================================================
# 7. Cross-Thread Concurrency
# ===========================================================================

class TestCrossThreadConcurrency:
    def test_cross_thread_limit_respected(self, orchestrator_cls):
        """Two threads calling generate() share the global concurrency limit."""
        max_c = 2
        orch = orchestrator_cls(max_concurrency=max_c)
        lm = MockLM(delay=0.1)

        def worker():
            orch.generate(lm, _make_batch(5))

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert lm.peak <= max_c

    def test_cross_thread_limit_three_threads(self, orchestrator_cls):
        max_c = 4
        orch = orchestrator_cls(max_concurrency=max_c)
        lm = MockLM(delay=0.08)

        def worker():
            orch.generate(lm, _make_batch(4))

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert lm.peak <= max_c

    def test_cross_thread_unlimited(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=-1)
        lm = MockLM(delay=0.05)

        def worker():
            orch.generate(lm, _make_batch(4))

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # With unlimited concurrency, all 8 tasks across 2 threads can run
        assert lm.peak > 1


# ===========================================================================
# 8. Error Handling
# ===========================================================================

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_single_error_propagates(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=4)
        lm = ErrorMockLM(error_indices={2})
        with pytest.raises(
            RuntimeError,
            match=r"LMOrchestrator:.*error\(s\).*out of 5 generation\(s\).*RuntimeError",
        ):
            await orch.agenerate(lm, _make_batch(5))

    @pytest.mark.asyncio
    async def test_all_errors_propagate(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=4)
        lm = ErrorMockLM(error_indices={0, 1, 2})
        with pytest.raises(
            RuntimeError,
            match=r"LMOrchestrator:.*error\(s\).*out of 3 generation\(s\).*RuntimeError",
        ):
            await orch.agenerate(lm, _make_batch(3))

    @pytest.mark.asyncio
    async def test_error_message_includes_count(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=4)
        lm = ErrorMockLM(error_indices={0, 2})
        with pytest.raises(RuntimeError) as exc_info:
            await orch.agenerate(lm, _make_batch(4))
        msg = str(exc_info.value)
        assert "out of 4 generation(s)" in msg
        assert "RuntimeError" in msg
        assert exc_info.value.__cause__ is not None

    @pytest.mark.asyncio
    async def test_semaphore_released_after_error(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=4)
        lm = ErrorMockLM(error_indices={1})

        with pytest.raises(RuntimeError):
            await orch.agenerate(lm, _make_batch(3))

        # Semaphore should be fully released
        assert _sem_value(orch) == 4

    @pytest.mark.asyncio
    async def test_next_batch_works_after_error(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=4)

        # First batch: errors
        error_lm = ErrorMockLM(error_indices={0})
        with pytest.raises(RuntimeError):
            await orch.agenerate(error_lm, _make_batch(2))

        # Second batch: should work fine
        ok_lm = MockLM(responses=["recovered"])
        result = await orch.agenerate(ok_lm, _make_batch(3))
        assert len(result) == 3
        assert all(r["content"] == "recovered" for r in result)


# ===========================================================================
# 9. Semaphore Leak / Release Safety
# ===========================================================================

class TestSemaphoreSafety:
    @pytest.mark.asyncio
    async def test_semaphore_restored_after_success(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=8)
        lm = MockLM(delay=0.01)

        initial = _sem_value(orch)
        await orch.agenerate(lm, _make_batch(5))
        assert _sem_value(orch) == initial

    @pytest.mark.asyncio
    async def test_semaphore_restored_after_failure(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=8)
        lm = ErrorMockLM(error_indices={0, 1, 2, 3, 4})

        initial = _sem_value(orch)
        with pytest.raises(RuntimeError):
            await orch.agenerate(lm, _make_batch(5))
        assert _sem_value(orch) == initial

    @pytest.mark.asyncio
    async def test_semaphore_restored_after_cancellation(self, orchestrator_cls):
        orch = orchestrator_cls(max_concurrency=4)
        lm = MockLM(delay=1.0)  # Long delay so we can cancel

        initial = _sem_value(orch)
        task = asyncio.create_task(orch.agenerate(lm, _make_batch(4)))
        await asyncio.sleep(0.05)  # Let tasks start acquiring
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Give executor threads a moment to finish releasing
        await asyncio.sleep(0.1)
        assert _sem_value(orch) == initial


# ===========================================================================
# 10. _ThreadSafeAsyncSemaphore Unit Tests
# ===========================================================================

class TestThreadSafeAsyncSemaphore:
    @pytest.mark.asyncio
    async def test_acquire_release(self):
        sem = _ThreadSafeAsyncSemaphore(2)
        assert sem._sem._value == 2
        async with sem:
            assert sem._sem._value == 1
        assert sem._sem._value == 2

    @pytest.mark.asyncio
    async def test_release_on_exception(self):
        sem = _ThreadSafeAsyncSemaphore(1)
        with pytest.raises(ValueError):
            async with sem:
                raise ValueError("boom")
        assert sem._sem._value == 1

    @pytest.mark.asyncio
    async def test_limits_concurrency(self):
        sem = _ThreadSafeAsyncSemaphore(1)
        order = []

        async def work(label):
            async with sem:
                order.append(f"{label}-start")
                await asyncio.sleep(0.05)
                order.append(f"{label}-end")

        await asyncio.gather(work("a"), work("b"))

        # With concurrency=1, one must finish before the other starts
        a_end = order.index("a-end")
        b_start = order.index("b-start")
        b_end = order.index("b-end")
        a_start = order.index("a-start")
        # Either a finishes before b starts, or b finishes before a starts
        assert (a_end < b_start) or (b_end < a_start)

    def test_works_across_threads(self):
        """Two threads sharing the same semaphore respect the limit."""
        sem = _ThreadSafeAsyncSemaphore(1)
        active = 0
        peak = 0
        lock = threading.Lock()

        async def work():
            nonlocal active, peak
            async with sem:
                with lock:
                    active += 1
                    peak = max(peak, active)
                await asyncio.sleep(0.05)
                with lock:
                    active -= 1

        def thread_fn():
            asyncio.run(work())

        threads = [threading.Thread(target=thread_fn) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert peak <= 1
