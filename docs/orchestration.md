# Orchestration Architecture

This document describes the orchestration architecture in its_hub, which provides structured concurrency for managing parallel language model calls.

## Overview

The `LMOrchestrator` eliminates code duplication across algorithms by providing a single component for parallel LM invocation. It implements the `AbstractOrchestrator` interface (`its_hub.api.orchestrator`).

**Key methods:**
- `agenerate(lm, messages_lst, ...)`: Manages parallel language model calls using `asyncio.TaskGroup` (Python 3.11+)
- `generate(lm, messages_lst, ...)`: Sync wrapper via `asyncio.run()`

**When to use:** Algorithms like `SelfConsistency` and `BestOfN` accept an optional `orchestrator` parameter. If not provided, they create a default `LMOrchestrator` internally. Pass your own orchestrator when you need to control concurrency limits.

## Why Orchestration Matters for Gateways

Inference-time scaling algorithms generate multiple LM calls per user request (e.g., Self-Consistency with `budget=5` fires 5 parallel calls). Without orchestration, this creates problems in gateway deployments:

- **Rate limit exhaustion**: Uncontrolled parallelism can exceed gateway or provider rate limits, causing cascading failures
- **Resource contention**: Unbounded concurrent requests can overwhelm backend LM servers
- **Error propagation**: A single failed LM call should cancel remaining calls and surface cleanly, not leave orphaned tasks

The `AbstractOrchestrator` interface solves these by centralizing parallel execution control. Gateway teams should implement this interface to enforce their own concurrency policies, rate limits, and error handling strategies.

### Implementing a Custom Orchestrator

For gateway deployments, implement `AbstractOrchestrator` to integrate with your infrastructure's concurrency and rate limiting:

```python
import asyncio
from its_hub import AbstractOrchestrator

class MyGatewayOrchestrator(AbstractOrchestrator):
    def __init__(self, rate_limiter):
        self.rate_limiter = rate_limiter

    async def agenerate(self, lm, messages_lst, temperature_list, **kwargs):
        async def _gen_coro(messages, temp):
            async with self.rate_limiter:
                return await lm.agenerate_single(messages, temperature=temp, **kwargs)

        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(_gen_coro(msgs, temp))
                for msgs, temp in zip(messages_lst, temperature_list)
            ]

        # Collect results in order
        result = [task.result() for task in tasks]
        return result
```

Alternatively, use the built-in `LMOrchestrator` (available with `its_hub[lm]`) which provides a sensible default with `asyncio.TaskGroup` and semaphore-based concurrency control.

## Architecture Diagram

```
┌─────────────────┐
│ self-consistency│────┐
└─────────────────┘    │
                       │
┌─────────────────┐    │    ┌──────────────────┐         ┌──────────────┐
│      bon        │────┘───►│  LMOrchestrator  │─task───►│      LM      │
└─────────────────┘         │ (uses TaskGroups)│         └──────────────┘
                            └──────────────────┘
┌─────────────────┐                   ▲
│  beam search    │────┐              │
└─────────────────┘    │    ┌─────────────────┐
                       │───►│ step generation │
┌─────────────────┐    │    └─────────────────┘
│ particle gibbs  │────┘
└─────────────────┘
```

Note: Currently only Self-Consistency and Best-of-N use the orchestrator. Experimental algorithms will be migrated in a future release.

## Core Implementation

its_hub ships two orchestrator implementations with identical APIs. Both accept the same `agenerate()` / `generate()` signatures and can be passed to any algorithm's `orchestrator` parameter.

### LMOrchestrator (Python)

Pure-Python implementation using `asyncio.TaskGroup` for structured concurrency.

**Key Features:**

- **TaskGroups (Python 3.11+)**: Uses `asyncio.TaskGroup` for structured concurrency with automatic cleanup
- **Thread-Safe Semaphore**: Controls concurrency across event loops
- **Error Handling**: First exception cancels all remaining tasks

```python
from its_hub import LMOrchestrator

orchestrator = LMOrchestrator(max_concurrency=32)  # Default: 32
```

**Parameters:**
- `max_concurrency` (int, default 32): Maximum number of parallel LM calls. Set to -1 for unlimited.

### RustOrchestrator

Rust-backed alternative that uses Tokio for concurrency control. It subclasses `AbstractOrchestrator` via a thin Python wrapper around the PyO3 `RustLMOrchestrator`, so it is a drop-in replacement for `LMOrchestrator` that satisfies ABC type-checks.

```python
from its_hub import RustOrchestrator

orchestrator = RustOrchestrator(max_concurrency=32)  # Default: 32
```

**Parameters:**
- `max_concurrency` (int, default 32): Maximum number of parallel LM calls. Set to -1 for unlimited.

The raw Rust class is also available as `from its_hub._rust import RustLMOrchestrator` for cases where the ABC wrapper is not needed.

### agenerate Method

```python
async def agenerate(
    self,
    lm: AbstractLanguageModel,
    messages_lst: list[list[ChatMessage]],
    stop: str | None = None,
    max_completion_tokens: int | None = None,
    temperature: float | list[float] | None = None,
    include_stop_str_in_output: bool | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
) -> list[dict]
```

Returns responses in the same order as the input `messages_lst`.

## Example: Sync Usage

```python
import asyncio

from its_hub import LMOrchestrator, OpenAICompatibleLanguageModel, SelfConsistency
from its_hub.api import ChatMessage, ChatMessages

# Control concurrency (e.g., limit to 2 parallel calls)
orchestrator = LMOrchestrator(max_concurrency=2)

lm = OpenAICompatibleLanguageModel(
    endpoint="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4o-mini"
)

messages = ChatMessages([
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="What is 847 * 293 + 156?")
])

sc = SelfConsistency(orchestrator=orchestrator)
result = sc.infer(lm, messages, budget=5)
print(result)

# Always close LM for resource cleanup
asyncio.run(lm.close())
```

## Example: Async Usage

```python
import asyncio

from its_hub import LMOrchestrator, OpenAICompatibleLanguageModel, SelfConsistency
from its_hub.api import ChatMessage, ChatMessages

async def main():
    orchestrator = LMOrchestrator(max_concurrency=4)

    async with OpenAICompatibleLanguageModel(
        endpoint="https://api.openai.com/v1",
        api_key="your-api-key",
        model_name="gpt-4o-mini"
    ) as lm:
        messages = ChatMessages([
            ChatMessage(role="user", content="Explain quantum computing briefly.")
        ])

        sc = SelfConsistency(orchestrator=orchestrator)
        result = await sc.ainfer(lm, messages, budget=5)
        print(result)

asyncio.run(main())
```
