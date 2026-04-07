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
from its_hub import AbstractOrchestrator

class MyGatewayOrchestrator(AbstractOrchestrator):
    def __init__(self, rate_limiter):
        self.rate_limiter = rate_limiter

    async def agenerate(self, lm, messages_lst, **kwargs):
        results = []
        for messages in messages_lst:
            await self.rate_limiter.acquire()
            result = await lm.agenerate_single(messages, **kwargs)
            results.append(result)
        return results
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

### LMOrchestrator

The `LMOrchestrator` class provides structured concurrency for parallel LM calls.

**Key Features:**

- **TaskGroups (Python 3.11+)**: Uses `asyncio.TaskGroup` for structured concurrency with automatic cleanup
- **Thread-Safe Semaphore**: Controls concurrency across event loops
- **Error Handling**: First exception cancels all remaining tasks

### Constructor

```python
from its_hub import LMOrchestrator

orchestrator = LMOrchestrator(max_concurrency=32)  # Default: 32
```

**Parameters:**
- `max_concurrency` (int, default 32): Maximum number of parallel LM calls. Set to -1 for unlimited.

### agenerate Method

```python
async def agenerate(
    self,
    lm: AbstractLanguageModel,
    messages_lst: list[list[ChatMessage]],
    stop: str | None = None,
    max_tokens: int | None = None,
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
