# Orchestration Architecture

This document describes the orchestration architecture in its_hub, which provides reusable components for managing language model calls across different inference-time scaling algorithms.

## Overview

The its_hub library uses an orchestration pattern to eliminate code duplication and provide structured concurrency for LM calls. The core package provides an inline implementation of the orchestrator interface (`its_hub.api.AbstractOrchestrator`). The implementation includes:

- `agenerate(lm, messages_lst, ...)`: Manages language model calls using taskgroups to generate responses for a batch of messages
- `generate(lm, messages_list, ...)`: Calls agenerate method

## Architecture Diagram

The workflow follows this pattern:

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

Note: At the time of writing, only the self-consistency and best-of-n algorithms use orchestrator. The experimental features will be migrated to the outlined workflow in the near future.

## Core Implementation

### LMOrchestrator

The `LMOrchestrator` class provides a unified interface for making language model calls with structured concurrency.

**Key Features:**

- **TaskGroups (Python 3.11+)**: Uses `asyncio.TaskGroup` for structured concurrency
- **Automatic Cleanup**: Tasks are automatically cancelled if any task raises an exception
- **Centralized Logging**: All LM calls are logged through the orchestrator
- **Error Handling**: First exception cancels all remaining tasks

**Example Usage:**

```python
from its_hub import LMOrchestrator, OpenAICompatibleLanguageModel, SelfConsistency
from its_hub.api import ChatMessage, ChatMessages

# Initialize orchestrator
orchestrator = LMOrchestrator(max_concurrency=2)

# Initialize language model
lm = OpenAICompatibleLanguageModel(
    endpoint="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4o-mini"
)

messages = ChatMessages([
    ChatMessage(
        role="system",
        content="You are a precise calculator. Always use the calculator tool for arithmetic."
    ),
    ChatMessage(
        role="user",
        content="What is 847 * 293 + 156?"
    )
])

# Use hierarchical tool voting
sc = SelfConsistency(tool_vote="tool_hierarchical", orchestrator=orchestrator)
result = sc.infer(
    lm,
    messages,
    budget=5,
    tools=tools,
    tool_choice="auto"
)
print(result)
```