# ITS-Hub Interface Design Specification

## Document Purpose

This document defines the **interface contracts** for the its-hub library. It specifies the abstractions, method signatures, inputs/outputs, and functional requirements needed for implementation.

**Target Audience**: Software engineers implementing or integrating with its-hub
**Usage**: Implementation guide and review checklist for ensuring compliance with the architecture

---

## ITS-Hub Architecture Overview

**its-hub** is a lightweight library providing inference-time scaling (ITS) algorithms for improving LLM response quality. The architecture separates gateway integration from algorithm implementation, enabling algorithms to work with any AI gateway.

<img src="assets/images/its-architecture-diagram.png" alt="ITS Architecture" width="600"/>

### Architecture Layers → Interface Design Mapping

| Layer | What It Is | Part of its-hub repo? | Where It Lives | Interface Section |
|-------|-----------|----------------------|----------------|-------------------|
| **Demo & Use Cases** (Top) | Agent applications (Langflow, LangChain, etc.) | ❌ No | External applications | Out of scope |
| **Gateway Layer** | AI Gateways (Portkey, LiteLLM, Envoy) | ❌ No | External systems | **Gateway Interface** |
| **Integration Layer** | Wraps gateway LM + Reward implementations | ❌ No | Gateway codebase (Direct PR) OR its-hub adapters (Plug-in) | **LM & Reward Interfaces** |
| **Algorithm Layer** | ITS algorithms (consume LM + Reward) | ✅ Yes | its-hub core | **Algorithm Interface** |

### What This Repository Contains

**its-hub core library provides**:
1. **Interface Definitions**: `AbstractLanguageModel`, `AbstractOutcomeRewardModel`, `AbstractScalingAlgorithm`
2. **Algorithm Implementations**: Self-Consistency, Best-of-N
3. **Default Helper**: `create_llm_judge()` - converts any LM into basic reward model

**Adapter Ecosystem** (separate packages):
- **its-hub-portkey**: Portkey gateway adapter (`pip install its-hub-portkey`)
- **its-hub-litellm**: LiteLLM gateway adapter (`pip install its-hub-litellm`)
- **its-hub-openai**: OpenAI-compatible adapter (`pip install its-hub-openai`)
- Each adapter package depends on: `its-hub` (core) + gateway-specific SDK

### Interface Definitions (Scope of its-hub)

This document defines **4 key interfaces** for inference-time scaling:

| Interface | Scope | Description |
|-----------|-------|-------------|
| **Gateway Interface** | ❌ External (out of scope) | Defines ITS request format (HTTP headers, parameters) - protocol contract only |
| **LM Interface** | ✅ its-hub core | Wraps gateway LM to match `AbstractLanguageModel` contract |
| **Reward Interface** | ✅ its-hub core | Wraps gateway reward OR uses `create_llm_judge()` helper |
| **Algorithm Interface** | ✅ its-hub core | Consumes both LM and Reward interfaces to execute ITS strategies |

**its-hub Scope**: This library provides **interface contracts (LM, Reward, Algorithm)** as abstractions in the core library, preparing them for flexible integration with any gateway. Gateway teams implement these interfaces; its-hub provides the algorithm logic that consumes them.

**Design Principles**:
- **Minimal dependencies** (core: `numpy`, `typing-extensions`)
- **Gateway agnostic** (works with any gateway via standard interfaces)
- **Clear contracts** (well-defined interfaces enable independent implementation)
- **Algorithm first** (focus on reasoning, not infrastructure)

---

## Gateway Interface

### What is an AI Gateway?

An **AI Gateway** is the primary interface through which users access LLMs in production. It acts as an intelligent proxy layer between applications and model providers, offering critical enterprise capabilities:

**Core Gateway Capabilities:**

1. **Authentication & Authorization**: API key management, user authentication, access control
2. **Rate Limiting**: Request throttling, quota management per user/tenant
3. **Translation & Parameter Mapping**: Normalize requests across different model providers (OpenAI, Anthropic, AWS Bedrock, etc.)
4. **Load Balancing & Reliability**: Distribute load, automatic failover, retry logic
5. **Post-Processing & Observability**: Cost tracking, distributed tracing, usage analytics, audit logging

**Examples of AI Gateways:**
- **Portkey**: AI gateway with observability, caching, and 200+ LLM provider support
- **LiteLLM**: Open-source proxy supporting 100+ LLM providers with unified API
- **OpenRouter**: Multi-model gateway with routing and load balancing across providers

**Examples of General Gateways (extensible for AI):**
- **Envoy**: High-performance proxy/gateway (can be extended with AI-specific filters)
- **Kong**: API gateway with plugin system (can add AI capabilities)

### Why Integrate its-hub with Gateways?

**The Gateway is the Marketplace**: AI Gateways serve as the primary access point where users consume LLMs. They aggregate models from various vendors, hosters, and providers into a unified interface.

**Avoid Duplication**: Gateways already provide authentication, rate-limiting, load-balancing, tracing, and more. its-hub should NOT reimplement these capabilities. Instead, its-hub should be a **plug-in or feature** that gateways can enable.

**Seamless User Experience**: Users activate inference-time scaling (ITS) algorithms using the same interface they already use for model inference. No new APIs or workflow changes needed.

### Request Protocol Specification

**Design Principle**: ITS requests are standard OpenAI-compatible requests with additional HTTP headers. No changes to request body format.

#### 1. ITS Activation Headers

**Header Specification**:

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `X-ITS-Algorithm` | string | Yes | Algorithm: `"self-consistency"`, `"best-of-n"` |
| `X-ITS-Budget` | integer | Yes | Number of generations (computational budget) |
| `X-ITS-Projection-Regex` | string | No | Answer extraction regex (Self-Consistency) |
| `X-ITS-Tool-Vote` | string | No | Tool voting mode: `"tool_hierarchical"`, etc. |
| `X-ITS-Exclude-Args` | string | No | Comma-separated args to exclude from voting |
| `X-ITS-Judge-Model` | string | No | Judge model name (Best-of-N) |
| `X-ITS-Judge-Endpoint` | string | No | Judge model endpoint URL |
| `X-ITS-Judge-API-Key` | string | No | Judge model API key |

#### 2. Base Request Format

**Requirement**: Gateway must have OpenAI-compatible `/v1/chat/completions` endpoint

**Required capabilities**:
- Support `n` parameter (n > 1) for multiple completions
- Support `temperature` parameter
- Handle concurrent requests

#### 3. Response Format

**Standard OpenAI-compatible response**: Same format with or without ITS headers

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-4o-mini",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "4"
    },
    "finish_reason": "stop"
  }]
}
```

### Request/Response Examples

```bash
# With ITS (Self-Consistency, budget=3)
curl -X POST https://gateway.example.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-ITS-Algorithm: self-consistency" \
  -H "X-ITS-Budget: 3" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'

# Without ITS (normal pass-through)
curl -X POST https://gateway.example.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

---

## LM Interface (Integration Layer)

### Overview

**Purpose**: The LM Interface bridges gateway infrastructure to its-hub algorithms, enabling algorithms to work with any gateway implementation.

**What this layer does**:
- Provides a standard contract for language model access
- Decouples its-hub algorithms from gateway-specific implementations
- Enables "write once, run anywhere" for algorithms

**Key benefit**: Algorithm developers write against one interface; gateway developers implement once to support all algorithms.

**Defined in**: `its_hub/base.py` as `AbstractLanguageModel`

### Interface Contract

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

class AbstractLanguageModel(ABC):
    """
    Abstract interface for language model access.

    Gateway integrators implement this to enable its-hub algorithm support.
    """
```

#### Method 1: `__init__` (Constructor)

**Purpose**: Initialize connection to gateway's model serving infrastructure

**Signature**:
```python
def __init__(
    self,
    endpoint: str,                      # Gateway API endpoint
    api_key: str | None,                # Authentication (if required)
    model_name: str,                    # Model identifier
    temperature: float = 1.0,           # Default sampling temperature
    max_tokens: int | None = None,      # Default token limit
    timeout: int = 60,                  # Request timeout (seconds)
    max_concurrent_requests: int = 10,  # Concurrency limit
    **kwargs                            # Gateway-specific config
) -> None:
    """Initialize language model client."""
```

**Requirements**:
- Store configuration for `agenerate` calls
- Set up client for gateway communication (HTTP, gRPC, SDK, etc.)
- Configure concurrency control (its-hub algorithms make parallel requests)

#### Method 2: `agenerate` (Core Method - REQUIRED)

**Purpose**: Generate model completion(s) asynchronously

**Signature**:
```python
@abstractmethod
async def agenerate(
    self,
    messages: Union[List[ChatMessage], List[List[ChatMessage]]],
    stop: Optional[str] = None,
    **kwargs
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Generate response(s) from the model.

    Args:
        messages: Single conversation or batch
            Single: [{"role": "user", "content": "..."}]
            Batch:  [[...], [...], ...]
        stop: Stop sequence
        **kwargs: temperature, max_tokens, n, tools, tool_choice, etc.

    Returns:
        Single: {"role": "assistant", "content": "...", "tool_calls": [...]}
        Batch:  [response1, response2, ...]
    """
```

**Output Specification**:

```python
{
    "role": "assistant",
    "content": str,                     # Generated text
    "tool_calls": List[dict] | None     # Tool calls if applicable
}
```

**Behavior Requirements**:
- **Batch detection**: Check if `isinstance(messages[0], list)` for batch mode
- **Concurrency control**: Respect `max_concurrent_requests` limit
- **Error handling**: Retry transient errors (rate limits, timeouts), fail fast on permanent errors (4xx)
- **Tool preservation**: If gateway returns tool_calls, include in output
- **Format normalization**: Return consistent format regardless of gateway's native format

### Type Definitions

```python
from typing import TypedDict, List, NotRequired

class ChatMessage(TypedDict):
    role: str                           # "system", "user", "assistant", "tool"
    content: str                        # Message content
    name: NotRequired[str]              # Optional: function name for tool messages
    tool_calls: NotRequired[List[dict]] # Optional: tool calls in assistant message
    tool_call_id: NotRequired[str]      # Optional: ID for tool response messages

ChatMessages = List[ChatMessage]
```

### Integration Approaches

Gateway teams have two architectural approaches for implementing this interface:

#### Approach 1: Direct Implementation (Code Contribution)

**Concept**: its-hub team contributes `AbstractLanguageModel` implementation to gateway's codebase via pull request. Gateway reviews, merges, and maintains the code.

**Characteristics**:
- Requires gateway company approval (e.g., LiteLLM, Portkey teams must review PR)
- Once merged, **gateway team owns maintenance**
- Native integration with gateway infrastructure
- Optimal performance (no extra layer)
- Significant effort to get accepted and merged

**Best for**: Gateways willing to accept and maintain contributions

#### Approach 2: Adapter Package (Separate Maintenance)

**Concept**: its-hub team maintains separate adapter packages (e.g., `its-hub-litellm`, `its-hub-portkey`). Gateway installs package and uses it.

**Characteristics**:
- **its-hub team maintains** adapter code (not gateway)
- No PR process - gateway just installs package
- Quick integration for gateway (minimal effort)
- Keeps core its-hub minimal (no gateway-specific dependencies)
- Less customization by gateway team

**Best for**: Fast integration without requiring gateway code changes or maintenance burden

### Comparison

| Aspect | Direct Implementation | Adapter Package |
|--------|----------------------|-----------------|
| **Implementation Effort** | Gateway team implements | Install pre-built package |
| **Maintenance** | Gateway team | its-hub team |
| **Customization** | Full control | Limited |
| **Performance** | Optimal | Good (extra layer) |
| **Dependencies** | None (its-hub core only) | its-hub core + adapter package |
| **Best For** | Deep integration, full control | Fast integration, minimal effort |

---

## Reward Interface (Integration Layer)

### Overview

**Purpose**: The Reward Interface defines how to score responses for quality/correctness selection. **Symmetric to LM Interface** - both are integration concerns.

**What this layer does**:
- Provides a standard contract for reward model access
- Decouples its-hub algorithms from reward-specific implementations
- Enables "write once, run anywhere" for algorithms

**Key benefit**: Algorithm developers write against one interface; integration provides implementations (gateway native OR its-hub helper).

**Defined in**: `its_hub/base.py` as `AbstractOutcomeRewardModel`

### Interface Contract

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional

class AbstractOutcomeRewardModel(ABC):
    """
    Abstract interface for reward models that score complete responses.

    Integration Layer implements this by wrapping gateway reward services
    OR using its-hub's create_llm_judge() helper.
    """
```

#### Method 1: `__init__` (Constructor)

**Purpose**: Initialize reward model with configuration

**Signature**: Implementation-specific.

#### Method 2: `score` (Core Method - REQUIRED)

**Purpose**: Score response(s) or conversation(s)

**Signature**:
```python
@abstractmethod
def score(
    self,
    messages: Union[List[Dict], List[List[Dict]]],
    **kwargs
) -> Union[float, List[float]]:
    """
    Score conversation(s).

    Args:
        messages: Single conversation or batch
            Single: [{"role": "user", ...}, {"role": "assistant", ...}]
            Batch:  [[conv1], [conv2], ...]
        **kwargs: Model-specific parameters

    Returns:
        Single: float (score)
        Batch:  List[float] (one score per conversation)

    Note: Higher score = better response
    """
```

**Behavior Requirements**:
- **Batch detection**: Check if `isinstance(messages[0], list)`
- **Consistent scoring**: Scores must be comparable across responses
- **Higher is better**: Convention that higher scores indicate better quality

#### Method 3: `ascore` (Async Scoring - OPTIONAL)

**Purpose**: Asynchronous scoring for models that make async calls (e.g., LLM judges)

**Note**: Default implementation raises `NotImplementedError`. Only implement if scoring requires async operations.

### Two Integration Approaches

Integration Layer has two options for implementing reward model support (parallel to LM Interface):

#### Approach 1: Wrap Gateway Native Abstractions

**Concept**: Most gateways already have evaluator/guardrail abstractions as part of their infrastructure. Integration Layer wraps these existing components to match `AbstractOutcomeRewardModel` contract.

**Typical gateway abstractions**:
- **Evaluator abstraction**: Scores model outputs for quality/correctness
- **Guardrail abstraction**: Validates safety, toxicity, policy compliance
- **Content moderation services**: Built-in classifiers and filters
- **Domain-specific validators**: Custom scoring for particular use cases

**Characteristics**:
- Leverages gateway's existing abstractions (no new infrastructure needed)
- Simple wrapper implementation to match interface
- No dependency on LLM calls for scoring
- Optimal performance (native service) and cost (no additional inference)

**Best for**: Gateways with existing evaluator/guardrail abstractions (most production gateways)

---

#### Approach 2: Use its-hub Helper (LLM-Judge)

**Concept**: Integration uses `create_llm_judge()` helper from its-hub Algorithm Layer. This helper wraps any `AbstractLanguageModel` into a basic LLM-judge reward model.

**What its-hub provides**:
- `create_llm_judge(lm, judge_prompt=None, fallback_score=5.0)` - helper function in Algorithm Layer
- Converts any LM implementation into reward model (no additional integration work)
- Default judging prompt (customizable)
- Fallback score handling for parsing failures

**Pros**:
- LLM-judge supported without any extra implementation in Integration Layer
- Reuses gateway's LM infrastructure (streaming, async, cost-tracking capabilities)
- Works immediately if LM Interface is implemented
- Good default for getting started

**Cons**:
- Only LLM-judge type of reward is supported (cannot leverage toxicity classifiers, safety filters, or other reward types)
- Simple implementation in its-hub (basic judging logic)
- Gateway's evaluator/guardrail may have much more features (batching, caching, custom prompts, advanced scoring)

**Best for**: Quick integration, gateways without native evaluator/guardrail abstractions

---

### Comparison

| Aspect | Gateway Native Abstractions | its-hub Helper (LLM-Judge) |
|--------|----------------------------|---------------------------|
| **Implementation Effort** | Wrap existing evaluator/guardrail | Call helper function (zero effort) |
| **Requires** | Gateway evaluator/guardrail abstraction | LM Interface only |
| **Reward Types Supported** | Multiple (toxicity, safety, quality, domain-specific) | LLM-judge only |
| **Infrastructure Reuse** | Native evaluator infrastructure | Reuses gateway LM (streaming, async, cost-tracking) |
| **Feature Richness** | Full-featured (batching, caching, custom prompts) | Simple implementation in its-hub |
| **Performance** | Fast (native service) | Slower (LLM calls) |
| **Cost** | Lower (no LLM calls) | Higher (LLM inference) |
| **Best For** | Gateways with evaluator/guardrail | Quick start, no native abstractions |

---

## Algorithm Interface (its-hub Core)

### Overview

**Purpose**: The Algorithm Interface defines how inference-time scaling algorithms are implemented and consumed.

**What this layer does**:
- Defines the contract for all ITS algorithms (Self-Consistency, Best-of-N, etc.)
- Specifies algorithm inputs (LM, Reward, prompt, budget) and outputs (selected response)
- Provides `create_llm_judge()` helper for immediate reward model usability
- Enables algorithm composability and testing

**Key benefit**: All algorithms follow the same interface → applications can swap algorithms without code changes. Algorithms consume both LM and Reward interfaces.

**Defined in**: `its_hub/base.py` as `AbstractScalingAlgorithm` and `AbstractScalingResult`

### Interface Contract

```python
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional

class AbstractScalingAlgorithm(ABC):
    """
    Abstract interface for inference-time scaling algorithms.

    All algorithms (Self-Consistency, Best-of-N, etc.) implement this.
    Algorithms consume both LM Interface and Reward Interface.
    """
```

#### Method 1: `__init__` (Constructor)

**Purpose**: Initialize algorithm with configuration

**Signature**: Algorithm-specific. Each algorithm defines its own configuration parameters.

**Examples**:
```python
# Self-Consistency (consumes LM only)
SelfConsistency(projection_fn=None, tool_vote=None, exclude_args=None)

# Best-of-N (consumes LM + Reward)
BestOfN(reward_model: AbstractOutcomeRewardModel)
```

#### Method 2: `ainfer` (Core Method - REQUIRED)

**Purpose**: Execute the algorithm with given LM and computational budget

**Signature**:
```python
@abstractmethod
async def ainfer(
    self,
    lm: AbstractLanguageModel,
    prompt_or_messages: Union[str, List[ChatMessage]],
    budget: int,
    return_response_only: bool = True,
    tools: Optional[List[Dict]] = None,
    tool_choice: Optional[Union[str, Dict]] = None,
) -> Union[Dict[str, Any], AbstractScalingResult]:
    """
    Run inference-time scaling algorithm.

    Args:
        lm: Language model (implements AbstractLanguageModel from LM Interface)
        prompt_or_messages: User input
            - str: "What is 2+2?"
            - List[ChatMessage]: [{"role": "user", "content": "..."}]
        budget: Computational budget (algorithm-specific interpretation)
        return_response_only:
            - True: return selected response dict
            - False: return full result with metadata
        tools: OpenAI-style tool definitions (optional)
        tool_choice: Tool selection strategy (optional)

    Returns:
        If return_response_only=True:
            {"role": "assistant", "content": "...", "tool_calls": [...]}
        If return_response_only=False:
            AbstractScalingResult instance
    """
```

**Budget Interpretation** (algorithm-specific):

| Algorithm | Budget Meaning |
|-----------|----------------|
| Self-Consistency | Number of parallel generations |
| Best-of-N | Number of parallel generations |

**Output Specification**:
```python
# Simple output (return_response_only=True)
{
    "role": "assistant",
    "content": str,
    "tool_calls": List[dict] | None
}
```

#### Method 3: `infer` (Synchronous Wrapper)

**Note**: Default implementation provided via `asyncio.run(self.ainfer(...))`. Algorithm implementations typically do not override this.

### Algorithm Result Interface

```python
class AbstractScalingResult(ABC):
    """
    Result object returned when return_response_only=False.
    """

    @property
    @abstractmethod
    def the_one(self) -> Dict[str, Any]:
        """
        The selected best response.

        Returns:
            {"role": "assistant", "content": "...", "tool_calls": [...]}
        """
        pass

    @property
    def candidates(self) -> List[Dict[str, Any]]:
        """All generated candidate responses."""
        return []

    @property
    def scores(self) -> Optional[List[float]]:
        """Scores for each candidate (if applicable)."""
        return None

    @property
    def metadata(self) -> Dict[str, Any]:
        """Algorithm-specific metadata."""
        return {}
```

### Critical Clarification: Orchestration vs Algorithm Logic

**IMPORTANT**: The Algorithm Interface defines WHAT the algorithm does (vote, score, select). The Integration Layer (LM & Reward Interfaces) handles HOW to execute it (parallel calls, concurrency, fan-out).

**Algorithm Interface Responsibility**:
- Define selection logic (e.g., vote on most common, select highest score)
- Specify budget interpretation
- Implement result construction
- Provide `create_llm_judge()` helper for immediate reward model access

**Integration Layer Responsibility** (LM & Reward Interfaces):
- Execute parallel LM calls (fan-out, concurrency management)
- Execute reward scoring (batching, caching)
- Manage concurrency limits (rate limiting, semaphores)
- Handle retries and error recovery
- Batch request optimization

**Separation of Concerns**:
- **Algorithm** says: "I need N responses" → calls `lm.agenerate()` N times
- **LM Interface** handles: Parallel execution, retries, concurrency limits
- **Algorithm** says: "Vote on most common" → implements voting logic
- **Reward Interface** handles: Batching scores, caching, async execution

### Helper Function: `create_llm_judge()`

**Purpose**: Convenience helper that wraps any `AbstractLanguageModel` into `AbstractOutcomeRewardModel`

**Location**: Algorithm Layer (its-hub core)

**Signature**:
```python
def create_llm_judge(
    lm: AbstractLanguageModel,
    judge_prompt: Optional[str] = None,
    fallback_score: float = 5.0
) -> AbstractOutcomeRewardModel:
    """
    Wrap any LM into LLM-judge reward model.

    Enables immediate Best-of-N usage when Integration Layer implements LM Interface.
    This is Approach 2 from Reward Interface - reuses gateway's LM infrastructure.

    Args:
        lm: Any AbstractLanguageModel implementation
        judge_prompt: Custom judging prompt (uses default if None)
        fallback_score: Score to use if parsing fails

    Returns:
        AbstractOutcomeRewardModel that uses LM for scoring
    """
```

**Usage Pattern**:
```python
from its_hub import create_llm_judge, BestOfN

# Integration provides LM
lm = MyGatewayLM(...)

# Algorithm Layer helper creates reward model
judge = create_llm_judge(lm)

# Now Best-of-N works immediately
bon = BestOfN(reward_model=judge)
result = await bon.ainfer(lm, prompt, budget=5)
```

### Officially Supported Algorithms

#### 1. Self-Consistency

**Purpose**: Generate N responses, vote on most common answer

**Constructor**:
```python
class SelfConsistency(AbstractScalingAlgorithm):
    def __init__(
        self,
        projection_fn: Optional[Callable] = None,
        tool_vote: Optional[str] = None,
        exclude_args: Optional[List[str]] = None
    ):
        """
        Args:
            projection_fn: Extract answer from response (e.g., regex)
            tool_vote: Voting mode for tool calls
                - "tool_name": vote on tool names
                - "tool_args": vote on arguments
                - "tool_hierarchical": vote on full structure
            exclude_args: Args to ignore in voting (e.g., ["timestamp", "id"])
        """
```

**Consumes**: LM Interface only

**Budget**: Number of parallel generations

**Selection Logic**: Vote on most common answer (majority wins)

**Usage**:
```python
from its_hub import SelfConsistency

sc = SelfConsistency()
result = await sc.ainfer(lm, "What is 2+2?", budget=5)
# Generates 5 responses, returns most common answer
```

#### 2. Best-of-N

**Purpose**: Generate N responses, select highest-scoring one

**Constructor**:
```python
class BestOfN(AbstractScalingAlgorithm):
    def __init__(self, reward_model: AbstractOutcomeRewardModel):
        """
        Args:
            reward_model: Scores responses (from Reward Interface)
                - Approach 1: Gateway native evaluator/guardrail (wrapped)
                - Approach 2: create_llm_judge(lm) helper
        """
```

**Consumes**: LM Interface + Reward Interface

**Budget**: Number of parallel generations

**Selection Logic**: Score all responses, return highest-scoring

**Usage**:
```python
from its_hub import BestOfN, create_llm_judge

# Approach 1: Wrap gateway native evaluator/guardrail
reward = MyGatewayReward(gateway.evaluator)
bon = BestOfN(reward_model=reward)

# Approach 2: Use helper (reuses LM infrastructure)
judge = create_llm_judge(lm)
bon = BestOfN(reward_model=judge)

result = await bon.ainfer(lm, "Write a sorting function", budget=10)
# Generates 10 responses, scores each, returns best
```

---

## Implementation Checklist

### For Gateway Developers (Gateway Interface)

Integrating its-hub with your gateway:

- [ ] Expose inference endpoint with `n` parameter support (n > 1)
- [ ] Support concurrent requests (recommend: 10+ parallel)
- [ ] Return structured errors with retryable/permanent distinction
- [ ] Support standard parameters: temperature, max_tokens, stop, tools
- [ ] (Optional) Support batch inference for efficiency
- [ ] (Optional) Provide tracing/logging hooks

### For LM Interface Implementers

Implementing `AbstractLanguageModel`:

- [ ] Implement `__init__` with standard parameters (endpoint, api_key, model_name)
- [ ] Implement `agenerate` method (REQUIRED)
  - [ ] Handle single conversation input
  - [ ] Handle batch conversation input
  - [ ] Preserve tool_calls in output
  - [ ] Implement retry logic for transient errors
  - [ ] Respect concurrency limits (orchestration responsibility)
- [ ] Use default `generate` method (sync wrapper)
- [ ] (Optional) Implement `mock_streaming` for testing
- [ ] (Optional) Implement tracing methods

### For Reward Interface Implementers

Implementing `AbstractOutcomeRewardModel`:

- [ ] Implement `__init__` with reward model config
- [ ] Implement `score` method (REQUIRED)
  - [ ] Handle single conversation input
  - [ ] Handle batch conversation input
  - [ ] Return consistent scores (higher = better)
  - [ ] Implement batching/caching if optimizing (orchestration responsibility)
- [ ] Implement `ascore` if model makes async calls (e.g., LLM judge)
- [ ] Document score range and interpretation
- [ ] **Quickstart option**: Use `create_llm_judge(lm)` helper instead of implementing

### For Algorithm Implementers

Implementing `AbstractScalingAlgorithm`:

- [ ] Implement `__init__` with algorithm-specific config
- [ ] Implement `ainfer` method (REQUIRED)
  - [ ] Normalize input (handle str and List[ChatMessage])
  - [ ] Validate budget > 0
  - [ ] Generate responses using LM interface (let Integration handle fan-out)
  - [ ] Implement selection logic (vote, score, etc.)
  - [ ] Support return_response_only flag
  - [ ] Handle tools/tool_choice parameters
- [ ] Use default `infer` method (sync wrapper)
- [ ] Document budget interpretation in docstring
- [ ] Implement `AbstractScalingResult` subclass if return_response_only=False

---

## Appendix: Complete Interface Summary

| Interface | Layer | Implementer | Consumer | Status |
|-----------|-------|-------------|----------|--------|
| **Gateway Interface** | Gateway Layer | Gateway developers | Integration Layer | Requirements only |
| **LM Interface** (`AbstractLanguageModel`) | Integration Layer | Wraps gateway LM | Algorithms | ✅ Defined |
| **Reward Interface** (`AbstractOutcomeRewardModel`) | Integration Layer | Wraps gateway reward OR uses helper | Algorithms (Best-of-N) | ✅ Defined |
| **Algorithm Interface** (`AbstractScalingAlgorithm`) | Algorithm Layer | its-hub core | End users | ✅ Defined |
| **Algorithm Result** (`AbstractScalingResult`) | Algorithm Layer | its-hub core | End users | ✅ Defined |
| **Helper** (`create_llm_judge()`) | Algorithm Layer | its-hub core | Integration Layer (quickstart) | ✅ Defined |

---

**Document Version**: 1.0
**Last Updated**: 2025-11-21
**Authors**: Red Hat AI Innovation Team
