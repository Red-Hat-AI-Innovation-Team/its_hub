# ITS-Hub Interface Design Specification

## Document Purpose

This document defines the **interface contracts** for the its-hub library. It specifies the abstractions, method signatures, inputs/outputs, and functional requirements needed for implementation.

**Target Audience**: Software engineers implementing or integrating with its-hub
**Usage**: Implementation guide and review checklist for ensuring compliance with the architecture

---

## Architecture Overview

**its-hub** is a lightweight library providing inference-time scaling algorithms for LLMs. The architecture consists of three layers with two key integration interfaces:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Gateway Layer (External Systems)              │
│  - Red Hat AI Gateway, vLLM, LangChain, OpenAI API     │
│  - Provides: Model serving, endpoints, infrastructure   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  │ Gateway must expose certain capabilities
                  ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 2: Integration Layer (Bridge Interface)          │
│  - LM Interface: AbstractLanguageModel                  │
│  - Implemented by: Gateway integrators                  │
│  - Enables: its-hub algorithms to use any LM provider   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  │ Algorithms consume LM Interface
                  ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Algorithm Layer (its-hub Library)             │
│  - Algorithm Interface: AbstractScalingAlgorithm        │
│  - Reward Interface: AbstractOutcomeRewardModel         │
│  - Implementations: SelfConsistency, BestOfN, etc.      │
└─────────────────────────────────────────────────────────┘
```

**Key Design Principles:**
- **Minimal Dependencies**: Core library only requires `numpy`, `typing-extensions`
- **Gateway Agnostic**: Works with any LM provider via the LM Interface
- **Clear Contracts**: Well-defined interfaces enable independent implementation and testing
- **Algorithm First**: Focus on reasoning algorithms, not infrastructure

---

## Interface 1: Gateway Layer Integration

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

### Interface Contract

**Design Principle**: Minimal gateway changes. Activate ITS via HTTP headers checked by a before-request hook.

#### 1. Before-Request Hook (Required)

**Purpose**: Check for ITS activation headers and route accordingly

**Signature**:
```python
def before_request_hook(request: HTTPRequest) -> HTTPResponse:
    """
    Intercept requests and check for X-ITS-* headers.

    Returns:
        ITS response if headers present, else normal pass-through
    """
    if has_its_headers(request):
        return handle_its_request(request)
    else:
        return handle_normal_request(request)
```

#### 2. ITS Activation Headers

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

#### 3. Gateway Inference Endpoint (Already Exists)

**Requirement**: Gateway must have OpenAI-compatible `/v1/chat/completions` endpoint

**Required capabilities**:
- Support `n` parameter (n > 1) for multiple completions
- Support `temperature` parameter
- Handle concurrent requests

### Usage Example

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

**Response format**: Same as standard OpenAI response (with or without ITS)

---

## Interface 2: LM Interface (Integration Layer)

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

**Input Specification**:

```python
ChatMessage = TypedDict('ChatMessage', {
    'role': str,                        # "system" | "user" | "assistant" | "tool"
    'content': str,                     # Message content
    'name': NotRequired[str],           # Optional: function name
    'tool_calls': NotRequired[List[dict]],  # Optional: tool calls
    'tool_call_id': NotRequired[str]    # Optional: tool response ID
})
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

### Two Integration Approaches

Gateway teams have two options for implementing its-hub support:

#### Approach 1: Direct Implementation (Pull Request to Gateway)

**How**: Gateway team implements `AbstractLanguageModel` directly in their codebase

**Implementation**:
```python
# In gateway codebase: gateway/its_integration.py
from its_hub import AbstractLanguageModel

class GatewayLM(AbstractLanguageModel):
    def __init__(self, model_name, **kwargs):
        # Use gateway's existing LM client
        self.client = gateway.get_model_client(model_name)

    async def agenerate(self, messages, stop=None, **kwargs):
        # Call gateway's native inference method
        response = await self.client.inference(messages, **kwargs)
        return self._normalize_response(response)
```

**Pros**:
- Native integration with gateway infrastructure
- Full control over implementation
- Optimal performance (no extra layer)
- Can leverage gateway-specific optimizations

**Cons**:
- Gateway team owns maintenance
- Requires gateway code changes

**Best for**: Gateways that want deep integration and full control

---

#### Approach 2: Plug-in Library (Adapter Pattern)

**How**: its-hub provides pre-built adapters; gateway imports and uses them

**Implementation**:
```python
# In gateway codebase: gateway/plugins/its_plugin.py
from its_hub.adapters import PortkeyAdapter  # Pre-built by its-hub team

class ITSPlugin:
    def __init__(self, gateway_config):
        # Adapter wraps gateway's existing client
        self.lm = PortkeyAdapter(
            client=gateway.client,
            model_name=gateway_config.default_model
        )

    async def handle_its_request(self, request):
        from its_hub import SelfConsistency, BestOfN
        # Use pre-built adapter with algorithms
        algorithm = self._get_algorithm(request.headers)
        return await algorithm.ainfer(self.lm, request.messages, request.budget)
```

**its-hub provides**:
```python
# its_hub/adapters/portkey.py
class PortkeyAdapter(AbstractLanguageModel):
    """Pre-built adapter for Portkey gateway"""
    # Implementation maintained by its-hub team

# its_hub/adapters/litellm.py
class LiteLLMAdapter(AbstractLanguageModel):
    """Pre-built adapter for LiteLLM gateway"""
    # Implementation maintained by its-hub team
```

**Pros**:
- Zero implementation effort for gateway team
- Maintained by its-hub team
- Quick integration (import and use)
- Easy to upgrade (update its-hub version)

**Cons**:
- Extra dependency on its-hub library
- Less customization
- May not leverage gateway-specific features

**Best for**: Gateways that want fast integration with minimal effort

---

### Comparison

| Aspect | Direct Implementation | Plug-in Library |
|--------|----------------------|-----------------|
| **Implementation Effort** | Gateway team implements | Import pre-built adapter |
| **Maintenance** | Gateway team | its-hub team |
| **Customization** | Full control | Limited |
| **Performance** | Optimal | Good (extra layer) |
| **Dependencies** | None (its-hub algorithms only) | its-hub library |
| **Integration Time** | Weeks | Days |
| **Best For** | Deep integration, full control | Fast integration, minimal effort |

### Usage Example

```python
from its_hub import AbstractLanguageModel, SelfConsistency

# Gateway implements the interface (either approach)
class MyGatewayLM(AbstractLanguageModel):
    def __init__(self, endpoint, api_key, model_name):
        self.client = gateway_sdk.Client(endpoint, api_key)
        self.model = model_name

    async def agenerate(self, messages, stop=None, **kwargs):
        response = await self.client.chat(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return {"role": "assistant", "content": response.text}

# Use with its-hub algorithms
lm = MyGatewayLM("https://gateway.com", "key", "gpt-4o-mini")
algorithm = SelfConsistency()
result = await algorithm.ainfer(lm, "What is 2+2?", budget=5)
```

---

## Interface 3: Algorithm Interface (its-hub Core)

### Overview
**Purpose**: Define how inference-time scaling algorithms are implemented and consumed

**Defined in**: `its_hub/base.py` as `AbstractScalingAlgorithm`

**Implementers**: its-hub library (SelfConsistency, BestOfN, etc.)

**Consumers**: End users, gateway integrators, application developers

### Interface Contract

```python
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional

class AbstractScalingAlgorithm(ABC):
    """
    Abstract base class for inference-time scaling algorithms.

    All algorithms implement this unified interface.
    """
```

### Method 1: `__init__` (Constructor)

**Purpose**: Initialize algorithm with configuration

**Signature**: Algorithm-specific, but common patterns:

```python
# Self-Consistency
def __init__(
    self,
    projection_fn: Optional[Callable] = None,      # Answer extraction function
    tool_vote: Optional[str] = None,               # Tool voting mode
    exclude_args: Optional[List[str]] = None       # Args to ignore in tool voting
) -> None:
    """Initialize Self-Consistency algorithm."""

# Best-of-N
def __init__(
    self,
    reward_model: AbstractOutcomeRewardModel       # Reward model for scoring
) -> None:
    """Initialize Best-of-N algorithm."""

# Beam Search (experimental)
def __init__(
    self,
    step_generation: StepGeneration,               # Step-wise generation config
    process_reward_model: AbstractProcessRewardModel,  # Step scoring model
    beam_width: int                                # Number of beams to maintain
) -> None:
    """Initialize Beam Search algorithm."""
```

**Requirements**:
- Store configuration for use in `ainfer`
- Validate parameters (e.g., beam_width > 0)
- Initialize dependencies (reward models, etc.)

### Method 2: `ainfer` (Primary Method - REQUIRED)

**Purpose**: Run inference with the given LM and computational budget

**Signature**:
```python
@abstractmethod
async def ainfer(
    self,
    lm: AbstractLanguageModel,
    prompt_or_messages: Union[str, List[ChatMessage], ChatMessages],
    budget: int,
    return_response_only: bool = True,
    tools: Optional[List[Dict]] = None,
    tool_choice: Optional[Union[str, Dict]] = None,
) -> Union[Dict[str, Any], AbstractScalingResult]:
    """
    Run inference asynchronously with computational budget.

    Args:
        lm: Language model instance (implements AbstractLanguageModel)
        prompt_or_messages: User input
            - str: Simple prompt, converted to [{"role": "user", "content": prompt}]
            - List[ChatMessage]: Full conversation history
        budget: Computational budget (interpretation varies by algorithm)
            - Self-Consistency/Best-of-N: number of parallel generations
            - Beam Search: total_generations = budget / beam_width
            - Particle Filtering: number of particles
        return_response_only: Output mode
            - True: return just the selected response dict
            - False: return full AbstractScalingResult with metadata
        tools: Optional OpenAI-style tool definitions
            [{"type": "function", "function": {"name": "...", "parameters": {...}}}]
        tool_choice: Optional tool choice strategy
            - "auto": model decides
            - "none": no tool calling
            - {"type": "function", "function": {"name": "..."}}: specific tool

    Returns:
        If return_response_only=True:
            Dict: {"role": "assistant", "content": "...", "tool_calls": [...]}

        If return_response_only=False:
            AbstractScalingResult with:
                - .the_one: selected best response
                - .candidates: all generated responses
                - .scores: scores for each candidate (if applicable)
                - .metadata: algorithm-specific data

    Raises:
        ValueError: If budget <= 0 or invalid for algorithm
        RuntimeError: If algorithm fails to produce result
    """
```

**Input Normalization**:
```python
# Handle different input types
if isinstance(prompt_or_messages, str):
    messages = [{"role": "user", "content": prompt_or_messages}]
elif isinstance(prompt_or_messages, list):
    messages = prompt_or_messages
else:
    raise ValueError("prompt_or_messages must be str or List[ChatMessage]")
```

**Budget Interpretation by Algorithm**:

| Algorithm | Budget Meaning | Example |
|-----------|----------------|---------|
| Self-Consistency | Number of parallel completions | budget=5 → generate 5 responses, vote |
| Best-of-N | Number of parallel completions | budget=10 → generate 10, score all, pick best |
| Beam Search | Total steps = budget / beam_width | budget=12, beam=3 → 4 steps × 3 beams |
| Particle Filtering | Number of particles | budget=8 → maintain 8 particles during sampling |

**Output Specifications**:

Simple output (return_response_only=True):
```python
{
    "role": "assistant",
    "content": "The answer is 4.",
    "tool_calls": []  # Empty if no tools used
}
```

Full output (return_response_only=False):
```python
class SelfConsistencyResult(AbstractScalingResult):
    def __init__(self, selected, candidates, vote_counts):
        self._selected = selected
        self._candidates = candidates
        self._vote_counts = vote_counts

    @property
    def the_one(self) -> Dict[str, Any]:
        """Return selected best response"""
        return self._selected

    @property
    def candidates(self) -> List[Dict[str, Any]]:
        """Return all generated responses"""
        return self._candidates

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return vote counts and other metadata"""
        return {"vote_counts": self._vote_counts}
```

**Implementation Pattern**:
```python
async def ainfer(self, lm, prompt_or_messages, budget, return_response_only=True, tools=None, tool_choice=None):
    # 1. Normalize input
    messages = self._normalize_input(prompt_or_messages)

    # 2. Validate budget
    if budget <= 0:
        raise ValueError(f"budget must be > 0, got {budget}")

    # 3. Generate responses (algorithm-specific)
    responses = await self._generate_responses(lm, messages, budget, tools, tool_choice)

    # 4. Select best response (algorithm-specific)
    selected = self._select_best(responses)

    # 5. Return based on mode
    if return_response_only:
        return selected
    else:
        return self._build_result(selected, responses)
```

### Method 3: `infer` (Synchronous Wrapper)

**Purpose**: Provide synchronous API

**Signature**:
```python
def infer(
    self,
    lm: AbstractLanguageModel,
    prompt_or_messages: Union[str, List[ChatMessage], ChatMessages],
    budget: int,
    return_response_only: bool = True,
    tools: Optional[List[Dict]] = None,
    tool_choice: Optional[Union[str, Dict]] = None,
) -> Union[Dict[str, Any], AbstractScalingResult]:
    """
    Synchronous wrapper around ainfer().

    Default implementation provided - typically not overridden.

    Args:
        Same as ainfer()

    Returns:
        Same as ainfer()
    """
    import asyncio
    return asyncio.run(
        self.ainfer(lm, prompt_or_messages, budget, return_response_only, tools, tool_choice)
    )
```

**Implementation**: Default implementation provided in base class.

### Algorithm Result Interface

```python
class AbstractScalingResult(ABC):
    """
    Result object returned by algorithms when return_response_only=False.
    """

    @property
    @abstractmethod
    def the_one(self) -> Dict[str, Any]:
        """
        Return the selected best response.

        Returns:
            Dict: {"role": "assistant", "content": "...", "tool_calls": [...]}
        """
        pass

    @property
    def candidates(self) -> List[Dict[str, Any]]:
        """
        All candidate responses generated.

        Returns:
            List of response dicts
        """
        return []

    @property
    def scores(self) -> Optional[List[float]]:
        """
        Scores assigned to each candidate (if applicable).

        Returns:
            List of scores (higher = better) or None if not scored
        """
        return None

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Algorithm-specific metadata.

        Returns:
            Dict with algorithm-specific data:
                - Self-Consistency: {"vote_counts": {...}}
                - Best-of-N: {"reward_model": "...", "scores": [...]}
                - Beam Search: {"beam_paths": [...], "step_scores": [...]}
        """
        return {}
```

### Algorithm Specifications

#### Self-Consistency

**Purpose**: Generate N responses and vote on most common answer

**Constructor**:
```python
def __init__(
    self,
    projection_fn: Optional[Callable[[str], str]] = None,
    tool_vote: Optional[str] = None,
    exclude_args: Optional[List[str]] = None
):
    """
    Args:
        projection_fn: Function to extract answer from response
            Example: lambda text: re.search(r'\\boxed\{([^}]+)\}', text).group(1)
        tool_vote: Voting mode for tool calls
            - None: vote on raw content
            - "tool_name": vote on tool names only
            - "tool_args": vote on tool arguments
            - "tool_hierarchical": vote on full tool call structure
        exclude_args: Arguments to exclude from tool voting
            Example: ["timestamp", "id"] to ignore non-semantic args
    """
```

**Budget Interpretation**: Number of parallel generations

**Selection Logic**:
- Extract answers using `projection_fn` (if provided)
- Count occurrences of each unique answer
- Return most common answer (majority vote)
- Tie-breaking: return first response among tied answers

**Example**:
```python
# Math problem voting
from its_hub.algorithms import SelfConsistency, create_regex_projection_function

# Extract answer from \boxed{...}
proj_fn = create_regex_projection_function(r'\\boxed\{([^}]+)\}')
sc = SelfConsistency(projection_fn=proj_fn)

result = await sc.ainfer(lm, "What is 2+2?", budget=5)
# Generates 5 responses, extracts answers, votes
```

#### Best-of-N

**Purpose**: Generate N responses and select highest-scoring one

**Constructor**:
```python
def __init__(self, reward_model: AbstractOutcomeRewardModel):
    """
    Args:
        reward_model: Model to score responses
            Examples: LLMJudge, MathVerifier, CustomClassifier
    """
```

**Budget Interpretation**: Number of parallel generations

**Selection Logic**:
- Generate N responses
- Score all responses using reward model
- Return response with highest score

**Example**:
```python
from its_hub import BestOfN
from its_hub.reward_models import LLMJudge

judge = LLMJudge(lm=judge_lm, fallback_score=5.0)
bon = BestOfN(reward_model=judge)

result = await bon.ainfer(lm, "Write a sorting function", budget=10)
# Generates 10 responses, judge scores each, returns best
```

---

## Interface 4: Reward Interface

### Overview
**Purpose**: Define how reward models score responses for Best-of-N and other algorithms

**Defined in**: `its_hub/base.py` as `AbstractOutcomeRewardModel`

**Implementers**: its-hub library and users creating custom reward models

**Consumers**: Best-of-N algorithm (and future meta-algorithms)

### Interface Contract

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union

class AbstractOutcomeRewardModel(ABC):
    """
    Abstract base class for outcome reward models.

    Scores complete responses/conversations for quality, correctness, etc.
    """
```

### Method 1: `__init__` (Constructor)

**Purpose**: Initialize reward model with configuration

**Signature**: Implementation-specific, common patterns:

```python
# LLM Judge
def __init__(
    self,
    lm: AbstractLanguageModel,              # LM to use as judge
    judge_prompt: Optional[str] = None,     # Custom prompt template
    fallback_score: float = 5.0,            # Default score on error
    score_range: tuple = (0, 10)            # Expected (min, max) score
) -> None:
    """Initialize LLM judge."""

# Verifier
def __init__(
    self,
    verification_fn: Callable,              # Function to verify correctness
    correct_score: float = 1.0,             # Score for correct answer
    incorrect_score: float = 0.0            # Score for incorrect answer
) -> None:
    """Initialize verifier."""
```

### Method 2: `score` (Synchronous Scoring - REQUIRED)

**Purpose**: Score responses synchronously

**Signature**:
```python
@abstractmethod
def score(
    self,
    messages: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    **kwargs
) -> Union[float, List[float]]:
    """
    Score response(s) or conversation(s).

    Args:
        messages: Single conversation or batch of conversations
            Single: [
                {"role": "user", "content": "Question?"},
                {"role": "assistant", "content": "Answer."}
            ]
            Batch: [
                [{"role": "user", ...}, {"role": "assistant", ...}],
                [{"role": "user", ...}, {"role": "assistant", ...}],
                ...
            ]
        **kwargs: Reward-model-specific parameters
            Common:
                - max_input_tokens: int (truncate long conversations)
                - return_reasoning: bool (return explanation with score)

    Returns:
        Single conversation: float (single score)
        Batch: List[float] (one score per conversation)

    Notes:
        - Higher score = better response
        - Scores should be comparable across different responses
        - No fixed range required, but 0-10 common for LLM judges
    """
```

**Input Type Detection**:
```python
def score(self, messages, **kwargs):
    # Check if batch or single
    is_batch = isinstance(messages[0], list)

    if is_batch:
        return [self._score_single(conv, **kwargs) for conv in messages]
    else:
        return self._score_single(messages, **kwargs)
```

**Example Implementation (LLM Judge)**:
```python
def score(self, messages, return_reasoning=False, **kwargs):
    # Format conversation
    conversation = self._format_conversation(messages)

    # Build judge prompt
    judge_input = self.judge_prompt.format(conversation=conversation)

    # Get judge's score
    response = self.lm.generate([{"role": "user", "content": judge_input}])

    # Parse score
    try:
        data = json.loads(response["content"])
        score = float(data["score"])

        if return_reasoning:
            return score, data.get("reasoning", "")
        return score
    except (json.JSONDecodeError, KeyError, ValueError):
        return self.fallback_score
```

### Method 3: `ascore` (Async Scoring - OPTIONAL)

**Purpose**: Score responses asynchronously (for LLM judges)

**Signature**:
```python
async def ascore(
    self,
    messages: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    **kwargs
) -> Union[float, List[float]]:
    """
    Async version of score().

    Args:
        Same as score()

    Returns:
        Same as score()

    Raises:
        NotImplementedError: If async scoring not supported

    Notes:
        - LLM judges should implement this for efficiency
        - Enables parallel scoring of multiple candidates
        - Default implementation raises NotImplementedError
    """
    raise NotImplementedError(
        f"{self.__class__.__name__} does not support async scoring."
    )
```

**When to implement**:
- MUST implement if reward model makes async calls (e.g., LLM judge)
- NOT needed for fast local computation (e.g., regex verifier)

### LLM Judge Specification

**Constructor**:
```python
class LLMJudge(AbstractOutcomeRewardModel):
    def __init__(
        self,
        lm: AbstractLanguageModel,
        judge_prompt: Optional[str] = None,
        fallback_score: float = 5.0,
        score_range: tuple = (0, 10)
    ):
        """
        Args:
            lm: Language model to use as judge
            judge_prompt: Custom prompt with {conversation} placeholder
                Must instruct LLM to return JSON: {"score": <number>}
            fallback_score: Score if parsing fails
            score_range: Expected (min, max) for validation
        """
```

**Default Judge Prompt**:
```python
DEFAULT_JUDGE_PROMPT = """Evaluate the following conversation for response quality.
Consider: accuracy, helpfulness, coherence, and relevance.

Conversation:
{conversation}

Return a JSON object with a score from 0-10.
Format: {{"score": <number>, "reasoning": "<explanation>"}}"""
```

**Custom Judge Prompts**:
```python
# Math-specific
MATH_JUDGE_PROMPT = """Evaluate the mathematical solution.
Consider: correctness, clarity, completeness.

Conversation:
{conversation}

Return JSON: {{"score": <0-10>, "reasoning": "<why>"}}"""

# Code quality
CODE_JUDGE_PROMPT = """Evaluate the code quality.
Consider: correctness, efficiency, readability, best practices.

Conversation:
{conversation}

Return JSON: {{"score": <0-10>, "reasoning": "<why>"}}"""
```

**Usage**:
```python
# Default judge
judge = LLMJudge(lm=judge_lm)

# Custom judge
math_judge = LLMJudge(lm=judge_lm, judge_prompt=MATH_JUDGE_PROMPT)

# Score single response
score = await judge.ascore([
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."}
])

# Score multiple responses (batch)
scores = await judge.ascore([
    [{"role": "user", ...}, {"role": "assistant", ...}],
    [{"role": "user", ...}, {"role": "assistant", ...}]
])
```

---

## Implementation Checklist

### For Gateway Developers (Layer 1)

Integrating its-hub with your gateway:

- [ ] Expose inference endpoint with `n` parameter support (n > 1)
- [ ] Support concurrent requests (recommend: 10+ parallel)
- [ ] Return structured errors with retryable/permanent distinction
- [ ] Support standard parameters: temperature, max_tokens, stop, tools
- [ ] (Optional) Support batch inference for efficiency
- [ ] (Optional) Provide tracing/logging hooks

### For LM Interface Implementers (Layer 2)

Implementing `AbstractLanguageModel`:

- [ ] Implement `__init__` with standard parameters (endpoint, api_key, model_name)
- [ ] Implement `agenerate` method (REQUIRED)
  - [ ] Handle single conversation input
  - [ ] Handle batch conversation input
  - [ ] Preserve tool_calls in output
  - [ ] Implement retry logic for transient errors
  - [ ] Respect concurrency limits
- [ ] Use default `generate` method (sync wrapper)
- [ ] (Optional) Implement `mock_streaming` for testing
- [ ] (Optional) Implement tracing methods

### For Algorithm Implementers (Layer 3)

Implementing `AbstractScalingAlgorithm`:

- [ ] Implement `__init__` with algorithm-specific config
- [ ] Implement `ainfer` method (REQUIRED)
  - [ ] Normalize input (handle str and List[ChatMessage])
  - [ ] Validate budget > 0
  - [ ] Generate responses using LM interface
  - [ ] Implement selection logic
  - [ ] Support return_response_only flag
  - [ ] Handle tools/tool_choice parameters
- [ ] Use default `infer` method (sync wrapper)
- [ ] Document budget interpretation in docstring
- [ ] Implement `AbstractScalingResult` subclass if return_response_only=False

### For Reward Model Implementers (Layer 4)

Implementing `AbstractOutcomeRewardModel`:

- [ ] Implement `__init__` with reward model config
- [ ] Implement `score` method (REQUIRED)
  - [ ] Handle single conversation input
  - [ ] Handle batch conversation input
  - [ ] Return consistent scores (higher = better)
- [ ] Implement `ascore` if model makes async calls (LLM judge)
- [ ] Document score range and interpretation

---

## Testing Requirements

### LM Interface Testing

```python
# Test single conversation
messages = [{"role": "user", "content": "Hello"}]
response = await lm.agenerate(messages)
assert response["role"] == "assistant"
assert "content" in response

# Test batch conversation
batch = [
    [{"role": "user", "content": "Hi"}],
    [{"role": "user", "content": "Bye"}]
]
responses = await lm.agenerate(batch)
assert len(responses) == 2
assert all(r["role"] == "assistant" for r in responses)

# Test with tools
tools = [{"type": "function", "function": {"name": "get_weather", ...}}]
response = await lm.agenerate(messages, tools=tools)
assert "tool_calls" in response or "content" in response
```

### Algorithm Interface Testing

```python
# Test basic inference
result = await algorithm.ainfer(lm, "What is 2+2?", budget=5)
assert isinstance(result, dict)
assert result["role"] == "assistant"

# Test full result
result = await algorithm.ainfer(lm, "What is 2+2?", budget=5, return_response_only=False)
assert hasattr(result, "the_one")
assert hasattr(result, "candidates")
assert len(result.candidates) == 5  # budget=5

# Test with tools
tools = [...]
result = await algorithm.ainfer(lm, "Query", budget=3, tools=tools)
assert isinstance(result, dict)
```

### Reward Model Testing

```python
# Test single scoring
conversation = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"}
]
score = reward_model.score(conversation)
assert isinstance(score, float)

# Test batch scoring
batch = [conversation, conversation]
scores = reward_model.score(batch)
assert len(scores) == 2
assert all(isinstance(s, float) for s in scores)
```

---

## Appendix: Complete Interface Summary

| Interface | Layer | Implementer | Consumer | Status |
|-----------|-------|-------------|----------|--------|
| Gateway Requirements | 1 | Gateway developers | LM Interface | Requirements only |
| `AbstractLanguageModel` | 2 | Gateway integrators | Algorithms | ✅ Defined |
| `AbstractScalingAlgorithm` | 3 | its-hub library | End users | ✅ Defined |
| `AbstractOutcomeRewardModel` | 3 | its-hub library, users | Best-of-N | ✅ Defined |
| `AbstractScalingResult` | 3 | Algorithm impls | End users | ✅ Defined |

---

**Document Version**: 1.0
**Last Updated**: 2025-11-21
**Authors**: Red Hat AI Innovation Team
