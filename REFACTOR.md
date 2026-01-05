# ITS-Hub Refactoring Guide

**Target Branch:** Create new branch from `main`
**Goal:** Transform its_hub into a minimal, algorithm-first library with clear architectural boundaries

---

## Part 1: High-Level Objectives

### 1.1 Core Principle: Algorithm-First Architecture

Transform its_hub from a bundled multi-purpose library into a **focused algorithm package** that:
- Provides core ITS algorithms (Self-Consistency, Best-of-N) with minimal dependencies
- Defines clean interfaces (`AbstractLanguageModel`, `AbstractScalingAlgorithm`, `AbstractOutcomeRewardModel`)
- Offers **optional** reference implementations for standalone use
- Enables **easy gateway integration** via simple interface implementation

### 1.2 Scope: This Repository Only

The ITS_DESIGN.md document covers broader architecture including gateway integration patterns. **This refactor focuses solely on the its_hub Python package itself**:

**In Scope:**
- ✅ Restructure package dependencies (minimal core + optional extras)
- ✅ Improve interface definitions and documentation
- ✅ Add LLMJudge reward model implementation
- ✅ Remove unnecessary code and dependencies
- ✅ Enhance type safety and clarity

**Out of Scope:**
- ❌ Gateway implementations (Portkey, LiteLLM, Envoy)
- ❌ Adapter packages (its-hub-portkey, its-hub-litellm)
- ❌ Production deployment infrastructure
- ❌ Cross-language integration (TypeScript, Go, Rust)

### 1.3 What Gets Kept vs. Removed

**KEEP - Core Algorithms (Always Available):**
- ✅ Self-Consistency algorithm (`its_hub/algorithms/self_consistency.py`)
- ✅ Best-of-N algorithm (`its_hub/algorithms/bon.py`)
- ✅ Abstract base classes (`its_hub/base.py`)
- ✅ Type definitions (`its_hub/types.py`)
- ✅ Utility functions (`its_hub/utils.py`)

**KEEP - Optional LM Implementations (`[lm]` extra):**
- ✅ OpenAI-compatible LM (`its_hub/lms.py` - OpenAICompatibleLanguageModel)
- ✅ StepGeneration (for experimental algorithms)
- ✅ Error handling (`its_hub/error_handling.py`)

**KEEP - Experimental Algorithms (`[experimental]` extra):**
- ✅ Beam Search (`its_hub/algorithms/beam_search.py`)
- ✅ Particle Filtering (`its_hub/algorithms/particle_gibbs.py`)
- ✅ Planning Wrapper (`its_hub/algorithms/planning_wrapper.py`)

**ADD - New Components:**
- ✅ LLMJudge reward model (`its_hub/reward_models.py`) - NEW FILE
- ✅ E2E tests with real API (`tests/e2e/`) - NEW DIRECTORY
- ✅ Environment template (`.env.example`) - NEW FILE

**KEEP & UPDATE - Documentation and Examples:**
- ✅ Documentation website (`docs/`) - Update for new architecture
- ✅ Jupyter notebooks (`notebooks/`) - Update imports and examples
- ✅ Benchmark scripts (`scripts/`) - Update for new interface
- ✅ All experimental algorithms tests - Update for new interface

**REMOVE - Intentional Deletions:**
- ❌ LiteLLM implementation (use OpenAI-compatible instead)
- ❌ IaaS service code (`its_hub/integration/iaas.py`)
- ❌ Reward-hub integration (`its_hub/integration/reward_hub.py`)
- ❌ Justfile (`justfile` - use standard tools: pytest, ruff, etc.)
- ❌ MetropolisHastings stub (not implemented)
- ❌ Stub classes (LocalVLLMLanguageModel, TransformersLanguageModel)

**REMOVE - Test Files for Removed Components:**
- ❌ `tests/test_iaas.py` (IaaS service removed)
- ❌ `tests/test_lms.py` (LiteLLM removed, OpenAI LM tested in E2E)
- ❌ `tests/test_reward_hub_integration.py` (reward-hub integration removed)

### 1.4 Key Objectives

1. **Minimal Core Dependencies**: Reduce from 15+ dependencies to just 2 (numpy, typing-extensions)
2. **Optional Extras System**: Clear separation between core and optional components
3. **Gateway-Friendly**: Simple interface implementation = full algorithm access
4. **Type Safety**: Fix type hints, remove pydantic from core
5. **Documentation**: Improve docstrings, add usage examples
6. **Validation**: Real API tests to ensure everything works

### 1.5 Success Criteria

- ✅ `pip install its_hub` installs with ≤2 dependencies
- ✅ Core algorithms work with any `AbstractLanguageModel` implementation
- ✅ Optional `[lm]` extra provides OpenAI-compatible reference implementation
- ✅ **All existing algorithm functionality preserved** (including experimental)
- ✅ **All examples and documentation updated** (docs/, notebooks/, scripts/)
- ✅ 100% test coverage maintained (unit + E2E)
- ✅ Type hints match actual implementation
- ✅ Minimal core dependencies while keeping full feature set

---

## Part 2: Step-by-Step Implementation Guide

### Phase 1: Dependency Restructuring (30 minutes)

#### Step 1.1: Update `pyproject.toml`

**Current state** (main branch has 15+ core dependencies)

**Target state:**

```toml
# pyproject.toml

[project]
dependencies = [
    "numpy",
    "typing-extensions>=4.12.2",
]

[project.scripts]
# its-iaas removed for MVP - will be added back in [iaas] extra

[project.optional-dependencies]

lm = [
    "openai>=1.68.2",
    "aiohttp>=3.9.0",
    "backoff>=2.2.0",
    "requests",
]

iaas = [
    "its_hub[lm]",
    "fastapi>=0.115.5",
    "uvicorn",
    "pydantic>=2.7.2",
    "click>=8.1.0",
]

dev = [
    "its_hub[lm,iaas]",
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.10.0",
    "python-dotenv>=1.0.0",
]

# Experimental - not officially supported in MVP
experimental = [
    "its_hub[lm]",
    "transformers>=4.53.2",
    "reward-hub>=0.1.7",
]

[tool.setuptools.packages.find]
include = [
    "its_hub",
    "its_hub.algorithms",
    # NOTE: "its_hub.integration" REMOVED
]
```

**Key changes:**
- Core deps: Only numpy + typing-extensions (remove openai, litellm, fastapi, etc.)
- Remove: `[vllm]`, `[prm]`, `[research]`, `[cloud]` extras
- Add: `[lm]`, `[iaas]`, `[experimental]` extras with clear purposes
- Remove: `its-iaas` script from main (move to `[iaas]` extra if needed)
- Remove: `its_hub.integration` from package includes

---

### Phase 2: Core Module Updates (60 minutes)

#### Step 2.1: Update `its_hub/__init__.py`

**Current state** (main branch has empty __init__.py)

**Target state:**

```python
"""
A Python library for inference-time scaling LLMs
"""

from importlib.metadata import version

__version__ = version("its_hub")

# Core abstractions - always available
from .base import (
    AbstractLanguageModel,
    AbstractScalingAlgorithm,
    AbstractOutcomeRewardModel,
    AbstractProcessRewardModel,
    AbstractScalingResult,
)

# Core algorithms - always available
from .algorithms.self_consistency import SelfConsistency
from .algorithms.bon import BestOfN

# Start with core exports
__all__ = [
    # Version
    "__version__",
    # Abstractions
    "AbstractLanguageModel",
    "AbstractScalingAlgorithm",
    "AbstractOutcomeRewardModel",
    "AbstractProcessRewardModel",
    "AbstractScalingResult",
    # Algorithms
    "SelfConsistency",
    "BestOfN",
]

# Optional LM implementations - only available if [lm] extra is installed
try:
    from .lms import OpenAICompatibleLanguageModel, StepGeneration
    from .reward_models import LLMJudge
    __all__.extend(["OpenAICompatibleLanguageModel", "StepGeneration", "LLMJudge"])
except ImportError:
    # LM implementations not available - install with: pip install its_hub[lm]
    pass
```

**Key changes:**
- Import core abstractions and algorithms (always available)
- Use try/except for optional LM implementations
- Dynamic `__all__` list based on available imports
- Clear comment indicating how to install missing components

---

#### Step 2.2: Update `its_hub/base.py`

**Critical changes:**

1. **Fix return type hints** (`str` → `dict`):

```python
class AbstractLanguageModel(ABC):
    """
    Abstract base class for language models.

    Gateway integrators should implement this interface to use its_hub algorithms
    with their existing LM infrastructure. Only async implementation is required.
    """

    @abstractmethod
    async def agenerate(
        self,
        messages: list[ChatMessage] | list[list[ChatMessage]],
        stop: str | None = None,
    ) -> dict | list[dict]:  # ← CHANGED from str | list[str]
        """
        Generate response(s) asynchronously.

        Args:
            messages: Single conversation or batch of conversations
            stop: Optional stop sequence for generation

        Returns:
            Single response dict or list of response dicts (for batched input)
            Response dict format: {"role": "assistant", "content": "...", "tool_calls": [...]}
        """
        pass
```

2. **Remove synchronous `generate()` method**:

```python
# DELETE this entire method from AbstractLanguageModel:
# @abstractmethod
# def generate(...) -> str | list[str]:
#     ...
```

3. **Update `AbstractScalingAlgorithm` return types**:

```python
class AbstractScalingAlgorithm(ABC):
    """
    Abstract base class for inference-time scaling algorithms.

    All algorithms (Self-Consistency, Best-of-N, etc.) implement this interface.
    """

    @abstractmethod
    async def ainfer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict | AbstractScalingResult:  # ← CHANGED from str | AbstractScalingResult
        """
        Run inference asynchronously with the given language model and prompt.

        Args:
            lm: Language model instance implementing AbstractLanguageModel
            prompt_or_messages: User prompt (string or structured messages)
            budget: Computational budget (interpretation varies by algorithm)
            return_response_only: If True, return just the selected response;
                                   if False, return full result object
            tools: Optional OpenAI-style tool definitions
            tool_choice: Optional tool choice strategy ("auto", "none", or specific tool)

        Returns:
            Selected response dict (if return_response_only=True) or
            AbstractScalingResult instance with full details
            Response dict format: {"role": "assistant", "content": "...", "tool_calls": [...]}
        """
        pass

    def infer(
        self,
        lm: AbstractLanguageModel,
        prompt_or_messages: str | list[ChatMessage] | ChatMessages,
        budget: int,
        return_response_only: bool = True,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict | AbstractScalingResult:  # ← CHANGED from str | AbstractScalingResult
        """
        Run inference synchronously with the given language model and prompt.

        Default implementation wraps ainfer() using asyncio.run().
        """
        import asyncio
        return asyncio.run(
            self.ainfer(lm, prompt_or_messages, budget, return_response_only, tools, tool_choice)
        )
```

4. **Update `AbstractOutcomeRewardModel` interface**:

```python
class AbstractOutcomeRewardModel(ABC):
    """
    Abstract base class for outcome reward models and judge models.

    This class supports both traditional reward models and LLM-based judge models
    that evaluate conversation outcomes and quality.
    """

    @abstractmethod
    def score(
        self,
        messages: list[list[dict]] | list[dict],
        **kwargs,
    ) -> list[float] | float:
        """
        Score conversations synchronously.

        Args:
            messages: Single conversation or batch of conversations
                Single: list[dict] (one conversation)
                Batch: list[list[dict]] (multiple conversations)
            **kwargs: Additional model-specific parameters

        Returns:
            Single score (float) or list of scores (list[float])
            Higher score = better response
        """
        pass

    async def ascore(
        self,
        messages: list[list[dict]] | list[dict],
        **kwargs,
    ) -> list[float] | float:
        """
        Score conversations asynchronously.

        Default implementation raises NotImplementedError.
        Override this for async-compatible reward models (e.g., LLM judges).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement async scoring. "
            "Override ascore() to support async scoring."
        )
```

5. **Update `AbstractScalingResult.the_one` return type**:

```python
class AbstractScalingResult(ABC):
    """
    Abstract base class for algorithm results.

    Algorithms return instances of this class when return_response_only=False.
    """

    @property
    @abstractmethod
    def the_one(self) -> dict:  # ← CHANGED from str
        """
        Return the selected best response.

        Returns:
            The response message dict selected by the algorithm
            Response dict format: {"role": "assistant", "content": "...", "tool_calls": [...]}
        """
        pass
```

**Summary of base.py changes:**
- Fix all type hints: `str` → `dict` for responses
- Remove sync `generate()` method from AbstractLanguageModel
- Improve all docstrings with detailed Args/Returns
- Add async `ascore()` with default NotImplementedError to AbstractOutcomeRewardModel
- Change interface signature: `(prompt, response)` → `(messages)` for reward models

---

#### Step 2.3: Update `its_hub/types.py`

**Critical changes:**

1. **Remove pydantic dependency**:

```python
# BEFORE (main branch):
from pydantic.dataclasses import dataclass

# AFTER:
from dataclasses import dataclass
```

2. **Remove unused classes**:

```python
# DELETE these entire classes:
# @dataclass
# class Function:
#     name: str
#     description: str | None = None
#     parameters: dict | None = None
#
# @dataclass
# class ToolCall:
#     id: str
#     type: Literal["function"] = "function"
#     function: Function | None = None
```

3. **Simplify `ChatMessage` class**:

```python
@dataclass
class ChatMessage:
    """A chat message with role and content."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict] | None
    tool_calls: list[dict] | None = None  # Store as plain dicts
    tool_call_id: str | None = None

    def to_dict(self) -> dict:
        """Convert ChatMessage to dictionary, excluding None values."""
        result = {"role": self.role}
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        return result
```

4. **Remove experimental methods from `ChatMessages`**:

```python
class ChatMessages:
    """Wrapper for handling both string prompts and conversation history."""

    def __init__(self, str_or_messages: str | list[ChatMessage]):
        self._str_or_messages = str_or_messages
        self._is_string = isinstance(str_or_messages, str)

    @classmethod
    def from_prompt_or_messages(cls, prompt_or_messages: str | list[ChatMessage]):
        if isinstance(prompt_or_messages, cls):
            return prompt_or_messages
        return cls(prompt_or_messages)

    # DELETE this entire method:
    # def to_prompt(self) -> str:
    #     """Convert to prompt string representation."""
    #     ...

    def to_chat_messages(self) -> list[ChatMessage]:
        """Convert to list of ChatMessage objects."""
        if self._is_string:
            return [ChatMessage(role="user", content=self._str_or_messages)]
        return self._str_or_messages

    def to_batch(self, size: int) -> list[list[ChatMessage]]:
        """Create a batch of identical message lists for parallel generation."""
        chat_messages = self.to_chat_messages()
        return [chat_messages for _ in range(size)]
```

**Summary of types.py changes:**
- Replace pydantic with standard dataclasses (remove dependency)
- Remove unused Function and ToolCall classes
- Remove extract_text_content() method from ChatMessage
- Remove to_prompt() method from ChatMessages
- Simplify to minimal working implementation (~60 lines, down from 148)

---

#### Step 2.4: Update `its_hub/lms.py`

**Critical change: Remove LiteLLM implementation**

```python
# DELETE these imports:
# import litellm
# logging.getLogger('litellm').setLevel(logging.WARNING)
# logging.getLogger('litellm.proxy').setLevel(logging.WARNING)
# logging.getLogger('litellm.logging').setLevel(logging.WARNING)

# DELETE entire LiteLLMLanguageModel class (lines 538-843):
# class LiteLLMLanguageModel(AbstractLanguageModel):
#     def __init__(...):
#         ...
#     ... [~300 lines]
#     ...

# DELETE stub classes:
# class LocalVLLMLanguageModel(AbstractLanguageModel):
#     pass
#
# class TransformersLanguageModel(AbstractLanguageModel):
#     pass
```

**Keep these components:**
- ✅ `StepGeneration` class (needed for experimental algorithms)
- ✅ `OpenAICompatibleLanguageModel` class (reference implementation)
- ✅ Helper functions: `rstrip_iff_entire()`
- ✅ All error handling logic

**Remove from OpenAICompatibleLanguageModel:**

```python
# DELETE sync generate() method:
# def generate(
#     self,
#     messages_or_messages_lst: list[ChatMessage] | list[list[ChatMessage]],
#     ...
# ) -> dict | list[dict]:
#     ...

# DELETE evaluate() method:
# def evaluate(self, prompt: str, generation: str) -> list[float]:
#     raise NotImplementedError("evaluate method not implemented")
```

**Result:** `lms.py` goes from ~843 lines to ~490 lines

---

#### Step 2.5: Create `its_hub/reward_models.py` (NEW FILE)

This is a completely new file. Create it with this content:

```python
"""Reward model implementations for production use."""

import json
import logging

from its_hub.base import AbstractLanguageModel, AbstractOutcomeRewardModel


class LLMJudge(AbstractOutcomeRewardModel):
    """
    LLM-based judge that scores conversations using generative reward.

    Reuses AbstractLanguageModel for API communication (retry, error handling, batching).
    Scores are generated via LLM prompting and parsed from structured JSON output.
    """

    DEFAULT_JUDGE_PROMPT = """Score the following conversation on a scale of 0-10.
Return only a JSON object with your score.

Conversation:
{conversation}

Format: {{"score": <number>}}"""

    def __init__(
        self,
        lm: AbstractLanguageModel,
        judge_prompt: str | None = None,
        fallback_score: float = 5.0,
    ):
        """
        Initialize LLM judge.

        Args:
            lm: Language model to use for scoring (reuses existing LM abstraction)
            judge_prompt: Custom judge prompt template. Use {conversation} placeholder.
                         If None, uses DEFAULT_JUDGE_PROMPT.
            fallback_score: Score to return if JSON parsing fails (default: 5.0)
        """
        self.lm = lm
        self.judge_prompt = judge_prompt or self.DEFAULT_JUDGE_PROMPT
        self.fallback_score = fallback_score

    def _format_conversation(self, messages: list[dict]) -> str:
        """Format conversation messages as readable text, including tool calls."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Format tool calls if present
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                tool_strs = []
                for tc in tool_calls:
                    if isinstance(tc, dict) and "function" in tc:
                        func = tc["function"]
                        func_name = func.get("name", "unknown")
                        func_args = func.get("arguments", "{}")
                        tool_strs.append(f"{func_name}({func_args})")
                if tool_strs:
                    lines.append(f"{role} [tool calls]: {', '.join(tool_strs)}")
                if content:  # Also include content if present
                    lines.append(f"{role}: {content}")
            else:
                # Regular message with content only
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _build_judge_prompt(self, conversation: list[dict]) -> list[dict]:
        """Build judge prompt from conversation."""
        conversation_text = self._format_conversation(conversation)
        prompt_text = self.judge_prompt.format(conversation=conversation_text)
        return [{"role": "user", "content": prompt_text}]

    def _parse_score(self, response_content: str) -> float:
        """Parse score from LLM response with fallback."""
        try:
            # Try to parse JSON
            parsed = json.loads(response_content)
            score = float(parsed.get("score", self.fallback_score))
            return score
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logging.warning(
                f"Failed to parse score from response: {response_content[:100]}. "
                f"Using fallback score {self.fallback_score}. Error: {e}"
            )
            return self.fallback_score

    def score(
        self,
        messages: list[list[dict]] | list[dict],
        **kwargs,
    ) -> list[float] | float:
        """
        Score conversations synchronously.

        Not implemented - LLMJudge requires async for API calls.
        Use ascore() instead.
        """
        raise NotImplementedError(
            "LLMJudge requires async API calls. Use ascore() instead of score()."
        )

    async def ascore(
        self,
        messages: list[list[dict]] | list[dict],
        **kwargs,
    ) -> list[float] | float:
        """
        Score conversations asynchronously using LLM.

        Args:
            messages: Single conversation or multiple conversations
            **kwargs: Additional parameters passed to LM (temperature, max_tokens, etc.)

        Returns:
            Single score or list of scores
        """
        # Detect batch vs single
        is_batch = messages and isinstance(messages[0], list)

        # Normalize to batch
        conversations = messages if is_batch else [messages]

        # Build judge prompts for all conversations
        judge_prompts = [self._build_judge_prompt(conv) for conv in conversations]

        # Leverage LM's async batching!
        responses = await self.lm.agenerate(judge_prompts, **kwargs)

        # Parse scores from responses
        scores = [self._parse_score(r.get("content", "")) for r in responses]

        # Return single or batch
        return scores if is_batch else scores[0]
```

**Key design:**
- Reuses `AbstractLanguageModel` for all API calls (no duplicate HTTP logic)
- Supports batch scoring via LM's async batching
- JSON parsing with graceful fallback
- Tool call formatting in conversation display
- ~137 lines total

---

#### Step 2.6: Update `its_hub/algorithms/__init__.py`

**Remove MetropolisHastings stub:**

```python
# BEFORE (main branch exports MetropolisHastings):
__all__ = [
    "BeamSearch",
    "BeamSearchResult",
    "BestOfN",
    "BestOfNResult",
    "MetropolisHastings",  # ← REMOVE THIS
    "MetropolisHastingsResult",  # ← REMOVE THIS
    "ParticleFiltering",
    "ParticleGibbs",
    "ParticleGibbsResult",
    "SelfConsistency",
    "SelfConsistencyResult",
]

# DELETE entire MetropolisHastings implementation (~40 lines)

# AFTER:
from .beam_search import BeamSearch, BeamSearchResult
from .bon import BestOfN, BestOfNResult
from .particle_gibbs import ParticleFiltering, ParticleGibbs, ParticleGibbsResult
from .self_consistency import SelfConsistency, SelfConsistencyResult

__all__ = [
    "BeamSearch",
    "BeamSearchResult",
    "BestOfN",
    "BestOfNResult",
    "ParticleFiltering",
    "ParticleGibbs",
    "ParticleGibbsResult",
    "SelfConsistency",
    "SelfConsistencyResult",
]
```

**Result:** File goes from 56 lines to 16 lines

---

#### Step 2.7: Update `its_hub/algorithms/bon.py`

**Key changes:**

1. **Replace pydantic with dataclasses**:

```python
# BEFORE:
from pydantic.dataclasses import dataclass

# AFTER:
from dataclasses import dataclass
```

2. **Add response deduplication logic**:

Add these helper functions at the top of the file:

```python
import json
from dataclasses import dataclass

def _response_to_hashable_key(response: dict) -> str:
    """
    Convert a response dict to a hashable key for deduplication.

    Handles:
    - content: None, str, or list[dict] (multi-modal)
    - tool_calls: Optional list of tool call dicts

    Returns canonical string representation for use as dict key.
    """
    # Handle content (can be None, str, or list[dict] for multi-modal)
    raw_content = response.get("content")
    if raw_content is None:
        content_str = ""
    elif isinstance(raw_content, str):
        content_str = raw_content
    elif isinstance(raw_content, list):
        # Multi-modal: extract text parts
        text_parts = [
            item.get("text", "")
            for item in raw_content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        content_str = " ".join(text_parts)
    else:
        content_str = str(raw_content)

    # Handle tool_calls (optional, may not exist)
    tool_calls_str = ""
    if response.get("tool_calls"):
        tool_parts = []
        for tc in response.get("tool_calls", []):
            if isinstance(tc, dict) and "function" in tc:
                func = tc["function"]
                func_name = func.get("name", "")
                # Use json.dumps with sort_keys to make arguments hashable
                func_args = json.dumps(func.get("arguments", {}), sort_keys=True)
                tool_parts.append(f"{func_name}:{func_args}")
        tool_calls_str = "|".join(tool_parts)

    # Combine into canonical key (|| separator to avoid content/tool_calls collision)
    return f"{content_str}||{tool_calls_str}"


def _dedupe_responses_with_inverse(responses: list[dict]) -> tuple[list[dict], list[int]]:
    """
    Deduplicate response dicts while preserving order and tracking original indices.

    Returns (uniques, inverse_idx) where:
    - uniques: list of unique response dicts in order of first appearance
    - inverse_idx: for each response in original list, its index in uniques

    Deduplication considers both content and tool_calls for semantic equality.

    Example:
        responses = [r1, r2, r1, r3, r2]  # where r1, r2, r3 are response dicts
        returns ([r1, r2, r3], [0, 1, 0, 2, 1])
    """
    uniques: list[dict] = []
    index_of: dict[str, int] = {}
    inverse_idx: list[int] = []

    for response in responses:
        key = _response_to_hashable_key(response)
        j = index_of.get(key)
        if j is None:
            j = len(uniques)
            index_of[key] = j
            uniques.append(response)  # Keep original dict with all fields
        inverse_idx.append(j)

    return uniques, inverse_idx
```

3. **Update BestOfNResult class**:

```python
@dataclass
class BestOfNResult(AbstractScalingResult):
    candidates: list[dict]  # ← CHANGED from list[str]
    scores: list[float]
    best_index: int

    @property
    def the_one(self) -> dict:  # ← CHANGED from str
        return self.candidates[self.best_index]
```

4. **Update BestOfN.ainfer() to use deduplication**:

In the `ainfer()` method, add deduplication before scoring:

```python
async def ainfer(
    self,
    lm: AbstractLanguageModel,
    prompt_or_messages: str | list[ChatMessage] | ChatMessages,
    budget: int,
    return_response_only: bool = True,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
) -> dict | BestOfNResult:
    # ... existing prompt conversion code ...

    # Generate N candidates
    candidates = await lm.agenerate(
        prompts_batch, tools=tools, tool_choice=tool_choice
    )

    # Deduplicate responses before scoring
    unique_candidates, inverse_idx = _dedupe_responses_with_inverse(candidates)

    # Score only unique candidates
    unique_scores = await self.reward_model.ascore(
        [[*chat_messages, cand] for cand in unique_candidates]
    )

    # Map scores back to all candidates (including duplicates)
    scores = [unique_scores[i] for i in inverse_idx]

    # Find best
    best_index = scores.index(max(scores))

    if return_response_only:
        return candidates[best_index]

    return BestOfNResult(
        candidates=candidates,
        scores=scores,
        best_index=best_index,
    )
```

**Summary:** Adds smart deduplication to avoid scoring identical responses multiple times

---

#### Step 2.8: Update `its_hub/algorithms/self_consistency.py`

**Change pydantic to dataclasses:**

```python
# BEFORE:
from pydantic.dataclasses import dataclass

# AFTER:
from dataclasses import dataclass
```

**Update result class:**

```python
@dataclass
class SelfConsistencyResult(AbstractScalingResult):
    candidates: list[dict]  # ← CHANGED from list[str]

    @property
    def the_one(self) -> dict:  # ← CHANGED from str
        # ... existing voting logic ...
        return self.candidates[selected_index]
```

**No other changes needed** - algorithm logic stays the same

---

#### Step 2.9: Update `its_hub/algorithms/beam_search.py` and `particle_gibbs.py`

**Only change: pydantic → dataclasses**

```python
# BEFORE:
from pydantic.dataclasses import dataclass

# AFTER:
from dataclasses import dataclass
```

**No other changes** - these are experimental algorithms, keep as-is

---

### Phase 3: Update Documentation & Examples (60 minutes)

#### Step 3.1: Update Jupyter Notebooks

**File: `notebooks/self-consistency.py`** (and `.ipynb` if it exists)

Update imports to work with new structure:

```python
# BEFORE (main branch):
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT

# AFTER:
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT  # Keep if it still exists

# Update algorithm usage to return dicts instead of strings
from its_hub import SelfConsistency

# Before:
# result = algorithm.infer(lm, prompt, budget=5)  # Returns str
# print(result)

# After:
result = algorithm.infer(lm, prompt, budget=5)  # Returns dict
print(result["content"])  # Extract content from response dict
```

**Key changes:**
- Update all algorithm calls to handle dict responses
- Use `result["content"]` or `result.get("content")` to access text
- Test that notebooks run end-to-end with new structure

#### Step 3.2: Update Benchmark Scripts

**File: `scripts/benchmark.py`**

Update imports and response handling:

```python
# Update imports - remove LiteLLM if used
# BEFORE:
# from its_hub.lms import LiteLLMLanguageModel

# AFTER:
from its_hub.lms import OpenAICompatibleLanguageModel

# Update algorithm response handling
# BEFORE:
# response = algorithm.infer(lm, prompt, budget=budget)
# answer = _extract_boxed(response)  # Expects string

# AFTER:
response = algorithm.infer(lm, prompt, budget=budget)
content = response.get("content", "") if isinstance(response, dict) else str(response)
answer = _extract_boxed(content)
```

**Key changes:**
- Replace LiteLLM with OpenAI-compatible LM
- Handle dict responses from algorithms
- Update reward model integrations if using reward-hub (may need to keep as experimental)

#### Step 3.3: Update Documentation Website

**Files in `docs/`:**

1. **`docs/installation.md`** - Update installation instructions:
```markdown
# Before:
pip install its_hub

# After - show all installation options:
# Core only (minimal dependencies)
pip install its_hub

# With LM support (for standalone use)
pip install its_hub[lm]

# With experimental algorithms
pip install its_hub[experimental]

# For development
pip install its_hub[dev]
```

2. **`docs/quick-start.md`** - Update quick start examples:
```python
# Show both gateway integration and standalone patterns
# Pattern 1: Gateway Integration (core only)
from its_hub import AbstractLanguageModel, SelfConsistency

class MyGatewayLM(AbstractLanguageModel):
    async def agenerate(self, messages, stop=None, **kwargs):
        # Your gateway's LM client
        return {"role": "assistant", "content": "..."}

# Pattern 2: Standalone with OpenAI (requires [lm] extra)
from its_hub.lms import OpenAICompatibleLanguageModel
lm = OpenAICompatibleLanguageModel(...)
```

3. **`docs/algorithms.md`** - Update algorithm examples:
```python
# Update all examples to show dict responses
result = algorithm.infer(lm, prompt, budget=5)
print(result)  # {"role": "assistant", "content": "...", ...}
```

4. **`docs/development.md`** - Update development setup:
```bash
# Show new dependency structure
uv sync --extra dev  # Installs all dev dependencies including [lm] and [experimental]
```

5. **`docs/iaas-service.md`** - Either remove or mark as deprecated:
```markdown
# Option 1: Remove this file entirely
# Option 2: Add deprecation notice
> **Deprecated**: The IaaS service has been removed from the core library.
> For production deployments, integrate its_hub algorithms into your existing
> gateway infrastructure.
```

#### Step 3.4: Delete Integration Code

```bash
# Remove integration directory
rm -rf its_hub/integration/
```

#### Step 3.5: Delete Justfile

```bash
# Use standard tools instead
rm justfile
```

#### Step 3.6: Clean Up Test Files

```bash
# Only remove tests for removed components
rm tests/test_iaas.py
rm tests/test_lms.py  # LiteLLM tests
rm tests/test_reward_hub_integration.py
```

**KEEP these test files** (update them instead):
- ✅ `tests/test_particle_gibbs_resampling.py` - Update for new interface
- ✅ `tests/test_planning_wrapper.py` - Update for new interface

Update these test files to:
- Use OpenAI-compatible LM instead of LiteLLM
- Handle dict responses instead of strings
- Import from correct locations

---

### Phase 4: Testing Infrastructure (45 minutes)

#### Step 4.0: Update Existing Test Files

**Before adding new tests, update existing tests to work with new structure:**

**File: `tests/test_particle_gibbs_resampling.py`**
```python
# Update imports
from its_hub.lms import OpenAICompatibleLanguageModel  # Remove LiteLLM
from its_hub.algorithms import ParticleFiltering

# Update response handling - algorithms now return dicts
# BEFORE:
# response = algorithm.infer(lm, prompt, budget=5)  # str
# AFTER:
response = algorithm.infer(lm, prompt, budget=5)  # dict
content = response.get("content") if isinstance(response, dict) else response
```

**File: `tests/test_planning_wrapper.py`**
```python
# Same pattern - update for dict responses
```

#### Step 4.1: Update `tests/conftest.py`

**Remove litellm imports:**

```python
# DELETE these imports:
# from its_hub.lms import LiteLLMLanguageModel

# KEEP the mock fixtures - they still work with new structure
```

#### Step 4.2: Update `tests/mocks/reward_models.py`

**Update to match new AbstractOutcomeRewardModel interface:**

```python
from its_hub.base import AbstractOutcomeRewardModel


class MockRewardModel(AbstractOutcomeRewardModel):
    """Mock reward model for testing."""

    def __init__(self, scores: list[float] | None = None):
        self.scores = scores or [1.0, 2.0, 3.0, 4.0, 5.0]
        self.call_count = 0

    def score(
        self,
        messages: list[list[dict]] | list[dict],
        **kwargs,
    ) -> list[float] | float:
        """Score conversations (mock implementation)."""
        is_batch = messages and isinstance(messages[0], list)

        if is_batch:
            # Return scores for batch
            num_convs = len(messages)
            result_scores = self.scores[:num_convs]
            self.call_count += 1
            return result_scores
        else:
            # Return single score
            score = self.scores[self.call_count % len(self.scores)]
            self.call_count += 1
            return score

    async def ascore(
        self,
        messages: list[list[dict]] | list[dict],
        **kwargs,
    ) -> list[float] | float:
        """Async scoring (just calls sync version for mocks)."""
        return self.score(messages, **kwargs)
```

#### Step 4.3: Create E2E Test Directory

```bash
mkdir -p tests/e2e
touch tests/e2e/__init__.py
```

#### Step 4.4: Create `tests/e2e/conftest.py`

```python
"""Fixtures for E2E tests with real API calls."""

import os
import pytest
from dotenv import load_dotenv

from its_hub.lms import OpenAICompatibleLanguageModel


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load environment variables from .env file."""
    load_dotenv()


@pytest.fixture(scope="session")
def openai_api_key():
    """Get OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-your-"):
        pytest.skip("OPENAI_API_KEY not set - skipping E2E tests")
    return api_key


@pytest.fixture(scope="session")
def openai_lm(openai_api_key):
    """Create OpenAI-compatible LM for testing."""
    return OpenAICompatibleLanguageModel(
        endpoint=os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1"),
        api_key=openai_api_key,
        model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.7,
        max_concurrent_requests=5,
    )
```

#### Step 4.5: Create `tests/e2e/test_real_openai.py`

Create a comprehensive E2E test file. Here's the structure (full file is ~470 lines, see the working version for complete implementation):

```python
"""E2E tests with real OpenAI API calls.

These tests make actual API calls to OpenAI and are skipped if OPENAI_API_KEY is not set.
Use budget=2 to minimize API costs while still validating functionality.
"""

import pytest

from its_hub import SelfConsistency, BestOfN
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.reward_models import LLMJudge


class TestSelfConsistencyE2E:
    """E2E tests for Self-Consistency with real API."""

    def test_self_consistency_basic(self, openai_lm):
        """Test basic self-consistency with real API."""
        algorithm = SelfConsistency()
        result = algorithm.infer(
            lm=openai_lm,
            prompt_or_messages="What is 2+2? Answer with just the number.",
            budget=2,  # Minimal budget to reduce costs
            return_response_only=True,
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert "role" in result
        assert result["role"] == "assistant"
        assert "content" in result
        assert isinstance(result["content"], str)
        assert len(result["content"]) > 0

    def test_self_consistency_with_tool_calls(self, openai_lm):
        """Test self-consistency with tool calls."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add_numbers",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]

        algorithm = SelfConsistency(tool_vote="tool_name")
        result = algorithm.infer(
            lm=openai_lm,
            prompt_or_messages="Add 2 and 3 using the add_numbers function.",
            budget=2,
            return_response_only=True,
            tools=tools,
            tool_choice="required",
        )

        # Verify tool calls present
        assert isinstance(result, dict)
        assert "tool_calls" in result
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) > 0

    @pytest.mark.asyncio
    async def test_self_consistency_async(self, openai_lm):
        """Test async self-consistency."""
        algorithm = SelfConsistency()
        result = await algorithm.ainfer(
            lm=openai_lm,
            prompt_or_messages="What is the capital of France?",
            budget=2,
            return_response_only=True,
        )

        assert isinstance(result, dict)
        assert "content" in result


class TestBestOfNE2E:
    """E2E tests for Best-of-N with real API."""

    def test_best_of_n_with_llm_judge_sync(self, openai_lm):
        """Test Best-of-N with LLM judge (sync wrapper)."""
        judge = LLMJudge(lm=openai_lm, fallback_score=5.0)
        algorithm = BestOfN(reward_model=judge)

        result = algorithm.infer(
            lm=openai_lm,
            prompt_or_messages="Write a haiku about programming.",
            budget=2,
            return_response_only=True,
        )

        assert isinstance(result, dict)
        assert "content" in result

    @pytest.mark.asyncio
    async def test_best_of_n_async(self, openai_lm):
        """Test async Best-of-N."""
        judge = LLMJudge(lm=openai_lm, fallback_score=5.0)
        algorithm = BestOfN(reward_model=judge)

        result = await algorithm.ainfer(
            lm=openai_lm,
            prompt_or_messages="What is 5+3?",
            budget=2,
            return_response_only=True,
        )

        assert isinstance(result, dict)


class TestLLMJudgeE2E:
    """E2E tests for LLMJudge with real API."""

    @pytest.mark.asyncio
    async def test_llm_judge_single_conversation(self, openai_lm):
        """Test LLMJudge scoring a single conversation."""
        judge = LLMJudge(lm=openai_lm, fallback_score=5.0)

        conversation = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]

        score = await judge.ascore(conversation)

        assert isinstance(score, float)
        assert 0 <= score <= 10

    @pytest.mark.asyncio
    async def test_llm_judge_batch_conversations(self, openai_lm):
        """Test LLMJudge batch scoring."""
        judge = LLMJudge(lm=openai_lm, fallback_score=5.0)

        conversations = [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "5"},  # Wrong answer
            ],
        ]

        scores = await judge.ascore(conversations)

        assert isinstance(scores, list)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    @pytest.mark.asyncio
    async def test_llm_judge_custom_prompt(self, openai_lm):
        """Test LLMJudge with custom prompt."""
        custom_prompt = """Rate the technical accuracy of this conversation on a scale of 0-10.

Conversation:
{conversation}

Return JSON: {{"score": <number>}}"""

        judge = LLMJudge(lm=openai_lm, judge_prompt=custom_prompt, fallback_score=5.0)

        conversation = [
            {"role": "user", "content": "Explain quicksort."},
            {"role": "assistant", "content": "Quicksort is a divide-and-conquer sorting algorithm."},
        ]

        score = await judge.ascore(conversation)
        assert isinstance(score, float)


class TestCoreInterfaceE2E:
    """Test core interface patterns."""

    def test_minimal_imports(self):
        """Test that core imports work without [lm] extra."""
        from its_hub import (
            AbstractLanguageModel,
            AbstractScalingAlgorithm,
            SelfConsistency,
            BestOfN,
        )

        assert AbstractLanguageModel is not None
        assert AbstractScalingAlgorithm is not None
        assert SelfConsistency is not None
        assert BestOfN is not None
```

#### Step 4.6: Create `.env.example`

```bash
# OpenAI API Configuration
# Copy this file to .env and fill in your actual API key
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_ENDPOINT=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

---

### Phase 5: Documentation Updates (30 minutes)

#### Step 5.1: Update `CLAUDE.md`

The current CLAUDE.md is already excellent. No changes needed - it already documents:
- ✅ Development commands
- ✅ Package installation extras
- ✅ Architecture overview
- ✅ Git workflow requirements

#### Step 5.2: Update `README.md`

Update the installation and quick start sections to reflect the new structure:

```markdown
# `its-hub`: A Python library for inference-time scaling

**its_hub** is a Python library for inference-time scaling of LLMs, focusing on mathematical reasoning tasks.

## Installation

**its_hub** provides a minimal core focused on algorithms, with optional language model implementations.

### Core Installation (Algorithms Only)

For **gateway integration** - just algorithms and interfaces, minimal dependencies:

```bash
pip install its_hub
```

This includes:
- ✓ Self-Consistency and Best-of-N algorithms
- ✓ Abstract base classes (`AbstractLanguageModel`, `AbstractOutcomeRewardModel`)
- ✓ Only 2 dependencies: `numpy`, `typing-extensions`

### With Language Model Support

For **standalone use** - includes OpenAI-compatible language model implementation:

```bash
pip install its_hub[lm]
```

Adds: `OpenAICompatibleLanguageModel`, `LLMJudge`, `StepGeneration` (requires `openai`, `aiohttp`, `backoff`)

### Development Installation

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub
pip install -e ".[dev]"
# or using uv:
uv sync --extra dev
```

## Quick Start

### Example 1: Gateway Integration (Core Installation)

**Installation required:** `pip install its_hub` (core only, minimal dependencies)

Gateway integration is simple: implement the `AbstractLanguageModel` interface with your existing LM client.

```python
from its_hub import AbstractLanguageModel, SelfConsistency

# Implement AbstractLanguageModel interface - this is the ONLY integration work required
class MyGatewayLM(AbstractLanguageModel):
    def __init__(self, gateway_client):
        self.client = gateway_client

    async def agenerate(self, messages, stop=None, **kwargs):
        response = await self.client.generate(messages, stop=stop, **kwargs)
        return {"role": "assistant", "content": response}

# Use its_hub algorithms with your gateway's LM
lm = MyGatewayLM(your_gateway_client)
algorithm = SelfConsistency()
result = await algorithm.ainfer(lm, "What is 2+2?", budget=5)
print(result)  # {"role": "assistant", "content": "4", ...}
```

### Example 2: Standalone Use with OpenAI-Compatible LM

**Installation required:** `pip install its_hub[lm]`

```python
from its_hub import SelfConsistency
from its_hub.lms import OpenAICompatibleLanguageModel

lm = OpenAICompatibleLanguageModel(
    endpoint="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4o-mini",
)

algorithm = SelfConsistency()
result = algorithm.infer(lm, "What is the capital of France?", budget=3)
print(result)  # Most common answer from 3 generations
```

### Example 3: Best-of-N with LLM Judge

**Installation required:** `pip install its_hub[lm]`

```python
from its_hub import BestOfN
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.reward_models import LLMJudge

lm = OpenAICompatibleLanguageModel(
    endpoint="https://api.openai.com/v1",
    api_key="your-api-key",
    model_name="gpt-4o-mini",
)

judge = LLMJudge(lm=lm, fallback_score=5.0)
algorithm = BestOfN(reward_model=judge)
result = await algorithm.ainfer(lm, "Write a sorting function", budget=5)
print(result)  # Best response as judged by LLM
```

## Key Features

- 🔬 **Multiple Algorithms**: Self-Consistency, Best-of-N, Beam Search (experimental), Particle Filtering (experimental)
- 🚀 **Gateway Integration**: Clean abstractions for easy integration with AI gateways
- 🧮 **Math-Optimized**: Built for mathematical reasoning tasks
- ⚡ **Async Support**: Concurrent generation with limits and error handling
- 🎯 **Minimal Core**: Only 2 dependencies (numpy, typing-extensions) for core install
```

---

### Phase 6: Final Validation (45 minutes)

#### Step 6.1: Run All Tests

```bash
# Install with dev extras
uv sync --extra dev

# Run all tests (unit + E2E)
uv run pytest tests/ -v

# Expected results:
# - All unit tests passing (core + experimental algorithms)
# - E2E tests passing (with real OpenAI API)
# - No import errors
# - No type errors
```

#### Step 6.2: Validate Examples

```bash
# Run Jupyter notebooks
cd notebooks/
jupyter nbconvert --to notebook --execute self-consistency.ipynb

# Run benchmark scripts (with small dataset)
cd scripts/
python benchmark.py --help  # Should show updated help
```

#### Step 6.3: Validate Documentation

```bash
# Check docs build (if using docsify)
cd docs/
# Open index.html in browser and verify all links work
```

#### Step 6.4: Test Installation from Scratch

```bash
# Test core installation
pip install -e .
python -c "from its_hub import SelfConsistency, BestOfN; print('Core OK')"

# Test [lm] installation
pip install -e ".[lm]"
python -c "from its_hub import OpenAICompatibleLanguageModel, LLMJudge; print('LM OK')"
```

#### Step 6.5: Run Linter

```bash
uv run ruff check its_hub/
uv run ruff format its_hub/
```

---

## Part 3: Commit Strategy

### Recommended Commit Sequence

```bash
# 1. Dependency restructuring
git add pyproject.toml
git commit -s -m "refactor: restructure dependencies with minimal core and optional extras

- Core: Only numpy and typing-extensions (2 deps)
- Add [lm] extra: OpenAI-compatible LM, LLMJudge
- Add [experimental] extra: Beam Search, Particle Filtering, reward-hub
- Remove: litellm from core dependencies
- Keep all algorithms and examples (update for new structure)
"

# 2. Core interface improvements
git add its_hub/base.py its_hub/types.py its_hub/__init__.py
git commit -s -m "refactor: improve core interfaces and type safety

- Fix return types: str -> dict for responses
- Remove pydantic from types.py (use standard dataclasses)
- Improve AbstractOutcomeRewardModel interface
- Add async ascore() with default NotImplementedError
- Update all docstrings with detailed Args/Returns
- Dynamic __all__ exports based on available imports
"

# 3. LM implementation cleanup
git add its_hub/lms.py
git commit -s -m "refactor: remove LiteLLM and stub implementations

- Remove LiteLLMLanguageModel class
- Remove LocalVLLMLanguageModel stub
- Remove TransformersLanguageModel stub
- Keep OpenAICompatibleLanguageModel as reference implementation
- Keep StepGeneration for experimental algorithms
"

# 4. Add reward model
git add its_hub/reward_models.py
git commit -s -m "feat: add LLMJudge reward model implementation

- Reuses AbstractLanguageModel for API calls
- Supports async batch scoring
- JSON parsing with fallback handling
- Tool call formatting in conversation display
"

# 5. Algorithm updates
git add its_hub/algorithms/
git commit -s -m "refactor: update algorithms for new interface

- Replace pydantic with standard dataclasses
- Add response deduplication to BestOfN
- Update return types: str -> dict
- Remove MetropolisHastings stub
- All algorithms now return dict responses
"

# 6. Remove integration code
git rm -rf its_hub/integration/ justfile
git rm tests/test_iaas.py tests/test_lms.py tests/test_reward_hub_integration.py
git commit -s -m "refactor: remove IaaS service and integration code

Removed:
- its_hub/integration/ (IaaS, reward-hub integrations)
- justfile (use standard tools: pytest, ruff)
- Test files for removed components (iaas, lms, reward-hub)

Kept:
- All algorithm implementations
- All documentation (updating in next commits)
- All examples (updating in next commits)
"

# 7. Update documentation
git add docs/
git commit -s -m "docs: update documentation for new architecture

- Update installation instructions with [lm] and [experimental] extras
- Update quick start examples for gateway integration pattern
- Update algorithm examples to show dict responses
- Remove/deprecate IaaS service documentation
- Keep all other documentation current
"

# 8. Update examples
git add notebooks/ scripts/
git commit -s -m "examples: update notebooks and scripts for new interface

- Update imports to use OpenAI-compatible LM
- Handle dict responses from algorithms
- Update benchmark scripts for new structure
- All examples tested and working
"

# 9. Add E2E tests
git add tests/e2e/ .env.example
git commit -s -m "test: add E2E tests with real OpenAI API

- Create tests/e2e/ directory with conftest.py
- Add comprehensive E2E tests for all algorithms
- Test Self-Consistency, Best-of-N, LLMJudge
- Test tool call support throughout
- Add .env.example for API key configuration
"

# 10. Update existing tests
git add tests/
git commit -s -m "test: update all tests for new interfaces

- Update mocks to match new AbstractOutcomeRewardModel
- Update experimental algorithm tests (particle_gibbs, planning_wrapper)
- Fix imports for new structure
- Remove litellm dependencies
- All tests passing
"

# 11. Final documentation
git add README.md CLAUDE.md ITS_DESIGN.md
git commit -s -m "docs: finalize documentation for new architecture

- Add ITS_DESIGN.md with full interface specification
- Update README.md with installation patterns
- Add gateway integration examples
- Document all extras and their purposes
"
```

---

## Part 4: Verification Checklist

Before creating PR, verify:

- [ ] `pip install its_hub` works with only 2 dependencies
- [ ] `pip install its_hub[lm]` provides OpenAI-compatible LM
- [ ] `pip install its_hub[experimental]` provides all experimental algorithms
- [ ] All unit tests pass (core + experimental)
- [ ] All E2E tests pass (with OPENAI_API_KEY set)
- [ ] All Jupyter notebooks run without errors
- [ ] Benchmark scripts run with updated interface
- [ ] Documentation website builds and links work
- [ ] Ruff linter passes with no errors
- [ ] Type hints are correct throughout
- [ ] CLAUDE.md is accurate and comprehensive
- [ ] README.md has updated installation instructions
- [ ] .env.example exists and is documented
- [ ] No litellm imports anywhere
- [ ] No pydantic in core dependencies
- [ ] Git commits use -s flag (sign-off)
- [ ] No mention of AI assistance in commits

---

## Part 5: Success Metrics

After refactoring:

**Dependencies:**
- ✅ Core: 2 dependencies (was 15+)
- ✅ [lm] extra: +4 dependencies
- ✅ [experimental] extra: +2 dependencies (transformers, reward-hub)
- ✅ Total core reduction: ~87% (15 → 2)

**Code Changes:**
- ✅ Core modules cleaned up (remove pydantic, LiteLLM)
- ✅ types.py: 60 lines (was 148) - 60% reduction
- ✅ lms.py: 490 lines (was 843) - 42% reduction
- ✅ algorithms/__init__.py: 16 lines (was 56) - 71% reduction
- ✅ Integration code removed (~1000+ lines)
- ✅ All examples updated and working
- ✅ All documentation updated

**Testing:**
- ✅ All unit tests passing (core + experimental)
- ✅ E2E tests with real API
- ✅ Example notebooks tested
- ✅ Benchmark scripts tested
- ✅ 100% functionality preserved

**Architecture:**
- ✅ Clean separation: algorithms vs LM implementations
- ✅ Gateway-friendly interface
- ✅ Type-safe throughout
- ✅ Comprehensive documentation
- ✅ Working examples and benchmarks

---

## Appendix: Common Issues and Solutions

### Issue 1: Import errors after removing pydantic

**Problem:** Tests fail with "cannot import pydantic"

**Solution:** Check all files for pydantic imports:
```bash
grep -r "from pydantic" its_hub/
grep -r "import pydantic" its_hub/
```

Replace all with `from dataclasses import dataclass`

### Issue 2: E2E tests skipped

**Problem:** E2E tests show "skipped" instead of running

**Solution:** Create `.env` file from `.env.example`:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Issue 3: Type errors with dict vs str

**Problem:** Type checker complains about str vs dict

**Solution:** Update all type hints in base.py:
- `str | list[str]` → `dict | list[dict]`
- Check all `the_one` property return types
- Check all `ainfer()` and `agenerate()` return types

### Issue 4: Tests fail with AttributeError

**Problem:** Tests fail with "dict object has no attribute content"

**Solution:** Update code that expects string responses:
```python
# BEFORE:
response = "Hello"

# AFTER:
response = {"role": "assistant", "content": "Hello"}
```

Use `response.get("content")` or `response["content"]` to access content

---

## Timeline Estimate

- **Phase 1** (Dependencies): 30 minutes
- **Phase 2** (Core modules): 60 minutes
- **Phase 3** (Update docs & examples): 60 minutes
- **Phase 4** (Tests): 45 minutes
- **Phase 5** (README updates): 30 minutes
- **Phase 6** (Validation): 45 minutes

**Total: ~4.5 hours** for a complete, tested refactor with all examples working

---

**End of Refactoring Guide**

This document should be used to create a new branch from `main` and systematically apply all changes. Each section provides both the rationale and exact code changes needed.
