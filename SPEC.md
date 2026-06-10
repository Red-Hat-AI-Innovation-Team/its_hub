# its_hub Repository Specification

Status: Draft v1

Purpose: Define the repository-level contract for `its_hub` as a library-first system for
inference-time scaling (ITS) of LLMs, while also documenting the two concrete gateway
implementations that currently exist in the repository family:

- the current OpenAI-compatible FastAPI IaaS service in `its_hub/integration/iaas.py`
- the Envoy external-processing profile implemented on `origin/envoy_ext_proc`

Source of truth for this specification version:

- `origin/v1`
  - normative for the core library surface, package layout, and install profiles
- current repository branch
  - normative for the current `its-iaas` FastAPI gateway profile
- `origin/envoy_ext_proc`
  - normative for the Envoy ext-proc gateway profile

When these sources differ, this specification MUST say which surface is being described.

Interpretation model for this document:

- Sections 3 through 10 define the intended repository-family design target unless they explicitly
  name a branch or concrete implementation.
- Sections 11 through 14 map that target onto the current Python repository family and its concrete
  gateway/runtime profiles.
- Python module paths in this document are reference mappings for the current Python branches, not
  mandatory shapes for future non-Python ports such as Rust.

## Normative Language

The key words `MUST`, `MUST NOT`, `REQUIRED`, `SHOULD`, `SHOULD NOT`, `RECOMMENDED`, `MAY`, and
`OPTIONAL` in this document are to be interpreted as described in RFC 2119.

`Implementation-defined` means the behavior is part of the implementation contract, but this
specification does not prescribe one universal policy. Implementations MUST document the selected
behavior.

## 1. Problem Statement

`its_hub` is a repository for inference-time scaling techniques that improve model outputs by
spending more compute during inference rather than by retraining the model.

The repository family currently serves three closely related needs:

1. `Library-first ITS`
   - reusable algorithms, LM adapters, reward-model abstractions, and orchestration contracts

2. `Gateway integration`
   - integration into existing serving layers without changing the downstream OpenAI-compatible
     client contract

3. `Research and evaluation`
   - examples, benchmarks, and math-oriented evaluation workflows

Important boundary:

- the center of gravity for `v1` is the library contract
- the gateway runtimes are integration profiles layered on top of that library contract
- no single gateway runtime is required for `v1` core conformance

## 2. Goals and Non-Goals

### 2.1 Goals

- Define stable abstractions for:
  - messages
  - language models
  - orchestration
  - ITS algorithms
  - reward models
- Support both string prompts and structured chat messages.
- Preserve OpenAI-compatible assistant message structure, including tool calls, where supported.
- Keep the stable core small enough for gateway integrators.
- Provide built-in implementations for common LM, orchestration, and judging workflows.
- Support multiple ITS strategies with a shared `budget` vocabulary.
- Document the current gateway profiles separately so design work can happen in `SPEC.md` before
  implementation changes land.

### 2.2 Non-Goals

- Defining the full OpenAI Chat Completions protocol.
- Requiring one mandatory service runtime for `v1`.
- Defining training, fine-tuning, or model-hosting internals.
- Standardizing streaming ITS behavior in this specification version.
- Guaranteeing that every experimental algorithm has the same stability level as the stable core.

## 3. Repository Model

### 3.1 Canonical `v1` Layout

The canonical `v1` design target uses a split API/core layout. `origin/v1` is the closest current
Python branch to that layout, but the diagram below is conceptual rather than a literal exhaustive
inventory of any single branch:

```text
its_hub/
  api/
    algorithm.py
    lm.py
    orchestrator.py
    types.py
    reward_models/
  core/
    algorithms/
    lms/
    reward_models/
    orchestrator.py
  algorithms/
benchmarking/
examples/
docs/
tests/
README.md
pyproject.toml
SPEC.md
```

Interpretation note:

- the diagram names the intended conceptual homes of repository surfaces
- supporting files such as `SPEC.md`, `scripts/`, `eval/`, or branch-specific docs MAY be added,
  omitted, or relocated without changing the core contract

Responsibilities:

- `its_hub/api/`
  - stable abstract contracts
- `its_hub/core/`
  - built-in implementations of those contracts
- `its_hub/algorithms/`
  - compatibility-oriented import surface
- `benchmarking/`, `examples/`, `docs/`, `tests/`
  - tooling, documentation, validation

### 3.2 Legacy Flat Layout Compatibility

Some active branches, including the current branch and `origin/envoy_ext_proc`, still use the older
flat layout:

```text
its_hub/
  base.py
  lms.py
  types.py
  algorithms/
  integration/
```

For this specification version:

- the `origin/v1` split layout is the canonical design target
- the flat layout branches are treated as equivalent implementations of the same conceptual layers
- gateway profiles that still live under the flat layout remain part of the repository-family
  contract until they are retired or migrated

### 3.3 Deliverables

The repository family currently contains these deliverables:

1. `Core library contract`
   - abstractions and built-in ITS implementations

2. `Gateway profiles`
   - current FastAPI IaaS service
   - Envoy ext-proc external processor

3. `Documentation, examples, and benchmarks`
   - install guides, quick starts, algorithm docs, benchmark tooling

4. `Tests and packaging metadata`
   - repository validation and distribution metadata

## 4. Distribution and Installation Profiles

### 4.1 `origin/v1` Profiles

The normative `v1` install profiles are:

- default / core
  - `pip install its_hub`
  - minimal dependency footprint
  - intended for integrators who provide their own LM and orchestrator

- `lm`
  - `pip install its_hub[lm]`
  - adds built-in OpenAI-compatible LM support, async HTTP dependencies, `LLMJudge`, and
    `LMOrchestrator`

- `iaas`
  - `pip install its_hub[iaas]`
  - adds service-oriented dependencies
  - in `origin/v1` this is a dependency profile, not a required built-in runtime contract

- `experimental`
  - `pip install its_hub[experimental]`
  - adds experimental algorithms and PRM-related dependencies

- `research`
  - `pip install its_hub[research]`
  - adds benchmark/evaluation dependencies

- `dev`
  - `pip install -e ".[dev]"` or `uv sync --extra dev`
  - adds development and test tooling

### 4.2 Legacy Branch Packaging Notes

The current branch and `origin/envoy_ext_proc` still ship older packaging shapes:

- current branch
  - includes a built-in `its-iaas` script in `pyproject.toml`
  - uses extras such as `vllm`, `prm`, `research`, and `cloud`
  - default dependencies already include a broader runtime/service footprint than the minimal
    `origin/v1` core profile

- `origin/envoy_ext_proc`
  - includes both:
    - `its-iaas`
    - `envoy-grpc`
  - default dependencies also include gRPC/runtime packages needed by that gateway profile

These older package shapes are implementation facts for those branches, but they are not the
normative `origin/v1` packaging model.

## 5. Runtime Entrypoints

### 5.1 `v1` Core Position

`origin/v1` does not require one built-in ITS service runtime as part of core conformance.

### 5.2 Current FastAPI IaaS Profile

The current branch exports:

- `its-iaas`
  - entrypoint: `its_hub.integration.iaas:main`

This starts the in-process FastAPI IaaS service.

### 5.3 Envoy ext-proc Profile

The `origin/envoy_ext_proc` branch exports:

- `envoy-grpc`
  - entrypoint: `its_hub.integration.ext_proc.processor:main`

This starts the Envoy external-processor gRPC service.

## 6. System Overview

### 6.1 Main Components

These are conceptual components. A concrete branch or port MAY collapse two or more components into
one module, service, or type so long as the externally visible behavior remains equivalent.

1. `Message Model`
   - normalized prompts, chat history, multimodal text-bearing messages, and tool calls

2. `Language Model Adapter Layer`
   - downstream LM call interface

3. `Orchestrator Layer`
   - parallel fanout, concurrency control, and argument forwarding

4. `ITS Algorithm Layer`
   - sampling, voting, scoring, and search logic

5. `Reward Layer`
   - outcome and process reward models

6. `Gateway Integration Layer`
   - service/proxy integrations such as FastAPI IaaS and Envoy ext-proc

7. `Research / Tooling Layer`
   - examples, benchmarks, docs, and tests

### 6.2 Abstraction Levels

The repository family is easiest to reason about in these canonical layers. A concrete branch MAY
collapse or inline some of them:

1. `Canonical Public API Contracts`
   - `ChatMessage`, `ChatMessages`
   - `AbstractLanguageModel`
   - `AbstractScalingAlgorithm`, `AbstractScalingResult`
   - `AbstractOutcomeRewardModel`, `AbstractProcessRewardModel`
   - `AbstractOrchestrator` in the canonical `origin/v1` model

2. `Canonical Built-in Implementations`
   - `OpenAICompatibleLanguageModel`
   - `SelfConsistency`, `BestOfN`
   - `LMOrchestrator`, `LLMJudge` in `origin/v1`
   - experimental search algorithms where implemented

3. `Gateway Profiles`
   - current FastAPI IaaS service
   - Envoy ext-proc external processor

4. `Tooling and Validation`
   - examples, benchmarks, docs, tests

## 7. Core Domain Model

### 7.1 Message Model

#### 7.1.1 `ChatMessage`

`ChatMessage` is the normalized conversation unit used across the repository family.

Fields:

- `role`
  - supported values:
    - `system`
    - `user`
    - `assistant`
    - `tool`
- `content`
  - `str` for plain text
  - `list[dict]` for OpenAI-style multimodal content
  - `null` when the message is represented primarily by tool calls
- `tool_calls` (OPTIONAL list of dicts)
- `tool_call_id` (OPTIONAL string)

Required semantics:

- tool-call-bearing assistant messages MUST preserve `tool_calls` when returned by the LM path
- multimodal text extraction MUST preserve text parts in order
- image parts MAY be ignored by text-only flows
- in the `origin/v1` API, `ChatMessage.from_dict(...)` ignores unknown inbound fields

#### 7.1.2 `ChatMessages`

`ChatMessages` wraps either:

- a string prompt
- a list of `ChatMessage`
- another `ChatMessages`

Required behaviors:

- `from_prompt_or_messages(...)`
- `to_chat_messages()`
- `to_batch(size)`
- `to_prompt()`

Contract notes:

- `to_chat_messages()` is the primary structured representation
- `to_prompt()` remains important for backward compatibility and experimental algorithms

### 7.2 Language Model Interface

#### 7.2.1 `AbstractLanguageModel`

The normative `origin/v1` contract is async-first.

Legacy flat-layout branches and future ports MAY expose narrower concrete method signatures or
inline batching logic, but they SHOULD map cleanly onto this async-first contract at the
specification level.

Required method:

- `agenerate(messages, stop=None, **kwargs)`

Important optional method:

- `agenerate_single(messages, stop=None, **kwargs)`

Required semantics:

- LM adapters SHOULD preserve OpenAI-compatible assistant message dicts, including `tool_calls`
- LM adapters MAY accept provider-specific kwargs such as:
  - `max_tokens`
  - `temperature`
  - `tools`
  - `tool_choice`
  - `response_format`

### 7.3 Orchestrator Interface

#### 7.3.1 `AbstractOrchestrator`

`AbstractOrchestrator` is a first-class `origin/v1` abstraction.

Legacy branches MAY instead place fanout/concurrency behavior inside LM adapters or gateway-specific
wrappers rather than exposing a first-class orchestrator symbol.

Primary method:

- `agenerate(lm, messages_lst, stop=None, **kwargs)`

Sync wrapper:

- `generate(...)`

Responsibilities:

- fan out one logical ITS request into multiple LM calls
- preserve result ordering
- forward LM-specific parameters such as:
  - `temperature`
  - `max_tokens`
  - `tools`
  - `tool_choice`
  - `response_format`
- enforce implementation-defined concurrency control

### 7.4 Scaling Algorithm Interfaces

#### 7.4.1 `AbstractScalingAlgorithm`

Primary method:

- `ainfer(lm, prompt_or_messages, budget, return_response_only=True, tools=None, tool_choice=None)`

Sync wrapper:

- `infer(...)`

Contract:

- algorithms MUST accept string prompts or normalized chat messages
- algorithms MUST interpret `budget` according to documented semantics
- algorithms SHOULD preserve tool-call-bearing assistant messages when the underlying LM does
- algorithms MAY depend on an orchestrator explicitly or through built-in wiring

#### 7.4.2 `AbstractScalingResult`

Required property:

- `the_one`
  - returns the selected assistant response dict

### 7.5 Reward Model Interfaces

#### 7.5.1 `AbstractOutcomeRewardModel`

Scores final responses or complete conversations.

Normative `origin/v1` shape:

- `score(messages, **kwargs)`
- OPTIONAL async method:
  - `ascore(messages, orchestrator=None, **kwargs)`

Legacy flat-layout branches use an older prompt-plus-response shape in some implementations.

#### 7.5.2 `AbstractProcessRewardModel`

Scores intermediate reasoning steps.

Required methods:

- `score(prompt_or_messages, steps)`
- `ascore(prompt_or_messages, steps)`

### 7.6 `StepGeneration`

`StepGeneration` is a built-in helper used by experimental step-wise algorithms.

Key fields include:

- `max_steps`
- exactly one of:
  - `step_token`
  - `tokens_per_step`
- `stop_token`
- `temperature`
- `include_stop_str_in_output`
- `temperature_switch`

Invariants:

- exactly one of `step_token` and `tokens_per_step` MUST be set
- `tokens_per_step` MUST be positive when used

## 8. Built-in Repository Components

### 8.1 Stable `origin/v1` Surfaces

The stable repository-facing surfaces include:

- `AbstractLanguageModel`
- `AbstractOrchestrator`
- `AbstractScalingAlgorithm`
- `AbstractOutcomeRewardModel`
- `AbstractProcessRewardModel`
- `ChatMessage`, `ChatMessages`
- `SelfConsistency`
- `BestOfN`

### 8.2 Built-in LM / Orchestration Components

When the `lm` extra is installed on `origin/v1`, the repository provides:

- `OpenAICompatibleLanguageModel`
- `LMOrchestrator`
- `LLMJudge`
- `StepGeneration`

Important `origin/v1` notes:

- `OpenAICompatibleLanguageModel` forwards tools, tool choice, and structured-output arguments
- `LMOrchestrator` is the default parallel fanout implementation
- `LLMJudge` is the built-in outcome-judging implementation

### 8.3 Stable Algorithms

#### 8.3.1 `SelfConsistency`

Behavior:

- generate multiple candidate responses
- project responses into a comparison space
- select the most common projection, with implementation-defined tie-breaking

Current capabilities across the repository family:

- content-based voting
- regex-based projection
- tool-call voting modes:
  - `tool_name`
  - `tool_args`
  - `tool_hierarchical`

#### 8.3.2 `BestOfN`

Behavior:

- generate `N` candidates
- score them with an outcome reward model or judge
- select the highest-scoring candidate

Current `origin/v1` note:

- deduplication considers both content and tool-call-bearing structure

### 8.4 Experimental Components

Experimental repository surfaces may include:

- `BeamSearch`
- `ParticleFiltering`
- `ParticleGibbs`
- `EntropicParticleFiltering`
- `PlanningWrapper`
- local PRM integration

These are part of the repository family, but they are not part of the same stable contract as
`SelfConsistency` and `BestOfN`.

### 8.5 Compatibility Surface

`its_hub/algorithms/` remains a compatibility-oriented import surface.

In `origin/v1`, the canonical implementation home is `its_hub/core/algorithms/`.

In legacy flat-layout branches, `its_hub/algorithms/` still contains the live implementations.

## 9. Budget Semantics

`budget` is the repository-wide compute-control vocabulary for one ITS execution.

Current semantics by algorithm:

- `SelfConsistency`
  - number of candidate generations
- `BestOfN`
  - number of candidate generations to score
- `BeamSearch`
  - total search effort, constrained by beam width
- `ParticleFiltering` / related particle methods
  - number of particles
- `PlanningWrapper`
  - planning cost plus downstream execution budget

Important note:

- `budget` is shared vocabulary, not a universal identical implementation detail
- each algorithm MUST document how it interprets `budget`

## 10. Core Library Contracts

### 10.1 Async-First Semantics

The `origin/v1` core is async-first.

Required behavior:

- primary contracts are async
- sync methods are convenience wrappers over async behavior

### 10.2 Input Flexibility

Algorithms and reward adapters MUST accept:

- string prompts
- `list[ChatMessage]`
- `ChatMessages`

Normalization MUST occur before algorithm-specific logic.

### 10.3 Tool-Calling Preservation

When assistant messages include tool calls:

- LM adapters SHOULD preserve them in returned message dicts
- algorithms SHOULD keep them attached to candidate responses and selected results
- comparison or deduplication logic MAY intentionally project tool calls into reduced forms when
  that is part of the documented algorithm

### 10.4 Structured Output Support

The repository's LM and orchestrator contracts MAY forward provider-native structured-output
arguments, including `response_format`.

This is part of the extensible LM call surface, not a universal algorithm-level requirement.

### 10.5 Prompt Fallback in Experimental Paths

Structured chat is the primary contract, but some experimental algorithms still rely on
`ChatMessages.to_prompt()` in current implementations.

Therefore:

- structured chat is the primary contract
- exact chat-history fidelity is weaker in some experimental paths than in the stable core

## 11. Gateway Profiles

### 11.1 Gateway Role in `v1`

In `origin/v1`, a gateway is an integration profile layered over the library contract.

The repository-family contract currently includes two concrete gateway profiles:

1. `Current FastAPI IaaS profile`
   - implemented on the current branch
   - request-body-driven ITS

2. `Envoy ext-proc profile`
   - implemented on `origin/envoy_ext_proc`
   - header-triggered interception in front of an upstream OpenAI-compatible API

### 11.2 Current FastAPI IaaS Profile

#### 11.2.1 Role

The current branch implements an in-process FastAPI service in `its_hub/integration/iaas.py`.

This profile is a real repository-family gateway implementation, even though it is not the
normative `origin/v1` core deployment model.

#### 11.2.2 Entrypoints and Endpoints

Entrypoint:

- `its-iaas`

HTTP endpoints:

- `POST /configure`
- `GET /v1/models`
- `POST /v1/chat/completions`

#### 11.2.3 Runtime Model

The current IaaS profile uses process-global mutable state:

- `LM_DICT`
  - configured model registry
- `SCALING_ALG`
  - currently active algorithm instance

This means:

- configuration is in-memory
- service state is lost on restart
- the profile is not horizontally scalable without refactoring

#### 11.2.4 Configuration Contract

`POST /configure` selects the active algorithm and model wiring.

Current required/important fields include:

- `provider`
  - current values:
    - `openai`
    - `litellm`
- `endpoint`
- `api_key`
  - REQUIRED when `provider == "openai"`
- `model`
- `alg`
  - current supported values:
    - `self-consistency`
    - `best-of-n`
    - `particle-filtering`
- other current implementation fields include:
  - `extra_args`
  - `step_token`
  - `stop_token`
  - `rm_device`
  - `rm_agg_method`

Algorithm-specific current behavior:

- `self-consistency`
  - currently requires `regex_patterns`
  - supports:
    - `tool_vote`
    - `exclude_tool_args`
- `best-of-n`
  - requires `rm_name`
  - supports `rm_name == "llm-judge"` for LLM-judge scoring
  - when `rm_name == "llm-judge"`, the current implementation also accepts judge-specific fields
    such as:
    - `judge_model`
    - `judge_base_url`
    - `judge_criterion`
    - `judge_mode`
    - `judge_top_n`
    - `judge_api_key`
    - `judge_temperature`
    - `judge_max_tokens`
    - `enable_judge_logging`
- `particle-filtering`
  - requires reward-model configuration
  - accepts step-generation-related fields such as `step_token` and `stop_token`
  - currently combines request-supplied fields with some hardcoded defaults in
    `its_hub/integration/iaas.py`; exact tuning is implementation-defined in this profile

#### 11.2.5 Request Contract

`POST /v1/chat/completions` accepts:

- `model`
- `messages`
- `budget`
- `temperature` (OPTIONAL)
- `max_tokens` (OPTIONAL; currently accepted by the request schema but not yet forwarded into the
  algorithm call path)
- `stream` (OPTIONAL, currently unsupported)
- `tools` (OPTIONAL)
- `tool_choice` (OPTIONAL)
- `return_response_only`

ITS activation in this profile is request-body-driven:

- the client sends `budget` in the JSON request body
- no Envoy-style `X-ITS-*` header protocol is used

#### 11.2.6 Response Contract

The current IaaS profile returns an OpenAI-compatible top-level response shape with:

- `id`
- `object = "chat.completion"`
- `created`
- `model`
- `choices`
- `usage`
- `metadata` (OPTIONAL, when algorithm metadata is returned)

Current limitations:

- streaming returns `501 Not Implemented`
- token usage is currently hardcoded to zero
- response metadata is algorithm-specific rather than standardized across all algorithms

### 11.3 Envoy ext-proc Profile

#### 11.3.1 Role

The `origin/envoy_ext_proc` branch implements an Envoy external-processor profile that intercepts
OpenAI-compatible requests and short-circuits them when ITS applies.

#### 11.3.2 Components

This profile consists of:

1. `Envoy HTTP Proxy`
   - front-door listener and routing layer

2. `External Processor Service`
   - gRPC service implementing Envoy ext-proc callbacks

3. `ITS Orchestrator`
   - request-scoped config plus long-lived LM client cache

4. `Provider Adapter`
   - OpenAI-compatible downstream LM client

5. `Response Shaper`
   - emits OpenAI-compatible response JSON

#### 11.3.3 Interception Target

The intended interception target is OpenAI-compatible chat-completions traffic.

The current Python implementation recognizes:

- requests whose `:path` starts with `/v1/chat/completions`

Requests to other paths pass through untouched.

#### 11.3.4 ITS Activation Headers

ITS activation is currently driven by:

- `X-ITS-Budget`
  - integer range `1..1000`
- `X-ITS-Endpoint`
- `X-ITS-API-Key` (OPTIONAL)

Current branch-specific note:

- per-request algorithm selection is not yet standardized
- invalid or out-of-range ITS header values fall back to pass-through behavior
- the ext-proc branch currently uses a fixed self-consistency policy with default content-based
  voting rather than per-request algorithm/tool-vote configuration

#### 11.3.5 Request-Scoped Config

The ext-proc profile constructs a request-scoped config equivalent to:

- `budget`
- `api_endpoint`
- `model`
- `api_key` (OPTIONAL)

The branch names this structure `ITSRequestConfig`.

Current implementation caveat:

- the long-lived LM client cache is keyed by `(api_endpoint, model)`, not by `api_key`
- therefore per-request API keys are parsed as request inputs, but they are not fully isolated when
  multiple requests reuse the same endpoint/model pair

#### 11.3.6 Request Resolution

For an intercepted request:

1. inspect request path and headers
2. if path does not start with `/v1/chat/completions` -> `pass_through`
3. if required ITS headers are absent or invalid -> `pass_through`
4. buffer and parse the request body
5. read:
   - `model`
   - `messages`
   - `tools`
   - `tool_choice`
6. if `model` is absent -> `pass_through`
7. run the fixed self-consistency orchestrator
8. return either:
   - immediate ITS response
   - or pass-through behavior on failure

#### 11.3.7 Pass-Through and Short-Circuit Semantics

The profile supports two outcomes:

1. `pass_through`
   - the original request continues upstream

2. `its_applied`
   - the gateway runs ITS
   - the gateway returns an OpenAI-compatible response directly
   - the original upstream request is short-circuited

#### 11.3.8 Response Shape

When ITS is applied, the response MUST be OpenAI-compatible and include:

- `id`
- `object = "chat.completion"`
- `created`
- `model`
- `choices`
  - `choices[0].message` is the selected assistant message
  - `choices[0].finish_reason` is typically `stop`
- `usage`

Additional profile behavior:

- the response sets `x-its-applied: true`
- usage is aggregated across the LM calls used by ITS

#### 11.3.9 Failure Model

The current Envoy profile favors safe fallback:

- invalid or incomplete ITS activation headers -> pass through
- non-chat-completions routes -> pass through
- missing `model` in request body -> pass through
- ITS processing failure after activation -> pass through

### 11.4 Streaming

This specification version does not standardize streaming ITS behavior.

Current profile notes:

- current FastAPI IaaS profile:
  - explicit `501 Not Implemented`
- Envoy ext-proc profile:
  - request bodies are buffered for interception
  - response streaming semantics are not standardized here

## 12. Documentation Surface

The repository-family documentation surface MAY include the documents below. Not every branch is
required to contain every document at the same path:

- `README.md`
  - top-level orientation and installation matrix
- `docs/installation.md`
  - install profiles and dependency expectations
- `docs/quick-start.md`
  - common usage flows
- `docs/algorithms.md`
  - algorithm-facing guidance
- `docs/orchestration.md` (`origin/v1`)
  - orchestrator contract and built-in orchestrator behavior
- `docs/benchmarking.md`
  - benchmark workflow
- `docs/development.md`
  - contributor guidance and architecture notes
- `docs/PLANNING_WRAPPER.md`
  - planning-wrapper behavior
- `docs/iaas-service.md`
  - current FastAPI IaaS profile and related gateway usage
- `its_hub/integration/iaas.md`
  - migration/architecture document for the Envoy proxy direction on legacy flat-layout Python
    branches
- `docs/usage.md`
  - optional user-facing usage guidance on some branches, including `origin/envoy_ext_proc`

Documentation is descriptive and user-facing.

This specification is normative for the intended repository contract, but branch-specific code
remains the source of truth for exact runtime behavior.

## 13. Research, Benchmarks, and Examples

### 13.1 Benchmark Surface

The benchmark surface includes dataset-driven evaluation entrypoints under:

- `benchmarking/` on `origin/v1`
- `scripts/benchmark.py` on legacy flat-layout branches

Current benchmark datasets include:

- MATH-500
- AIME-2024

### 13.2 Examples

The repository family includes runnable examples and notebook-like artifacts, typically under
`examples/`, `scripts/`, `notebooks/`, or equivalent branch-specific locations, for:

- self-consistency usage
- math-oriented reasoning flows
- experimental reasoning/search workflows where installed

## 14. Development, Testing, and Quality

### 14.1 Development Setup

The repository supports:

- `uv sync --extra dev`
- `pip install -e ".[dev]"`

Python expectations:

- `origin/v1`: Python `>= 3.11`
- current branch / `origin/envoy_ext_proc`: Python `>= 3.10`

### 14.2 Test Surface

Across the repository family, tests and validation assets MAY cover behaviors such as:

- message and normalization behavior
- orchestrator behavior and concurrency semantics
- stable algorithms
- tool-calling behavior
- judge behavior
- planning wrapper / experimental behavior where present
- gateway-specific behavior for the IaaS profile and, where present, Envoy-related flows

Concrete coverage differs by branch. For example, current Python coverage is centered on the core
algorithms and the FastAPI IaaS profile, while the Envoy gateway profile currently also relies on
branch-specific deployment artifacts and script-level validation.

### 14.3 Quality Tooling

Current repository quality tooling includes:

- `pytest`
- `ruff`

## 15. Validation Matrix

A conforming implementation or faithful port SHOULD include tests for the behaviors below.

### 15.1 Message and Input Model

- string prompts normalize into a single `user` message
- chat histories preserve role/content/tool-call fields
- multimodal text extraction behaves consistently
- tool-call-bearing assistant messages preserve tool-call structures

### 15.2 LM and Orchestrator Contracts

- LM adapters return one response per logical request
- orchestrators preserve input ordering
- concurrency limits behave as documented
- tools, tool choice, and `response_format` are forwarded correctly

### 15.3 Stable Algorithms

- self-consistency selects repeated candidates correctly
- tool-voting modes behave as documented
- best-of-n uses outcome scoring correctly
- result objects expose `the_one`

### 15.4 Experimental Algorithms

If experimental algorithms are implemented or installed:

- beam search uses step generation and beam width correctly
- particle methods handle budget semantics correctly
- prompt-string fallback behavior is documented

### 15.5 Current FastAPI IaaS Profile

- `/configure` validates supported algorithm and provider shapes
- `/v1/models` reflects configured models
- `/v1/chat/completions` requires configured model state
- request-body `budget` drives ITS execution
- tool calls are preserved when produced by the selected algorithm path
- streaming returns `501 Not Implemented`
- usage behavior matches the current zero-token implementation or later documented replacement

### 15.6 Envoy ext-proc Profile

- requests without ITS activation pass through
- requests with valid ITS activation may be short-circuited
- missing-model behavior matches documented fallback
- OpenAI-compatible response shaping is correct
- `x-its-applied: true` is present on ITS responses
- aggregated usage behavior matches implementation

## 16. Implementation Checklist

### 16.1 Core Repository Checklist

- message normalization layer
- abstract language-model contract
- abstract orchestrator contract
- abstract ITS algorithm contract
- abstract reward-model contracts
- stable algorithms (`SelfConsistency`, `BestOfN`)
- docs and tests for supported core surfaces

### 16.2 Optional Built-in Implementation Checklist

- `OpenAICompatibleLanguageModel`
- `LMOrchestrator`
- `LLMJudge`
- `StepGeneration`
- experimental algorithms behind the documented optional profiles

### 16.3 Gateway Checklist

For the current FastAPI IaaS profile:

- `/configure` runtime setup
- `/v1/models` model enumeration
- OpenAI-compatible chat-completions response shaping
- request-body `budget` handling
- documented non-streaming behavior

For the Envoy ext-proc profile:

- request-scoped ITS config parsing
- header-driven ITS activation
- pass-through fallback behavior
- OpenAI-compatible response shaping
- documented usage accounting mode
- documented non-streaming behavior

## Appendix A. Non-Normative Notes

1. `Repository center of gravity`
   - `origin/v1` is the design baseline for the core library surface.

2. `Why this spec documents multiple branch states`
   - The repository family currently has a split between the `origin/v1` library refactor and
     gateway implementations that still live on flat-layout branches.
   - This specification deliberately keeps the core contract and the gateway profiles separate so
     future design changes can be made in `SPEC.md` first.

3. `Current gateway inventory`
   - The current branch still has a concrete FastAPI IaaS service.
   - `origin/envoy_ext_proc` has a concrete Envoy external-processing gateway profile.

4. `Migration direction`
   - The long-term design direction is library-first with gateway runtimes treated as integrations,
     not as the center of the repository contract.
