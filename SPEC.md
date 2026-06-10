# its_hub Repository Specification

Status: Draft v1 (language-agnostic repository contract)

Purpose: Define the repository-level contract for `its_hub` as a library-first system for
inference-time scaling (ITS) of LLMs, including:

- the core Python package and its abstractions
- built-in algorithms, model adapters, and reward-model integrations
- the repository's benchmarking and example tooling
- the gateway/serving contract used by both direct-service and intercepting implementations

This specification is intentionally broader than the gateway. The gateway is one part of the
repository and is specified here as a subsystem.

## Normative Language

The key words `MUST`, `MUST NOT`, `REQUIRED`, `SHOULD`, `SHOULD NOT`, `RECOMMENDED`, `MAY`, and
`OPTIONAL` in this document are to be interpreted as described in RFC 2119.

`Implementation-defined` means the behavior is part of the implementation contract, but this
specification does not prescribe one universal policy. Implementations MUST document the selected
behavior.

## 1. Problem Statement

`its_hub` is a repository for runtime techniques that improve LLM outputs by spending more compute
during inference rather than by retraining the model. It is especially oriented toward:

- mathematical reasoning
- multi-step reasoning
- tool-calling reliability
- benchmarking the quality/computation trade-off across ITS algorithms

The repository is not only a Python library. It also contains:

- provider adapters for OpenAI-compatible and LiteLLM-backed inference
- reward-model integrations
- a serving/gateway layer for OpenAI-compatible chat completions
- benchmark and example scripts
- repository documentation describing installation, usage, and development workflows

Important boundary:

- `its_hub` is a runtime orchestration library and accompanying tooling.
- It is not a training framework.
- It does not require one model vendor, one reward-model runtime, or one gateway framework.
- The repository's public behavior is defined by abstractions and contracts, not by one specific web
  stack or proxy runtime.

## 2. Goals and Non-Goals

### 2.1 Goals

- Define clean interfaces between language models, ITS algorithms, and reward models.
- Support both string-prompt and chat-message inputs, including tool-calling flows.
- Support sync and async execution from the same conceptual algorithm surface.
- Preserve OpenAI-compatible message structures where possible.
- Support multiple ITS strategies with a shared `budget` concept.
- Support both outcome-based and process-based scoring.
- Expose an OpenAI-compatible gateway contract for serving ITS.
- Provide benchmarking and example tooling for math-focused evaluation.
- Keep the repository portable enough that future implementations in other languages can follow the
  same conceptual contract.

### 2.2 Non-Goals

- Defining the full OpenAI Chat Completions protocol.
- Standardizing one deployment topology for gateway/service implementations.
- Requiring streaming ITS responses in this version.
- Requiring one persistent configuration store or control plane.
- Requiring all exported classes to be equally stable or production-ready.
- Defining training, fine-tuning, or model-hosting internals.

## 3. Repository Overview

### 3.1 Deliverables

The repository contains six main deliverables:

1. `Core Package`
   - The `its_hub` Python package.
   - Defines abstractions, implementations, and utilities for inference-time scaling.

2. `Integrations`
   - Reward-model adapters.
   - Gateway/serving implementations and related architecture.

3. `Scripts`
   - Benchmark tooling.
   - Example scripts for manual evaluation and local testing.

4. `Documentation`
   - Installation, quick-start, algorithm, gateway, benchmarking, and development guides.

5. `Tests`
   - Unit and integration-style tests for algorithms, language models, tooling, and wrappers.

6. `Packaging Metadata`
   - Project metadata, extras, CLI entrypoints, and dependency profiles.

### 3.2 Canonical Repository Layout

The repository is organized around these top-level surfaces:

```text
its_hub/
  base.py
  types.py
  lms.py
  utils.py
  error_handling.py
  algorithms/
  integration/
docs/
scripts/
tests/
README.md
pyproject.toml
SPEC.md
```

Logical responsibilities:

- `its_hub/base.py`
  - abstract interfaces
- `its_hub/types.py`
  - normalized message and tool-call types
- `its_hub/lms.py`
  - language-model adapters and stepwise generation helpers
- `its_hub/algorithms/`
  - ITS algorithm implementations
- `its_hub/integration/`
  - reward-model integration and serving/gateway integration
- `its_hub/utils.py`
  - prompts and helper utilities
- `docs/`
  - user-facing documentation
- `scripts/`
  - research, benchmarking, and example entrypoints
- `tests/`
  - validation of the above surfaces

### 3.3 Distribution and Installation Profiles

The repository defines multiple installation profiles through package extras.

Current profiles:

- `core`
  - installed by `pip install its_hub`
  - intended for self-consistency, best-of-n, OpenAI-compatible inference, and LLM-judge usage
- `prm`
  - installed by `pip install its_hub[prm]`
  - adds process-reward-model dependencies for stepwise reasoning algorithms
- `cloud`
  - installed by `pip install its_hub[cloud]`
  - adds cloud-provider SDK support such as Bedrock and Vertex AI
- `research`
  - installed by `pip install its_hub[research]`
  - adds benchmark/evaluation dependencies
- `dev`
  - installed by `pip install -e ".[dev]"` or `uv sync --extra dev`
  - adds testing, linting, notebook, and contributor tooling

### 3.4 CLI Entrypoints

The repository currently defines this package CLI entrypoint in the main branch:

- `its-iaas`
  - starts the direct HTTP gateway/service profile

Additional gateway entrypoints MAY exist in other implementation profiles or active development
branches. This specification defines the gateway contract independently of one entrypoint name.

## 4. System Overview

### 4.1 Main Components

1. `Message Model`
   - Represents prompts, chat history, tool calls, and normalized conversation batches.

2. `Language Model Adapter Layer`
   - Adapts downstream LLM APIs to a shared generation interface.

3. `ITS Algorithm Layer`
   - Implements sampling, voting, search, and selection logic.

4. `Reward Layer`
   - Scores either final answers or intermediate reasoning steps.

5. `Integration Layer`
   - Bridges the core package to external systems such as reward runtimes and gateways.

6. `Gateway Layer`
   - Exposes ITS as an OpenAI-compatible serving surface.

7. `Research / Tooling Layer`
   - Benchmarks algorithms on standardized datasets and provides example scripts.

8. `Documentation and Quality Layer`
   - Explains usage, tests behavior, and enforces repository quality.

### 4.2 Abstraction Levels

The repository is easiest to reason about when kept in these layers:

1. `Input Normalization Layer`
   - `ChatMessage`, `ChatMessages`

2. `Provider Layer`
   - `AbstractLanguageModel` and concrete model adapters

3. `Algorithm Layer`
   - `AbstractScalingAlgorithm`, `AbstractScalingResult`, concrete ITS implementations

4. `Scoring Layer`
   - `AbstractOutcomeRewardModel`, `AbstractProcessRewardModel`, concrete adapters

5. `Serving Layer`
   - configuration and transport surfaces that turn ITS into an API

6. `Evaluation Layer`
   - benchmark datasets, evaluation logic, and research scripts

### 4.3 Portability Boundary

The repository is implemented in Python, but the conceptual architecture is portable because:

- algorithms depend on abstract model and reward interfaces
- chat/message normalization is explicit
- the gateway is described in terms of transport-agnostic execution config
- the benchmark layer depends on algorithm behavior rather than one server implementation

## 5. Core Domain Model

### 5.1 Message Model

#### 5.1.1 ChatMessage

`ChatMessage` is the normalized conversation unit used across the repository.

Fields:

- `role` (enum/string)
  - current supported values:
    - `system`
    - `user`
    - `assistant`
    - `tool`
- `content` (string, list of content parts, or null)
  - string for plain text
  - list of dicts for OpenAI-style multimodal content
  - null when the message is represented primarily by tool calls
- `tool_calls` (OPTIONAL list of dicts)
- `tool_call_id` (OPTIONAL string)

Semantics:

- The repository MUST preserve tool calls in assistant messages when they are part of the selected
  response.
- Multimodal content is represented structurally, but most current reasoning paths extract and use
  text content only.

#### 5.1.2 ChatMessages

`ChatMessages` is a wrapper that normalizes either:

- a string prompt
- a list of `ChatMessage`
- another `ChatMessages`

into one internal representation.

Required behaviors:

- `from_prompt_or_messages(...)`
  - normalize strings, message lists, or wrapper instances
- `to_prompt()`
  - produce a string representation of the conversation
- `to_chat_messages()`
  - produce a list of normalized messages
- `to_batch(size)`
  - duplicate one logical request into `size` parallel candidate requests

Important compatibility note:

- String prompts are normalized as a single `user` message.
- `to_prompt()` exists for compatibility and utility flows. It is not the preferred representation
  for chat-native or tool-native execution paths.

#### 5.1.3 Text Extraction

For multimodal message content:

- text parts are concatenated for text-only reasoning paths
- image parts may be ignored by text-only internal flows
- unsupported content part types MAY raise an error

This behavior is current repository behavior, not a universal multimodal standard.

### 5.2 Language Model Interfaces

#### 5.2.1 AbstractLanguageModel

`AbstractLanguageModel` is the base contract for downstream model adapters.

Required methods:

- `agenerate(messages, stop=None, ...)`
- `generate(messages, stop=None, ...)`

Optional method:

- `evaluate(prompt, generation)`

Expected semantics:

- input may be one conversation or a batch of conversations
- output may be one response or a list of responses
- the response shape SHOULD preserve OpenAI-compatible assistant message structure when available

### 5.3 Scaling Algorithm Interfaces

#### 5.3.1 AbstractScalingAlgorithm

`AbstractScalingAlgorithm` is the base contract for ITS algorithms.

Primary method:

- `ainfer(lm, prompt_or_messages, budget, return_response_only=True, tools=None, tool_choice=None)`

Sync wrapper:

- `infer(...)`
  - conceptually wraps the async implementation

Contract:

- algorithms MUST accept either string prompts or normalized chat messages
- algorithms MUST interpret `budget` according to their documented semantics
- algorithms MAY support tool-calling inputs by forwarding `tools` and `tool_choice`

#### 5.3.2 AbstractScalingResult

`AbstractScalingResult` defines the result-object surface for algorithms that return structured
state instead of only the selected message.

Required property:

- `the_one`
  - returns the selected response

### 5.4 Reward Model Interfaces

#### 5.4.1 AbstractOutcomeRewardModel

Scores final responses.

Required methods:

- `ascore(prompt_or_messages, response)`
- `score(prompt_or_messages, response)`

#### 5.4.2 AbstractProcessRewardModel

Scores intermediate reasoning steps.

Required methods:

- `ascore(prompt_or_messages, steps)`
- `score(prompt_or_messages, steps)`

### 5.5 StepGeneration

`StepGeneration` is the repository's step-wise generation helper for algorithms that build or score
responses incrementally.

Key fields:

- `max_steps`
- `step_token` or `tokens_per_step`
- `stop_token`
- `temperature`
- `include_stop_str_in_output`
- `temperature_switch`

Invariants:

- exactly one of `step_token` and `tokens_per_step` MUST be set
- `tokens_per_step` MUST be positive when used

Behavior:

- generates one or many next-step continuations
- reconstructs current assistant reasoning state from prior steps
- stops when `max_steps` or `stop_token` conditions are met

## 6. Core Library Contracts

### 6.1 Sync and Async Semantics

The repository is async-first conceptually.

Required behavior:

- concrete implementations SHOULD treat async methods as the primary execution path
- sync methods MAY be thin wrappers around async implementations

Caller guidance:

- async callers SHOULD use async methods directly
- sync wrappers are a convenience layer and may rely on `asyncio.run()`

### 6.2 Input Flexibility

Algorithms and reward adapters MUST accept:

- string prompts
- `list[ChatMessage]`
- `ChatMessages`

Normalization MUST occur before algorithm-specific logic.

### 6.3 Tool-Calling Preservation

When assistant messages include tool calls:

- tool calls MUST remain attached to message objects during candidate generation and selection
- algorithms MUST NOT flatten tool calls into plain text unless that behavior is explicitly part of
  a documented projection or comparison strategy

### 6.4 Provider Adapter Expectations

A concrete language-model adapter SHOULD provide:

- batched generation
- retry handling for transient provider failures
- concurrency control where supported
- propagation of request features such as:
  - stop conditions
  - max tokens
  - temperature
  - tools
  - tool choice

### 6.5 Error Handling Boundary

Provider adapters are responsible for:

- transport interaction with downstream providers
- retry classification where implemented
- mapping malformed or non-success upstream behavior into repository-level errors

Algorithms are responsible for:

- handling candidate sets and selection logic
- surfacing algorithm-level failures when valid selection is impossible

## 7. Built-in Library Components

### 7.1 Language Model Implementations

#### 7.1.1 OpenAICompatibleLanguageModel

Primary documented provider adapter for:

- OpenAI-compatible APIs
- local vLLM-style chat-completion endpoints

Current repository behaviors include:

- OpenAI-compatible request shaping
- support for system prompts
- retry/backoff and error parsing
- concurrency control for async generation
- support for tool definitions and tool choice
- configurable SSL verification behavior

#### 7.1.2 LiteLLMLanguageModel

Adapter for provider access through LiteLLM.

Purpose:

- unify multiple downstream providers behind a common generation surface
- enable cloud/provider routing without changing algorithm code

#### 7.1.3 Other Model Classes

`its_hub/lms.py` also contains additional classes such as:

- `LocalVLLMLanguageModel`
- `TransformersLanguageModel`

Current note:

- the repository's primary documented and actively described model surfaces are
  `OpenAICompatibleLanguageModel` and `LiteLLMLanguageModel`
- other classes MAY be experimental, utility, or less central to the public contract

### 7.2 Utility Prompts and Helpers

The repository includes prompt and helper utilities in `its_hub/utils.py`.

Important current examples:

- `SAL_STEP_BY_STEP_SYSTEM_PROMPT`
- `QWEN_SYSTEM_PROMPT`

These are repository utilities for reasoning workflows, especially math tasks. They are not protocol
requirements for algorithm correctness.

### 7.3 Reward Model Integrations

#### 7.3.1 LocalVllmProcessRewardModel

Adapter around `reward_hub` process-reward-model support.

Purpose:

- score step-wise reasoning trajectories
- support algorithms such as beam search and particle filtering

Current dependencies:

- local reward runtime via `reward_hub`
- device selection
- aggregation method selection

#### 7.3.2 LLMJudgeRewardModel

Adapter around `reward_hub` LLM-judge support.

Purpose:

- score complete candidate responses
- enable best-of-n without a local process reward model

Current judge concepts:

- `pointwise`
- `groupwise`

### 7.4 Algorithm Implementations

#### 7.4.1 SelfConsistency

Behavior:

- generate multiple candidate responses
- project each candidate into a comparison space
- select the most common projection, with implementation-defined tie-breaking

Supported comparison modes:

- default exact-content projection
- regex-based projection
- tool-call voting modes:
  - `tool_name`
  - `tool_args`
  - `tool_hierarchical`

Current use cases:

- tool-calling consistency
- math answers with structured final-answer extraction

#### 7.4.2 BestOfN

Behavior:

- generate `N` candidates
- score them with an outcome reward model or LLM judge
- select the highest-scoring candidate

Current use cases:

- quality-focused selection with a judge/scorer
- cloud-API scenarios that do not require a local PRM

#### 7.4.3 BeamSearch

Behavior:

- perform step-wise generation with fixed beam width
- score partial trajectories using a process reward model

Current use cases:

- structured multi-step reasoning
- problems where partial trajectories are meaningful and scoreable

#### 7.4.4 ParticleGibbs Family

The repository includes a particle-based family in `its_hub/algorithms/particle_gibbs.py`.

Important public classes:

- `ParticleGibbs`
- `ParticleFiltering`
- `EntropicParticleFiltering`

Shared conceptual behavior:

- maintain multiple reasoning trajectories
- resample or retain trajectories according to algorithm-specific selection rules
- use step-wise scoring for long-horizon reasoning

Current emphasis in docs and examples is on:

- `ParticleFiltering`
- `EntropicParticleFiltering`

#### 7.4.5 PlanningWrapper

`PlanningWrapper` is a repository utility that adds a planning phase to another ITS algorithm.

Behavior:

- use one generation to propose several approaches
- allocate remaining budget across approaches
- run the wrapped algorithm for each approach
- select the best approach/result using result-level heuristics

Current note:

- this is part of the repository surface
- it is a wrapper utility rather than one of the foundational ITS primitives

### 7.5 Result Object Semantics

Algorithm result objects generally:

- preserve candidate responses or trajectories
- expose the selected response through `the_one`
- MAY include algorithm-specific fields such as:
  - response counts
  - scores
  - selected index
  - path/particle state
  - planning metadata

### 7.6 Experimental or Incomplete Surfaces

`its_hub/algorithms/__init__.py` currently exports `MetropolisHastings`, but the implementation is
not yet complete and raises `NotImplementedError`.

Therefore:

- it is part of the repository surface area
- it is not a required stable algorithm contract in this specification version

## 8. Budget Semantics

`budget` is the repository-wide compute control for one ITS execution.

Current semantics by algorithm:

- `SelfConsistency`
  - number of parallel generations to compare
- `BestOfN`
  - number of candidate responses to score
- `BeamSearch`
  - total search effort; commonly interpreted as `beam_width * effective_depth`
- `ParticleFiltering`
  - number of particles maintained
- `EntropicParticleFiltering`
  - number of particles maintained
- `PlanningWrapper`
  - spends one unit on planning, then distributes the remainder across approaches

Important note:

- `budget` is shared vocabulary, not a universal identical implementation detail.
- Each algorithm MUST document how it interprets `budget`.

## 9. Integration Layer

### 9.1 reward_hub Integration Contract

The repository integrates with `reward_hub` rather than re-implementing all reward-model logic.

Required repository behavior:

- adapt reward-hub interfaces to the abstract outcome/process reward model contracts
- normalize prompt/message inputs into the format expected by reward-hub clients
- preserve algorithm-facing interfaces regardless of reward-hub internals

### 9.2 Serving and Gateway Integration

The repository includes a serving layer that turns ITS into an OpenAI-compatible chat-completions
surface.

This serving layer is specified in Section 10 as a subsystem of the repository, not as the entire
repository contract.

## 10. Gateway Specification

### 10.1 Gateway Role

The gateway exposes inference-time scaling behind an OpenAI-compatible chat-completions interface.

It does more than proxy one upstream request. It may:

- make multiple downstream model calls
- run voting or reward-based selection
- aggregate usage across multiple calls
- return either a selected final response or pass the original request through

### 10.2 Gateway Profiles

This specification defines two gateway profiles.

#### 10.2.1 Configured Gateway Service Profile

A direct HTTP service that:

- exposes OpenAI-compatible endpoints itself
- accepts runtime ITS configuration through a management endpoint
- stores effective policy in process-local state
- applies ITS directly inside `/v1/chat/completions`

This profile matches the current direct-service implementation style.

#### 10.2.2 Intercepting Gateway Profile

A proxy-integrated or callback-based service that:

- inspects requests headed to an upstream OpenAI-compatible chat endpoint
- decides per request whether ITS should be applied
- either returns a final ITS response immediately or lets the original request continue upstream

This profile matches the current proxy/ext-proc implementation style.

### 10.3 Gateway Components

1. `Transport Adapter`
   - HTTP handler, proxy hook, ext-proc handler, or equivalent

2. `Configuration Resolver`
   - resolves profile-specific inputs into one normalized execution config

3. `Model / Provider Registry`
   - OPTIONAL depending on profile
   - stores long-lived provider registrations

4. `ITS Orchestrator`
   - executes one normalized ITS request

5. `Algorithm Registry`
   - resolves algorithm names to implementations

6. `Provider Adapter`
   - performs downstream OpenAI-compatible or provider-backed calls

7. `Response Shaper`
   - emits an OpenAI-compatible response envelope

8. `Observability`
   - logs and OPTIONAL metrics/traces

### 10.4 Gateway ExecutionConfig

The gateway MUST normalize serving inputs into a request-scoped `ExecutionConfig`.

Fields:

- `model` (string)
- `algorithm` (string)
- `budget` (integer, `1..1000`)
- `provider` (object)
  - `kind`
  - `endpoint`
  - `api_key`
  - `extra_args` (OPTIONAL)
- `algorithm_config` (object or null)
- `reward_config` (object or null)
- `response_mode`
  - `selected_message_only`
  - `selected_message_with_metadata`

### 10.5 Base Protocol Contract

The OpenAI-compatible Chat Completions API is the source of truth for the base wire schema.

This repository specification defines only the ITS-related extensions and handling rules.

#### 10.5.1 Required ITS Response Shape

When the gateway returns an ITS-generated response, it MUST be OpenAI-compatible and include:

- `id`
- `object = "chat.completion"`
- `created`
- `model`
- `choices`
  - `choices[0].message` is the selected assistant message
  - `choices[0].finish_reason` is implementation-defined but typically `stop`
- `usage`

Tool-call preservation:

- if the selected response is a tool call, the tool-call structure MUST remain present in
  `choices[0].message`

#### 10.5.2 Gateway Extensions

Current standardized repository extensions:

- configured-service request body fields:
  - `budget`
  - `return_response_only`
- configured-service response body field:
  - `metadata`
- intercepting-profile request headers:
  - `X-ITS-Budget`
  - `X-ITS-Endpoint`
  - `X-ITS-API-Key` (OPTIONAL)
- intercepting-profile diagnostic response header:
  - `x-its-applied: true` (OPTIONAL)

#### 10.5.3 Streaming

This specification version standardizes non-streaming ITS behavior only.

- handling of `stream=true` is implementation-defined
- implementations MUST document whether they reject, ignore, downgrade, or pass through streaming
  requests

### 10.6 Configured Gateway Service Profile

#### 10.6.1 Required Endpoints

Required endpoints:

1. `POST /configure`
2. `POST /v1/chat/completions`

Optional but expected endpoint:

3. `GET /v1/models`

#### 10.6.2 Configuration Payload

The configuration surface MUST support these core fields:

- `provider`
- `endpoint`
- `api_key`
- `model`
- `alg`

It MAY also support algorithm-specific fields such as:

- self-consistency:
  - `regex_patterns`
  - `tool_vote`
  - `exclude_tool_args`
- step-wise / PRM algorithms:
  - `step_token`
  - `stop_token`
  - `rm_name`
  - `rm_device`
  - `rm_agg_method`
- LLM judge:
  - `judge_model`
  - `judge_base_url`
  - `judge_criterion`
  - `judge_mode`
  - `judge_top_n`
  - `judge_api_key`
  - `judge_temperature`
  - `judge_max_tokens`
  - `enable_judge_logging`
- provider-specific:
  - `extra_args`

#### 10.6.3 Effective Behavior

On successful configuration:

- the gateway registers or updates a model/provider mapping
- the gateway replaces the active ITS policy for subsequent requests
- the effective configuration is local to the process and not required to persist across restart

Important current repository constraint:

- the configured-service profile maintains a model registry but only one active ITS algorithm policy
  at a time

#### 10.6.4 Request Resolution

For `POST /v1/chat/completions`, execution config is resolved from:

1. request body:
   - `model`
   - `messages`
   - `budget`
   - `tools`
   - `tool_choice`
   - `return_response_only`
2. active gateway policy:
   - algorithm
   - provider mapping
   - reward config
   - algorithm-specific config

Expected error classes:

- unknown model
- missing active service configuration
- invalid `budget`
- empty `messages`

### 10.7 Intercepting Gateway Profile

#### 10.7.1 Interception Target

The intercepting profile MUST at minimum recognize:

- `POST /v1/chat/completions`

Requests to other paths MAY be passed through untouched.

#### 10.7.2 ITS Activation Headers

In this specification version, ITS activation is driven by:

- `X-ITS-Budget`
- `X-ITS-Endpoint`
- `X-ITS-API-Key` (OPTIONAL)

Current note:

- per-request algorithm selection is not standardized in the intercepting profile in this version
- current prototype behavior uses a fixed/default self-consistency policy once ITS is activated

#### 10.7.3 Request Resolution

For an intercepted request:

1. parse headers
2. if required ITS headers are absent or invalid -> `pass_through`
3. otherwise parse the request body
4. read:
   - `model`
   - `messages`
   - `tools`
   - `tool_choice`
5. resolve the profile's default/fixed algorithm policy
6. build `ExecutionConfig`

If `model` is absent from the request body, the gateway MAY:

- pass through
- or return an explicit error

The implementation MUST document which behavior it chooses.

#### 10.7.4 Pass-Through and Short-Circuit Semantics

The intercepting profile supports two outcomes:

1. `pass_through`
   - the original request continues upstream

2. `its_applied`
   - the gateway runs ITS
   - the gateway returns an OpenAI-compatible response directly
   - the original upstream call is short-circuited

### 10.8 Gateway Orchestration Contract

For one ITS-applied request, the gateway orchestrator MUST:

1. validate the normalized execution config
2. build or reuse a downstream provider client
3. build or reuse any required reward/judge client
4. normalize messages into the algorithm's expected input representation
5. run the selected algorithm with:
   - provider client
   - messages
   - budget
   - tools
   - tool choice
6. select the final assistant message
7. aggregate or synthesize usage
8. shape the final OpenAI-compatible response

### 10.9 Usage Accounting

Current repository family supports two usage-accounting modes:

1. `aggregated_actual`
   - usage reflects the sum of all downstream calls used by ITS

2. `placeholder`
   - usage fields are present but populated with placeholder values such as zero

Implementations MUST document which mode they use.

### 10.10 Gateway Failure Model

Failure classes:

1. `Request Validation Failures`
   - empty `messages`
   - invalid `budget`
   - missing `model`

2. `Gateway Configuration Failures`
   - unknown model registration
   - missing active policy
   - unsupported algorithm

3. `Provider Failures`
   - transport failure
   - non-success upstream status
   - malformed payload

4. `Reward Failures`
   - missing reward runtime
   - judge call failure
   - scorer integration failure

5. `Algorithm Execution Failures`
   - selection failure
   - invalid candidate set after filtering

Configured-service profile:

- SHOULD return explicit errors for invalid or unconfigured requests

Intercepting profile:

- MUST support pass-through when ITS is not activated
- MUST document whether activated ITS failures pass through, return explicit errors, or vary by
  error class

## 11. Research, Benchmarking, and Example Tooling

### 11.1 Benchmark Script

`scripts/benchmark.py` is the repository's primary research/benchmark entrypoint.

Current responsibilities:

- load benchmark datasets
- instantiate a selected algorithm and model adapter
- run across one or more budgets
- optionally evaluate results
- save and display outputs

Current benchmark datasets include:

- MATH-500
- AIME-2024

Current benchmark-facing algorithm set includes:

- self-consistency
- beam-search
- particle-filtering
- entropic-particle-filtering

### 11.2 Example Script

`scripts/test_math_example.py` is a runnable example showing:

- local OpenAI-compatible/vLLM inference
- step generation
- a process reward model
- particle filtering on math prompts

### 11.3 Research Dependencies

The `research` extra adds repository tooling for:

- dataset loading
- answer verification
- visualization and result analysis

## 12. Documentation Surface

### 12.1 Documentation Paths

The repository's documentation surface includes:

- `README.md`
  - top-level orientation
- `docs/installation.md`
  - installation profiles and dependency expectations
- `docs/quick-start.md`
  - common usage flows
- `docs/algorithms.md`
  - algorithm-level user guidance
- `docs/benchmarking.md`
  - research and evaluation workflow
- `docs/iaas-service.md`
  - gateway/service usage
- `docs/development.md`
  - contributor guidance
- `docs/PLANNING_WRAPPER.md`
  - planning-wrapper usage and behavior

### 12.2 Documentation Contract

Documentation is descriptive and user-facing.

This specification is normative for the repository contract when the docs and implementation differ
at a conceptual level, but implementation details remain the ultimate source of truth for exact
runtime behavior.

## 13. Development, Testing, and Quality

### 13.1 Development Setup

The repository supports:

- `uv sync --extra dev`
- `pip install -e ".[dev]"`

Python version expectations:

- Python `>= 3.10`

### 13.2 Test Surface

Tests live under `tests/` and cover core repository behaviors such as:

- algorithms
- language-model adapters
- tool-calling behavior
- planning wrapper
- reward-model integration
- gateway/service behavior

### 13.3 Code Quality

Current repository quality tooling:

- `pytest`
- `ruff check`
- `ruff format`

### 13.4 Notebooks and Examples

Development and research workflows MAY include notebooks and synced notebook artifacts. Their exact
presence is repository-version-dependent and not part of the core conformance contract.

## 14. Test and Validation Matrix

A conforming repository implementation or faithful port SHOULD include tests for the behaviors below.

### 14.1 Message and Input Model

- string prompts normalize into a single `user` message
- chat histories preserve role/content/tool-call fields
- multimodal text extraction behaves consistently
- tool-call-bearing assistant messages preserve tool-call structures

### 14.2 Provider Adapters

- OpenAI-compatible request shaping is valid
- batched generation returns one response per candidate
- tool definitions and tool choice are forwarded correctly
- transient provider failures are handled according to documented retry policy

### 14.3 Algorithms

- self-consistency selects from repeated candidates correctly
- tool-voting modes behave as documented
- best-of-n uses outcome scoring correctly
- beam search uses step generation and beam width correctly
- particle-filtering family handles particle budget semantics correctly
- planning wrapper allocates budget across approaches correctly
- exported incomplete algorithms are either clearly disabled or tested as intentionally unimplemented

### 14.4 Reward Integrations

- outcome reward adapters accept prompt/message inputs
- process reward adapters accept step-wise inputs
- reward-hub integration preserves expected score shapes

### 14.5 Gateway Profiles

- configured-service profile:
  - `/configure` applies the active policy
  - `/v1/chat/completions` resolves request + policy into execution config
  - tool-call responses are preserved
  - error cases behave as documented
- intercepting profile:
  - requests without ITS activation pass through
  - requests with ITS activation may be short-circuited
  - missing-model and ITS-failure fallback behavior matches documentation

### 14.6 Benchmark and Tooling

- benchmark dataset loading succeeds for supported datasets
- benchmark CLI parses budgets and algorithm names correctly
- example scripts instantiate documented components correctly

## 15. Implementation Checklist

### 15.1 Core Repository Checklist

- message normalization layer
- abstract language-model contract
- abstract ITS algorithm contract
- abstract reward-model contracts
- step-wise generation helper
- at least one concrete model adapter
- at least one concrete ITS algorithm
- at least one reward-model integration
- docs and tests for supported surfaces

### 15.2 Current Repository-Family Checklist

- `OpenAICompatibleLanguageModel`
- `LiteLLMLanguageModel`
- `SelfConsistency`
- `BestOfN`
- `BeamSearch`
- particle-filtering family
- `reward_hub` integration
- direct-service gateway profile
- benchmark and example scripts

### 15.3 Gateway Checklist

- normalized gateway execution config
- OpenAI-compatible response shaping
- configured-service profile support
- intercepting-profile contract support if that profile is implemented
- documented streaming behavior
- documented usage-accounting mode
- documented ITS failure behavior

## 16. Reference Algorithms (Language-Agnostic)

### 16.1 Generic ITS Execution

```text
function run_its_request(lm, algorithm, prompt_or_messages, budget, tools, tool_choice):
  normalized_messages = ChatMessages.from_prompt_or_messages(prompt_or_messages)

  result = algorithm.ainfer(
    lm=lm,
    prompt_or_messages=normalized_messages,
    budget=budget,
    return_response_only=false,
    tools=tools,
    tool_choice=tool_choice
  )

  return result.the_one
```

### 16.2 Configured-Service Gateway Handling

```text
function handle_configured_chat_completion(request):
  validate request.messages is non-empty
  validate request.budget in 1..1000

  if active_gateway_policy is missing:
    return error("service_not_configured")

  provider_config = active_gateway_policy.registered_models[request.model]
  if provider_config is missing:
    return error("unknown_model")

  execution_config = {
    model: request.model,
    algorithm: active_gateway_policy.active_algorithm,
    budget: request.budget,
    provider: provider_config,
    algorithm_config: active_gateway_policy.active_algorithm_config,
    reward_config: active_gateway_policy.active_reward_config,
    response_mode: request.return_response_only
      ? "selected_message_only"
      : "selected_message_with_metadata"
  }

  result = orchestrator.run(
    execution_config,
    messages=request.messages,
    tools=request.tools,
    tool_choice=request.tool_choice
  )

  return shape_openai_response(result)
```

### 16.3 Intercepting Gateway Handling

```text
function handle_intercepted_chat_completion(request_path, headers, body):
  if request_path is not "/v1/chat/completions":
    return pass_through()

  budget = parse_int(headers["x-its-budget"])
  endpoint = headers["x-its-endpoint"]
  api_key = headers["x-its-api-key"]

  if budget invalid or endpoint missing:
    return pass_through()

  request = parse_json(body)
  if request.model missing:
    return documented_fallback_for_missing_model()

  execution_config = {
    model: request.model,
    algorithm: interceptor_default_algorithm,
    budget: budget,
    provider: {
      kind: interceptor_provider_kind,
      endpoint: endpoint,
      api_key: api_key
    },
    algorithm_config: interceptor_algorithm_config,
    reward_config: interceptor_reward_config,
    response_mode: "selected_message_only"
  }

  result = orchestrator.run(
    execution_config,
    messages=request.messages,
    tools=request.tools,
    tool_choice=request.tool_choice
  )

  if result succeeded:
    return immediate_openai_response(result, headers={"x-its-applied": "true"})

  return documented_fallback_for_its_failure()
```

### 16.4 Benchmark Execution

```text
function run_benchmark(dataset, algorithm_name, model, budgets):
  problems = load_dataset(dataset)

  for budget in budgets:
    algorithm = init_algorithm(algorithm_name, ...)

    for problem in problems:
      response = algorithm.infer(model, problem.prompt, budget)
      record_result(problem, budget, response)

  return aggregate_results()
```

## Appendix A. Non-Normative Side Notes

These notes are intentionally outside the conformance requirements. They identify areas where the
current repository implementation family may evolve in a future revision.

1. `Gateway-only vs repository-wide scope`
   - This `SPEC.md` is intentionally repository-wide.
   - The gateway specification is included here as one subsystem rather than standing alone.

2. `Configured-service policy granularity`
   - The current configured-service style keeps one active ITS policy for all requests, even though
     it also keeps a model registry.
   - A future revision may define per-model or per-route policies.

3. `Configured-service validation drift`
   - Current configured-service examples and exact implementation validation rules are not perfectly
     aligned in every case.
   - This specification captures the conceptual contract rather than every accidental detail.

4. `Intercepting profile algorithm selection`
   - The current intercepting profile activates ITS through headers but does not yet standardize full
     per-request algorithm selection.

5. `Usage accounting`
   - Current repository-family implementations differ between placeholder usage and aggregated actual
     usage.

6. `Streaming`
   - Streaming ITS semantics are not standardized in this version.

7. `Incomplete exported surfaces`
   - Some exported classes are intentionally incomplete or experimental.
   - Future revisions may separate stable and experimental exports more explicitly.
