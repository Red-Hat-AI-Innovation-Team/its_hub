# its_hub System Specification

Status: Draft v1 (language-agnostic)

Purpose: Define a library-first system for inference-time scaling (ITS) of LLMs.

This specification defines the system contract, including:

- a reusable core ITS library contract
- stable and experimental algorithm profiles
- two OPTIONAL gateway profiles:
  - `Direct ITS Gateway`
  - `External-Processing Gateway`
- documentation, benchmarking, and validation expectations

The intent is that implementations evolve against this specification. A current codebase is one
possible implementation of the spec, not the source of truth for the spec itself.

## Normative Language

The key words `MUST`, `MUST NOT`, `REQUIRED`, `SHOULD`, `SHOULD NOT`, `RECOMMENDED`, `MAY`, and
`OPTIONAL` in this document are to be interpreted as described in RFC 2119.

`Implementation-defined` means the behavior is part of the implementation contract, but this
specification does not prescribe one universal policy. Implementations MUST document the selected
behavior.

## 1. Problem Statement

`its_hub` is a system for improving LLM outputs by spending more compute at inference time rather
than by retraining the model.

The system addresses four related problems:

- It defines reusable contracts for inference-time scaling algorithms, language-model adapters,
  orchestration, and optional scoring capabilities.
- It allows ITS to be integrated into serving systems without redefining the downstream
  chat-completions client contract.
- It supports both production-facing gateway integration and research-oriented experimentation.
- It provides enough common structure that future implementations in other languages can remain
  behaviorally compatible.

Important boundary:

- `its_hub` is library-first.
- Gateways are OPTIONAL integration profiles layered on top of the library contract.
- The specification defines behavior and interfaces, not one language, one packaging model, or one
  deployment topology.
- A conforming implementation can be a library, a gateway, both, or a strict subset according to
  the capability profiles in Section 3.3.

## 2. Goals and Non-Goals

### 2.1 Goals

- Define stable abstractions for:
  - message normalization
  - language-model generation
  - orchestration/fanout
  - ITS algorithms
  - optional scoring capabilities
- Support both string prompts and structured chat messages.
- Preserve canonical tool-invocation structure and support lossless mapping to common
  chat-completions protocols where needed.
- Provide a shared `budget` vocabulary for compute allocation across ITS algorithms.
- Support both outcome-based and process-based scoring.
- Define two interoperable gateway profiles:
  - a direct service profile
  - an external-processing interception profile
- Keep the contract portable across multiple implementations and languages.
- Support design-first workflow where changes to system behavior are specified before
  implementation.

### 2.2 Non-Goals

- Defining the full OpenAI Chat Completions protocol.
- Requiring a single mandatory service runtime.
- Defining model training, fine-tuning, or model-hosting internals.
- Requiring every implementation to ship every algorithm or gateway profile.
- Standardizing streaming ITS behavior in this specification version.
- Guaranteeing that experimental algorithms have the same stability level as the stable core.

## 3. System Overview

### 3.1 Main Components

1. `Message Model`
   - Normalizes prompts, chat history, structured content, and tool-invocation-bearing messages.

2. `Language Model Capability`
   - Produces normalized assistant responses from normalized message inputs.

3. `Orchestration Capability`
   - Fans out one logical ITS request into multiple LM calls, preserves ordering, and applies
     concurrency control.

4. `ITS Algorithm Capability`
   - Implements sampling, voting, scoring, or search logic over one or more LM calls.

5. `Scoring Capability` (OPTIONAL)
   - Scores final candidates and/or intermediate reasoning steps for algorithm profiles that require
     it.

6. `Gateway Adapter` (OPTIONAL)
   - Exposes or intercepts chat-completions-like requests and maps them onto the core library
     contract.

7. `Research / Tooling Surface` (OPTIONAL)
   - Provides examples, benchmarks, and evaluation helpers.

8. `Observability Surface` (OPTIONAL)
   - Exposes logs, metrics, traces, or implementation-defined runtime status.

### 3.2 Abstraction Levels

`its_hub` is easiest to port when kept in these layers:

1. `Core Domain Layer`
   - Messages, tool invocations, candidates, results, and budget semantics.

2. `Execution Layer`
   - LM generation, orchestration, and algorithm execution.

3. `Optional Scoring Layer`
   - Outcome and process scoring capabilities used only by profiles that require them.

4. `Gateway Layer`
   - Direct or intercepting request handling built on top of the execution layer.

5. `Research and Tooling Layer`
   - Documentation, observability, examples, benchmarks, validation assets.

An implementation MAY collapse multiple layers into one module or service so long as the observable
behavior remains conformant.

### 3.3 Capability Profiles

This specification defines these conformance profiles:

1. `Core Library Conformance`
   - Message model
   - generic candidate/result model
   - LM capability
   - orchestration capability or equivalent
   - scaling algorithm execution contract

2. `Outcome Reward Extension`
   - Scores final candidates for profiles that require outcome evaluation.

3. `Process Reward Extension`
   - Scores intermediate reasoning steps for profiles that require process evaluation.

4. `Stable SelfConsistency Extension`
   - Candidate-generation and voting profile.

5. `Stable BestOfN Extension`
   - Candidate-generation and outcome-scoring profile.

6. `Experimental Search Extension`
   - Step-wise search family
   - beam search
   - particle methods
   - planning wrapper

7. `Direct Gateway Extension`
   - Standalone service that terminates client requests and applies ITS directly.

8. `External-Processing Gateway Extension`
   - Intercepting gateway that conditionally applies ITS in front of an upstream API.

9. `Research Toolkit Extension`
   - Examples, benchmark runners, dataset evaluation tooling.

Composition rules:

- Every implementation that claims any conformance profile from this specification MUST satisfy
  `Core Library Conformance`.
- An implementation MAY claim `Core Library Conformance` without implementing a stable or
  experimental algorithm profile.
- An implementation MAY claim full algorithm conformance only if it implements at least one stable
  algorithm profile from Section 7.
- `Stable BestOfN Extension` requires `Outcome Reward Extension`.
- `Experimental Search Extension` MAY require `Process Reward Extension` according to the selected
  profile.
- `Experimental Search Extension` alone is not sufficient for full algorithm conformance.
- Sections marked OPTIONAL are extension profiles.
- An implementation MAY support multiple extensions at once.

### 3.4 External Dependencies

Common external dependencies include:

- one or more downstream language-model providers
- OPTIONAL scoring services or local scoring runtimes
- OPTIONAL HTTP, proxy, or gRPC runtime for gateway implementations
- local filesystem for docs, examples, test fixtures, benchmark assets, or logs
- host environment authentication and secret management

The specification does not require any one vendor, network topology, or packaging format.

## 4. Core Domain Model

### 4.1 Entities

#### 4.1.1 `ContentPart`

`ContentPart` is the canonical provider-neutral unit for structured message content.

Fields:

- `type` (string)
- type-specific payload fields

Minimum standardized shape:

- `type = "text"`
  - requires `text` (string)

Other content-part types MAY be supported, but implementations MUST document:

- required payload fields
- text-extraction behavior
- whether unsupported parts are rejected, ignored, or preserved as opaque metadata

#### 4.1.2 `ToolInvocation`

`ToolInvocation` is the canonical provider-neutral representation of one assistant-initiated tool
request.

Fields:

- `name` (string)
- `arguments` (structured value)
- `invocation_id` (OPTIONAL string)
- `provider_metadata` (OPTIONAL object)

Required semantics:

- `arguments` SHOULD be structured rather than serialized text when a lossless structured form is
  available.
- Implementations MAY preserve provider-native tool-invocation payloads in `provider_metadata`, but
  the canonical fields remain the interoperability surface.

#### 4.1.3 `ChatMessage`

`ChatMessage` is the normalized conversation unit used throughout the system.

Fields:

- `role` (string)
  - Supported values:
    - `system`
    - `user`
    - `assistant`
    - `tool`
- `content`
  - `string` for plain text
  - `list[ContentPart]` for structured content
  - `null` when the message is represented primarily by tool invocations or tool results
- `tool_invocations` (OPTIONAL list of `ToolInvocation`)
- `in_reply_to_tool_invocation` (OPTIONAL string)

Required semantics:

- Assistant messages that request tools MUST preserve `tool_invocations`.
- Tool-result messages SHOULD identify the invocation they are replying to when that relationship is
  available.
- Text-bearing content MUST preserve text parts in order.
- Implementations SHOULD tolerate unknown inbound fields for forward compatibility.

#### 4.1.4 `ChatMessages`

`ChatMessages` is the normalized conversation container.

It wraps either:

- a string prompt
- a list of `ChatMessage`
- an implementation-equivalent normalized conversation object

Required behaviors:

- normalize string prompts into structured messages
- return structured chat messages as the primary representation
- produce repeated batches of equivalent conversations for parallel generation
- provide a prompt-string fallback representation for compatibility-oriented paths when needed

#### 4.1.5 `ScalingCandidate`

One candidate assistant response produced during ITS execution.

Logical fields:

- `message`
  - selected or candidate assistant message
- `provider_metadata` (OPTIONAL)
  - provider-native generation metadata
- `usage` (OPTIONAL)
  - generation-usage record for this candidate

`ScalingCandidate` is a logical model; implementations MAY inline it into plain message records so
long as equivalent information can be expressed.

#### 4.1.6 `ScalingResult`

The result of one logical ITS execution.

Logical fields:

- `selected`
  - the chosen assistant response
- `candidates` (OPTIONAL)
  - all candidate responses considered by the algorithm
- `metadata` (OPTIONAL)
  - algorithm-specific decision metadata such as vote counts, scores, selected index, or traces
- `usage` (OPTIONAL)
  - usage aggregated across the logical ITS execution

Required semantics:

- A selected response MUST always be available.
- Detailed result objects are OPTIONAL, but implementations SHOULD make them available when research,
  debugging, or observability is an intended use case.

#### 4.1.7 `GenerationUsage`

When token usage is exposed, the interoperable shape is:

- `prompt_tokens`
- `completion_tokens`
- `total_tokens`

All fields are non-negative integers.

If an implementation cannot determine usage exactly, it MUST do one of the following:

- omit usage entirely
- provide a documented estimate
- provide a documented sentinel representation

### 4.2 Normalization and Mapping Rules

- A string prompt MUST normalize into one `user` message.
- Structured chat is the primary cross-implementation contract.
- Prompt-string fallback MAY exist for compatibility-oriented or experimental paths.
- Tool-invocation-bearing assistant messages SHOULD remain attached to candidates and selected
  results.
- Provider-native fields MAY be preserved in metadata, but canonical fields remain authoritative.
- Implementations that map to or from OpenAI-compatible payloads SHOULD map:
  - `tool_calls` -> `tool_invocations`
  - `tool_call_id` -> `in_reply_to_tool_invocation`
- Structured content parts MUST either:
  - match a documented content-part shape, or
  - fail with an explicit validation error

## 5. Core Library Contract

### 5.1 Input Normalization

Algorithms MUST accept normalized conversation input in one of these forms:

- string prompt
- list of normalized messages
- implementation-equivalent `ChatMessages` wrapper

Normalization MUST occur before algorithm-specific logic.

### 5.2 Language Model Capability

A conforming implementation MUST provide LM generation behavior equivalent to:

- generate one assistant response from one normalized conversation

Batch generation can be implemented in either of two ways:

- the LM capability accepts batched inputs directly
- a separate orchestration capability fans out multiple single-generation LM calls

Required semantics:

- LM generation MUST return normalized assistant messages.
- Implementations SHOULD preserve canonical tool-invocation structure when the underlying provider
  supports tool use.
- Retry, backoff, session reuse, and connection pooling are implementation-defined.

### 5.3 Orchestration Capability

Orchestration is a required behavior, but it does not need to exist as a first-class public type.

Responsibilities:

- fan out one logical ITS request into multiple LM calls
- preserve input/output ordering
- forward generation arguments consistently
- apply implementation-defined concurrency control
- batch requests directly or emulate batching through repeated single calls

Required semantics:

- Orchestration MUST preserve deterministic positional alignment between input conversations and
  returned assistant responses.
- Concurrency policy is implementation-defined, but the implementation MUST document whether and how
  concurrency is bounded.

### 5.4 Scaling Algorithm Capability

Every ITS algorithm MUST implement behavior equivalent to:

- accept normalized input
- allocate compute according to `budget`
- invoke LM generation and, where needed, optional scoring extensions
- select a final assistant response

Required semantics:

- Algorithms MUST accept both prompt-style and chat-style input.
- Algorithms MUST document how they interpret `budget`.
- Algorithms SHOULD preserve tool-invocation-bearing assistant messages when the LM capability does.
- Algorithms MAY depend on an explicit orchestration capability or inline equivalent batching/fanout
  behavior.
- Algorithms MAY expose either:
  - selected response only
  - selected response plus full algorithm result metadata

### 5.5 Concurrency and Convenience Interfaces

This specification is concurrency-oriented, not tied to one language's async model.

Required behavior:

- Implementations MUST support concurrent execution of multiple independent LM calls, either in the
  LM capability itself or through orchestration.
- Synchronous blocking interfaces MAY be provided as convenience layers.

This specification does not require any one event-loop, thread-pool, callback, future, or coroutine
model.

### 5.6 Provider-Native Argument Forwarding and Invocation Mapping

Provider-native generation arguments are OPTIONAL parts of the extensible execution surface.

Implementations MAY support arguments such as:

- `max_tokens`
- `temperature`
- tool-availability hints
- tool-selection hints
- structured-output hints

Required semantics:

- If provider-native tool-use structures are supported, the implementation MUST map them onto the
  canonical `ToolInvocation` model without silent semantic loss.
- If structured-output features are supported, the implementation MUST document whether algorithms
  merely forward those features or interpret them directly.

### 5.7 Usage and Detailed Result Reporting

Usage and detailed result reporting are OPTIONAL parts of the core contract.

If exposed:

- per-candidate usage MUST refer to one LM generation
- aggregated usage MUST refer to one logical ITS execution
- algorithm metadata MUST be documented as algorithm-specific, not assumed to be universal

### 5.8 Configuration Surface

Every conforming implementation MUST document how the following are configured:

- downstream model or provider selection
- algorithm-profile selection
- concurrency policy
- timeout policy
- budget defaults and ceilings
- scoring configuration for profiles that require scoring
- which values are supplied:
  - at initialization time
  - per execution/request
  - or by an external control surface

Configuration model:

- Configuration MAY be static, dynamic, file-based, API-based, embedded in the caller, or supplied
  by an external control plane.
- Dynamic reload is OPTIONAL.
- Invalid configuration MUST be detected no later than:
  - initialization time, or
  - request acceptance for the affected execution path

### 5.9 Validation and Required Failure Signaling

Implementations MUST validate:

- message normalization inputs
- `budget`
- provider-native generation arguments that they claim to support
- candidate/result cardinality assumptions
- scoring cardinality assumptions for profiles that use scoring

Required semantics:

- Invalid input MUST produce an explicit failure.
- Unsupported optional provider-native arguments MAY be rejected explicitly.
- Malformed provider responses MUST produce an explicit failure.
- A gateway profile MAY choose fallback only when the gateway profile explicitly permits fallback.

## 6. Optional Scoring Extensions

### 6.1 `Outcome Reward Extension`

Outcome scoring evaluates final candidates.

Required behavior:

- score one or more complete candidates with enough prompt or conversation context to make the score
  meaningful
- higher score means better candidate
- return exactly one score per scored candidate

Required invariants:

- score cardinality MUST equal scored candidate cardinality
- if score cardinality does not match, fail with `invalid_reward_cardinality` or equivalent

Interface freedom:

- An implementation MAY score:
  - prompt + candidate
  - full conversation including candidate
  - batched conversations
- An implementation MAY expose blocking, non-blocking, or both

### 6.2 `Process Reward Extension`

Process scoring evaluates intermediate reasoning steps.

Required behavior:

- accept prompt or conversation context plus ordered reasoning steps
- return exactly one score per step, or an equivalent structure that preserves one-to-one
  step-aligned evaluation meaning

Required invariants:

- score cardinality MUST equal step cardinality
- score ordering MUST remain aligned with step ordering
- if score cardinality does not match, fail with `invalid_reward_cardinality` or equivalent

The minimal interoperable shape is a list of numeric scores aligned with the step list.

## 7. Stable Algorithm Profiles

### 7.1 Shared `budget` Vocabulary

`budget` is the shared compute-control vocabulary for one ITS execution.

Required semantics:

- `budget` MUST be a positive integer
- stable algorithms MUST specify their `budget` semantics in this section
- identical `budget` values across different algorithms do not imply identical cost

Candidate availability policy for stable candidate-set algorithms:

- If one or more candidate generations succeed, the algorithm MUST continue using the successful
  subset.
- If zero candidate generations succeed, the algorithm MUST fail with `insufficient_candidates` or
  equivalent.
- Implementations MUST NOT synthesize placeholder candidates for failed generations.

### 7.2 `SelfConsistency`

Behavior:

- Generate multiple candidate responses.
- Project each candidate into a comparison space.
- Select the most common projection, with implementation-defined tie-breaking.

Projection behavior:

- The default comparison space MAY be exact or normalized text content.
- Implementations MAY support explicit projection functions, including regex-based projections.
- If a projection fails to match, the implementation MUST document whether the candidate is:
  - ignored
  - grouped under a null value
  - or treated as raw content

Tool-invocation behavior:

- Invocation-bearing responses MAY participate in consistency voting.
- If invocation-based voting is supported, the implementation MUST document the voting modes.
- Common modes include:
  - tool identity only
  - tool arguments only
  - hierarchical combination of tool identity and arguments

Result metadata MAY include:

- all candidate responses
- vote counts
- selected index

Budget semantics:

- `budget` is the number of candidate generations attempted.

### 7.3 `BestOfN`

Behavior:

- Generate `N` candidate responses.
- Score the successful candidate set with the `Outcome Reward Extension`.
- Select the highest-scoring candidate, with implementation-defined tie-breaking.

Scoring behavior:

- Implementations MAY score all candidates directly.
- Implementations MAY deduplicate semantically equivalent candidates before scoring to reduce cost.
- If invocation support is present, semantic equivalence SHOULD consider both content and canonical
  tool-invocation structure.

Required invariants:

- every scored candidate MUST receive exactly one score
- if scoring of the surviving candidate set is incomplete, the execution MUST fail explicitly

Result metadata MAY include:

- all candidate responses
- scores
- selected index

Budget semantics:

- `budget` is the number of candidate generations attempted.

## 8. Experimental Algorithm Profiles

Experimental algorithms are OPTIONAL extensions.

Cross-implementation conformance for experimental profiles is intentionally limited in v1:

- the explicit invariants in this section are normative
- behavior beyond those invariants is implementation-defined and MUST be documented
- experimental profile support does not imply cross-implementation behavioral interchangeability in
  the same way as the stable profiles in Section 7

### 8.1 `StepGenerationConfig`

`StepGenerationConfig` is used by step-wise search profiles.

Fields:

- `max_steps` (positive integer)
- exactly one of:
  - `step_token` (string literal marking step boundaries in generated text)
  - `tokens_per_step` (positive integer, maximum generated tokens per step)
- `stop_token` (OPTIONAL string)
- `temperature` (OPTIONAL numeric value)
- `include_stop_str_in_output` (OPTIONAL boolean)
- `temperature_switch` (OPTIONAL implementation-defined temperature-switch rule)

Validation rules:

- If both `step_token` and `tokens_per_step` are set, fail with explicit validation error.
- If neither `step_token` nor `tokens_per_step` is set, fail with explicit validation error.
- `tokens_per_step` MUST be positive when used.

### 8.2 Step-Wise Search Family

Common characteristics:

- use `StepGenerationConfig`
- generate partial reasoning trajectories rather than only final answers
- MAY depend on `Process Reward Extension`
- may rely on prompt-string compatibility paths more than the stable core

### 8.3 `BeamSearch`

Minimum invariants:

- maintain a bounded frontier of partial trajectories
- expand frontier members in repeated search rounds
- score or rank expanded partial trajectories according to a documented policy
- retain no more than the documented frontier bound after each pruning step

Budget semantics:

- implementation-defined in v1
- the implementation MUST document how `budget` bounds total search effort

Required documentation:

- whether `Process Reward Extension` is required by the selected beam-search variant
- the ranking signal used when `Process Reward Extension` is not required
- frontier-bound policy
- step stopping criteria
- score aggregation policy

### 8.4 Particle Methods

This family includes profiles such as:

- `ParticleFiltering`
- `ParticleGibbs`
- entropic or annealed particle variants

Minimum invariants:

- maintain multiple evolving trajectories
- update or resample trajectories according to a documented particle policy
- use a documented scoring or weighting policy to guide evolution

Budget semantics:

- implementation-defined in v1
- the implementation MUST document whether `budget` represents:
  - particle count
  - total particle effort
  - or another documented particle-control quantity

Required documentation:

- whether `Process Reward Extension` is required by the selected particle variant
- the weighting or ranking signal used when `Process Reward Extension` is not required
- update/resampling policy
- score or weight aggregation policy
- stopping criteria

### 8.5 `PlanningWrapper`

Minimum invariants:

- allocate some compute to planning
- execute a downstream algorithm conditioned on the planning result

Budget semantics:

- implementation-defined in v1
- the implementation MUST document how planning cost and downstream execution cost share `budget`

Required documentation:

- whether planning, execution, or both depend on `Outcome Reward Extension` and/or
  `Process Reward Extension`

## 9. Gateway Profiles

### 9.1 Gateway Role

Gateways are OPTIONAL integration profiles layered over the core library contract.

They do not redefine algorithm behavior. Instead, they:

- accept or intercept client requests
- map those requests onto the core ITS contract
- shape results into an OpenAI-compatible response surface

This specification version defines two gateway profiles:

1. `Direct ITS Gateway`
2. `External-Processing Gateway`

### 9.2 Shared Gateway Requirements

All gateway profiles that claim conformance MUST:

- target an OpenAI-compatible chat-completions-like request/response surface
- accept a client-visible `model` selection or documented equivalent
- accept normalized `messages` or a compatible chat body
- preserve selected assistant-message content and tool invocation structure when mapped into the
  gateway protocol
- document how downstream LM routing and algorithm wiring are configured
- document timeout, retry, and failure behavior

Minimum OpenAI-compatible response shape:

- `id`
- `object`
- `created`
- `model`
- `choices`

`usage` and algorithm-specific `metadata` are OPTIONAL but RECOMMENDED when meaningful.

### 9.3 `Direct ITS Gateway` (OPTIONAL)

#### 9.3.1 Role

A direct gateway is a standalone service that terminates client requests and applies ITS directly.

#### 9.3.2 Activation Model

ITS activation MUST be explicit in the request payload.

Canonical behavior:

- the client includes `budget` or an implementation-equivalent compute-control field in the request
  body

The direct gateway profile MUST NOT rely on proxy-only transport metadata as its primary activation
mechanism.

#### 9.3.3 Configuration Surface

A direct gateway MUST provide a documented configuration mechanism for:

- downstream LM routing
- algorithm selection or policy selection
- scoring wiring where applicable
- credentials and secret handling

The mechanism MAY be:

- static startup configuration
- a configuration file
- an admin/control API
- an external control plane

#### 9.3.4 Request Contract

A direct gateway MUST accept:

- `model`
- `messages`

A direct gateway SHOULD accept:

- `budget`

A direct gateway MAY accept:

- generation-control fields such as maximum token count or temperature
- tool-availability hints
- tool-selection hints
- `stream`
- implementation-defined metadata fields

Unsupported optional request fields SHOULD produce explicit documented behavior rather than silent
reinterpretation.

#### 9.3.5 Response Contract

When ITS is applied, the response MUST be OpenAI-compatible and include:

- one selected assistant message in `choices[0].message`

Example minimum response body:

```json
{
  "id": "its-req-123",
  "object": "chat.completion",
  "created": 1730000000,
  "model": "example-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "example output"
      },
      "finish_reason": "stop"
    }
  ]
}
```

The response MAY also include:

- `usage`
- algorithm-specific `metadata`

If usage is exposed, the implementation MUST document whether it is:

- aggregated across all LM calls used by ITS
- estimated
- or otherwise computed

#### 9.3.6 Failure Model

Configuration or wiring failures SHOULD produce explicit service errors.

Generation failures MAY be handled by:

- explicit service error
- documented fallback behavior

Direct gateways MUST NOT silently forward the request to an unknown upstream unless that behavior is
an explicit part of the documented design.

#### 9.3.7 State and Scaling Notes

Direct gateways MAY use:

- static configuration
- in-memory mutable state
- external configuration/state stores

Restart behavior, persistence, and horizontal scaling are implementation-defined and MUST be
documented.

### 9.4 `External-Processing Gateway` (OPTIONAL)

#### 9.4.1 Role

An external-processing gateway sits in front of an upstream OpenAI-compatible API and conditionally
applies ITS.

#### 9.4.2 Activation Model

ITS activation MUST be conveyed out-of-band relative to the standard request body.

Examples include:

- HTTP headers
- proxy route metadata
- gRPC metadata
- implementation-defined request context

Required semantics:

- activation metadata MUST include compute-control information equivalent to `budget`
- the implementation MUST document all recognized activation fields and validation rules

#### 9.4.3 Outcomes

The profile supports two outcomes:

1. `pass_through`
   - the original request continues to the upstream service

2. `its_applied`
   - the gateway runs ITS
   - the gateway returns an OpenAI-compatible response directly
   - the upstream request is short-circuited

#### 9.4.4 Request Resolution

For an intercepted request:

1. inspect route and activation metadata
2. if route is unsupported -> `pass_through`
3. if activation metadata is absent or invalid -> `pass_through`
4. buffer or inspect the request body as needed
5. if required body fields are absent:
   - `pass_through`, or
   - fail explicitly according to documented policy
6. run ITS
7. on success, return immediate OpenAI-compatible response
8. on failure, either:
   - `pass_through`, or
   - explicit error according to documented policy

#### 9.4.5 Response Contract

When ITS is applied, the response MUST be OpenAI-compatible and include:

- the selected assistant message

Example minimum response body:

```json
{
  "id": "its-req-123",
  "object": "chat.completion",
  "created": 1730000000,
  "model": "example-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "example output"
      },
      "finish_reason": "stop"
    }
  ]
}
```

The response SHOULD also include an implementation-defined signal that ITS was applied, such as:

- response header
- response metadata
- proxy-local observability event

If usage is exposed, it SHOULD refer to the full logical ITS execution, not only the selected
candidate.

#### 9.4.6 Failure Model

Safe fallback is RECOMMENDED.

If safe fallback is chosen:

- invalid activation metadata -> `pass_through`
- unsupported route -> `pass_through`
- missing required request-body fields -> `pass_through`
- ITS execution failure after activation -> `pass_through`

If explicit errors are chosen instead, the implementation MUST document that policy clearly.

#### 9.4.7 Safety Notes

External-processing gateways SHOULD sanitize or strip internal ITS activation metadata before
forwarding pass-through requests upstream when that metadata is not intended for upstream services.

### 9.5 Streaming

This specification version does not standardize streaming ITS behavior.

Implementations MAY:

- reject streaming explicitly
- buffer streaming requests and return non-streaming ITS responses
- support streaming according to an implementation-defined design

If streaming is not supported, the implementation SHOULD return an explicit documented error rather
than silently pretending to stream.

## 10. Documentation, Observability, Benchmarking, and Example Surface

### 10.1 Documentation

Documentation SHOULD cover:

- installation or setup guidance
- core algorithm concepts
- gateway profile behavior where implemented
- development and contribution notes
- benchmarking and evaluation workflows

This specification does not require fixed file paths or one documentation toolchain.

### 10.2 Observability

If observability is implemented, the implementation SHOULD expose structured information for each
logical ITS execution such as:

- execution identifier
- algorithm profile
- `budget`
- candidate count
- selected index when detailed outputs are available
- duration or latency
- usage, if available
- failure code, if the execution failed
- gateway outcome (`pass_through` or `its_applied`) for gateway profiles

Observability outputs MAY be logs, traces, metrics, snapshots, or another documented surface.

Observability failures MUST NOT change the correctness of the ITS result.

### 10.3 Benchmarking Surface

A research-oriented implementation MAY provide benchmark tooling for:

- mathematical reasoning tasks
- multi-step reasoning tasks
- quality/cost trade-off measurement across algorithms

Benchmark datasets, scripts, and storage formats are implementation-defined.

If benchmark tooling is shipped, the implementation SHOULD document:

- dataset assumptions
- evaluation procedure
- reproducibility expectations

### 10.4 Examples

Examples MAY be provided for:

- `SelfConsistency`
- `BestOfN`
- experimental search profiles
- gateway profile usage

Examples are descriptive and user-facing, not normative.

## 11. Failure Model and Recovery Strategy

### 11.1 Shared Error Vocabulary

Implementations SHOULD use or map onto a shared error vocabulary that includes codes such as:

- `invalid_input`
- `invalid_message`
- `invalid_content_part`
- `invalid_tool_invocation`
- `invalid_budget`
- `unsupported_generation_argument`
- `lm_generation_failed`
- `malformed_lm_response`
- `orchestration_failed`
- `partial_generation_failure`
- `insufficient_candidates`
- `reward_failed`
- `invalid_reward_cardinality`
- `gateway_configuration_error`
- `gateway_request_invalid`
- `gateway_interception_failed`
- `gateway_response_shaping_failed`

The exact public error envelope is implementation-defined, but the implementation SHOULD document a
stable mapping from its error surface to these concepts.

### 11.2 Failure Classes

1. `Input / Normalization Failures`
   - invalid message shape
   - unsupported content form
   - malformed tool invocation structure

2. `LM Generation Failures`
   - provider or network failure
   - timeout
   - provider rejection
   - malformed provider response

3. `Orchestration Failures`
   - batch fanout failure
   - concurrency-control failure
   - partial candidate-generation failure

4. `Scoring Failures`
   - scoring-service failure
   - invalid score payload
   - inconsistent score cardinality

5. `Gateway Configuration Failures`
   - missing routing configuration
   - invalid algorithm wiring
   - missing credentials

6. `Gateway Runtime Failures`
   - request interception failure
   - body parsing failure
   - response shaping failure
   - proxy transport failure

7. `Observability / Tooling Failures`
   - logging sink failure
   - benchmark I/O failure
   - metrics/status reporting failure

### 11.3 Recovery Behavior

- Input and normalization failures:
  - fail the current request or run attempt explicitly

- LM transient failures:
  - MAY be retried according to documented policy before a candidate is treated as failed

- Orchestration failures:
  - MUST NOT silently reorder candidate results

- Direct gateway configuration failures:
  - SHOULD fail explicitly and operator-visibly

- External-processing gateway failures:
  - MUST follow the documented fallback-or-error policy from Section 9.4.6

- Observability and tooling failures:
  - MUST NOT corrupt the correctness of the core ITS result

### 11.4 Partial Failure Rules

Stable candidate-set algorithms MUST follow these rules:

- If one or more candidate generations succeed, proceed using the successful subset only.
- If zero candidate generations succeed, fail with `insufficient_candidates` or equivalent.
- Synthetic filler candidates are forbidden.

Scoring-related partial failures:

- If a scored profile requires one score per surviving candidate, incomplete scoring MUST fail
  explicitly.
- Implementations MAY retry scoring failures according to documented policy before failing.

Gateway partial-failure rule:

- `pass_through` is a documented gateway outcome, not an internal library success state.

### 11.5 Restart and State Notes

The core library can be stateless.

Gateway implementations MAY maintain:

- in-memory configuration
- client/session caches
- request-scoped caches
- externalized state

Persistence across restarts is OPTIONAL and implementation-defined.

## 12. Security and Operational Safety

### 12.1 Trust Boundary Assumption

Implementations SHOULD assume that all of the following may be partially or fully untrusted:

- prompts and chat messages
- structured content parts
- tool invocations and their arguments
- activation metadata
- downstream model outputs
- benchmark inputs

The specification intentionally does not mandate one trust posture, but implementations MUST
document their own.

### 12.2 Secret Handling

Implementations SHOULD:

- avoid logging raw credentials or API tokens
- validate the presence of secrets without printing them
- document how secrets are supplied and scoped

### 12.3 Tool Invocation and Structured Input Safety

Preservation of tool invocation structure is part of the contract, but actual execution of tools is
out of scope unless an implementation explicitly adds such a feature.

Implementations SHOULD:

- preserve tool invocation structure faithfully
- avoid mutating invocation arguments silently
- document any normalization applied to structured content or invocations

### 12.4 Gateway Safety

Gateway implementations SHOULD document:

- authentication model
- timeout policy
- retry policy
- rate limiting or quota behavior
- whether internal ITS metadata is stripped before upstream forwarding

### 12.5 Resource Exhaustion and Budget Controls

ITS can amplify LM call volume significantly.

Implementations SHOULD document guardrails for:

- maximum accepted `budget`
- concurrency limits
- timeout ceilings
- gateway backpressure behavior
- scoring-service resource limits

## 13. Non-Normative Reference Procedures

This section is NON-NORMATIVE.

It illustrates one conforming family of procedures, but it does not add requirements beyond
Sections 4 through 12. If any procedure in this section appears to conflict with a normative
section, the normative section controls.

### 13.1 Input Normalization Procedure

Input:

- string prompt, or
- normalized message list, or
- implementation-equivalent chat container

Illustrative procedure:

1. Convert string input into one `user` message.
2. Extract structured messages from any normalized chat container.
3. Pass through already-normalized message lists unchanged.
4. Reject unsupported input forms explicitly.

### 13.2 Self-Consistency Reference Procedure

Input:

- normalized or normalizable conversation input
- LM capability
- `budget`
- documented projection rule

Illustrative procedure:

1. Normalize the input conversation.
2. Attempt candidate generation `budget` times.
3. Discard failed candidates according to the stable candidate-availability rule.
4. Project surviving candidates into the comparison space.
5. Count projections.
6. Select the winning projection according to the documented tie-break policy.
7. Return the associated candidate as the selected response.

### 13.3 Best-of-N Reference Procedure

Input:

- normalized or normalizable conversation input
- LM capability
- outcome scoring extension
- `budget`

Illustrative procedure:

1. Normalize the input conversation.
2. Attempt candidate generation `budget` times.
3. Discard failed candidates according to the stable candidate-availability rule.
4. Optionally deduplicate semantically equivalent surviving candidates if the implementation
   documents such behavior.
5. Score the surviving candidate set.
6. Fail if score cardinality is incomplete.
7. Select the highest-scoring candidate according to the documented tie-break policy.

### 13.4 Direct Gateway Reference Procedure

Input:

- OpenAI-compatible chat-completions-like request
- direct gateway configuration

Illustrative procedure:

1. Validate gateway configuration.
2. Read the request's model selection, messages, and compute-control field.
3. Forward supported optional generation fields.
4. Run the configured ITS profile.
5. Shape the result into an OpenAI-compatible response.

### 13.5 External-Processing Gateway Reference Procedure

Input:

- intercepted upstream request
- activation metadata
- external-processing gateway configuration

Illustrative procedure:

1. Determine whether the route is eligible for ITS handling.
2. If the route is not eligible, choose `pass_through`.
3. Parse activation metadata.
4. If activation metadata is absent or invalid, choose `pass_through`.
5. Read or buffer the request body.
6. If required request-body fields are unavailable, follow the documented fallback-or-error policy.
7. If ITS should be applied, run the configured ITS profile.
8. If ITS succeeds, return an immediate OpenAI-compatible response and short-circuit the upstream
   request.
9. If ITS fails, follow the documented fallback-or-error policy.

## 14. Test and Validation Matrix

A conforming implementation SHOULD include tests that cover the behaviors defined in this
specification.

Validation profiles:

- `Core Conformance`: REQUIRED for all conforming implementations
- `Extension Conformance`: REQUIRED only for OPTIONAL profiles that an implementation chooses to
  ship
- `Real Integration Profile`: RECOMMENDED environment-dependent checks before production use

Unless otherwise noted, Section 14.1 is `Core Conformance`. Bullets that begin with `If ... is
implemented` are `Extension Conformance`.

### 14.1 Core Library Conformance

- string prompts normalize into one `user` message
- normalized chat preserves role/content/invocation fields
- structured content validation behaves as documented
- canonical tool invocation mapping is preserved
- LM capability returns one response per logical input
- orchestration preserves input ordering
- orchestration forwards documented generation arguments consistently
- invalid input and invalid `budget` are rejected explicitly
- malformed provider responses fail explicitly

### 14.2 Outcome Reward Extension

If the `Outcome Reward Extension` is implemented:

- one score is returned per scored candidate
- score cardinality mismatches fail explicitly
- score ordering remains aligned with candidate ordering

### 14.3 Process Reward Extension

If the `Process Reward Extension` is implemented:

- one score is returned per scored step, or an equivalent step-aligned structure is returned
- score cardinality mismatches fail explicitly
- score ordering remains aligned with step ordering

### 14.4 Stable SelfConsistency Extension

If the `Stable SelfConsistency Extension` is implemented:

- repeated candidates are selected correctly
- documented projection behavior is applied consistently
- invocation-based voting behavior matches documentation if implemented
- partial candidate-generation failure follows the stable candidate-availability rule
- `budget` semantics match Section 7.2

### 14.5 Stable BestOfN Extension

If the `Stable BestOfN Extension` is implemented:

- successful candidates are scored correctly
- deduplication behavior matches documentation if implemented
- incomplete scoring fails explicitly
- result objects or detailed outputs expose the selected response correctly
- `budget` semantics match Section 7.3

### 14.6 Experimental Search Extension

If experimental search is implemented:

- `StepGenerationConfig` validation rules are enforced
- declared scoring dependencies are documented and consistent with the selected profile
- beam-search frontier bounds follow the documented policy
- particle methods follow documented update/resampling policy
- planning-wrapper budget allocation follows documented policy
- prompt-fallback behavior is documented where structured-chat fidelity is reduced

### 14.7 Direct Gateway Extension

If the `Direct ITS Gateway` profile is implemented:

- request-body activation drives ITS execution
- the gateway rejects or handles unsupported optional fields as documented
- configuration failures are surfaced explicitly
- OpenAI-compatible response shaping is correct
- canonical tool invocation mapping is preserved when tool use is exposed through the gateway
- usage behavior matches documented semantics
- non-streaming behavior matches documented semantics

### 14.8 External-Processing Gateway Extension

If the `External-Processing Gateway` profile is implemented:

- unsupported routes pass through
- missing or invalid activation metadata follows documented fallback
- missing required body fields follow documented fallback or error policy
- ITS-applied responses short-circuit upstream behavior correctly
- OpenAI-compatible response shaping is correct
- ITS-applied signaling behavior matches documentation
- usage aggregation behavior matches documentation if usage is exposed

### 14.9 Research Toolkit Extension

If benchmark or research tooling is implemented:

- benchmark inputs and outputs are reproducible enough for the documented workflow
- dataset and scoring assumptions are documented
- example code remains consistent with the documented public contract

### 14.10 Real Integration Profile (RECOMMENDED)

These checks are RECOMMENDED before production use and MAY be skipped in CI when credentials or
network access are unavailable.

- Run a real downstream LM smoke test with valid credentials.
- If an outcome scoring extension is part of the deployment, run a real scoring-path smoke test.
- If a direct gateway is implemented, run an end-to-end chat-completions smoke test.
- If an external-processing gateway is implemented, test both:
  - pass-through behavior
  - ITS-applied short-circuit behavior
- Report skipped real-integration tests as skipped rather than silently passing them.

## 15. Implementation Checklist (Definition of Done)

Use the same validation profiles as Section 14:

- Section 15.1 = `Core Conformance`
- Section 15.2 = `Extension Conformance`
- Section 15.3 = `Real Integration Profile`

### 15.1 REQUIRED for Core Conformance

- normalized message model with canonical tool invocation fields
- generic candidate, result, and usage model
- LM generation capability
- orchestration behavior or equivalent batch/fanout layer
- scaling algorithm execution contract
- documented `budget` semantics
- configuration and validation surface
- documentation of failure behavior and trust boundary
- deterministic tests for supported core behavior

Additional rule for full algorithm conformance:

- implement at least one stable algorithm profile from Section 7

### 15.2 RECOMMENDED Extensions

- `Outcome Reward Extension`
- `Process Reward Extension`
- `Stable SelfConsistency Extension`
- `Stable BestOfN Extension`
- `Experimental Search Extension`
- structured-output forwarding support
- detailed result metadata and usage accounting
- `Direct ITS Gateway` profile
- `External-Processing Gateway` profile
- observability surface
- benchmark and example toolkit

### 15.3 Operational Validation Before Production

- Run the `Real Integration Profile` from Section 14.10 with valid credentials and network access.
- Verify timeout, retry, and fallback behavior under representative failure conditions.
- Verify budget and concurrency guardrails on the target deployment.
- Verify secret handling and log redaction behavior on the target environment.

## Appendix A. Spec Evolution

### A.1 Versioning and Breaking Changes

This specification is versioned by draft/release label.

Breaking changes include:

- removing or renaming canonical fields
- changing required behaviors for an existing conformance profile
- changing gateway activation or response requirements incompatibly

Minor or additive changes include:

- adding OPTIONAL fields
- adding OPTIONAL extension profiles
- tightening documentation requirements without changing behavior

### A.2 Experimental Graduation

Experimental profiles MAY graduate to stable in a future version when:

- their budget semantics are specified normatively
- their invariants are strong enough for cross-implementation conformance testing
- their failure behavior and validation rules are specified precisely enough for independent
  implementations to converge
