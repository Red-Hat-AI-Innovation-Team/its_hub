# its_hub Repository Specification

Status: Draft v1 (language-agnostic)

Purpose: Define a library-first system for inference-time scaling (ITS) of LLMs.

This specification defines the repository-family contract, including:

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
  orchestration, and reward models.
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
  - reward models
- Support both string prompts and structured chat messages.
- Preserve OpenAI-compatible assistant message structure, including tool calls, where supported.
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
   - Normalizes prompts, chat history, multimodal content, and tool-call-bearing messages.

2. `Language Model Capability`
   - Produces assistant responses from normalized message inputs.

3. `Orchestration Capability`
   - Fans out one logical ITS request into multiple LM calls, preserves ordering, and applies
     concurrency control.

4. `ITS Algorithm Capability`
   - Implements sampling, voting, scoring, or search logic over one or more LM calls.

5. `Reward Capability`
   - Scores final candidates and/or intermediate reasoning steps.

6. `Gateway Adapter` (OPTIONAL)
   - Exposes or intercepts OpenAI-compatible requests and maps them onto the core library
     contract.

7. `Research / Tooling Surface` (OPTIONAL)
   - Provides examples, benchmarks, and evaluation helpers.

8. `Observability Surface` (OPTIONAL)
   - Exposes logs, metrics, and implementation-defined runtime status.

### 3.2 Abstraction Levels

`its_hub` is easiest to port when kept in these layers:

1. `Core Domain Layer`
   - Messages, candidates, results, budget semantics, step-generation configuration.

2. `Execution Layer`
   - LM generation, orchestration, reward-model scoring, algorithm execution.

3. `Gateway Layer`
   - Direct or intercepting request handling built on top of the execution layer.

4. `Research and Tooling Layer`
   - Documentation, examples, benchmarks, validation assets.

An implementation MAY collapse multiple layers into one module or service so long as the observable
behavior remains conformant.

### 3.3 Capability Profiles

This specification defines these conformance profiles:

1. `Core Library Conformance`
   - Message model
   - LM capability
   - orchestration capability or equivalent
   - scaling algorithm capability
   - reward capability

2. `Stable Algorithm Extension`
   - `SelfConsistency`
   - `BestOfN`

3. `Experimental Search Extension`
   - Step-wise search family
   - beam search
   - particle methods
   - planning wrapper

4. `Direct Gateway Extension`
   - Standalone service that terminates client requests and applies ITS directly.

5. `External-Processing Gateway Extension`
   - Intercepting gateway that conditionally applies ITS in front of an upstream API.

6. `Research Toolkit Extension`
   - Examples, benchmark runners, dataset evaluation tooling.

Composition rules:

- Every conforming implementation MUST satisfy `Core Library Conformance`.
- Sections marked OPTIONAL are extension profiles.
- An implementation MAY support multiple extensions at once.

### 3.4 External Dependencies

Common external dependencies include:

- one or more downstream language-model providers
- OPTIONAL reward-model services or local scoring runtimes
- OPTIONAL HTTP, proxy, or gRPC runtime for gateway implementations
- local filesystem for docs, examples, test fixtures, or benchmark assets
- host environment authentication and secret management

The specification does not require any one vendor, network topology, or packaging format.

## 4. Core Domain Model

### 4.1 Entities

#### 4.1.1 `ChatMessage`

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
  - `list[object]` for structured or multimodal content
  - `null` when the message is represented primarily by tool calls
- `tool_calls` (OPTIONAL list of structured tool-call records)
- `tool_call_id` (OPTIONAL string)

Required semantics:

- Tool-call-bearing assistant messages MUST preserve `tool_calls` when returned by LM or gateway
  paths.
- Text-bearing content MUST preserve text parts in order.
- Image-bearing content MAY be ignored by text-only flows, but that behavior MUST be documented.
- Implementations SHOULD tolerate unknown inbound fields for forward compatibility.

#### 4.1.2 `ChatMessages`

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

#### 4.1.3 `ScalingCandidate`

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

#### 4.1.4 `ScalingResult`

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
- Full result objects are OPTIONAL, but implementations SHOULD support an output mode that exposes
  detailed result metadata for debugging and research.

#### 4.1.5 `GenerationUsage`

When token usage is exposed, the interoperable shape is:

- `prompt_tokens`
- `completion_tokens`
- `total_tokens`

All fields are non-negative integers.

If an implementation cannot determine usage exactly, it MUST document whether usage is omitted,
estimated, or reported as zero/unknown.

#### 4.1.6 `StepGenerationConfig`

Configuration used by step-wise search algorithms.

Fields:

- `max_steps`
- exactly one of:
  - `step_token`
  - `tokens_per_step`
- `stop_token` (OPTIONAL)
- `temperature` (OPTIONAL)
- `include_stop_str_in_output` (OPTIONAL)
- `temperature_switch` (OPTIONAL)

Invariants:

- Exactly one of `step_token` and `tokens_per_step` MUST be set.
- `tokens_per_step` MUST be positive when used.

### 4.2 Normalization and Compatibility Rules

- A string prompt MUST normalize into one `user` message.
- Structured chat is the primary cross-implementation contract.
- Prompt-string fallback MAY exist for compatibility-oriented or experimental paths.
- Tool-call-bearing assistant messages SHOULD remain attached to candidates and selected results.
- Implementations MAY accept additional provider-native content parts, but they MUST document how
  text extraction behaves for those parts.

## 5. Core Library Contract

### 5.1 Input Normalization

Algorithms and reward capabilities MUST accept normalized conversation input in one of these forms:

- string prompt
- list of normalized messages
- implementation-equivalent `ChatMessages` wrapper

Normalization MUST occur before algorithm-specific logic.

### 5.2 Language Model Capability

The core contract is async-first.

A conforming implementation MUST provide asynchronous LM generation behavior equivalent to:

- generate one assistant response from one normalized conversation

Batch generation can be implemented in either of two ways:

- the LM capability accepts batched inputs directly
- a separate orchestration capability fans out multiple single-generation LM calls

Required semantics:

- LM generation SHOULD return assistant messages in OpenAI-compatible structure where possible.
- LM generation SHOULD preserve `tool_calls` when the provider returns them.
- LM generation MAY accept provider-native arguments such as:
  - `max_tokens`
  - `temperature`
  - `tools`
  - `tool_choice`
  - `response_format`
- Retry, backoff, session reuse, and connection pooling are implementation-defined.

Convenience behavior:

- A synchronous wrapper MAY be provided.
- An implementation MAY expose both single-conversation and batched generation operations.

### 5.3 Orchestration Capability

Orchestration is a required behavior, but it does not need to exist as a first-class public type.

Responsibilities:

- fan out one logical ITS request into multiple LM calls
- preserve input/output ordering
- forward LM-generation arguments consistently
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
- invoke LM generation and, where needed, reward capabilities
- select a final assistant response

Required semantics:

- Algorithms MUST accept both prompt-style and chat-style input.
- Algorithms MUST document how they interpret `budget`.
- Algorithms SHOULD preserve tool-call-bearing assistant messages when the LM capability does.
- Algorithms MAY depend on an explicit orchestration capability or inline equivalent batching/fanout
  behavior.
- Algorithms MAY expose either:
  - selected response only
  - selected response plus full algorithm result metadata

### 5.5 Outcome Reward Capability

Outcome reward models score final candidates.

Required behavior:

- score one or more complete candidates with enough prompt/conversation context to make the score
  meaningful
- higher score means better candidate

Interface freedom:

- An implementation MAY score:
  - prompt + candidate
  - full conversation including candidate
  - batched conversations
- An implementation MAY expose sync, async, or both, so long as the overall algorithm behavior is
  conformant

### 5.6 Process Reward Capability

Process reward models score intermediate reasoning steps.

Required behavior:

- accept prompt or conversation context plus ordered reasoning steps
- return per-step scores or an equivalent structure that preserves per-step evaluation meaning

The minimal interoperable shape is a list of numeric scores aligned with the step list.

### 5.7 Async-First and Sync Wrappers

The specification is async-first.

Required behavior:

- primary execution contracts SHOULD be async
- synchronous wrappers MAY be provided as convenience layers

This specification does not require any one event-loop or threading model.

### 5.8 Structured Output, Tool Calling, and Provider-Native Args

Structured-output and tool-calling support are part of the extensible LM call surface.

Required semantics:

- Implementations that support tool calling MUST preserve returned tool-call structure on assistant
  messages.
- Implementations that support provider-native structured output MAY forward arguments such as
  `response_format`.
- Algorithms are not required to interpret provider-native structured-output arguments themselves,
  but they SHOULD forward them when that is part of the documented design.

### 5.9 Usage and Metadata

Usage and metadata are OPTIONAL parts of the core contract.

If exposed:

- per-candidate usage MUST refer to one LM generation
- aggregated usage MUST refer to one logical ITS execution
- algorithm metadata MUST be documented as algorithm-specific, not assumed to be universal

## 6. Stable Algorithm Profiles

### 6.1 Shared `budget` Vocabulary

`budget` is the shared compute-control vocabulary for one ITS execution.

Required semantics:

- `budget` MUST be a positive integer
- all algorithms MUST document how `budget` maps to compute
- identical `budget` values across different algorithms do not imply identical cost

### 6.2 `SelfConsistency`

Behavior:

- Generate multiple candidate responses.
- Project each candidate into a comparison space.
- Select the most common projection, with implementation-defined tie-breaking.

Projection behavior:

- The default comparison space MAY be exact or normalized text content.
- Implementations MAY support explicit projection functions, including regex-based projections.
- If a projection fails to match, the implementation MUST document whether the candidate is ignored,
  grouped under a null value, or treated as raw content.

Tool-calling behavior:

- Tool-call-bearing responses MAY participate in consistency voting.
- If tool-call voting is supported, the implementation MUST document the voting modes.
- Common modes include:
  - tool identity only
  - tool arguments only
  - hierarchical combination of tool identity and arguments

Result metadata MAY include:

- all candidate responses
- vote counts
- selected index

Budget semantics:

- `budget` is the number of candidate generations.

### 6.3 `BestOfN`

Behavior:

- Generate `N` candidate responses.
- Score them with an outcome reward capability.
- Select the highest-scoring candidate, with implementation-defined tie-breaking.

Scoring behavior:

- Implementations MAY score all candidates directly.
- Implementations MAY deduplicate semantically equivalent candidates before scoring to reduce cost.
- If tool-calling is supported, semantic equivalence SHOULD consider both content and
  tool-call-bearing structure.

Result metadata MAY include:

- all candidate responses
- scores
- selected index

Budget semantics:

- `budget` is the number of candidate generations to score.

## 7. Experimental Algorithm Profiles

Experimental algorithms are OPTIONAL extensions.

They are part of the repository-family design, but they are not part of the stable core contract in
the same way as `SelfConsistency` and `BestOfN`.

### 7.1 Step-Wise Search Family

Common characteristics:

- use `StepGenerationConfig`
- generate partial reasoning trajectories rather than only final answers
- may depend on process reward scoring
- may rely on prompt-string compatibility paths more than the stable core

### 7.2 `BeamSearch`

Behavior:

- expand a bounded set of partial trajectories
- retain the best partial candidates according to implementation-defined scoring
- continue until stopping criteria are met

Budget semantics:

- `budget` represents total search effort rather than a universal number of final candidates

Required documentation:

- beam-width policy
- step stopping criteria
- score aggregation policy

### 7.3 Particle Methods

This family includes profiles such as:

- `ParticleFiltering`
- `ParticleGibbs`
- entropic or annealed particle variants

Behavior:

- maintain multiple evolving trajectories
- resample or update trajectories according to implementation-defined particle rules
- use process reward or equivalent scoring to guide evolution

Budget semantics:

- `budget` typically represents number of particles or equivalent total particle effort

Required documentation:

- resampling/update policy
- score aggregation policy
- stopping criteria

### 7.4 `PlanningWrapper`

Behavior:

- allocate some compute to planning
- execute a downstream algorithm conditioned on the planning result

Budget semantics:

- `budget` combines planning cost and downstream execution cost according to a documented policy

## 8. Gateway Profiles

### 8.1 Gateway Role

Gateways are OPTIONAL integration profiles layered over the core library contract.

They do not redefine algorithm behavior. Instead, they:

- accept or intercept client requests
- map those requests onto the core ITS contract
- shape results into an OpenAI-compatible response surface

This specification version defines two gateway profiles:

1. `Direct ITS Gateway`
2. `External-Processing Gateway`

### 8.2 Shared Gateway Requirements

All gateway profiles that claim conformance MUST:

- target an OpenAI-compatible chat-completions-like request/response surface
- accept a client-visible `model` selection or documented equivalent
- accept normalized `messages` or a compatible chat body
- preserve the selected assistant message structure, including `tool_calls` where supported
- document how downstream LM routing and algorithm wiring are configured
- document timeout, retry, and failure behavior

Minimum OpenAI-compatible response shape:

- `id`
- `object`
- `created`
- `model`
- `choices`

`usage` and `metadata` are OPTIONAL but RECOMMENDED when meaningful.

### 8.3 `Direct ITS Gateway` (OPTIONAL)

#### 8.3.1 Role

A direct gateway is a standalone service that terminates client requests and applies ITS directly.

#### 8.3.2 Activation Model

ITS activation MUST be explicit in the request payload.

Canonical behavior:

- the client includes `budget` or an implementation-equivalent compute-control field in the request
  body

The direct gateway profile MUST NOT rely on proxy-only transport metadata as its primary activation
mechanism.

#### 8.3.3 Configuration Surface

A direct gateway MUST provide a documented configuration mechanism for:

- downstream LM endpoint/provider selection
- algorithm selection or policy selection
- reward-model wiring where applicable
- credentials and secret handling

The mechanism MAY be:

- static startup configuration
- a configuration file
- an admin/control API
- an external control plane

This specification does not require a single configuration topology.

#### 8.3.4 Request Contract

A direct gateway MUST accept:

- `model`
- `messages`

A direct gateway SHOULD accept:

- `budget`

A direct gateway MAY accept:

- `temperature`
- `max_tokens`
- `tools`
- `tool_choice`
- `stream`
- implementation-defined metadata fields

Unsupported optional request fields SHOULD produce explicit documented behavior rather than silent
reinterpretation.

#### 8.3.5 Response Contract

When ITS is applied, the response MUST be OpenAI-compatible and include:

- one selected assistant message in `choices[0].message`

The response MAY also include:

- `usage`
- algorithm-specific `metadata`

If usage is exposed, the implementation MUST document whether it is:

- aggregated across all LM calls used by ITS
- estimated
- or otherwise computed

#### 8.3.6 Failure Model

Configuration or wiring failures SHOULD produce explicit service errors.

Generation failures MAY be handled by:

- explicit service error
- documented fallback behavior

Direct gateways MUST NOT silently forward the request to an unknown upstream unless that behavior is
an explicit part of the documented design.

#### 8.3.7 State and Scaling Notes

Direct gateways MAY use:

- static configuration
- in-memory mutable state
- external configuration/state stores

Restart behavior, persistence, and horizontal scaling are implementation-defined and MUST be
documented.

### 8.4 `External-Processing Gateway` (OPTIONAL)

#### 8.4.1 Role

An external-processing gateway sits in front of an upstream OpenAI-compatible API and conditionally
applies ITS.

#### 8.4.2 Activation Model

ITS activation MUST be conveyed out-of-band relative to the standard request body.

Examples include:

- HTTP headers
- proxy route metadata
- gRPC metadata
- implementation-defined request context

Required semantics:

- activation metadata MUST include compute-control information equivalent to `budget`
- the implementation MUST document all recognized activation fields and validation rules

#### 8.4.3 Outcomes

The profile supports two outcomes:

1. `pass_through`
   - the original request continues to the upstream service

2. `its_applied`
   - the gateway runs ITS
   - the gateway returns an OpenAI-compatible response directly
   - the upstream request is short-circuited

#### 8.4.4 Request Resolution

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

#### 8.4.5 Response Contract

When ITS is applied, the response MUST be OpenAI-compatible and include:

- the selected assistant message

The response SHOULD also include an implementation-defined signal that ITS was applied, such as:

- response header
- response metadata
- proxy-local observability event

If usage is exposed, it SHOULD refer to the full logical ITS execution, not only the selected
candidate.

#### 8.4.6 Failure Model

Safe fallback is RECOMMENDED.

If safe fallback is chosen:

- invalid activation metadata -> `pass_through`
- unsupported route -> `pass_through`
- missing required request-body fields -> `pass_through`
- ITS execution failure after activation -> `pass_through`

If explicit errors are chosen instead, the implementation MUST document that policy clearly.

#### 8.4.7 Safety Notes

External-processing gateways SHOULD sanitize or strip internal ITS activation metadata before
forwarding pass-through requests upstream when that metadata is not intended for upstream services.

### 8.5 Streaming

This specification version does not standardize streaming ITS behavior.

Implementations MAY:

- reject streaming explicitly
- buffer streaming requests and return non-streaming ITS responses
- support streaming according to an implementation-defined design

If streaming is not supported, the implementation SHOULD return an explicit documented error rather
than silently pretending to stream.

## 9. Documentation, Benchmarking, and Example Surface

### 9.1 Documentation

Repository-family documentation SHOULD cover:

- installation or setup guidance
- core algorithm concepts
- gateway profile behavior where implemented
- development and contribution notes
- benchmarking and evaluation workflows

This specification does not require fixed file paths or one documentation toolchain.

### 9.2 Benchmarking Surface

A research-oriented implementation MAY provide benchmark tooling for:

- mathematical reasoning tasks
- multi-step reasoning tasks
- quality/cost trade-off measurement across algorithms

Benchmark datasets, scripts, and storage formats are implementation-defined.

If benchmark tooling is shipped, the implementation SHOULD document:

- dataset assumptions
- evaluation procedure
- reproducibility expectations

### 9.3 Examples

Examples MAY be provided for:

- `SelfConsistency`
- `BestOfN`
- experimental search profiles
- gateway profile usage

Examples are descriptive and user-facing, not normative.

## 10. Failure Model and Recovery Strategy

### 10.1 Failure Classes

1. `Input / Normalization Failures`
   - invalid message shape
   - unsupported content form
   - malformed tool-call structure

2. `LM Generation Failures`
   - provider/network failure
   - timeout
   - provider rejection
   - malformed provider response

3. `Orchestration Failures`
   - batch fanout failure
   - concurrency-control failure
   - partial batch failure

4. `Reward Failures`
   - reward-service failure
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

### 10.2 Recovery Behavior

- Input/normalization failures:
  - fail the current request or run attempt explicitly

- LM or reward transient failures:
  - MAY be retried according to implementation-defined policy

- Orchestration failures:
  - SHOULD fail the current logical ITS execution explicitly
  - MUST NOT silently reorder candidate results

- Direct gateway configuration failures:
  - SHOULD fail explicitly and operator-visibly

- External-processing gateway failures:
  - SHOULD follow the documented fallback or explicit-error policy from Section 8.4.6

- Observability/tooling failures:
  - SHOULD NOT corrupt the correctness of the core ITS result

### 10.3 Restart and State Notes

The core library can be stateless.

Gateway implementations MAY maintain:

- in-memory configuration
- client/session caches
- request-scoped caches
- externalized state

Persistence across restarts is OPTIONAL and implementation-defined.

## 11. Security and Operational Safety

### 11.1 Trust Boundary Assumption

Implementations SHOULD assume that all of the following may be partially or fully untrusted:

- prompts and chat messages
- tool definitions
- tool-call arguments
- activation metadata
- downstream model outputs
- benchmark inputs

The specification intentionally does not mandate one trust posture, but implementations MUST
document their own.

### 11.2 Secret Handling

Implementations SHOULD:

- avoid logging raw credentials or API tokens
- validate the presence of secrets without printing them
- document how secrets are supplied and scoped

### 11.3 Tool-Calling and Structured Input Safety

Tool-calling preservation is part of the contract, but actual execution of tools is out of scope for
this specification unless an implementation explicitly adds such a feature.

Implementations SHOULD:

- preserve tool-call structure faithfully
- avoid mutating tool-call arguments silently
- document any normalization applied to structured content or tool calls

### 11.4 Gateway Safety

Gateway implementations SHOULD document:

- authentication model
- timeout policy
- retry policy
- rate limiting or quota behavior
- whether internal ITS metadata is stripped before upstream forwarding

### 11.5 Resource Exhaustion and Budget Controls

ITS can amplify LM call volume significantly.

Implementations SHOULD document guardrails for:

- maximum accepted `budget`
- concurrency limits
- timeout ceilings
- gateway backpressure behavior
- reward-model resource limits

## 12. Reference Procedures (Language-Agnostic)

This section is intentionally non-executable. It describes reference procedures that illustrate the
intended behavior of a conforming implementation without prescribing any programming language,
runtime, or internal representation.

### 12.1 Input Normalization Procedure

Input:

- string prompt, or
- normalized message list, or
- implementation-equivalent chat container

Required procedure:

1. If the input is a string prompt, convert it into a single `user` message.
2. If the input is already a normalized chat container, extract its structured message
   representation.
3. If the input is already a normalized message list, use it unchanged.
4. If the input cannot be interpreted as one of the supported forms, fail with an explicit
   input-normalization error.

Required outcome:

- downstream algorithm logic receives a normalized structured message sequence

### 12.2 Self-Consistency Reference Procedure

Input:

- normalized or normalizable conversation input
- LM capability
- `budget`
- documented projection rule

Required procedure:

1. Normalize the input conversation.
2. Create `budget` equivalent candidate-generation requests from that normalized conversation.
3. Generate one assistant response for each candidate-generation request.
4. Project each candidate response into the documented comparison space.
5. Count how many times each projected value occurs.
6. Select the winning projected value using the documented tie-break policy.
7. Return the candidate associated with the winning projected value as the selected response.

Recommended detailed output:

- selected response
- all candidate responses
- vote counts
- selected index

### 12.3 Best-of-N Reference Procedure

Input:

- normalized or normalizable conversation input
- LM capability
- outcome reward capability
- `budget`

Required procedure:

1. Normalize the input conversation.
2. Create `budget` equivalent candidate-generation requests from that normalized conversation.
3. Generate one assistant response for each candidate-generation request.
4. If the implementation documents candidate deduplication, collapse semantically equivalent
   candidates before scoring.
5. Score each candidate, or each unique candidate, using the outcome reward capability.
6. If deduplication was used, map scores back onto the original candidate set.
7. Select the candidate with the highest score using the documented tie-break policy.
8. Return that candidate as the selected response.

Recommended detailed output:

- selected response
- all candidate responses
- per-candidate scores
- selected index

### 12.4 Direct Gateway Reference Procedure

Input:

- OpenAI-compatible chat-completions-like request
- direct gateway configuration

Required procedure:

1. Validate that the gateway has enough configuration to route LM requests and apply the selected
   ITS policy.
2. Read the client-visible request fields required by the direct gateway profile, including at
   minimum:
   - `model`
   - `messages`
   - request-body compute control equivalent to `budget`, when the profile requires explicit ITS
     activation
3. Forward supported optional generation fields such as tools or tool choice according to the
   documented gateway behavior.
4. Run the configured ITS algorithm against the normalized request.
5. Shape the result into an OpenAI-compatible response.

Required outcome:

- one direct gateway response representing either:
  - successful ITS application, or
  - a documented explicit error

### 12.5 External-Processing Gateway Reference Procedure

Input:

- intercepted upstream request
- activation metadata
- external-processing gateway configuration

Required procedure:

1. Determine whether the intercepted route is eligible for ITS handling.
2. If the route is not eligible, choose `pass_through`.
3. Parse activation metadata from the documented out-of-band channel.
4. If activation metadata is absent or invalid, choose `pass_through`.
5. Read or buffer the request body as needed to obtain the required chat-completions fields.
6. If the required body fields are unavailable, follow the documented fallback-or-error policy.
7. If ITS should be applied, run the configured ITS algorithm using:
   - the upstream request's logical model selection
   - the request messages
   - the activation metadata's compute-control value
   - any forwarded optional fields such as tools or tool choice
8. If ITS succeeds, return an OpenAI-compatible immediate response and short-circuit the upstream
   request.
9. If ITS fails after activation, follow the documented fallback-or-error policy.

Required outcomes:

- `pass_through`, or
- `its_applied`

## 13. Test and Validation Matrix

A conforming implementation SHOULD include tests that cover the behaviors defined in this
specification.

Validation profiles:

- `Core Conformance`: REQUIRED for all conforming implementations
- `Extension Conformance`: REQUIRED only for OPTIONAL profiles that an implementation chooses to
  ship
- `Real Integration Profile`: RECOMMENDED environment-dependent checks before production use

Unless otherwise noted, Sections 13.1 and 13.2 are `Core Conformance`. Bullets that begin with
`If ... is implemented` are `Extension Conformance`.

### 13.1 Core Library Conformance

- string prompts normalize into one `user` message
- normalized chat preserves role/content/tool-call fields
- text extraction from structured content is consistent
- tool-call-bearing assistant messages preserve tool-call structure
- LM capability returns one response per logical input
- orchestration preserves input ordering
- orchestration forwards generation arguments consistently
- outcome reward scoring is aligned with final candidates
- process reward scoring is aligned with intermediate steps

### 13.2 Stable Algorithm Extension

- `SelfConsistency` selects repeated candidates correctly
- documented projection behavior is applied consistently
- tool-call voting behavior matches documentation if implemented
- `BestOfN` scores candidates correctly
- deduplication behavior matches documentation if implemented
- result objects or detailed outputs expose the selected response correctly
- `budget` semantics match documentation

### 13.3 Experimental Search Extension

If experimental search is implemented:

- step-generation invariants are enforced
- beam-search expansion/pruning follows documented policy
- particle methods follow documented resampling/update policy
- planning-wrapper budget allocation follows documented policy
- prompt-fallback behavior is documented where structured-chat fidelity is reduced

### 13.4 Direct Gateway Extension

If the `Direct ITS Gateway` profile is implemented:

- request-body activation drives ITS execution
- gateway rejects or handles unsupported optional fields as documented
- configuration failures are surfaced explicitly
- OpenAI-compatible response shaping is correct
- tool calls are preserved when produced by the algorithm path
- usage behavior matches documented semantics
- non-streaming behavior matches documented semantics

### 13.5 External-Processing Gateway Extension

If the `External-Processing Gateway` profile is implemented:

- unsupported routes pass through
- missing or invalid activation metadata follows documented fallback
- missing required body fields follow documented fallback or error policy
- ITS-applied responses short-circuit upstream behavior correctly
- OpenAI-compatible response shaping is correct
- ITS-applied signaling behavior matches documentation
- usage aggregation behavior matches documentation if usage is exposed

### 13.6 Research Toolkit Extension

If benchmark or research tooling is implemented:

- benchmark inputs and outputs are reproducible enough for the documented workflow
- dataset and scoring assumptions are documented
- example code remains consistent with the documented public contract

### 13.7 Real Integration Profile (RECOMMENDED)

These checks are RECOMMENDED before production use and MAY be skipped in CI when credentials or
network access are unavailable.

- Run a real downstream LM smoke test with valid credentials.
- If a reward model is part of the deployment, run a real reward-path smoke test.
- If a direct gateway is implemented, run an end-to-end chat-completions smoke test.
- If an external-processing gateway is implemented, test both:
  - pass-through behavior
  - ITS-applied short-circuit behavior
- Report skipped real-integration tests as skipped rather than silently passing them.

## 14. Implementation Checklist (Definition of Done)

Use the same validation profiles as Section 13:

- Section 14.1 = `Core Conformance`
- Section 14.2 = `Extension Conformance`
- Section 14.3 = `Real Integration Profile`

### 14.1 REQUIRED for Core Conformance

- normalized message model
- async-first LM generation capability
- orchestration behavior or equivalent batch/fanout layer
- scaling algorithm execution contract
- documented `budget` semantics
- outcome and/or process reward capability as required by shipped algorithms
- tool-call preservation for supported tool-calling providers
- documentation of failure behavior and trust boundary
- deterministic tests for supported core behavior

### 14.2 RECOMMENDED Extensions

- stable algorithms:
  - `SelfConsistency`
  - `BestOfN`
- experimental search family
- structured-output forwarding support
- detailed result metadata and usage accounting
- `Direct ITS Gateway` profile
- `External-Processing Gateway` profile
- benchmark and example toolkit

### 14.3 Operational Validation Before Production

- Run the `Real Integration Profile` from Section 13.7 with valid credentials and network access.
- Verify timeout, retry, and fallback behavior under representative failure conditions.
- Verify budget and concurrency guardrails on the target deployment.
- Verify secret handling and log redaction behavior on the target environment.
