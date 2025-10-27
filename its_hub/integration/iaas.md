# Inference-Time Scaling Proxy Architecture

## Background
The current MVP (`its_hub/integration/iaas.py`) exposes inference-time scaling (ITS) through an in-process FastAPI server. It keeps algorithm instances (`LM_DICT`, `SCALING_ALG`) as global state, reads a `budget` field from the JSON body, and directly orchestrates ITS loops before proxying responses back in OpenAI format. Reward functions (`its_hub/integration/reward_hub.py`) execute inside the same process and may call local vLLM workers or LLM-as-a-judge endpoints synchronously. While this approach works for experimentation, it lacks production guardrails (auth, rate limiting, circuit breaking, observability) and does not scale horizontally.

As ITS moves toward production, we need a proxy layer that:
- Remains OpenAI-compatible while ingesting ITS-specific metadata from headers (for example, `X-ITS-Budget`, `X-ITS-Alg`).
- Executes ITS algorithms across multiple downstream LLM calls safely, including reward-model scoring that can depend on GPU-heavy models or remote evaluators.
- Leverages industry-proven, lightweight gateways instead of bespoke web servers, so we inherit battle-tested reliability, security, and operational tooling.

Our first production implementation will be built on Envoy with the external processing (ext_proc) filter. Envoy gives us hardened L7 proxy capabilities (mTLS, authentication, rate limiting, retries, circuit breaking, observability) while letting us inject custom logic via a gRPC ext_proc service that can run ITS orchestration. This document lays out that architecture, details the implementation steps, and then surveys alternative gateway options that can host the same ITS logic later.

## Implementation Status

### Current State (MVP)
**Location**: `its_hub/integration/iaas.py:1`, `its_hub/integration/reward_hub.py`

**What Works**:
- OpenAI-compatible FastAPI server with `/v1/chat/completions` endpoint
- Support for self-consistency, best-of-n, and particle-filtering algorithms
- Budget-based inference-time scaling via request body parameter
- Integration with vLLM process reward models (`LocalVllmProcessRewardModel`)
- LLM-as-a-Judge support via `LLMJudgeRewardModel` adapter
- Async generation with `OpenAICompatibleLanguageModel` and `LiteLLMLanguageModel`
- Tool calling support for self-consistency algorithm
- Configuration endpoint (`/configure`) for runtime setup

**Current Limitations**:
- Global mutable state (`LM_DICT`, `SCALING_ALG`) prevents multi-tenancy and horizontal scaling
- No authentication, authorization, or API key validation
- No rate limiting or quota enforcement per tenant/user
- Synchronous reward model scoring blocks request processing (uses `asyncio.to_thread`)
- No circuit breaking or graceful degradation for upstream failures
- Minimal observability (basic logging only, no metrics or distributed tracing)
- No streaming support for chat completions
- Hardcoded token counting (returns 0 for all usage statistics)
- Single-node deployment model with no failover capability
- Configuration stored in-memory (lost on restart)

### Target State (Production Architecture)
**What We're Building**:
- Stateless ext_proc service with request-scoped configuration lookup
- Envoy-managed TLS termination, authentication (JWT/API keys), and rate limiting
- External reward service with GPU resource pooling and queuing
- Distributed tracing (OpenTelemetry), Prometheus metrics, and structured logging
- Multi-tenant configuration with Redis/PostgreSQL persistence
- Circuit breakers, retries with backoff, and timeout enforcement
- Horizontal scaling for both ext_proc and reward services
- Kubernetes-native deployment with autoscaling and GPU scheduling

### Migration Path
**Phase 1: Library Refactoring**
- Extract orchestration logic from `iaas.py` into reusable `ITSOrchestrator` class
- Abstract LM client to support Envoy-backed HTTP calls
- Refactor reward models for remote execution (`GrpcRewardClient`, `HttpRewardClient`)
- Define configuration schema for models, algorithms, and reward endpoints

**Phase 2: ext_proc Service**
- Implement gRPC external processing service skeleton
- Integrate refactored `ITSOrchestrator` with request/response handling
- Add header-based ITS metadata parsing (`X-ITS-Budget`, `X-ITS-Alg`)
- Implement pass-through mode for non-ITS requests

**Phase 3: Reward Service Isolation**
- Containerize `LocalVllmProcessRewardModel` with GPU runtime
- Build gRPC/HTTP API with concurrency controls
- Deploy with resource limits and autoscaling

**Phase 4: Envoy Configuration & Integration**
- Configure Envoy listeners, clusters, and ext_proc filter
- Enable mTLS, rate limiting, and circuit breakers
- Set up observability stack (Prometheus, Jaeger/Zipkin, ELK/Loki)

**Phase 5: Validation & Rollout**
- Integration testing with docker-compose
- Load testing for ext_proc and reward service sizing
- Staged rollout with feature flags and traffic mirroring
- Gradual cutover with error budget monitoring

## Envoy External Processing Architecture
```mermaid
flowchart TD
      Client["Client<br/>(OpenAI-compatible caller)"]

      subgraph Gateway["Existing Envoy Gateway"]
          Listener["Listener / TLS / Auth / Routing"]
      end

      subgraph ITSFilterBlock["ITS ext_proc filter"]
          ITSFilter["HTTP ext_proc filter"]
      end
      style ITSFilterBlock stroke-dasharray: 5 5;

      subgraph ITSProcessorBlock["ITS ext-proc gRPC service"]
          ExtProcService["Ext-Proc handler (policy lookup)"]
          ConfigCache["Config registry<br/>(model → algorithm)"]
      end
      style ITSProcessorBlock stroke-dasharray: 5 5;

      subgraph ITSHub["its_hub library"]
          ITSHubCore["ITS orchestration<br/>(validation, algorithm, LM client)"]
      end

      LlmDRoute["Existing Envoy Gateway<br/>(llm-d backed inference)"]
      ExternalProvider["External LLM provider<br/>(OpenAI, etc.)"]

      Client -- "1" --> Listener
      Listener -- "2" --> ITSFilter
      ITSFilter -. "3 (gRPC)" .-> ExtProcService
      ExtProcService -- "4" --> ConfigCache
      ExtProcService -- "5" --> ITSHubCore

      ITSHubCore -- "6a" --> LlmDRoute
      ITSHubCore -- "6b" --> ExternalProvider
      LlmDRoute -- "7a" --> ITSHubCore
      ExternalProvider -- "7b" --> ITSHubCore

      ITSHubCore -- "8" --> ExtProcService
      ExtProcService -- "9" --> ITSFilter
      ITSFilter -- "10" --> Listener
      Listener -- "11" --> Client
```

### High-Level Flow
```
client → Envoy (HTTP filter chain)
          ├─ ext_proc filter → ITS Orchestrator Service (gRPC)
          │                    ├─ ITS Core Library (refactored from iaas.py)
          │                    ├─ Reward Service (local vLLM or remote judge)
          │                    └─ Downstream LLM Calls (via Envoy cluster or direct HTTP)
          └─ Upstream cluster → OpenAI-compatible provider(s)
```

1. Client sends a standard OpenAI Chat Completions request with additional ITS headers.
2. Envoy performs front-door auth/rate limiting and invokes the ext_proc filter.
3. ext_proc forwards the request to a dedicated ITS Orchestrator service.
4. The orchestrator reads ITS headers, runs the scaling algorithm using the refactored `its_hub` library, issues multiple LLM calls (via Envoy or direct), and invokes reward models as needed.
5. Once the orchestrator selects the final response, it returns an OpenAI-compatible payload back to Envoy, which relays it to the client.

### Component Responsibilities
- **Envoy Gateway**: terminates TLS, authenticates, enforces rate/quotas, fuses in observability (access logs, metrics, distributed tracing), applies retries/circuit breaking, and forwards ext_proc gRPC messages.
- **ITS Orchestrator Service**: stateless gRPC service that hosts the refactored ITS algorithms, manages per-request execution (budget enforcement, tool routing, failures), and coordinates reward scoring through asynchronous workers.
- **Reward Service**: optional out-of-process component that wraps `LocalVllmProcessRewardModel` or LLM Judge, exposing a small gRPC/HTTP API with concurrency controls to avoid blocking the orchestrator.
- **Downstream LLM Connectivity**: either reuse Envoy clusters (preferred) so the orchestrator sends HTTP requests through Envoy, or embed a hardened HTTP client with mTLS and retries if direct access is required.

## Implementation Guide
The steps below are designed to bootstrap engineering work or hand implementation to another team. Each step assumes a Kubernetes-style deployment, but the same instructions apply to VM-based environments with minor adjustments.

### 1. Refactor ITS core for gateway integration
1. Extract orchestration logic from `its_hub/integration/iaas.py` into a new module, e.g., `its_hub.integration.orchestrator`.
   - Provide an `ITSOrchestrator` class with methods `configure_model`, `run_chat_completion`, and `shutdown`.
   - Remove FastAPI globals; instead accept explicit dependencies (model registry, algorithm registry, reward client).
2. Replace direct instantiation of `OpenAICompatibleLanguageModel` with a provider interface that can call through Envoy (HTTP client abstraction).
   - Create adapters for streaming vs. non-streaming responses, even if streaming is deferred.
3. Update reward wrappers (`its_hub/integration/reward_hub.py`) to support remote execution.
   - Factor out synchronous calls into `RewardClient` abstractions (e.g., `GrpcRewardClient`, `HttpRewardClient`) that the orchestrator can await.
4. Document configuration schema (YAML/JSON) for models, algorithms, budgets, and reward endpoints; store in Redis/Postgres or ConfigMap for runtime lookup.

### 2. Build the Envoy ext_proc service
1. Define the gRPC service using Envoy’s `external_processing.proto`. The service must implement:
   - `Process(Request) → Response` for header/body interception.
   - Optional streaming interfaces if we need to inspect chunks (not required for the first iteration).
2. Implement the service in Python (gRPC) or Go (for higher performance). Recommended layout:
   ```
   its_ext_proc/
     server.py              # gRPC server bootstrap
     processor.py           # ExtProc handler class
     orchestrator_client.py # Thin wrapper around ITSOrchestrator
     envoy/
       external_processing.proto
   ```
3. Request handling flow inside `processor.py`:
   - Parse request metadata (method, path, headers). Validate required headers (`X-ITS-Budget`, optional `X-ITS-Alg`, `X-ITS-Reward-Profile`).
   - Read/parse JSON body from `HttpBody` message. Forward to `ITSOrchestrator.run_chat_completion()`.
   - Pass along client headers needed for downstream LLM auth (`Authorization`, `OpenAI-Organization`, etc.). Encode them in the orchestrator request context.
   - Short-circuit if ITS headers are absent: return `CONTINUE` so Envoy forwards request upstream without modification.
   - On ITS execution:
     1. Call orchestrator with budget/algorithm.
     2. Orchestrator issues N LLM calls. Calls should use Envoy cluster by making HTTP requests to `http://envoy-cluster-name/v1/chat/completions` (loop back through Envoy) or by leveraging direct provider clients with built-in retry policies.
     3. Orchestrator invokes reward client; enforce timeouts and fallbacks (e.g., degrade to single completion if reward service unavailable).
     4. Construct final OpenAI-compatible response JSON.
   - Populate `HttpResponse` with final payload, status 200, and pass headers like `Content-Type: application/json`.
4. Add resilience:
   - Wrap orchestrator calls with deadlines shorter than Envoy timeouts.
   - Map exceptions to appropriate HTTP statuses (`429` for quota exhaustion, `503` for upstream failure, `500` for internal error).
   - Emit structured logs (request id, user, model, algorithm, budget, latency, reward score).
5. Package the service (Dockerfile with minimal base, e.g., `python:3.11-slim`). Include health/metrics endpoints (e.g., `/healthz`, `/metrics`) served via aiohttp or Prometheus client.

### 3. Deploy reward model microservice
1. For vLLM-based scoring:
   - Containerize `reward_hub.vllm.reward.VllmProcessRewardModel` with GPU runtime (CUDA image).
   - Expose gRPC/HTTP API: `ScoreRequest { messages[], responses[] } → ScoreResponse { scores[] }`.
   - Implement concurrency limits and queueing (e.g., asyncio semaphore) to prevent GPU overload.
2. For LLM Judge:
   - Create a service that proxies LiteLLM requests; store provider keys securely (Vault/KMS).
   - Cache judge prompts per criterion to reduce latency.
3. Provide deployment manifests with resource requests, tolerations for GPU nodes, and autoscaling rules.
4. Update orchestrator configuration to point to reward service endpoints, with per-tenant overrides.

### 4. Configure Envoy
1. Define clusters:
   - `its_ext_proc` (gRPC) → orchestrator service.
   - `openai_primary` (HTTP) → OpenAI endpoint or upstream gateway.
   - Optional `reward_service` if orchestrator wants to proxy via Envoy.
2. Configure listeners and HTTP connection manager:
   ```yaml
   static_resources:
     listeners:
       - name: its_listener
         address:
           socket_address: { address: 0.0.0.0, port_value: 8443 }
         filter_chains:
           - filters:
               - name: envoy.filters.network.http_connection_manager
                 typed_config:
                   "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                   stat_prefix: its_hcm
                   route_config:
                     name: default
                     virtual_hosts:
                       - name: its
                         domains: ["*"]
                         routes:
                           - match: { prefix: "/v1/chat/completions" }
                             route:
                               cluster: openai_primary
                               timeout: 30s
                   http_filters:
                     - name: envoy.filters.http.ext_proc
                       typed_config:
                         "@type": type.googleapis.com/envoy.extensions.filters.http.ext_proc.v3.ExternalProcessor
                         grpc_service:
                           envoy_grpc:
                             cluster_name: its_ext_proc
                         processing_mode:
                           request_body_mode: BUFFERED
                           response_body_mode: BUFFERED
                           request_header_mode: SEND
                           response_header_mode: SKIP
                     - name: envoy.filters.http.router
   ```
3. Enable production safeguards:
   - mTLS between Envoy and ext_proc/reward services.
   - Rate limiting via `envoy.filters.http.local_ratelimit` (per API key) plus global rate limits via Redis.
   - Circuit breakers (`max_requests`, `max_connections`) on `openai_primary` and `its_ext_proc`.
   - Retries with backoff on upstream cluster (respect provider limits).
4. Observability:
   - Enable access logs with additional headers (`x-request-id`, `x-its-budget`, `x-its-alg`).
   - Emit Prometheus metrics; configure tracing (Zipkin/OTel) for ext_proc interactions.
5. Security:
   - Require API key/JWT validation via `envoy.filters.http.jwt_authn` or ext_authz.
   - Sanitize ITS headers before forwarding to upstream providers to avoid leaking internal metadata.

### 5. Observability, testing, and rollout
1. Integration tests: spin up Envoy + ext_proc in docker-compose; test scenarios (with/without ITS headers, reward timeout, upstream failure).
2. Load testing: simulate concurrent ITS workloads to size ext_proc instances and GPU reward workers.
3. Staging rollout: deploy behind feature flag; mirror production traffic with ITS disabled, validate latency/availability.
4. Cutover: gradually enable ITS per tenant or per API key. Monitor error budgets, fallback to pass-through mode if SLA risk detected.

## Alternative Implementations
While Envoy ext_proc is our first target, other open-source gateways can host ITS logic with less custom infrastructure.

### LiteLLM Proxy
- Python-based OpenAI-compatible gateway with built-in auth, rate limiting, provider fallbacks, caching, and Prometheus metrics.
- Integration approach: implement a LiteLLM custom provider or pre-call hook that invokes the refactored `ITSOrchestrator`. Downstream calls reuse LiteLLM’s provider clients, and reward services stay external.
- Pros: minimal glue code, existing ecosystem, straightforward deployment. Cons: Python runtime limits (GIL), need to isolate reward workloads to avoid blocking.

### Helicone Gateway
- Rust gateway focused on analytics and request logging for OpenAI-compatible APIs (Postgres + ClickHouse).
- Integration approach: use Helicone transformers/webhooks to route ITS-specific requests to an orchestrator service that returns responses to Helicone for delivery.
- Pros: strong observability out of the box. Cons: requires external orchestrator for ITS loops; less control over inline request modification.

### Envoy/Kong Front with LiteLLM Core
- Run a managed ingress proxy (Envoy Gateway or Kong) for edge policies while delegating ITS orchestration to LiteLLM behind it.
- Provides layered defense: enterprise ingress features plus Python-based extensibility for ITS logic.

## Open Questions
- Streaming support: ext_proc currently buffers bodies; decide if/when to support streaming completions.
- Reward model isolation: determine acceptable latency/throughput for reward scoring and whether synchronous scoring suffices.
- Configuration management: choose between service discovery (Redis/Postgres) or declarative configs (GitOps) for models, budgets, and tenant overrides.

Document owners should revisit this plan after the initial Envoy build to confirm assumptions and decide whether to invest in alternate gateways for specific deployment environments.
