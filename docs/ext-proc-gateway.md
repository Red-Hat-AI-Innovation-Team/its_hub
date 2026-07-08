# External-Processing Gateway (Envoy ext_proc)

The External-Processing Gateway implements the `External-Processing Gateway Extension`
profile from [SPEC.md](../SPEC.md) Section 9.4. It sits in front of an upstream
OpenAI-compatible API as an Envoy external processor and conditionally applies
inference-time scaling to chat completion requests.

## Activation Model

ITS activation is conveyed **out-of-band** via HTTP request headers, not the request
body. The gateway recognizes these headers (case-insensitive):

| Header | Required | Description | Validation |
|---|---|---|---|
| `X-ITS-Budget` | Yes | Number of candidate generations (compute budget) | Integer, 1-1000 inclusive |
| `X-ITS-Endpoint` | Yes | Downstream LLM API endpoint URL | Non-empty string |
| `X-ITS-API-Key` | No | API key for the downstream LLM endpoint | Passed through as-is |

The `model` field is read from the **request body** (standard OpenAI `model` field),
not from headers.

### Validation Rules

- If `X-ITS-Budget` or `X-ITS-Endpoint` is absent, the request passes through without
  ITS.
- If `X-ITS-Budget` is not a valid integer, the request passes through.
- If `X-ITS-Budget` is outside the range 1-1000, the request passes through.
- If the request body does not contain a `model` field, the request passes through.

### Header Sanitization

All `X-ITS-*` headers are **stripped** from the request before it is forwarded upstream.
Upstream services never see ITS activation metadata, regardless of whether ITS is
applied or the request passes through.

## Outcomes

The gateway produces one of two outcomes for each intercepted request:

### pass_through

The original request continues to the upstream LLM service unmodified (except for
`X-ITS-*` header removal). This happens when:

- The request path is not `/v1/chat/completions`
- ITS activation headers are absent or invalid
- The request body is missing the `model` field
- ITS execution fails (safe fallback)

### its_applied

The gateway runs the ITS algorithm, short-circuits the upstream request, and returns an
OpenAI-compatible response directly. The response includes an `x-its-applied: true`
header to signal that ITS was applied.

## Algorithm Wiring

The gateway uses `SelfConsistency` as its algorithm. The algorithm is initialized once at
service startup and reused across all requests. The `budget` header controls the number
of parallel candidate generations per request.

Algorithm selection is not configurable per-request in this version.

## LM Routing

Downstream LLM routing is controlled **per-request** via the `X-ITS-Endpoint` header.
The gateway maintains an in-memory cache of LM clients keyed by
`(endpoint, model, hashed_api_key)` to reuse HTTP connections across requests. Different
requests can target different LLM endpoints. The API key is SHA-256 hashed (truncated to
16 hex chars) in the cache key to prevent credential cross-contamination between requests
using different API keys for the same endpoint/model pair.

## Concurrency Control

The gateway uses `LMOrchestrator` with an `asyncio.Semaphore` to bound the number of
concurrent LM calls. The default concurrency limit is set by `LMOrchestrator`'s default.
All requests share the same orchestrator instance and concurrency pool.

## Response Contract

When ITS is applied, the response is OpenAI-compatible:

```json
{
  "id": "chatcmpl-its-<hash>",
  "object": "chat.completion",
  "created": 1730000000,
  "model": "the-model-name",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "selected response"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 200,
    "total_tokens": 300
  }
}
```

### Usage Reporting

The `usage` field reports token counts **aggregated across all LM calls** in the ITS
execution, not just the selected candidate. Usage is accumulated by summing
`prompt_tokens` and `completion_tokens` from each individual LM generation call. If the
downstream LLM does not report usage, all fields are zero.

## Failure Policy

The gateway follows a **safe fallback** policy throughout:

| Failure | Behavior |
|---|---|
| Invalid or missing activation headers | pass_through |
| Budget out of range (1-1000) | pass_through |
| Missing `model` in request body | pass_through |
| ITS algorithm execution error | pass_through |
| Unsupported route (not `/v1/chat/completions`) | pass_through |

The gateway never returns an error response to the client for ITS-related failures.
Failed ITS requests are logged at ERROR level and fall back to the upstream service.

## Timeout and Retry Behavior

- **Timeouts**: The gateway does not impose its own timeout on ITS execution. Timeout
  behavior is inherited from the underlying `OpenAICompatibleLanguageModel`, which uses
  the OpenAI client's default timeout settings and `backoff` retry logic for transient
  API errors.
- **Retries**: Individual LM generation calls are retried with exponential backoff for
  transient errors (rate limits, connection errors, server errors). The retry policy is
  defined by `OpenAICompatibleLanguageModel`'s backoff configuration, not the gateway.
- **gRPC deadline**: Envoy may impose its own deadline on the ext_proc stream. If the
  gRPC stream is terminated before ITS completes, the gateway logs the error and the
  request is handled by Envoy's configured failure policy.

## Trust Boundary

All client-supplied inputs are treated as untrusted:

- **Headers**: `X-ITS-Budget` is validated to integer within 1–1000. `X-ITS-Endpoint` and
  `X-ITS-API-Key` are used as-is but never logged.
- **Request body**: `model` and `messages` are passed to the algorithm without sanitization.
  Malformed JSON is caught and results in pass-through.
- **Tool arguments**: `tools` and `tool_choice` are forwarded to the downstream LLM as-is.
- **Model outputs**: Downstream LLM responses are returned to the client without
  sanitization.

## Secret Handling

- API keys are passed via `X-ITS-API-Key` header per request. They are **never logged**.
- API keys are not stored persistently. They exist only for the duration of the LM client
  that uses them, cached in memory by `(endpoint, model, hashed_api_key)`.
- Secrets are not read from config files. The gateway has no configuration file; all
  per-request credentials come from headers.
- `X-ITS-*` headers (including `X-ITS-API-Key`) are stripped before forwarding to
  upstream.

## Restart and Scaling

- **State**: The gateway holds an in-memory LM client cache and algorithm instance. No
  state is persisted to disk.
- **Restart**: Restarting the service clears all cached LM clients. Pending requests are
  lost. No warm-up or state recovery is needed.
- **Horizontal scaling**: Multiple gateway instances can run independently. Each instance
  maintains its own LM client cache and concurrency pool. There is no shared state
  between instances.

## Streaming

Streaming is not supported. If the client sends `"stream": true` in the request body,
the gateway ignores the field and returns a non-streaming response when ITS is applied.
Pass-through requests preserve the client's `stream` field for the upstream to handle.

## Running the Service

```bash
# Start the ext_proc gRPC server
envoy-grpc --port 50051

# With debug logging
envoy-grpc --port 50051 --log-level DEBUG
```

The service binds a gRPC server that implements the Envoy External Processor protocol.
Configure Envoy to route to this service as an external processor. See
`config/envoy/ext_proc.yaml` for a reference Envoy configuration.
