# IaaS Service Setup Guide

This guide provides comprehensive instructions for setting up and running the its_hub Inference-as-a-Service (IaaS) with inference-time scaling algorithms.

## Overview

The IaaS service provides an OpenAI-compatible API with inference-time scaling algorithms. **Optimized for tool-calling applications** including agents, function calling, and multi-step reasoning. Currently supports **Self-Consistency**, with **Best-of-N** coming soon.

### Architecture

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  Client App  │───►│   IaaS Service  │───►│  LLM Provider    │
│              │    │                 │    │                  │
│              │    │ - Best-of-N     │    │  - OpenAI        │
│              │    │ - Self-Consist. │    │  - AWS Bedrock   │
│              │    │ - LLM Judge     │    │  - vLLM (local)  │
└──────────────┘    └─────────────────┘    └──────────────────┘
```

## Activation Model

ITS activation is conveyed **in-band** via the request body and optional HTTP headers.
The `budget` field in the request body triggers ITS; without it, the service default is
used.

### Per-Request Configuration

| Source | Field / Header | Description | Priority |
|---|---|---|---|
| HTTP header | `X-ITS-Budget` | Override compute budget | 1 (highest) |
| HTTP header | `X-ITS-Endpoint` | Override LLM endpoint | 1 |
| HTTP header | `X-ITS-API-Key` | Override API key | 1 |
| Request body | `budget` | Compute budget | 2 |
| `/configure` | `budget`, `endpoint`, `api_key` | Service defaults | 3 (lowest) |

Priority chain: **header > body > service default**. Headers are intended for Envoy
ext_proc routing but are also accepted on the standalone IaaS endpoint.

### Algorithm Selection

The algorithm is configured globally via `POST /configure` with the `alg` field.
Per-request algorithm selection is not supported. Currently only `self-consistency` is
available through the gateway.

## API Key Handling

API keys can enter the system through three paths:

1. **`/configure` body** — stored in memory as the service default; used when no
   per-request key is provided
2. **`X-ITS-API-Key` header** — per-request override, highest priority
3. **Request body** — not supported; keys must come via `/configure` or headers

**Security properties:**

- Keys are **never logged** — the gateway logs endpoint and model but not credentials
- Keys are **hashed** (SHA-256, truncated to 16 hex chars) in LM cache keys to prevent
  credential cross-contamination between requests using different API keys
- Keys are **not persisted** to disk — they exist only in memory for the lifetime of the
  service process
- On shutdown, all cached LM clients (and their associated keys) are cleared

## Prerequisites

- **Software**: Python 3.11+, its_hub library
- **API Access**: OpenAI API key, AWS credentials, or local vLLM server
- **GPU**: Optional (only if using local vLLM or local reward models)

## Quick Start

### Start IaaS Service (standalone)

```bash
uv run its-iaas --port 8109
```

**Parameters:**
- `--host`: Host to bind to (default: `127.0.0.1`, localhost only)
- `--port 8109`: Default port for IaaS service
- `--dev`: Optional development mode with auto-reload
- `--print-config`: Print the bundled Envoy config to stdout and exit

### Configure IaaS Service

The IaaS service supports different algorithm configurations based on your use case.

### Self-Consistency with Tool Voting

Best for: Tool-calling models where you want to vote on tool usage patterns.

```bash
curl -X POST http://localhost:8109/configure \
  -H "Content-Type: application/json" \
  -d '{
    "endpoint": "https://api.openai.com/v1",
    "api_key": "your-openai-api-key",
    "model": "gpt-4o-mini",
    "alg": "self-consistency",
    "regex_patterns": ["\\\\boxed\\{([^}]+)\\}"],
    "tool_vote": "tool_hierarchical",
    "exclude_tool_args": ["timestamp", "request_id", "id", "type"]
  }'
```

**Parameters:**
- `regex_patterns`: Required for self-consistency — each pattern needs a capturing group that extracts the answer to vote on (used for text responses; tool responses vote via `tool_vote`)
- `tool_vote`: Voting strategy - `"tool_name"`, `"tool_args"`, or `"tool_hierarchical"` (recommended)
- `exclude_tool_args`: List of argument names to exclude from voting (e.g., timestamps, IDs)

---

### Best-of-N with LLM Judge (coming soon)

Best for: When you want LLM-based scoring without local reward models. Not yet available through the IaaS gateway — currently only supported via the library API directly.

---

### Common Parameters

All configurations support:
- `endpoint`: OpenAI-compatible API endpoint URL
- `api_key`: API key for the provider
- `model`: Model identifier
- `alg`: Algorithm name - `"self-consistency"` or `"best-of-n"`

## Usage Examples

### Tool Calling Example (Recommended)

Tool calling is the primary use case for IaaS with Self-Consistency and Best-of-N algorithms.

```bash
curl -X POST http://localhost:8109/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "system",
        "content": "You are a precise calculator. Always use the calculator tool for arithmetic."
      },
      {
        "role": "user",
        "content": "What is 847 * 293 + 156?"
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "calculator",
          "description": "Perform arithmetic calculations",
          "parameters": {
            "type": "object",
            "properties": {
              "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate"
              }
            },
            "required": ["expression"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "budget": 5
  }'
```

### Python Client with Tool Calling

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8109/v1",
    api_key="dummy-key"  # Not validated for local use
)

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform arithmetic calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a precise calculator. Always use the calculator tool for arithmetic."},
        {"role": "user", "content": "What is 847 * 293 + 156?"}
    ],
    tools=tools,
    tool_choice="auto",
    extra_body={"budget": 5}  # IaaS-specific parameter
)

# Access tool calls
tool_calls = response.choices[0].message.tool_calls
print(f"Tool: {tool_calls[0].function.name}")
print(f"Arguments: {tool_calls[0].function.arguments}")
```

### Basic Text Request

```bash
curl -X POST http://localhost:8109/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in one sentence"}
    ],
    "budget": 4
  }'
```

### Budget Parameter

The `budget` parameter controls the computational effort:
- `budget=1`: Single generation (no scaling)
- `budget=4`: Generate 4 responses, select best
- `budget=8`: Generate 8 responses, select best
- Higher budget = better quality but slower response

## External Access via SSH Tunneling

### Single Port Forward

```bash
# Forward IaaS service only
ssh -L 8109:localhost:8109 user@server-ip

# Forward vLLM service only  
ssh -L 8100:localhost:8100 user@server-ip
```

### Multiple Port Forward

```bash
# Forward both services
ssh -L 8100:localhost:8100 -L 8109:localhost:8109 user@server-ip
```

### Background SSH Tunnel

```bash
# Run tunnel in background
ssh -f -N -L 8100:localhost:8100 -L 8109:localhost:8109 user@server-ip
```

### Access from Local Machine

After establishing the tunnel, access services on your local machine:

```bash
# Test vLLM direct access
curl -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_completion_tokens": 50
  }'

# Test IaaS with scaling
curl -X POST http://localhost:8109/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "budget": 2
  }'
```

## Service Management

### Check Service Status

```bash
# Check if services are running
ss -tlnp | grep 8100  # vLLM
ss -tlnp | grep 8109  # IaaS

# Check GPU usage
nvidia-smi
```

### Stop Services

```bash
# Find process IDs
ss -tlnp | grep 8109

# Kill specific process
kill -9 <PID>

# Kill all vLLM processes
pkill -f "vllm serve"

# Kill all IaaS processes  
pkill -f "its-iaas"
```

### Background Execution

```bash
# Run vLLM in background
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct \
  --dtype float16 --host 127.0.0.1 --port 8100 > vllm.log 2>&1 &

# Run IaaS in background
CUDA_VISIBLE_DEVICES=1 uv run its-iaas \
  --host 127.0.0.1 --port 8109 > iaas.log 2>&1 &
```

## Failure Policy

| Failure | HTTP Status | Response |
|---|---|---|
| Service not configured (no `/configure` call) | 200 (SSE) / 400 | `"Service not configured"` error in stream or HTTP 400 |
| Invalid budget (non-integer, out of range) | 400 | `"Bad Request"` with detail |
| Unsupported algorithm | 422 | `"Algorithm 'X' not supported"` (Pydantic validation on `/configure`) |
| Invalid regex pattern in `/configure` | 400 | Regex compilation error detail |
| Downstream LLM unreachable / timeout | 500 | `"Generation failed. Check server logs for details."` |
| Algorithm execution error | 500 | `"Generation failed. Check server logs for details."` |
| `/configure` internal error | 500 | `"Configuration failed. Check server logs for details."` |

Error responses in the streaming path (`stream: true`) are delivered as SSE `data:` frames
with an `error` field, followed by `data: [DONE]`. The HTTP status is always 200 for
streaming responses (per SSE convention).

Non-streaming errors return standard HTTP error responses with a `detail` field.
Internal error details are **never exposed** to clients — they are logged server-side at
ERROR level.

## Streaming

The IaaS service accepts `"stream": true` in the request body. However, ITS algorithms
require all candidate responses before selecting the best one, so **true token-level
streaming is not possible**.

Instead, the service **buffers the full ITS result** and then emits it as SSE chunks:

1. The algorithm generates all candidates and selects the winner
2. The selected response is emitted as one or more `data:` frames in OpenAI SSE format
3. A final `data: [DONE]` frame signals completion

For content responses, a single content chunk is emitted. For tool call responses, each
tool call is emitted as a separate chunk followed by a `finish_reason: "tool_calls"` frame.

Clients using the OpenAI SDK with `stream=True` will work correctly — the response just
arrives as a burst rather than incrementally.

## Restart and Scaling

- **State**: The service holds an in-memory LM client cache (LRU, default 64 entries),
  a gateway instance, and the service config set via `/configure`. No state is persisted
  to disk.
- **Restart**: Restarting clears all cached LM clients and the service config.
  `/configure` must be called again after restart.
- **Horizontal scaling**: Multiple IaaS instances can run independently. Each maintains
  its own LM client cache, config, and gateway. There is no shared state between
  instances. A load balancer must route `/configure` to all instances or each instance
  must be configured independently.

## API Endpoints

### Configuration
- `POST /configure` - Configure the service
- `GET /v1/models` - List available models

### Chat Completions
- `POST /v1/chat/completions` - Generate responses with scaling

### Health Check
- `GET /docs` - API documentation
- `GET /health` - Service health (if available)

## Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Check what's using the port
ss -tlnp | grep 8109
# Kill the process
kill -9 <PID>
```

**2. CUDA Out of Memory**
```bash
# Check GPU memory
nvidia-smi
# Reduce model size or use smaller batch size
```

**3. Model Not Found**
```bash
# Verify model is downloaded
huggingface-cli download Qwen/Qwen2.5-Math-1.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-Math-PRM-7B
```

**4. Connection Refused**
```bash
# Check if service is running
curl -X GET http://localhost:8109/docs
# Check firewall settings
# Verify host binding
```

**5. Slow Responses**
- This is expected behavior for inference-time scaling
- Reduce `budget` parameter for faster responses
- Best-of-N with budget=4 typically takes 30-60 seconds

### Log Files

```bash
# View vLLM logs
tail -f vllm.log

# View IaaS logs  
tail -f iaas.log

# Check Python traceback
python -c "import traceback; traceback.print_exc()"
```

## Performance Optimization

### Memory Management
- Use `float16` for models to save memory
- Monitor GPU memory with `nvidia-smi`
- Adjust batch sizes based on available memory

### Response Time
- Lower `budget` values for faster responses
- Use `temperature=0.001` for more deterministic generation
- Consider using `particle-filtering` for different quality/speed trade-offs

### Scaling Considerations
- vLLM on GPU 0 (main model, 74GB memory)
- IaaS + Reward model on GPU 1 (14GB memory)
- Ensure adequate cooling for sustained high GPU usage

## Security Considerations

- Services bind to `127.0.0.1` (localhost only) by default — use `--host 0.0.0.0` only when network access is intentional
- `POST /configure` is an unauthenticated admin endpoint — when binding to `0.0.0.0`, protect it with network-level access control (firewall, Envoy, reverse proxy)
- Use SSH tunneling for secure remote access
- Consider adding authentication for production use
- Monitor resource usage to prevent abuse

## Envoy Integration

The IaaS service can be deployed behind Envoy with an ext_proc filter that routes requests with `X-ITS-*` headers to the IaaS backend.

### Architecture

```
Client → Envoy (port 8108)
   ├── /v1/chat/completions + X-ITS-* headers  →  ext_proc → IaaS (port 8109)  [transparent ITS]
   └── /* (no ITS headers)  →  LLM upstream (port 8100)                        [passthrough]
```

The ext_proc acts as a lightweight router — it checks for `X-ITS-Budget` and redirects matching requests to the IaaS backend. The IaaS service handles algorithm execution.

### Starting the Full Stack

```bash
# Start all three services together
make envoy-iaas-stack

# Or start individually:
its-iaas-ext-proc --port 50051 &  # ext_proc gRPC router
its-iaas --host 127.0.0.1 --port 8109 &  # IaaS service
its-iaas --print-config | envoy -c /dev/stdin &  # Envoy proxy
```

### Using ITS Through Envoy

Requests with `X-ITS-*` headers are intercepted by ext_proc and routed to IaaS:

```bash
curl -X POST http://localhost:8108/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-ITS-Budget: 4" \
  -H "X-ITS-Endpoint: http://localhost:8100/v1" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

### Header Stripping

When deployed behind Envoy, `X-ITS-*` headers are stripped in two layers to ensure
upstream LLM services never see ITS metadata:

1. **Envoy route-level**: The pass-through route includes `request_headers_to_remove`
   for `X-ITS-Budget`, `X-ITS-Endpoint`, and `X-ITS-API-Key`, stripping them before
   forwarding to the LLM upstream.
2. **ext_proc code-level**: The external processor strips any stray `X-ITS-*` headers
   from requests it processes, as a defense-in-depth measure.

On the IaaS route, headers are preserved — the IaaS service reads them for per-request
configuration overrides.

### Generating the Envoy Config

The bundled Envoy config can be printed and customized:

```bash
# Print config to stdout
its-iaas --print-config

# Save to file for customization
its-iaas --print-config > envoy_config.yaml
```

See the config comments for customization options (ports, timeouts, failure modes).

For standalone ext_proc deployment (without IaaS), see [ext_proc Gateway Guide](ext-proc-gateway.md).
