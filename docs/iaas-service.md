# IaaS Service Setup Guide

This guide provides comprehensive instructions for setting up and running the its_hub Inference-as-a-Service (IaaS) with inference-time scaling algorithms.

## Overview

The IaaS service provides an OpenAI-compatible API with inference-time scaling algorithms. **Optimized for tool-calling applications** including agents, function calling, and multi-step reasoning. Currently supports **Best-of-N** and **Self-Consistency** algorithms.

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

## Prerequisites

- **Software**: Python 3.11+, its_hub library
- **API Access**: OpenAI API key, AWS credentials, or local vLLM server
- **GPU**: Optional (only if using local vLLM or local reward models)

## Quick Start

### Start IaaS Service

```bash
uv run its-iaas --host 0.0.0.0 --port 8108
```

**Parameters:**
- `--host 0.0.0.0`: Listen on all interfaces
- `--port 8108`: Default port for IaaS service
- `--dev`: Optional development mode with auto-reload

### 3. Configure IaaS Service

The IaaS service supports different algorithm configurations based on your use case.

## Algorithm Configurations

### Self-Consistency with Tool Voting

Best for: Tool-calling models where you want to vote on tool usage patterns.

**With OpenAI:**
```bash
curl -X POST http://localhost:8108/configure \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "litellm",
    "endpoint": "auto",
    "api_key": "your-openai-api-key",
    "model": "gpt-4o-mini",
    "alg": "self-consistency",
    "tool_vote": "tool_hierarchical",
    "exclude_args": ["timestamp", "request_id", "id", "type"]
  }'
```

**With AWS Bedrock:**
```bash
curl -X POST http://localhost:8108/configure \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "litellm",
    "endpoint": "auto",
    "api_key": null,
    "model": "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "alg": "self-consistency",
    "tool_vote": "tool_hierarchical",
    "exclude_args": ["timestamp", "request_id", "id"],
    "extra_args": {
      "aws_access_key_id": "your-access-key",
      "aws_secret_access_key": "your-secret-key",
      "aws_region_name": "us-east-1"
    }
  }'
```

**Parameters:**
- `tool_vote`: Voting strategy - `"tool_name"`, `"tool_args"`, or `"tool_hierarchical"` (recommended)
- `exclude_args`: List of argument names to exclude from voting (e.g., timestamps, IDs)

---

### Best-of-N with LLM Judge

Best for: Cloud APIs where you want LLM-based scoring without local reward models.

**With OpenAI:**
```bash
curl -X POST http://localhost:8108/configure \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "litellm",
    "endpoint": "auto",
    "api_key": "your-openai-api-key",
    "model": "gpt-4o-mini",
    "alg": "best-of-n",
    "rm_name": "llm-judge",
    "judge_model": "gpt-4o-mini",
    "judge_base_url": "auto",
    "judge_mode": "groupwise",
    "judge_criterion": "multi_step_tool_judge",
    "judge_api_key": "your-openai-api-key",
    "judge_temperature": 0.7,
    "judge_max_tokens": 2048
  }'
```

**With AWS Bedrock:**
```bash
curl -X POST http://localhost:8108/configure \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "litellm",
    "endpoint": "auto",
    "api_key": null,
    "model": "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "alg": "best-of-n",
    "rm_name": "llm-judge",
    "judge_model": "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "judge_base_url": "auto",
    "judge_mode": "groupwise",
    "judge_criterion": "multi_step_tool_judge",
    "judge_api_key": null,
    "judge_temperature": 0.7,
    "judge_max_tokens": 2048,
    "extra_args": {
      "aws_access_key_id": "your-access-key",
      "aws_secret_access_key": "your-secret-key",
      "aws_region_name": "us-east-1"
    }
  }'
```

**Parameters:**
- `rm_name`: Set to `"llm-judge"` to use LLM-based scoring
- `judge_model`: LiteLLM model name for the judge
- `judge_mode`: `"pointwise"` or `"groupwise"` (groupwise recommended)
- `judge_criterion`: Criterion for judging (e.g., `"overall_quality"`, `"multi_step_tool_judge"`)
- `judge_temperature`: Temperature for judge generation (0.0-1.0)

---

### Common Parameters

All configurations support:
- `provider`: `"litellm"` for multi-cloud support
- `endpoint`: API endpoint URL or `"auto"` for LiteLLM auto-detection
- `api_key`: API key for the provider (use `null` for Bedrock with credentials in `extra_args`)
- `model`: Model identifier (format depends on provider)
- `alg`: Algorithm name - `"self-consistency"` or `"best-of-n"`

## Usage Examples

### Tool Calling Example (Recommended)

Tool calling is the primary use case for IaaS with Self-Consistency and Best-of-N algorithms.

```bash
curl -X POST http://localhost:8108/v1/chat/completions \
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
    base_url="http://localhost:8108/v1",
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
curl -X POST http://localhost:8108/v1/chat/completions \
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
ssh -L 8108:localhost:8108 user@server-ip

# Forward vLLM service only  
ssh -L 8100:localhost:8100 user@server-ip
```

### Multiple Port Forward

```bash
# Forward both services
ssh -L 8100:localhost:8100 -L 8108:localhost:8108 user@server-ip
```

### Background SSH Tunnel

```bash
# Run tunnel in background
ssh -f -N -L 8100:localhost:8100 -L 8108:localhost:8108 user@server-ip
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
curl -X POST http://localhost:8108/v1/chat/completions \
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
ss -tlnp | grep 8108  # IaaS

# Check GPU usage
nvidia-smi
```

### Stop Services

```bash
# Find process IDs
ss -tlnp | grep 8108

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
  --dtype float16 --host 0.0.0.0 --port 8100 > vllm.log 2>&1 &

# Run IaaS in background
CUDA_VISIBLE_DEVICES=1 uv run its-iaas \
  --host 0.0.0.0 --port 8108 > iaas.log 2>&1 &
```

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
ss -tlnp | grep 8108
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
curl -X GET http://localhost:8108/docs
# Check firewall settings
# Verify host binding (0.0.0.0 vs 127.0.0.1)
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

- Services bind to `0.0.0.0` for external access
- Use SSH tunneling for secure remote access
- Consider adding authentication for production use
- Monitor resource usage to prevent abuse

## Integration Examples

### Watson Orchestrate
The service is compatible with Watson Orchestrate's OpenAI-compatible API:

```python
# Watson Orchestrate integration
import openai

client = openai.OpenAI(
    base_url="http://localhost:8108/v1",
    api_key="dummy-key"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Math-1.5B-Instruct",
    messages=[{"role": "user", "content": "Solve this math problem"}],
    extra_body={"budget": 4}  # IaaS-specific parameter
)
```

### Custom Applications
The service follows OpenAI's API format with the addition of the `budget` parameter for controlling inference-time scaling.

## Next Steps

1. **Production Deployment**: Consider using Docker containers and orchestration
2. **Monitoring**: Add metrics collection and alerting
3. **Authentication**: Implement API key management
4. **Load Balancing**: Scale horizontally with multiple instances
5. **Model Management**: Implement model versioning and hot-swapping

For more information, see the [its_hub documentation](https://github.com/your-org/its_hub) and [API reference](http://localhost:8108/docs).