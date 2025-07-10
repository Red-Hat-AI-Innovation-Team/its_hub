# IaaS Service Setup Guide

This guide provides comprehensive instructions for setting up and running the its_hub Inference-as-a-Service (IaaS) with inference-time scaling algorithms.

## Overview

The IaaS service provides an OpenAI-compatible API that applies inference-time scaling algorithms like Best-of-N and Particle Filtering to improve the quality of language model responses.

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │───►│   IaaS Service  │───►│   vLLM Server   │
│ (Watson Orc.)   │    │   (GPU 1)       │    │   (GPU 0)       │
│                 │    │ - Best-of-N     │    │ - Main Model    │
│                 │    │ - Reward Model  │    │ - Generation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

- **Hardware**: Multi-GPU setup (minimum 2 GPUs recommended)
- **Software**: Python 3.11+, CUDA, vLLM, its_hub library
- **Models**: Pre-downloaded language model and reward model
- **Memory**: Sufficient GPU memory for both models

## Step-by-Step Setup

### 1. Start vLLM Server (Main Model)

The vLLM server hosts the main language model for generation.

```bash
# Start on GPU 0
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct \
  --dtype float16 \
  --host 0.0.0.0 \
  --port 8100
```

**Parameters:**
- `--dtype float16`: Use half precision for memory efficiency
- `--host 0.0.0.0`: Listen on all interfaces (enables external access)
- `--port 8100`: Default port for vLLM service

### 2. Start IaaS Service (Scaling Algorithms)

The IaaS service applies inference-time scaling algorithms and hosts the reward model.

```bash
# Start on GPU 1
CUDA_VISIBLE_DEVICES=1 uv run its-iaas \
  --host 0.0.0.0 \
  --port 8108
```

**Parameters:**
- `--host 0.0.0.0`: Listen on all interfaces
- `--port 8108`: Default port for IaaS service
- `--dev`: Optional development mode with auto-reload

### 3. Configure IaaS Service

Configure the service to connect to vLLM and set up the scaling algorithm.

```bash
curl -X POST http://localhost:8108/configure \
  -H "Content-Type: application/json" \
  -d '{
    "endpoint": "http://localhost:8100/v1",
    "api_key": "NO_API_KEY",
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "alg": "best-of-n",
    "rm_name": "Qwen/Qwen2.5-Math-PRM-7B",
    "rm_device": "cuda:1",
    "rm_agg_method": "model"
  }'
```

**Configuration Parameters:**
- `endpoint`: vLLM server URL
- `api_key`: API key (use "NO_API_KEY" for local vLLM)
- `model`: Model name (must match vLLM model)
- `alg`: Algorithm (`"best-of-n"` or `"particle-filtering"`)
- `rm_name`: Reward model name
- `rm_device`: GPU device for reward model (`"cuda:0"`, `"cuda:1"`, etc.)
- `rm_agg_method`: Reward aggregation method (`"model"` recommended)

## Usage Examples

### Basic Request

```bash
curl -X POST http://localhost:8108/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "budget": 4
  }'
```

### Python Client

```python
import requests

url = "http://localhost:8108/v1/chat/completions"
payload = {
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "messages": [
        {"role": "user", "content": "Solve: x^2 + 5x + 6 = 0"}
    ],
    "budget": 4  # Generate 4 responses, select best
}

response = requests.post(url, json=payload)
result = response.json()
print(result['choices'][0]['message']['content'])
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
    "max_tokens": 50
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