# Installation

## Prerequisites

- Python 3.11 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- 20GB+ GPU memory for full examples (H100 80GB recommended)

## Production Installation

For production use, install from PyPI:

```bash
pip install its_hub
```

## Development Installation

For development or to run examples:

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/its_hub.git
cd its_hub
pip install -e ".[dev]"
```

The development installation includes:
- All core dependencies
- Testing frameworks (pytest, coverage)
- Code formatting and linting tool (ruff)
- Example scripts and notebooks

## Verification

Verify your installation by running:

```bash
python -c "import its_hub; print('its-hub installed successfully')"
```

## GPU Setup

For GPU-accelerated inference, ensure you have:

1. **NVIDIA drivers** compatible with your GPU
2. **CUDA toolkit** (version 11.8 or higher recommended)
3. **PyTorch with CUDA support**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

Verify GPU setup:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Dependencies

Core dependencies include:
- `torch` - Deep learning framework
- `transformers` - Hugging Face transformers library
- `vllm` - High-performance LLM serving
- `fastapi` - Web framework for IaaS API
- `numpy` - Numerical computations
- `asyncio` - Asynchronous programming support