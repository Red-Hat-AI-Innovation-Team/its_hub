#!/bin/bash
# Create the ONE conda env that serves and drives all three models.
set -euo pipefail
RUN16_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN16_DIR/config.sh"
cd "$REPO_ROOT"

if [ ! -x "$EPF_PY" ]; then
  echo "creating conda env '$EPF_ENV_NAME' (python 3.11)"
  conda create -n "$EPF_ENV_NAME" python=3.11 -y
fi
"$EPF_PY" -m pip install -r requirements-epf.txt
"$EPF_PY" -m pip install -e ".[dev,benchmark]"
# server-side audio decode deps (PyAV is vLLM's fallback decoder — without it: "Invalid audio file")
# + hf_transfer for fast downloads of the 44 GiB data.zip and the checkpoints
"$EPF_PY" -m pip install librosa soundfile av resampy hf_transfer

"$EPF_PY" - <<'PYEOF'
import torch, vllm
print(f"ENV OK: vllm {vllm.__version__} | torch {torch.__version__} | CUDA {torch.version.cuda} "
      f"| GPUs visible: {torch.cuda.device_count()}")
PYEOF

cat <<'NOTE'
--------------------------------------------------------------------------------
requirements-epf.txt pins CUDA-13.0 builds of torch 2.11.0 / vLLM 0.22.1 (the
reference Blackwell box). If the pip install failed on torch/vllm, install wheels
matching YOUR CUDA stack first, then re-run this script — the remaining pins apply.
ONE env serves all three models; only serve-time flags differ (see config.sh).
--------------------------------------------------------------------------------
NOTE
