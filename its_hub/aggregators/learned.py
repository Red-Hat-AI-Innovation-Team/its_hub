from __future__ import annotations

import numpy as np

from its_hub.base import AbstractTrajectoryAggregator


def _extract_features(step_scores: list[float]) -> np.ndarray:
    """Convert variable-length step scores to a fixed 10-dim feature vector.

    Features (all position indices normalised by length):
      0  mean
      1  min
      2  max
      3  last step score
      4  trajectory length (raw count)
      5  variance
      6  normalised position of min (argmin / (len-1), or 0 if len==1)
      7  normalised position of max
      8  last minus first score
      9  score gap at min position (score_before_min - score_at_min), or 0 at boundary
    """
    n = len(step_scores)
    if n == 0:
        return np.zeros(10, dtype=np.float32)

    arr = np.array(step_scores, dtype=np.float64)
    mean = float(arr.mean())
    mn = float(arr.min())
    mx = float(arr.max())
    last = float(arr[-1])
    length = float(n)
    var = float(arr.var())

    if n > 1:
        pos_min = float(arr.argmin()) / (n - 1)
        pos_max = float(arr.argmax()) / (n - 1)
    else:
        pos_min = 0.0
        pos_max = 0.0

    delta_last_first = last - float(arr[0])

    amin = int(arr.argmin())
    gap_at_min = float(arr[amin - 1] - arr[amin]) if amin > 0 else 0.0

    return np.array(
        [mean, mn, mx, last, length, var, pos_min, pos_max, delta_last_first, gap_at_min],
        dtype=np.float32,
    )


class LearnedMLPAggregator(AbstractTrajectoryAggregator):
    """Trajectory aggregator backed by a trained MLP checkpoint.

    The MLP maps a 10-dim feature vector (derived from per-step scores) to a
    scalar trajectory score via a sigmoid output.  Checkpoint format is a plain
    PyTorch state-dict saved alongside the architecture hyper-parameters:

        torch.save({"state_dict": model.state_dict(), "hidden_width": 16}, path)

    Requires torch; raises ImportError with a clear message if absent.
    """

    def __init__(self, checkpoint_path: str):
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "LearnedMLPAggregator requires PyTorch. "
                "Install it with: pip install torch"
            ) from exc

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        hidden_width = checkpoint.get("hidden_width", 16)

        self._model = _TrajectoryMLP(input_dim=10, hidden_width=hidden_width)
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.eval()
        self._torch = torch

    def aggregate(self, step_scores: list[float]) -> float:
        features = _extract_features(step_scores)
        x = self._torch.tensor(features, dtype=self._torch.float32).unsqueeze(0)
        with self._torch.no_grad():
            score = self._model(x).item()
        return score


class _TrajectoryMLP:
    """Minimal 2-layer MLP; kept here so the aggregator is self-contained."""

    def __init__(self, input_dim: int = 10, hidden_width: int = 16):
        try:
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError("torch is required") from exc

        import torch.nn as nn

        self._net = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 1),
            nn.Sigmoid(),
        )

    def load_state_dict(self, state_dict):
        self._net.load_state_dict(state_dict)

    def eval(self):
        self._net.eval()

    def __call__(self, x):
        return self._net(x).squeeze(-1)
