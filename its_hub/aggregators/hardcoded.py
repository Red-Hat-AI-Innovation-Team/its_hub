import math
from typing import Literal

from its_hub.base import AbstractTrajectoryAggregator


class HardcodedAggregator(AbstractTrajectoryAggregator):
    """Reduces per-step PRM scores with a fixed aggregation rule.

    Wraps the three reductions previously hardcoded inside PRM implementations,
    making the choice explicit and pluggable at the algorithm level.

    prod: product of step probabilities (log-space: sum of log-probs)
    min:  worst-case step score
    mean: length-invariant average
    """

    def __init__(self, reduction: Literal["prod", "min", "mean"] = "prod"):
        self.reduction = reduction

    def aggregate(self, step_scores: list[float]) -> float:
        if not step_scores:
            return 0.0
        if self.reduction == "prod":
            return math.prod(step_scores)
        elif self.reduction == "min":
            return min(step_scores)
        elif self.reduction == "mean":
            return sum(step_scores) / len(step_scores)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction!r}")
