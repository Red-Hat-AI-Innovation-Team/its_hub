"""Answer evaluation and result tracking for e2e tests."""

import re
from dataclasses import dataclass, field

import math_verify


def extract_boxed(s: str) -> str:
    """Extract the last ``\\boxed{...}`` content from *s*."""
    matches = re.findall(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", s)
    return matches[-1] if matches else ""


def evaluate_answer(predicted: str, expected: str) -> bool:
    """Return *True* if *predicted* is mathematically equivalent to *expected*."""
    try:
        return math_verify.verify(
            math_verify.parse(expected),
            math_verify.parse(predicted),
        )
    except Exception:
        return False


@dataclass
class TestResult:
    """Aggregated result for one (algorithm, dataset) pair."""

    algorithm: str
    dataset: str
    total: int = 0
    correct: int = 0
    errors: int = 0
    error_messages: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    latencies: list[float] = field(default_factory=list)

    @property
    def evaluated(self) -> int:
        return self.total - self.errors

    @property
    def accuracy(self) -> float:
        return self.correct / self.evaluated if self.evaluated else 0.0

    @property
    def passed(self) -> bool:
        return self.errors == 0

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def min_latency(self) -> float:
        return min(self.latencies) if self.latencies else 0.0

    @property
    def max_latency(self) -> float:
        return max(self.latencies) if self.latencies else 0.0

