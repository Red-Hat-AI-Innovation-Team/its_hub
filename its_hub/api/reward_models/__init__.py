"""Reward model abstract base classes."""

from .orm import AbstractOutcomeRewardModel
from .prm import AbstractProcessRewardModel

__all__ = [
    "AbstractOutcomeRewardModel",
    "AbstractProcessRewardModel",
]
