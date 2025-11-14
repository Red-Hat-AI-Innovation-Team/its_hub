"""Dummy reward model implementations for testing and demonstration."""

from its_hub.base import AbstractOutcomeRewardModel


class DummyRewardModel(AbstractOutcomeRewardModel):
    """
    A simple dummy reward model that returns a fixed score.

    This is useful for testing algorithms without requiring a real reward model.
    For production use, implement AbstractOutcomeRewardModel with your own
    reward scoring logic.
    """

    def __init__(self, fixed_score: float = 0.5):
        """
        Initialize the dummy reward model.

        Args:
            fixed_score: The fixed score to return for all evaluations (default: 0.5)
        """
        self.fixed_score = fixed_score

    async def ascore(self, prompt_or_messages, response):
        """
        Score a response or list of responses asynchronously.

        Args:
            prompt_or_messages: The input prompt (ignored)
            response: The model's response or list of responses (ignored)

        Returns:
            The fixed score (or list of scores if response is a list)
        """
        if isinstance(response, list):
            return [self.fixed_score for _ in response]
        return self.fixed_score

    def score(self, prompt_or_messages, response):
        """
        Score a response or list of responses synchronously.

        Args:
            prompt_or_messages: The input prompt (ignored)
            response: The model's response or list of responses (ignored)

        Returns:
            The fixed score (or list of scores if response is a list)
        """
        if isinstance(response, list):
            return [self.fixed_score for _ in response]
        return self.fixed_score
