"""Mock reward models for testing."""


from its_hub.base import AbstractOutcomeRewardModel


class MockOutcomeRewardModel(AbstractOutcomeRewardModel):
    """Mock outcome reward model with configurable scores."""

    def __init__(self, scores: list[float] | float):
        if isinstance(scores, float):
            self.scores = [scores]
        else:
            self.scores = scores
        self.call_count = 0  # tracks total number of individual scores returned
        self.score_call_count = 0  # tracks number of times score() is called

    async def ascore(self, prompt: str, response: str | list[str]) -> float | list[float]:
        return self.score(prompt, response)

    def score(self, prompt: str, response: str | list[str]) -> float | list[float]:
        self.score_call_count += 1
        if isinstance(response, list):
            scores = []
            for i in range(len(response)):
                score_idx = (self.call_count + i) % len(self.scores)
                scores.append(self.scores[score_idx])
            self.call_count += len(response)
            return scores
        else:
            score = self.scores[self.call_count % len(self.scores)]
            self.call_count += 1
            return score


class MockProcessRewardModel:
    """Mock process reward model with configurable scores."""

    def __init__(self, scores: list[float] | list[list[float]]):
        if isinstance(scores[0], list):
            # Flatten nested lists
            self.scores = [score for sublist in scores for score in sublist]
        else:
            self.scores = scores
        self.call_count = 0

    async def ascore(self, prompt: str, response: str | list[str]) -> float | list[float]:
        return self.score(prompt, response)

    def score(self, prompt: str, response: str | list[str]) -> float | list[float]:
        if isinstance(response, list):
            scores = []
            for i in range(len(response)):
                score_idx = (self.call_count + i) % len(self.scores)
                scores.append(self.scores[score_idx])
            self.call_count += len(response)
            return scores
        else:
            score = self.scores[self.call_count % len(self.scores)]
            self.call_count += 1
            return score


class HighVarianceRewardModel:
    """Mock reward model with high variance for testing edge cases."""

    def __init__(self):
        self.scores = [0.0, 1.0, 0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4]
        self.call_count = 0

    async def ascore(self, prompt: str, response: str | list[str]) -> float | list[float]:
        return self.score(prompt, response)

    def score(self, prompt: str, response: str | list[str]) -> float | list[float]:
        if isinstance(response, list):
            scores = []
            for i in range(len(response)):
                score_idx = (self.call_count + i) % len(self.scores)
                scores.append(self.scores[score_idx])
            self.call_count += len(response)
            return scores
        else:
            score = self.scores[self.call_count % len(self.scores)]
            self.call_count += 1
            return score


class ErrorRewardModel:
    """Mock reward model that can simulate errors."""

    def __init__(self, scores: list[float], error_on_calls: list[int] | None = None):
        self.scores = scores
        self.error_on_calls = error_on_calls or []
        self.call_count = 0

    async def ascore(self, prompt: str, response: str | list[str]) -> float | list[float]:
        return self.score(prompt, response)

    def score(self, prompt: str, response: str | list[str]) -> float | list[float]:
        if self.call_count in self.error_on_calls:
            self.call_count += 1
            raise Exception("Simulated reward model error")

        if isinstance(response, list):
            scores = []
            for i in range(len(response)):
                if (self.call_count + i) in self.error_on_calls:
                    raise Exception("Simulated reward model error in batch")
                score_idx = (self.call_count + i) % len(self.scores)
                scores.append(self.scores[score_idx])
            self.call_count += len(response)
            return scores
        else:
            score = self.scores[self.call_count % len(self.scores)]
            self.call_count += 1
            return score
