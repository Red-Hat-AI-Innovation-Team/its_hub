"""Test PlanningWrapper with multiple ITS algorithms."""

import re

import pytest

from its_hub.algorithms import BestOfN, ParticleFiltering, SelfConsistency
from its_hub.algorithms.planning_wrapper import (
    PlanningWrapper,
    create_planning_best_of_n,
    create_planning_particle_filtering,
    create_planning_self_consistency,
)
from its_hub.base import (
    AbstractLanguageModel,
    AbstractOutcomeRewardModel,
    AbstractProcessRewardModel,
)
from its_hub.lms import StepGeneration


def extract_boxed(s: str) -> str:
    """Extract answer from boxed format."""
    boxed_matches = re.findall(r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}', s)
    return boxed_matches[-1] if boxed_matches else ""


class MockLanguageModel(AbstractLanguageModel):
    """Mock language model for testing."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or [
            "Let me solve this step by step.\n\nFirst, I'll use algebraic methods.\n\nSolving: 2x + 3 = 7\n2x = 4\nx = 2\n\n\\boxed{2}",
            "I'll approach this differently.\n\nUsing substitution method:\nLet y = 2x + 3\ny = 7\n2x = 4\nx = 2\n\n\\boxed{2}",
            "Using geometric interpretation:\n\nThis represents a line equation.\nSolving: 2x + 3 = 7\n\n\\boxed{2}",
            "Step 1: Set up equation\n\nStep 2: Simplify\n\nFinal answer: \\boxed{2}"
        ]
        self.planning_response = "APPROACH 1: Direct algebraic approach using standard techniques\nAPPROACH 2: Alternative method using different mathematical properties\nAPPROACH 3: Geometric or graphical interpretation approach"
        self.call_count = 0

    def generate(self, messages, stop=None, max_tokens=None, include_stop_str_in_output=False, temperature=None, **kwargs):
        # Handle both single and batch generation
        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], list):
            # Batch generation
            batch_size = len(messages)
            results = []
            for i in range(batch_size):
                response_idx = (self.call_count + i) % len(self.responses)
                results.append(self.responses[response_idx])
            self.call_count += batch_size
            return results
        else:
            # Single generation (for planning)
            return self.planning_response

    def evaluate(self, prompt: str, generation: str) -> list[float]:
        """Return dummy evaluation scores."""
        return [0.5] * len(generation.split())


class MockProcessRewardModel(AbstractProcessRewardModel):
    """Mock process reward model for testing."""

    def __init__(self, scores: list[float] | None = None):
        self.scores = scores or [0.1, 0.5, 0.9]
        self.call_count = 0

    def score(self, prompt: str, response: str | list[str]) -> float | list[float]:
        import random
        if isinstance(response, str):
            return random.uniform(0.1, 0.9)
        else:  # List of responses
            return [random.uniform(0.1, 0.9) for _ in response]


class ProcessToOutcomeRewardModel(AbstractOutcomeRewardModel):
    """Convert process reward model to outcome reward model."""

    def __init__(self, process_rm: AbstractProcessRewardModel):
        self.process_rm = process_rm

    def score(self, prompt: str, responses: str | list[str]) -> float | list[float]:
        """Convert process reward to outcome reward by aggregating scores."""
        if isinstance(responses, list):
            scores = []
            for response in responses:
                try:
                    process_scores = self.process_rm.score(prompt, response)
                    if isinstance(process_scores, list) and len(process_scores) > 0:
                        final_score = process_scores[-1]
                    else:
                        final_score = process_scores if process_scores else 0.0
                    scores.append(final_score)
                except Exception:
                    scores.append(0.0)
            return scores
        else:
            try:
                process_scores = self.process_rm.score(prompt, responses)
                if isinstance(process_scores, list) and len(process_scores) > 0:
                    return process_scores[-1]
                else:
                    return process_scores if process_scores else 0.0
            except Exception:
                return 0.0


class TestPlanningWrapper:
    """Test suite for PlanningWrapper functionality."""

    @pytest.fixture
    def mock_language_model(self):
        """Create a mock language model for testing."""
        return MockLanguageModel()

    @pytest.fixture
    def mock_process_reward_model(self):
        """Create a mock process reward model for testing."""
        return MockProcessRewardModel()

    @pytest.fixture
    def mock_outcome_reward_model(self, mock_process_reward_model):
        """Create a mock outcome reward model for testing."""
        return ProcessToOutcomeRewardModel(mock_process_reward_model)

    @pytest.fixture
    def step_generation(self):
        """Create a StepGeneration instance for testing."""
        return StepGeneration("\n\n", 32, r"\boxed")

    @pytest.fixture
    def test_problem(self):
        """Sample test problem for algorithms."""
        return "Solve for x: 2x + 3 = 7"

    def test_planning_self_consistency_creation(self):
        """Test that planning self-consistency algorithm can be created."""
        planning_sc = create_planning_self_consistency(extract_boxed)
        assert isinstance(planning_sc, PlanningWrapper)
        assert hasattr(planning_sc, 'infer')

    def test_planning_best_of_n_creation(self, mock_outcome_reward_model):
        """Test that planning best-of-n algorithm can be created."""
        planning_bon = create_planning_best_of_n(mock_outcome_reward_model)
        assert isinstance(planning_bon, PlanningWrapper)
        assert hasattr(planning_bon, 'infer')

    def test_planning_particle_filtering_creation(self, step_generation, mock_process_reward_model):
        """Test that planning particle filtering algorithm can be created."""
        planning_pf = create_planning_particle_filtering(step_generation, mock_process_reward_model)
        assert isinstance(planning_pf, PlanningWrapper)
        assert hasattr(planning_pf, 'infer')

    def test_planning_wrapper_has_required_methods(self):
        """Test that PlanningWrapper has all required methods."""
        # Create a basic planning wrapper
        base_algorithm = SelfConsistency(extract_boxed)
        planning_wrapper = PlanningWrapper(base_algorithm)

        assert hasattr(planning_wrapper, 'infer')
        assert hasattr(planning_wrapper, 'base_algorithm')
        assert planning_wrapper.base_algorithm == base_algorithm

    def test_planning_self_consistency_inference(self, mock_language_model, test_problem):
        """Test that planning self-consistency can perform inference."""
        planning_sc = create_planning_self_consistency(extract_boxed)

        # Test inference
        result = planning_sc.infer(mock_language_model, test_problem, budget=4, return_response_only=False)

        # Verify result structure
        assert hasattr(result, 'the_one')
        assert hasattr(result, 'approaches')
        assert hasattr(result, 'best_approach')

        # Verify response contains expected content
        assert isinstance(result.the_one, str)
        assert len(result.approaches) > 0
        assert result.best_approach is not None

    def test_planning_best_of_n_inference(self, mock_language_model, mock_outcome_reward_model, test_problem):
        """Test that planning best-of-n can perform inference."""
        planning_bon = create_planning_best_of_n(mock_outcome_reward_model)

        # Test inference
        result = planning_bon.infer(mock_language_model, test_problem, budget=4, return_response_only=False)

        # Verify result structure
        assert hasattr(result, 'the_one')
        assert hasattr(result, 'approaches')
        assert hasattr(result, 'best_approach')

        # Verify response contains expected content
        assert isinstance(result.the_one, str)
        assert len(result.approaches) > 0
        assert result.best_approach is not None

    def test_planning_particle_filtering_inference(self, mock_language_model, step_generation, mock_process_reward_model, test_problem):
        """Test that planning particle filtering can perform inference."""
        planning_pf = create_planning_particle_filtering(step_generation, mock_process_reward_model)

        # Test inference
        result = planning_pf.infer(mock_language_model, test_problem, budget=4, return_response_only=False)

        # Verify result structure
        assert hasattr(result, 'the_one')
        assert hasattr(result, 'approaches')
        assert hasattr(result, 'best_approach')

        # Verify response contains expected content
        assert isinstance(result.the_one, str)
        assert len(result.approaches) > 0
        assert result.best_approach is not None

    def test_planning_vs_vanilla_self_consistency(self, mock_language_model, test_problem):
        """Test that planning and vanilla self-consistency both work."""
        # Vanilla self-consistency
        vanilla_sc = SelfConsistency(extract_boxed)
        vanilla_result = vanilla_sc.infer(mock_language_model, test_problem, budget=4, return_response_only=False)

        # Planning self-consistency
        planning_sc = create_planning_self_consistency(extract_boxed)
        planning_result = planning_sc.infer(mock_language_model, test_problem, budget=4, return_response_only=False)

        # Both should produce results
        assert isinstance(vanilla_result.the_one, str)
        assert isinstance(planning_result.the_one, str)

        # Planning result should have additional attributes
        assert hasattr(planning_result, 'approaches')
        assert hasattr(planning_result, 'best_approach')
        assert not hasattr(vanilla_result, 'approaches')

    def test_planning_vs_vanilla_best_of_n(self, mock_language_model, mock_outcome_reward_model, test_problem):
        """Test that planning and vanilla best-of-n both work."""
        # Vanilla best-of-n
        vanilla_bon = BestOfN(mock_outcome_reward_model)
        vanilla_result = vanilla_bon.infer(mock_language_model, test_problem, budget=4, return_response_only=False)

        # Planning best-of-n
        planning_bon = create_planning_best_of_n(mock_outcome_reward_model)
        planning_result = planning_bon.infer(mock_language_model, test_problem, budget=4, return_response_only=False)

        # Both should produce results
        assert isinstance(vanilla_result.the_one, str)
        assert isinstance(planning_result.the_one, str)

        # Planning result should have additional attributes
        assert hasattr(planning_result, 'approaches')
        assert hasattr(planning_result, 'best_approach')
        assert not hasattr(vanilla_result, 'approaches')

    def test_planning_vs_vanilla_particle_filtering(self, mock_language_model, step_generation, mock_process_reward_model, test_problem):
        """Test that planning and vanilla particle filtering both work."""
        # Vanilla particle filtering
        vanilla_pf = ParticleFiltering(step_generation, mock_process_reward_model)
        vanilla_result = vanilla_pf.infer(mock_language_model, test_problem, budget=4, return_response_only=False)

        # Planning particle filtering
        planning_pf = create_planning_particle_filtering(step_generation, mock_process_reward_model)
        planning_result = planning_pf.infer(mock_language_model, test_problem, budget=4, return_response_only=False)

        # Both should produce results
        assert isinstance(vanilla_result.the_one, str)
        assert isinstance(planning_result.the_one, str)

        # Planning result should have additional attributes
        assert hasattr(planning_result, 'approaches')
        assert hasattr(planning_result, 'best_approach')
        assert not hasattr(vanilla_result, 'approaches')

    def test_extract_boxed_function(self):
        """Test the extract_boxed utility function."""
        # Test with valid boxed answer
        text_with_boxed = "The answer is \\boxed{42}"
        assert extract_boxed(text_with_boxed) == "42"

        # Test with nested braces
        text_with_nested = "The answer is \\boxed{x^2 + 1}"
        assert extract_boxed(text_with_nested) == "x^2 + 1"

        # Test with no boxed answer
        text_without_boxed = "This has no boxed answer"
        assert extract_boxed(text_without_boxed) == ""

        # Test with multiple boxed answers (should return last one)
        text_with_multiple = "First \\boxed{1} then \\boxed{2}"
        assert extract_boxed(text_with_multiple) == "2"

    def test_process_to_outcome_reward_model_conversion(self, mock_process_reward_model):
        """Test ProcessToOutcomeRewardModel conversion."""
        outcome_rm = ProcessToOutcomeRewardModel(mock_process_reward_model)

        # Test single response
        single_score = outcome_rm.score("test prompt", "test response")
        assert isinstance(single_score, float)
        assert 0.0 <= single_score <= 1.0

        # Test multiple responses
        multiple_scores = outcome_rm.score("test prompt", ["response1", "response2"])
        assert isinstance(multiple_scores, list)
        assert len(multiple_scores) == 2
        assert all(isinstance(score, float) for score in multiple_scores)

    def test_planning_wrapper_return_response_only(self, mock_language_model, test_problem):
        """Test that return_response_only parameter works correctly."""
        planning_sc = create_planning_self_consistency(extract_boxed)

        # Test with return_response_only=True
        result_only = planning_sc.infer(mock_language_model, test_problem, budget=4, return_response_only=True)
        assert isinstance(result_only, str)

        # Test with return_response_only=False
        result_full = planning_sc.infer(mock_language_model, test_problem, budget=4, return_response_only=False)
        assert hasattr(result_full, 'the_one')
        assert hasattr(result_full, 'approaches')
        assert hasattr(result_full, 'best_approach')

    def test_planning_wrapper_with_different_budgets(self, mock_language_model, test_problem):
        """Test planning wrapper with different budget values."""
        planning_sc = create_planning_self_consistency(extract_boxed)

        # Test with different budgets
        for budget in [2, 4, 8]:
            result = planning_sc.infer(mock_language_model, test_problem, budget=budget, return_response_only=False)
            assert isinstance(result.the_one, str)
            assert len(result.approaches) > 0

    def test_mock_language_model_batch_generation(self):
        """Test that mock language model handles batch generation correctly."""
        lm = MockLanguageModel()

        # Test batch generation
        batch_messages = [
            [{"role": "user", "content": "Problem 1"}],
            [{"role": "user", "content": "Problem 2"}],
            [{"role": "user", "content": "Problem 3"}]
        ]

        results = lm.generate(batch_messages)
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(result, str) for result in results)

    def test_mock_language_model_single_generation(self):
        """Test that mock language model handles single generation correctly."""
        lm = MockLanguageModel()

        # Test single generation
        single_message = [{"role": "user", "content": "Single problem"}]
        result = lm.generate(single_message)
        assert isinstance(result, str)
        assert "APPROACH" in result

