"""Planning Wrapper for any ITS algorithm implementation."""

import re
from dataclasses import dataclass

from its_hub.base import (
    AbstractLanguageModel,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
)
from its_hub.types import ChatMessage


@dataclass
class PlanningWrappedResult(AbstractScalingResult):
    """Result object for Planning-Enhanced algorithms."""

    plan: str
    approaches: list[str]
    approach_results: dict[str, AbstractScalingResult]
    approach_budgets: dict[str, int]
    combined_responses: list[str]
    best_approach: str
    best_approach_result: AbstractScalingResult

    @property
    def the_one(self) -> str:
        return self.best_approach_result.the_one


class PlanningPromptTemplate:
    """Template for generating planning prompts."""

    PLANNING_TEMPLATE = """Before solving this problem, I want you to first create a plan with different approaches to explore. This will help generate diverse solution strategies.

Problem: {problem}

Please provide a plan with 3 distinct approaches or hypotheses for solving this problem. Format your response as:

APPROACH 1: [Brief description of first method/strategy]
APPROACH 2: [Brief description of second method/strategy] 
APPROACH 3: [Brief description of third method/strategy]

Make sure each approach represents a genuinely different mathematical strategy or perspective for tackling this problem."""

    @classmethod
    def create_planning_prompt(cls, problem: str) -> str:
        """Create a planning prompt for the given problem."""
        return cls.PLANNING_TEMPLATE.format(problem=problem)


class PlanParser:
    """Parser to extract approaches from planning output."""

    @staticmethod
    def extract_approaches(plan: str) -> list[str]:
        """Extract approaches from the planning output."""
        approaches = []

        # Look for patterns like "APPROACH 1:", "APPROACH 2:", etc.
        approach_pattern = r"APPROACH\s+(\d+):\s*([^\n]+(?:\n(?!APPROACH)[^\n]*)*)"
        matches = re.findall(approach_pattern, plan, re.IGNORECASE | re.MULTILINE)

        for match in matches:
            approach_num, approach_desc = match
            # Clean up the approach description
            approach_desc = approach_desc.strip()
            approaches.append(approach_desc)

        # Fallback: if no structured approaches found, try to split by numbered points
        if not approaches:
            lines = plan.split("\n")
            for line in lines:
                line = line.strip()
                # Look for numbered approaches like "1.", "2.", "3."
                if re.match(r"^\d+\.", line):
                    approach = re.sub(r"^\d+\.\s*", "", line).strip()
                    if approach:
                        approaches.append(approach)

        # Ensure we have at least 2 approaches, fallback to generic ones
        if len(approaches) < 2:
            approaches = [
                "Direct algebraic approach using standard techniques",
                "Alternative method using different mathematical properties",
                "Geometric or graphical interpretation approach",
            ][: max(2, len(approaches))]

        return approaches[:3]  # Limit to 3 approaches


class ApproachPromptTemplate:
    """Template for generating approach-specific prompts."""

    APPROACH_TEMPLATE = """Using the {approach} method from your plan, solve this problem step by step:

Problem: {problem}

Approach to use: {approach}

Please solve the problem following this specific approach and show your work clearly. Make sure to box your final answer using \\boxed{{answer}}."""

    @classmethod
    def create_approach_prompt(cls, problem: str, approach: str) -> str:
        """Create an approach-specific prompt."""
        return cls.APPROACH_TEMPLATE.format(problem=problem, approach=approach)


class PlanningWrapper(AbstractScalingAlgorithm):
    """
    Planning Wrapper that can enhance any ITS algorithm with a planning phase.

    This wrapper adds a planning step before running the base algorithm, where:
    1. Model generates a plan with distinct approaches/hypotheses
    2. Budget is divided equally across planned approaches
    3. Each approach is executed with the base algorithm
    4. Best result across all approaches is selected
    """

    def __init__(self, base_algorithm: AbstractScalingAlgorithm):
        """Initialize Planning Wrapper.

        Args:
            base_algorithm: The base ITS algorithm to enhance (e.g., SelfConsistency,
                           ParticleFiltering, BestOfN, BeamSearch)
        """
        self.base_algorithm = base_algorithm
        self.plan_parser = PlanParser()

    def infer(
        self,
        lm: AbstractLanguageModel,
        prompt: str,
        budget: int,
        return_response_only: bool = True,
    ) -> str | PlanningWrappedResult:
        """Run Planning-Enhanced version of the base algorithm.

        Args:
            lm: Language model for generation
            prompt: Problem prompt
            budget: Total computational budget
            return_response_only: If True, return only the best response

        Returns:
            Best response string or full result object
        """
        # Step 1: Generate plan (uses 1 generation from budget)
        planning_prompt = PlanningPromptTemplate.create_planning_prompt(prompt)
        plan = lm.generate([ChatMessage(role="user", content=planning_prompt)])

        # Step 2: Parse approaches from plan
        approaches = self.plan_parser.extract_approaches(plan)

        # Step 3: Allocate remaining budget across approaches
        remaining_budget = budget - 1  # Subtract 1 for planning
        budget_per_approach = max(1, remaining_budget // len(approaches))

        # Handle remainder by giving extra budget to first approaches
        approach_budgets = {}
        total_allocated = 0
        for i, approach in enumerate(approaches):
            base_budget = budget_per_approach
            # Give remainder to first few approaches
            if total_allocated + base_budget < remaining_budget:
                if i < (remaining_budget % len(approaches)):
                    base_budget += 1
            approach_budgets[approach] = base_budget
            total_allocated += base_budget

        # Step 4: Run base algorithm for each approach
        approach_results = {}
        combined_responses = []

        for approach in approaches:
            approach_budget = approach_budgets[approach]

            # Create approach-specific prompt
            approach_prompt = ApproachPromptTemplate.create_approach_prompt(
                prompt, approach
            )

            # Run base algorithm for this approach
            approach_result = self.base_algorithm.infer(
                lm, approach_prompt, approach_budget, return_response_only=False
            )

            # Store approach-specific result
            approach_results[approach] = approach_result

            # Collect responses for overall analysis
            if hasattr(approach_result, "responses"):
                # For algorithms like BestOfN, SelfConsistency
                combined_responses.extend(approach_result.responses)
            elif hasattr(approach_result, "all_responses"):
                # For algorithms like ParticleFiltering
                combined_responses.extend(approach_result.all_responses)
            elif hasattr(approach_result, "response_lists"):
                # For algorithms like BeamSearch
                for response_list in approach_result.response_lists:
                    combined_responses.extend(response_list)
            else:
                # Fallback: treat as single response
                combined_responses.append(str(approach_result.the_one))

        # Step 5: Select best approach based on algorithm-specific criteria
        best_approach, best_result = self._select_best_approach(approach_results)

        # Create result object
        result = PlanningWrappedResult(
            plan=plan,
            approaches=approaches,
            approach_results=approach_results,
            approach_budgets=approach_budgets,
            combined_responses=combined_responses,
            best_approach=best_approach,
            best_approach_result=best_result,
        )

        return result.the_one if return_response_only else result

    def _select_best_approach(
        self, approach_results: dict[str, AbstractScalingResult]
    ) -> tuple[str, AbstractScalingResult]:
        """Select the best approach based on algorithm-specific criteria."""

        # Default: select based on highest confidence/score if available
        best_approach = None
        best_result = None
        best_score = float("-inf")

        for approach, result in approach_results.items():
            score = self._get_result_score(result)

            if score > best_score:
                best_score = score
                best_approach = approach
                best_result = result

        # Fallback to first approach if no scoring available
        if best_approach is None:
            best_approach = list(approach_results.keys())[0]
            best_result = approach_results[best_approach]

        return best_approach, best_result

    def _get_result_score(self, result: AbstractScalingResult) -> float:
        """Extract a score from the algorithm result for comparison."""

        # Try different score attributes that algorithms might have
        score_attrs = [
            "best_score",
            "max_score",
            "score",
            "confidence",
            "probability",
            "weight",
        ]

        for attr in score_attrs:
            if hasattr(result, attr):
                score_val = getattr(result, attr)
                if isinstance(score_val, (int, float)):
                    return float(score_val)
                elif isinstance(score_val, list) and score_val:
                    return float(max(score_val))

        # Try to get scores from response collections
        if hasattr(result, "scores") and result.scores:
            return float(max(result.scores))
        elif hasattr(result, "all_scores") and result.all_scores:
            return float(max(result.all_scores))
        elif hasattr(result, "log_weights_lst") and result.log_weights_lst:
            # For ParticleFiltering
            all_weights = []
            for weights in result.log_weights_lst:
                all_weights.extend(weights)
            return float(max(all_weights)) if all_weights else 0.0

        # Fallback: use response length as a proxy (longer = more detailed)
        response = str(result.the_one)
        return float(len(response)) / 1000.0  # Normalize to reasonable range


# Convenience functions for common combinations


def create_planning_self_consistency(extract_fn=None):
    """Create Planning-Enhanced Self-Consistency algorithm."""
    from its_hub.algorithms import SelfConsistency

    base_alg = SelfConsistency(extract_fn)
    return PlanningWrapper(base_alg)


def create_planning_particle_filtering(sg, prm, selection_method="argmax"):
    """Create Planning-Enhanced Particle Filtering algorithm."""
    from its_hub.algorithms import ParticleFiltering

    base_alg = ParticleFiltering(sg, prm, selection_method)
    return PlanningWrapper(base_alg)


def create_planning_best_of_n(orm):
    """Create Planning-Enhanced Best-of-N algorithm."""
    from its_hub.algorithms import BestOfN

    base_alg = BestOfN(orm)
    return PlanningWrapper(base_alg)


def create_planning_beam_search(sg, prm, beam_width=4):
    """Create Planning-Enhanced Beam Search algorithm."""
    from its_hub.algorithms import BeamSearch

    base_alg = BeamSearch(sg, prm, beam_width)
    return PlanningWrapper(base_alg)
