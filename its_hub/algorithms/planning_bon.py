"""Planning-Enhanced Best-of-N algorithm implementation."""

import re
from typing import List, Dict, Any
from pydantic.dataclasses import dataclass

from its_hub.base import (
    AbstractLanguageModel,
    AbstractOutcomeRewardModel,
    AbstractScalingAlgorithm,
    AbstractScalingResult,
)
from its_hub.types import ChatMessage
from its_hub.algorithms.bon import BestOfN


@dataclass
class PlanningBestOfNResult(AbstractScalingResult):
    """Result object for Planning-Enhanced Best-of-N."""
    plan: str
    approaches: List[str]
    approach_responses: Dict[str, List[str]]
    approach_scores: Dict[str, List[float]]
    all_responses: List[str]
    all_scores: List[float]
    selected_index: int
    approach_budgets: Dict[str, int]

    @property
    def the_one(self) -> str:
        return self.all_responses[self.selected_index]


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
    def extract_approaches(plan: str) -> List[str]:
        """Extract approaches from the planning output."""
        approaches = []
        
        # Look for patterns like "APPROACH 1:", "APPROACH 2:", etc.
        approach_pattern = r'APPROACH\s+(\d+):\s*([^\n]+(?:\n(?!APPROACH)[^\n]*)*)'
        matches = re.findall(approach_pattern, plan, re.IGNORECASE | re.MULTILINE)
        
        for match in matches:
            approach_num, approach_desc = match
            # Clean up the approach description
            approach_desc = approach_desc.strip()
            approaches.append(approach_desc)
        
        # Fallback: if no structured approaches found, try to split by numbered points
        if not approaches:
            lines = plan.split('\n')
            for line in lines:
                line = line.strip()
                # Look for numbered approaches like "1.", "2.", "3."
                if re.match(r'^\d+\.', line):
                    approach = re.sub(r'^\d+\.\s*', '', line).strip()
                    if approach:
                        approaches.append(approach)
        
        # Ensure we have at least 2 approaches, fallback to generic ones
        if len(approaches) < 2:
            approaches = [
                "Direct algebraic approach using standard techniques",
                "Alternative method using different mathematical properties",
                "Geometric or graphical interpretation approach"
            ][:max(2, len(approaches))]
        
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


class PlanningBestOfN(AbstractScalingAlgorithm):
    """Planning-Enhanced Best-of-N algorithm."""
    
    def __init__(self, orm: AbstractOutcomeRewardModel):
        """Initialize Planning-Enhanced Best-of-N.
        
        Args:
            orm: Outcome reward model for scoring responses
        """
        self.orm = orm
        self.plan_parser = PlanParser()
    
    def infer(
        self,
        lm: AbstractLanguageModel,
        prompt: str,
        budget: int,
        return_response_only: bool = True,
    ) -> str | PlanningBestOfNResult:
        """Run Planning-Enhanced Best-of-N inference.
        
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
        
        # Step 4: Generate responses for each approach
        all_responses = []
        all_scores = []
        approach_responses = {}
        approach_scores = {}
        
        for approach in approaches:
            approach_budget = approach_budgets[approach]
            
            # Create approach-specific prompt
            approach_prompt = ApproachPromptTemplate.create_approach_prompt(prompt, approach)
            
            # Generate responses for this approach using vanilla Best-of-N
            vanilla_bon = BestOfN(self.orm)
            approach_result = vanilla_bon.infer(
                lm, approach_prompt, approach_budget, return_response_only=False
            )
            
            # Store approach-specific results
            approach_responses[approach] = approach_result.responses
            approach_scores[approach] = approach_result.scores
            
            # Add to global pool
            all_responses.extend(approach_result.responses)
            all_scores.extend(approach_result.scores)
        
        # Step 5: Select best response across all approaches
        selected_index = all_scores.index(max(all_scores))
        
        # Create result object
        result = PlanningBestOfNResult(
            plan=plan,
            approaches=approaches,
            approach_responses=approach_responses,
            approach_scores=approach_scores,
            all_responses=all_responses,
            all_scores=all_scores,
            selected_index=selected_index,
            approach_budgets=approach_budgets,
        )
        
        return result.the_one if return_response_only else result