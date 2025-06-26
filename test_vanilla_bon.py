#!/usr/bin/env python3
"""Simple test script to verify vanilla Best-of-N works properly."""

import sys
sys.path.insert(0, '.')

from its_hub.algorithms.bon import BestOfN
from tests.mocks.reward_models import MockOutcomeRewardModel
from tests.mocks.language_models import StepMockLanguageModel

def test_vanilla_best_of_n():
    """Test vanilla Best-of-N with mock components."""
    print("Testing vanilla Best-of-N...")
    
    # Create mock language model that returns different responses
    mock_responses = [
        "Response 1: The answer is 5",
        "Response 2: The answer is 10", 
        "Response 3: The answer is 7",
        "Response 4: The answer is 3"
    ]
    lm = StepMockLanguageModel(step_responses=mock_responses)
    
    # Create mock outcome reward model with scores favoring response 2
    mock_scores = [0.3, 0.9, 0.5, 0.2]  # Response 2 has highest score
    orm = MockOutcomeRewardModel(scores=mock_scores)
    
    # Create Best-of-N algorithm
    bon = BestOfN(orm=orm)
    
    # Test with budget of 4 (should generate 4 responses and pick best)
    prompt = "What is 2 + 3?"
    budget = 4
    
    # Get full result object first
    result = bon.infer(lm, prompt, budget, return_response_only=False)
    
    print(f"Generated {len(result.responses)} responses:")
    for i, (response, score) in enumerate(zip(result.responses, result.scores)):
        marker = " <- SELECTED" if i == result.selected_index else ""
        print(f"  {i+1}. Score: {score:.2f} | {response}{marker}")
    
    print(f"\nBest response: {result.the_one}")
    
    # Verify the highest scoring response was selected
    expected_best_index = mock_scores.index(max(mock_scores))
    assert result.selected_index == expected_best_index, f"Expected index {expected_best_index}, got {result.selected_index}"
    
    # Test return_response_only=True
    best_response_only = bon.infer(lm, prompt, budget, return_response_only=True)
    assert best_response_only == result.the_one, "return_response_only should return same result"
    
    print("✓ Vanilla Best-of-N test passed!")
    return True

if __name__ == "__main__":
    try:
        test_vanilla_best_of_n()
        print("\n✅ All tests passed! Vanilla Best-of-N is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)