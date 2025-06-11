#!/usr/bin/env python3
"""
Test script to verify error_as_message functionality works correctly.
"""

from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.types import ChatMessage

def test_error_as_message():
    """Test that error_as_message flag works correctly."""
    
    # Test with error_as_message=True - should return error message after backoff exhausted
    lm_with_error_handling = OpenAICompatibleLanguageModel(
        endpoint="http://invalid-endpoint:8000/v1",  # Invalid endpoint to trigger error
        api_key="dummy-key",
        model_name="test-model",
        error_as_message=True,
        error_message="[GENERATION ERROR]",
        max_tries=2  # Reduce for faster testing
    )
    
    # Test with error_as_message=False - should raise exception
    lm_without_error_handling = OpenAICompatibleLanguageModel(
        endpoint="http://invalid-endpoint:8000/v1",  # Invalid endpoint to trigger error
        api_key="dummy-key", 
        model_name="test-model",
        error_as_message=False
    )
    
    test_messages = [ChatMessage(role="user", content="What is 2+2?")]
    
    print("Testing error_as_message=True (should retry with backoff then return error message)...")
    try:
        result = lm_with_error_handling.generate(test_messages)
        print(f"✓ Got error message after backoff: '{result}'")
        assert result == "[GENERATION ERROR]", f"Expected '[GENERATION ERROR]', got '{result}'"
        print("✓ Single generation test passed")
    except Exception as e:
        print(f"✗ Unexpected exception with error_as_message=True: {e}")
        return False
    
    print("\nTesting batch generation with error_as_message=True...")
    try:
        batch_messages = [test_messages, test_messages]
        results = lm_with_error_handling.generate(batch_messages)
        print(f"✓ Got batch error messages: {results}")
        assert all(r == "[GENERATION ERROR]" for r in results), f"Expected all '[GENERATION ERROR]', got {results}"
        print("✓ Batch generation test passed")
    except Exception as e:
        print(f"✗ Unexpected exception in batch generation: {e}")
        return False
    
    print("\nTesting error_as_message=False (should raise exception)...")
    try:
        result = lm_without_error_handling.generate(test_messages)
        print(f"✗ Expected exception but got result: '{result}'")
        return False
    except Exception as e:
        print(f"✓ Got expected exception: {e}")
        print("✓ Exception test passed")
    
    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    test_error_as_message()