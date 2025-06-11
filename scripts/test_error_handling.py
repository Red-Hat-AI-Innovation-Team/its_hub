#!/usr/bin/env python3
"""
Test script to demonstrate enhanced error handling for API failures.

This script shows how different types of API errors are now handled with
specific error messages and appropriate retry logic.
"""

import sys
sys.path.append('..')

from its_hub.error_handling import (
    parse_api_error, enhanced_on_backoff, should_retry, 
    format_non_retryable_error, RateLimitError, ContextLengthError,
    AuthenticationError, BadRequestError
)


def test_error_parsing():
    """Test error parsing for different API error scenarios."""
    
    print("🧪 Testing API Error Parsing\n")
    
    # Test cases with different error scenarios
    test_cases = [
        {
            "name": "Rate Limit Error",
            "status_code": 429,
            "error_text": '{"error": {"message": "Rate limit exceeded. Please wait and try again.", "type": "rate_limit_exceeded"}}',
            "expected_type": RateLimitError,
            "should_retry": True
        },
        {
            "name": "Context Length Error",
            "status_code": 400,
            "error_text": '{"error": {"message": "This model\'s maximum context length is 4096 tokens.", "type": "invalid_request_error"}}',
            "expected_type": ContextLengthError,
            "should_retry": False
        },
        {
            "name": "Authentication Error",
            "status_code": 401,
            "error_text": '{"error": {"message": "Invalid API key provided", "type": "invalid_request_error"}}',
            "expected_type": AuthenticationError,
            "should_retry": False
        },
        {
            "name": "Bad Request Error",
            "status_code": 400,
            "error_text": '{"error": {"message": "Invalid parameter: temperature must be between 0 and 2", "type": "invalid_request_error"}}',
            "expected_type": BadRequestError,
            "should_retry": False
        }
    ]
    
    for test_case in test_cases:
        print(f"Testing: {test_case['name']}")
        
        # Parse the error
        parsed_error = parse_api_error(test_case['status_code'], test_case['error_text'])
        
        # Check if it's the expected type
        is_correct_type = isinstance(parsed_error, test_case['expected_type'])
        retry_behavior = should_retry(parsed_error)
        
        print(f"  ✓ Error type: {type(parsed_error).__name__} ({'✓' if is_correct_type else '✗'})")
        print(f"  ✓ Should retry: {retry_behavior} ({'✓' if retry_behavior == test_case['should_retry'] else '✗'})")
        print(f"  ✓ Message: {parsed_error.message}")
        
        if not should_retry(parsed_error):
            print(f"  ✓ User-friendly message:")
            formatted_msg = format_non_retryable_error(parsed_error)
            for line in formatted_msg.split('\n'):
                print(f"     {line}")
        
        print()


def test_backoff_callback():
    """Test the enhanced backoff callback."""
    
    print("🧪 Testing Enhanced Backoff Callback\n")
    
    # Simulate backoff details for different error types
    rate_limit_error = RateLimitError("Rate limit exceeded", status_code=429)
    context_error = ContextLengthError("Context length exceeded", status_code=400)
    
    test_details = [
        {
            "exception": rate_limit_error,
            "wait": 2.5,
            "tries": 2,
            "target": "fetch_response"
        },
        {
            "exception": context_error,
            "wait": 1.0,
            "tries": 1,
            "target": "fetch_response"
        }
    ]
    
    for details in test_details:
        print(f"Simulating backoff for {type(details['exception']).__name__}:")
        enhanced_on_backoff(details)
        print()


if __name__ == "__main__":
    print("🚀 Testing Enhanced Error Handling for Issue #35\n")
    
    test_error_parsing()
    test_backoff_callback()
    
    print("✅ All tests completed!")
    print("\n📋 Summary of Improvements:")
    print("  • Specific error types with targeted messages")
    print("  • Only retry appropriate errors (rate limits, server errors)")
    print("  • Clear user guidance for non-retryable errors")
    print("  • Detailed backoff messages showing error type and reason")