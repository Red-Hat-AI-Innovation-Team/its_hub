"""Tests for message object format changes."""

import pytest
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.algorithms.self_consistency import SelfConsistency
from its_hub.types import ChatMessage
from tests.conftest import TEST_CONSTANTS


class MockMessageLanguageModel:
    """Mock language model that returns message objects."""
    
    def __init__(self):
        self.call_count = 0
        
    def generate(self, messages_or_messages_lst, messages_output=False):
        """Return message objects or content strings based on messages_output flag."""
        is_single = not isinstance(messages_or_messages_lst[0], list)
        
        if is_single:
            # Single message list
            messages = messages_or_messages_lst
            # Get the last user message
            user_messages = [msg.content for msg in messages if msg.role == "user"]
            user_content = user_messages[-1] if user_messages else "default"
            
            # Return based on messages_output flag
            message_obj = {
                "role": "assistant", 
                "content": f"Response to: {user_content}"
            }
            return message_obj if messages_output else message_obj["content"]
        else:
            # Multiple message lists
            responses = []
            for messages in messages_or_messages_lst:
                # Get the last user message
                user_messages = [msg.content for msg in messages if msg.role == "user"]
                user_content = user_messages[-1] if user_messages else "default"
                message_obj = {
                    "role": "assistant",
                    "content": f"Response to: {user_content}"
                }
                responses.append(message_obj if messages_output else message_obj["content"])
            return responses


class TestMessageObjectFormat:
    """Test that the system properly handles message objects."""
    
    def test_lm_returns_message_object_single(self):
        """Test that language model returns message object for single generation."""
        mock_lm = MockMessageLanguageModel()
        messages = [ChatMessage(role="user", content="Hello")]
        
        response = mock_lm.generate(messages, messages_output=True)
        
        # Should return a message object, not a string
        assert isinstance(response, dict)
        assert response["role"] == "assistant"
        assert response["content"] == "Response to: Hello"
    
    def test_lm_returns_message_object_batch(self):
        """Test that language model returns message objects for batch generation."""
        mock_lm = MockMessageLanguageModel()
        messages_list = [
            [ChatMessage(role="user", content="Hello")],
            [ChatMessage(role="user", content="Hi there")]
        ]
        
        responses = mock_lm.generate(messages_list, messages_output=True)
        
        # Should return a list of message objects
        assert isinstance(responses, list)
        assert len(responses) == 2
        assert all(isinstance(resp, dict) for resp in responses)
        assert responses[0]["content"] == "Response to: Hello"
        assert responses[1]["content"] == "Response to: Hi there"
    
    def test_self_consistency_with_string_prompt_legacy_output(self):
        """Test self-consistency algorithm with string prompt returning string (backward compatibility)."""
        mock_lm = MockMessageLanguageModel()
        
        def simple_projection(text):
            # Extract the last word as the "answer"
            return text.split()[-1] if text.strip() else "unknown"
        
        algorithm = SelfConsistency(simple_projection)
        
        # Test with string prompt and legacy string output (default)
        result = algorithm.infer(mock_lm, "What is 2+2?", budget=3, messages_output=False)
        
        # Should return a content string
        assert isinstance(result, str)
        assert "2+2?" in result
    
    def test_self_consistency_with_string_prompt_message_output(self):
        """Test self-consistency algorithm with string prompt returning message object."""
        mock_lm = MockMessageLanguageModel()
        
        def simple_projection(text):
            # Extract the last word as the "answer"
            return text.split()[-1] if text.strip() else "unknown"
        
        algorithm = SelfConsistency(simple_projection)
        
        # Test with string prompt and message object output
        result = algorithm.infer(mock_lm, "What is 2+2?", budget=3, messages_output=True)
        
        # Should return a message object
        assert isinstance(result, dict)
        assert result["role"] == "assistant"
        assert "2+2?" in result["content"]
    
    def test_self_consistency_with_conversation_history(self):
        """Test self-consistency algorithm with full conversation history."""
        mock_lm = MockMessageLanguageModel()
        
        def simple_projection(text):
            return text.split()[-1] if text.strip() else "unknown"
        
        algorithm = SelfConsistency(simple_projection)
        
        # Test with conversation history
        conversation = [
            ChatMessage(role="system", content="You are a math tutor."),
            ChatMessage(role="user", content="What is 2+2?"),
            ChatMessage(role="assistant", content="2+2 equals 4."),
            ChatMessage(role="user", content="What about 3+3?")
        ]
        
        result = algorithm.infer(mock_lm, conversation, budget=2, messages_output=True)
        
        # Should return a message object
        assert isinstance(result, dict)
        assert result["role"] == "assistant"
        assert "3+3?" in result["content"]
    
    def test_self_consistency_returns_full_result(self):
        """Test that self-consistency can return full result with metadata."""
        mock_lm = MockMessageLanguageModel()
        
        def simple_projection(text):
            return "answer"  # Always return same projection for consistency
        
        algorithm = SelfConsistency(simple_projection)
        
        # Get full result with message objects
        result = algorithm.infer(mock_lm, "Test question", budget=3, return_response_only=False, messages_output=True)
        
        # Should return SelfConsistencyResult object
        assert hasattr(result, 'the_one')
        assert hasattr(result, 'responses')
        assert hasattr(result, 'response_counts')
        assert hasattr(result, 'selected_index')
        
        # The selected response should be a message object
        selected = result.the_one
        assert isinstance(selected, dict)
        assert selected["role"] == "assistant"
        
        # All responses should be message objects
        assert len(result.responses) == 3
        assert all(isinstance(resp, dict) for resp in result.responses)
        assert all(resp["role"] == "assistant" for resp in result.responses)
        
        # Test with string output format
        result_str = algorithm.infer(mock_lm, "Test question", budget=3, return_response_only=False, messages_output=False)
        
        # The selected response should be a string
        selected_str = result_str.the_one
        assert isinstance(selected_str, str)
        
        # All responses should be strings
        assert len(result_str.responses) == 3
        assert all(isinstance(resp, str) for resp in result_str.responses)