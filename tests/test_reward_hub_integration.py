"""Integration tests for reward_hub integration."""

from unittest.mock import MagicMock, patch

import pytest
from reward_hub.base import AggregationMethod

from its_hub.integration.reward_hub import LocalVllmProcessRewardModel


class TestLocalVllmProcessRewardModelIntegration:
    """Test the integration between its_hub and reward_hub."""

    @pytest.fixture
    def mock_vllm_model(self):
        """Create a mock VllmProcessRewardModel."""
        with patch('reward_hub.vllm.reward.VllmProcessRewardModel') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            yield mock_instance

    def test_single_response_scoring(self, mock_vllm_model):
        """Test scoring a single response with proper message format."""
        # Setup mock to return a single score
        mock_vllm_model.score.return_value = [0.85]

        # Create the reward model
        model = LocalVllmProcessRewardModel(
            model_name="test-model",
            device="cpu",
            aggregation_method=AggregationMethod.PRODUCT
        )

        # Score a single response
        prompt = "What is 2+2?"
        response = "2+2 = 4"
        score = model.score(prompt, response)

        # Verify the score is returned correctly
        assert score == 0.85

        # Verify the mock was called with correct message format
        mock_vllm_model.score.assert_called_once()
        call_args = mock_vllm_model.score.call_args

        # Check that messages are in dict format, not ChatMessage objects
        messages = call_args[1]['messages']
        assert len(messages) == 1  # Single conversation
        assert len(messages[0]) == 2  # User + assistant messages

        # Verify message format - should be dicts, not ChatMessage objects
        user_msg = messages[0][0]
        assistant_msg = messages[0][1]

        assert isinstance(user_msg, dict)
        assert isinstance(assistant_msg, dict)
        assert user_msg == {"role": "user", "content": prompt}
        assert assistant_msg == {"role": "assistant", "content": response}

        # Verify other parameters
        assert call_args[1]['aggregation_method'] == AggregationMethod.PRODUCT
        assert call_args[1]['return_full_prm_result'] is False

    def test_multiple_responses_scoring(self, mock_vllm_model):
        """Test scoring multiple responses with proper message format."""
        # Setup mock to return multiple scores
        mock_vllm_model.score.return_value = [0.85, 0.72, 0.91]

        # Create the reward model
        model = LocalVllmProcessRewardModel(
            model_name="test-model",
            device="cuda:0",
            aggregation_method=AggregationMethod.MIN
        )

        # Score multiple responses
        prompt = "Solve this math problem: 3x + 5 = 14"
        responses = [
            "3x + 5 = 14\n3x = 9\nx = 3",
            "Let me solve step by step:\n3x = 14 - 5 = 9\nx = 3",
            "x = (14-5)/3 = 3"
        ]
        scores = model.score(prompt, responses)

        # Verify scores are returned correctly
        assert scores == [0.85, 0.72, 0.91]

        # Verify the mock was called with correct message format
        mock_vllm_model.score.assert_called_once()
        call_args = mock_vllm_model.score.call_args

        # Check that messages are in dict format for all responses
        messages = call_args[1]['messages']
        assert len(messages) == 3  # Three conversations

        for i, conversation in enumerate(messages):
            assert len(conversation) == 2  # User + assistant messages

            user_msg = conversation[0]
            assistant_msg = conversation[1]

            # Verify message format - should be dicts, not ChatMessage objects
            assert isinstance(user_msg, dict)
            assert isinstance(assistant_msg, dict)
            assert user_msg == {"role": "user", "content": prompt}
            assert assistant_msg == {"role": "assistant", "content": responses[i]}

        # Verify other parameters
        assert call_args[1]['aggregation_method'] == AggregationMethod.MIN
        assert call_args[1]['return_full_prm_result'] is False

    def test_different_aggregation_methods(self, mock_vllm_model):
        """Test that different aggregation methods are passed correctly."""
        mock_vllm_model.score.return_value = [0.5]

        for agg_method in [AggregationMethod.PRODUCT, AggregationMethod.MIN, AggregationMethod.LAST]:
            model = LocalVllmProcessRewardModel(
                model_name="test-model",
                device="cpu",
                aggregation_method=agg_method
            )

            model.score("test prompt", "test response")

            # Check that the aggregation method was passed correctly
            call_args = mock_vllm_model.score.call_args
            assert call_args[1]['aggregation_method'] == agg_method

    def test_message_format_compatibility(self, mock_vllm_model):
        """Test that the message format is compatible with reward_hub expectations.

        This test specifically addresses the bug from issue #73 where ChatMessage
        objects were used instead of dict format.
        """
        mock_vllm_model.score.return_value = [0.7]

        model = LocalVllmProcessRewardModel(
            model_name="test-model",
            device="cpu",
            aggregation_method=AggregationMethod.PRODUCT
        )

        # Score a response
        model.score("Test prompt", "Test response")

        # Get the messages that were passed to the reward_hub model
        call_args = mock_vllm_model.score.call_args
        messages = call_args[1]['messages']

        # Verify that each message is a plain dict (not a ChatMessage object)
        for conversation in messages:
            for message in conversation:
                # Should be a dict with 'role' and 'content' keys
                assert isinstance(message, dict)
                assert 'role' in message
                assert 'content' in message
                assert len(message) == 2  # Only 'role' and 'content'

                # Should not have any class-specific attributes
                assert not hasattr(message, '__class__') or message.__class__ is dict

                # Role should be string
                assert isinstance(message['role'], str)
                assert message['role'] in ['user', 'assistant']

                # Content should be string
                assert isinstance(message['content'], str)

    def test_error_handling(self, mock_vllm_model):
        """Test that errors from reward_hub are properly propagated."""
        # Setup mock to raise an exception
        mock_vllm_model.score.side_effect = Exception("reward_hub error")

        model = LocalVllmProcessRewardModel(
            model_name="test-model",
            device="cpu",
            aggregation_method=AggregationMethod.PRODUCT
        )

        # Verify that the exception is propagated
        with pytest.raises(Exception, match="reward_hub error"):
            model.score("test prompt", "test response")

    @pytest.mark.parametrize("device", ["cpu", "cuda:0", "cuda:1"])
    def test_device_parameter_passing(self, mock_vllm_model, device):
        """Test that device parameter is passed correctly to VllmProcessRewardModel."""
        with patch('reward_hub.vllm.reward.VllmProcessRewardModel') as mock_class:
            LocalVllmProcessRewardModel(
                model_name="test-model",
                device=device,
                aggregation_method=AggregationMethod.PRODUCT
            )

            # Verify VllmProcessRewardModel was initialized with correct device
            mock_class.assert_called_once_with(model_name="test-model", device=device)

    def test_model_name_parameter_passing(self, mock_vllm_model):
        """Test that model_name parameter is passed correctly to VllmProcessRewardModel."""
        test_model_names = [
            "microsoft/DialoGPT-medium",
            "meta-llama/Llama-2-7b-chat-hf",
            "custom-model-name"
        ]

        for model_name in test_model_names:
            with patch('reward_hub.vllm.reward.VllmProcessRewardModel') as mock_class:
                LocalVllmProcessRewardModel(
                    model_name=model_name,
                    device="cpu",
                    aggregation_method=AggregationMethod.PRODUCT
                )

                # Verify VllmProcessRewardModel was initialized with correct model name
                mock_class.assert_called_once_with(model_name=model_name, device="cpu")

    def test_regression_chatmessage_format_bug(self, mock_vllm_model):
        """Regression test for issue #73: ChatMessage objects vs dict format.

        This test simulates what would happen if ChatMessage objects were used
        instead of dict format, which was the bug fixed in PR #73.
        """
        # Setup mock to be strict about message format
        def strict_score_check(messages, **kwargs):
            # This simulates reward_hub expecting dict format
            for conversation in messages:
                for message in conversation:
                    # If this were a ChatMessage object, it would have additional attributes
                    # and methods that are not expected by reward_hub
                    if not isinstance(message, dict):
                        raise TypeError(f"Expected dict, got {type(message)}")

                    # Check that it only has the expected keys
                    expected_keys = {'role', 'content'}
                    if set(message.keys()) != expected_keys:
                        raise ValueError(f"Message has unexpected keys: {set(message.keys())}")

                    # Check that values are strings
                    if not isinstance(message['role'], str):
                        raise TypeError(f"Role should be string, got {type(message['role'])}")
                    if not isinstance(message['content'], str):
                        raise TypeError(f"Content should be string, got {type(message['content'])}")

            return [0.5]

        mock_vllm_model.score.side_effect = strict_score_check

        model = LocalVllmProcessRewardModel(
            model_name="test-model",
            device="cpu",
            aggregation_method=AggregationMethod.PRODUCT
        )

        # This should work fine with the current implementation
        score = model.score("Test prompt", "Test response")
        assert score == 0.5

        # Verify that the format check passed (no exception was raised)
        mock_vllm_model.score.assert_called_once()

    def test_demonstrates_chatmessage_compatibility_issue(self):
        """Test that demonstrates what the issue #73 bug would look like.

        This is a demonstration test showing how ChatMessage objects would fail
        with reward_hub's expected dict format.
        """
        from dataclasses import dataclass

        # Simulate a ChatMessage-like object (what was causing the bug)
        @dataclass
        class ChatMessage:
            role: str
            content: str

            def to_dict(self):
                return {"role": self.role, "content": self.content}

        # Create messages in the old (broken) format
        user_msg = ChatMessage(role="user", content="What is 2+2?")
        assistant_msg = ChatMessage(role="assistant", content="2+2 = 4")

        # This would be what the old code might have done
        broken_messages = [[user_msg, assistant_msg]]

        # Simulate reward_hub's expectation (dict format)
        def check_dict_format(messages):
            for conversation in messages:
                for message in conversation:
                    if not isinstance(message, dict):
                        raise TypeError(f"Expected dict, got {type(message)}")
                    if 'role' not in message or 'content' not in message:
                        raise ValueError("Message missing required keys")

        # This would fail with the old implementation
        with pytest.raises(TypeError, match="Expected dict, got"):
            check_dict_format(broken_messages)

        # But this works with the current implementation (dict format)
        correct_messages = [[{"role": "user", "content": "What is 2+2?"},
                            {"role": "assistant", "content": "2+2 = 4"}]]

        # This should pass
        check_dict_format(correct_messages)  # No exception should be raised

