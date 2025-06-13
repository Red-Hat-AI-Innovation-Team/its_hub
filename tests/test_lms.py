"""Clean tests for language models with improved organization."""

import unittest
from unittest.mock import patch
import pytest

from its_hub.lms import OpenAICompatibleLanguageModel, StepGeneration
from tests.mocks.language_models import SimpleMockLanguageModel
from tests.mocks.test_data import TestDataFactory, TEST_SCENARIOS
from tests.conftest import TEST_CONSTANTS


class TestOpenAICompatibleLanguageModel:
    """Test the OpenAICompatibleLanguageModel class using fixtures."""
    
    def test_generate_single_message(self, openai_server):
        """Test generating a response for a single message."""
        model = OpenAICompatibleLanguageModel(
            endpoint=openai_server,
            api_key=TEST_CONSTANTS["DEFAULT_API_KEY"],
            model_name=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            system_prompt="You are a helpful assistant.",
            max_tries=2
        )
        
        messages = TestDataFactory.create_chat_messages("Hello, world!")
        response = model.generate(messages)
        assert response == "Response to: Hello, world!"

    @pytest.mark.parametrize("scenario_name", ["simple_chat", "math_problem", "with_system_prompt"])
    def test_generate_scenarios(self, openai_server, scenario_name):
        """Test generation with various predefined scenarios."""
        scenario = TEST_SCENARIOS[scenario_name]
        
        if scenario.get("should_error"):
            pytest.skip("Error scenarios tested separately")
            
        model = OpenAICompatibleLanguageModel(
            endpoint=openai_server,
            api_key=TEST_CONSTANTS["DEFAULT_API_KEY"],
            model_name=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            max_tries=2
        )
        
        messages = TestDataFactory.create_chat_messages(
            scenario["user_content"],
            scenario.get("system_content")
        )
        
        response = model.generate(messages)
        assert response == scenario["expected_response"]

    @pytest.mark.parametrize("stop_token,include_stop,expected_suffix", [
        (None, False, ""),
        ("STOP", False, ""),
        ("STOP", True, "STOP"),
    ])
    def test_stop_token_handling(self, openai_server, stop_token, include_stop, expected_suffix):
        """Test stop token handling with different configurations."""
        model = OpenAICompatibleLanguageModel(
            endpoint=openai_server,
            api_key=TEST_CONSTANTS["DEFAULT_API_KEY"],
            model_name=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            max_tries=2
        )
        
        messages = TestDataFactory.create_chat_messages("Hello, world!")
        response = model.generate(
            messages, 
            stop=stop_token, 
            include_stop_str_in_output=include_stop
        )
        
        expected = "Response to: Hello, world!" + expected_suffix
        assert response == expected

    def test_batch_generation(self, openai_server):
        """Test generating responses for multiple message sets."""
        model = OpenAICompatibleLanguageModel(
            endpoint=openai_server,
            api_key=TEST_CONSTANTS["DEFAULT_API_KEY"],
            model_name=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            max_tries=2
        )
        
        messages_lst = [
            TestDataFactory.create_chat_messages("Hello, world!"),
            TestDataFactory.create_chat_messages("How are you?")
        ]
        
        responses = model.generate(messages_lst)
        expected = ["Response to: Hello, world!", "Response to: How are you?"]
        assert responses == expected

    def test_async_generation(self, openai_server):
        """Test async generation functionality."""
        async_model = OpenAICompatibleLanguageModel(
            endpoint=openai_server,
            api_key=TEST_CONSTANTS["DEFAULT_API_KEY"],
            model_name=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            is_async=True,
            max_tries=2
        )
        
        messages_lst = [
            TestDataFactory.create_chat_messages(f"Message {i}") 
            for i in range(4)
        ]
        
        responses = async_model.generate(messages_lst)
        expected = [f"Response to: Message {i}" for i in range(4)]
        assert responses == expected

    @pytest.mark.parametrize("max_concurrency,expected_semaphore_value", [
        (2, 2),
        (-1, 5),  # Should use length of messages_lst
    ])
    def test_concurrency_control(self, openai_server, max_concurrency, expected_semaphore_value):
        """Test concurrency control with different settings."""
        with patch('asyncio.Semaphore') as mock_semaphore:
            model = OpenAICompatibleLanguageModel(
                endpoint=openai_server,
                api_key=TEST_CONSTANTS["DEFAULT_API_KEY"],
                model_name=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
                is_async=True,
                max_concurrency=max_concurrency
            )
            
            messages_lst = [
                TestDataFactory.create_chat_messages(f"Message {i}") 
                for i in range(5)
            ]
            
            model.generate(messages_lst)
            mock_semaphore.assert_called_once_with(expected_semaphore_value)

    def test_error_handling_with_retries(self, openai_server):
        """Test error handling with retries."""
        model = OpenAICompatibleLanguageModel(
            endpoint=openai_server,
            api_key=TEST_CONSTANTS["DEFAULT_API_KEY"],
            model_name=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            max_tries=2
        )
        
        messages = TestDataFactory.create_chat_messages(TEST_CONSTANTS["ERROR_TRIGGER"])
        
        with pytest.raises(Exception) as exc_info:
            model.generate(messages)
        
        assert "Server error" in str(exc_info.value)

    @pytest.mark.parametrize("error_message,expected_result", [
        ("[CUSTOM ERROR]", "[CUSTOM ERROR]"),
        ("", ""),
        (None, Exception),  # Should raise exception
    ])
    def test_replace_error_with_message(self, openai_server, error_message, expected_result):
        """Test replace_error_with_message functionality."""
        model = OpenAICompatibleLanguageModel(
            endpoint=openai_server,
            api_key=TEST_CONSTANTS["DEFAULT_API_KEY"],
            model_name=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            max_tries=1,
            replace_error_with_message=error_message
        )
        
        messages = TestDataFactory.create_chat_messages(TEST_CONSTANTS["ERROR_TRIGGER"])
        
        if expected_result == Exception:
            with pytest.raises(Exception):
                model.generate(messages)
        else:
            result = model.generate(messages)
            assert result == expected_result

    def test_replace_error_with_message_batch(self, openai_server):
        """Test error replacement in batch requests."""
        error_message = "[BATCH ERROR]"
        model = OpenAICompatibleLanguageModel(
            endpoint=openai_server,
            api_key=TEST_CONSTANTS["DEFAULT_API_KEY"],
            model_name=TEST_CONSTANTS["DEFAULT_MODEL_NAME"],
            max_tries=1,
            replace_error_with_message=error_message
        )
        
        messages_lst = [
            TestDataFactory.create_chat_messages("Hello, world!"),
            TestDataFactory.create_chat_messages(TEST_CONSTANTS["ERROR_TRIGGER"]),
            TestDataFactory.create_chat_messages("How are you?")
        ]
        
        results = model.generate(messages_lst)
        expected = [
            "Response to: Hello, world!",
            error_message,
            "Response to: How are you?"
        ]
        assert results == expected


class TestStepGeneration:
    """Test the StepGeneration class with improved organization."""
    
    @pytest.mark.parametrize("step_token,max_steps,stop_token,temperature,include_stop", [
        ("\n", 5, None, 0.8, False),
        ("\n", 3, "END", 0.5, True),
        (">>", 10, "STOP", 1.0, False),
    ])
    def test_initialization(self, step_token, max_steps, stop_token, temperature, include_stop):
        """Test StepGeneration initialization with various parameters."""
        step_gen = StepGeneration(
            step_token=step_token,
            max_steps=max_steps,
            stop_token=stop_token,
            temperature=temperature,
            include_stop_str_in_output=include_stop
        )
        
        assert step_gen.step_token == step_token
        assert step_gen.max_steps == max_steps
        assert step_gen.stop_token == stop_token
        assert step_gen.temperature == temperature
        assert step_gen.include_stop_str_in_output == include_stop

    def test_initialization_validation(self):
        """Test that initialization validates parameters correctly."""
        with pytest.raises(AssertionError):
            StepGeneration(step_token=None, max_steps=5, include_stop_str_in_output=True)

    @pytest.mark.parametrize("steps,stopped,include_stop,expected", [
        (["step1", "step2", "step3"], False, False, "step1\nstep2\nstep3\n"),
        (["step1", "step2", "step3"], True, False, "step1\nstep2\nstep3"),
        (["step1", "step2", "step3"], False, True, "step1step2step3"),
    ])
    def test_post_process(self, steps, stopped, include_stop, expected):
        """Test post-processing with different configurations."""
        step_gen = StepGeneration(
            step_token="\n", 
            max_steps=5, 
            include_stop_str_in_output=include_stop
        )
        
        result = step_gen._post_process(steps, stopped=stopped)
        assert result == expected

    def test_forward_single_prompt(self):
        """Test forward generation with single prompt."""
        mock_lm = SimpleMockLanguageModel(["response1", "response2", "response3"])
        step_gen = StepGeneration(step_token="\n", max_steps=5)
        
        # Basic forward
        next_step, is_stopped = step_gen.forward(mock_lm, "test prompt")
        assert next_step == "response1"
        assert not is_stopped
        
        # With steps_so_far
        next_step, is_stopped = step_gen.forward(mock_lm, "test prompt", steps_so_far=["step1"])
        assert next_step == "response2"
        assert not is_stopped

    def test_forward_max_steps_reached(self):
        """Test forward generation when max steps is reached."""
        mock_lm = SimpleMockLanguageModel(["response3"])
        step_gen = StepGeneration(step_token="\n", max_steps=5)
        
        next_step, is_stopped = step_gen.forward(
            mock_lm, 
            "test prompt", 
            steps_so_far=["step1"] * 5
        )
        assert next_step == "response3"
        assert is_stopped

    def test_forward_stop_token_detection(self):
        """Test forward generation with stop token detection."""
        mock_lm = SimpleMockLanguageModel(["response with END"])
        step_gen = StepGeneration(step_token="\n", max_steps=5, stop_token="END")
        
        next_step, is_stopped = step_gen.forward(mock_lm, "test prompt")
        assert next_step == "response with END"
        assert is_stopped

    def test_forward_multiple_prompts(self):
        """Test forward generation with multiple prompts."""
        mock_lm = SimpleMockLanguageModel(["response1", "response2"])
        step_gen = StepGeneration(step_token="\n", max_steps=5)
        
        prompts = ["prompt1", "prompt2"]
        steps_so_far = [["step1"], ["step2"]]
        results = step_gen.forward(mock_lm, prompts, steps_so_far)
        
        assert len(results) == 2
        assert results[0][0] == "response1"
        assert not results[0][1]
        assert results[1][0] == "response2"
        assert not results[1][1]

    def test_forward_multiple_prompts_with_stop_token(self):
        """Test forward generation with multiple prompts and stop token detection."""
        mock_lm = SimpleMockLanguageModel(["response1", "response with END"])
        step_gen = StepGeneration(step_token="\n", max_steps=5, stop_token="END")
        
        prompts = ["prompt1", "prompt2"]
        steps_so_far = [["step1"], ["step2"]]
        results = step_gen.forward(mock_lm, prompts, steps_so_far)
        
        assert len(results) == 2
        assert results[0][0] == "response1"
        assert not results[0][1]
        assert results[1][0] == "response with END"
        assert results[1][1]  # Should be stopped due to END token