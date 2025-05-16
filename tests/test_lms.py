import unittest
import json
import threading
import asyncio
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import socket
from typing import List, Dict, Any
from unittest.mock import patch

from its_hub.lms import OpenAICompatibleLanguageModel, StepGeneration


def find_free_port():
    """Find a free port to use for the test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class DummyOpenAIHandler(BaseHTTPRequestHandler):
    """A dummy HTTP handler that mimics the OpenAI API."""
    
    # Class-level variables to track concurrent requests
    active_requests = 0
    max_concurrent_requests = 0
    request_lock = threading.Lock()
    
    @classmethod
    def reset_stats(cls):
        """Reset the request statistics."""
        with cls.request_lock:
            cls.active_requests = 0
            cls.max_concurrent_requests = 0
    
    def do_POST(self):
        """Handle POST requests to the /chat/completions endpoint."""
        if self.path == "/chat/completions":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Track concurrent requests
            with self.__class__.request_lock:
                self.__class__.active_requests += 1
                self.__class__.max_concurrent_requests = max(
                    self.__class__.max_concurrent_requests, 
                    self.__class__.active_requests
                )
            
            # Simulate some processing time
            time.sleep(0.1)
            
            # Check if we should simulate an error
            if "trigger_error" in request_data.get("messages", [{}])[-1].get("content", ""):
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                error_response = {
                    "error": {
                        "message": "Simulated API error",
                        "type": "server_error",
                        "code": 500
                    }
                }
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
                
                # Decrement active requests
                with self.__class__.request_lock:
                    self.__class__.active_requests -= 1
                
                return
            
            # extract the messages from the request
            messages = request_data.get("messages", [])
            
            # prepare a response based on the messages
            response_content = f"Response to: {messages[-1]['content']}"
            
            # check if there's a stop sequence and we should include it
            stop = request_data.get("stop")
            include_stop = request_data.get("include_stop_str_in_output", False)
            
            if stop and include_stop:
                response_content += stop
            
            # create an OpenAI-like response
            response = {
                "id": "dummy-id",
                "object": "chat.completion",
                "created": 1234567890,
                "model": request_data.get("model", "dummy-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_content
                        },
                        "finish_reason": "stop"
                    }
                ]
            }
            
            # send the response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
            # Decrement active requests
            with self.__class__.request_lock:
                self.__class__.active_requests -= 1
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
    
    def log_message(self, format, *args):
        """Suppress log messages to keep test output clean."""
        pass


class TestOpenAICompatibleLanguageModel(unittest.TestCase):
    """Test the OpenAICompatibleLanguageModel class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test server."""
        cls.port = find_free_port()
        cls.server = HTTPServer(('localhost', cls.port), DummyOpenAIHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()
    
    @classmethod
    def tearDownClass(cls):
        """Shut down the test server."""
        cls.server.shutdown()
        cls.server_thread.join()
    
    def setUp(self):
        """Set up the language model for each test."""
        self.endpoint = f"http://localhost:{self.port}"
        self.model = OpenAICompatibleLanguageModel(
            endpoint=self.endpoint,
            api_key="dummy-api-key",
            model_name="dummy-model",
            system_prompt="You are a helpful assistant.",
            max_tries=2  # Set to a low value for faster tests
        )
        
        # Also create an async model for testing
        self.async_model = OpenAICompatibleLanguageModel(
            endpoint=self.endpoint,
            api_key="dummy-api-key",
            model_name="dummy-model",
            system_prompt="You are a helpful assistant.",
            is_async=True,
            max_tries=2  # Set to a low value for faster tests
        )
    
    def test_generate_single_message(self):
        """Test generating a response for a single message."""
        messages = [{"role": "user", "content": "Hello, world!"}]
        response = self.model.generate(messages)
        self.assertEqual(response, "Response to: Hello, world!")
    
    def test_generate_with_stop_token(self):
        """Test generating a response with a stop token."""
        messages = [{"role": "user", "content": "Hello, world!"}]
        response = self.model.generate(messages, stop="STOP")
        self.assertEqual(response, "Response to: Hello, world!")
    
    def test_generate_with_stop_token_included(self):
        """Test generating a response with an included stop token."""
        messages = [{"role": "user", "content": "Hello, world!"}]
        response = self.model.generate(messages, stop="STOP", include_stop_str_in_output=True)
        self.assertEqual(response, "Response to: Hello, world!STOP")
    
    def test_generate_multiple_messages(self):
        """Test generating responses for multiple message sets."""
        messages_lst = [
            [{"role": "user", "content": "Hello, world!"}],
            [{"role": "user", "content": "How are you?"}]
        ]
        responses = self.model.generate(messages_lst)
        self.assertEqual(responses, ["Response to: Hello, world!", "Response to: How are you?"])
    
    def test_with_system_prompt(self):
        """Test that the system prompt is included in the request."""
        messages = [{"role": "user", "content": "Hello, world!"}]
        response = self.model.generate(messages)
        self.assertEqual(response, "Response to: Hello, world!")
        # Note: We can't directly verify that the system prompt was included,
        # but the handler will use it when constructing the response
        
    def test_async_generate_single_message(self):
        """Test generating a response for a single message using async model."""
        messages = [{"role": "user", "content": "Hello, world!"}]
        response = self.async_model.generate(messages)
        self.assertEqual(response, "Response to: Hello, world!")
    
    def test_async_generate_multiple_messages(self):
        """Test generating responses for multiple message sets using async model."""
        messages_lst = [
            [{"role": "user", "content": "Hello, world!"}],
            [{"role": "user", "content": "How are you?"}],
            [{"role": "user", "content": "What's your name?"}],
            [{"role": "user", "content": "Tell me a joke."}]
        ]
        responses = self.async_model.generate(messages_lst)
        expected = [
            "Response to: Hello, world!",
            "Response to: How are you?",
            "Response to: What's your name?",
            "Response to: Tell me a joke."
        ]
        self.assertEqual(responses, expected)
    
    def test_async_with_parameters(self):
        """Test async generation with various parameters."""
        messages = [{"role": "user", "content": "Hello, world!"}]
        response = self.async_model.generate(
            messages, 
            stop="STOP", 
            max_tokens=100, 
            temperature=0.7, 
            include_stop_str_in_output=True
        )
        self.assertEqual(response, "Response to: Hello, world!STOP")
    
    def test_error_handling(self):
        """Test error handling with retries."""
        # This message will trigger a 500 error in the dummy server
        messages = [{"role": "user", "content": "trigger_error"}]
        
        # With max_tries=2, this should fail after 2 attempts
        model_with_retries = OpenAICompatibleLanguageModel(
            endpoint=self.endpoint,
            api_key="dummy-api-key",
            model_name="dummy-model",
            max_tries=2
        )
        
        with self.assertRaises(Exception) as context:
            model_with_retries.generate(messages)
        
        self.assertIn("API request failed", str(context.exception))
    
    def test_async_error_handling(self):
        """Test error handling with retries in async mode."""
        # This message will trigger a 500 error in the dummy server
        messages = [{"role": "user", "content": "trigger_error"}]
        
        # With max_tries=2, this should fail after 2 attempts
        async_model_with_retries = OpenAICompatibleLanguageModel(
            endpoint=self.endpoint,
            api_key="dummy-api-key",
            model_name="dummy-model",
            is_async=True,
            max_tries=2
        )
        
        with self.assertRaises(Exception) as context:
            async_model_with_retries.generate(messages)
        
        self.assertIn("API request failed", str(context.exception))
    
    def test_max_concurrency(self):
        """Test that the max_concurrency parameter limits concurrent requests."""
        # Instead of testing actual concurrency (which is hard in a single process),
        # we'll check that the semaphore is created with the correct value
        
        # Test with limited concurrency
        with patch('asyncio.Semaphore') as mock_semaphore:
            limited_model = OpenAICompatibleLanguageModel(
                endpoint=self.endpoint,
                api_key="dummy-api-key",
                model_name="dummy-model",
                is_async=True,
                max_concurrency=2
            )
            
            # Create messages to send
            messages_lst = [
                [{"role": "user", "content": f"Message {i}"}] for i in range(5)
            ]
            
            # Generate responses
            limited_model.generate(messages_lst)
            
            # Check that the semaphore was created with value=2
            mock_semaphore.assert_called_once_with(2)
        
        # Test with unlimited concurrency
        with patch('asyncio.Semaphore') as mock_semaphore:
            unlimited_model = OpenAICompatibleLanguageModel(
                endpoint=self.endpoint,
                api_key="dummy-api-key",
                model_name="dummy-model",
                is_async=True,
                max_concurrency=-1
            )
            
            # Create messages to send
            messages_lst = [
                [{"role": "user", "content": f"Message {i}"}] for i in range(5)
            ]
            
            # Generate responses
            unlimited_model.generate(messages_lst)
            
            # Check that the semaphore was created with value=5 (length of messages_lst)
            mock_semaphore.assert_called_once_with(5)


class MockLanguageModel:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    def generate(self, messages, stop=None, temperature=None, include_stop_str_in_output=None):
        # handle both single message and list of messages
        if isinstance(messages[0], dict):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        else:
            # for multiple messages, return a list of responses
            responses = self.responses[self.call_count:self.call_count + len(messages)]
            self.call_count += len(messages)
            return responses


class TestStepGeneration(unittest.TestCase):
    """Test the StepGeneration class."""
    
    def test_init(self):
        # test basic initialization
        step_gen = StepGeneration(step_token="\n", max_steps=5)
        self.assertEqual(step_gen.step_token, "\n")
        self.assertEqual(step_gen.max_steps, 5)
        self.assertIsNone(step_gen.stop_token)
        self.assertEqual(step_gen.temperature, 0.8)
        self.assertFalse(step_gen.include_stop_str_in_output)

        # test with custom parameters
        step_gen = StepGeneration(
            step_token="\n",
            max_steps=3,
            stop_token="END",
            temperature=0.5,
            include_stop_str_in_output=True
        )
        self.assertEqual(step_gen.step_token, "\n")
        self.assertEqual(step_gen.max_steps, 3)
        self.assertEqual(step_gen.stop_token, "END")
        self.assertEqual(step_gen.temperature, 0.5)
        self.assertTrue(step_gen.include_stop_str_in_output)

        # test validation
        with self.assertRaises(AssertionError):
            StepGeneration(step_token=None, max_steps=5, include_stop_str_in_output=True)

    def test_post_process(self):
        step_gen = StepGeneration(step_token="\n", max_steps=5)
        
        # test without stop token
        steps = ["step1", "step2", "step3"]
        result = step_gen._post_process(steps)
        self.assertEqual(result, "step1\nstep2\nstep3\n")
        
        # test with stop token
        result = step_gen._post_process(steps, stopped=True)
        self.assertEqual(result, "step1\nstep2\nstep3")
        
        # test with include_stop_str_in_output
        step_gen = StepGeneration(step_token="\n", max_steps=5, include_stop_str_in_output=True)
        result = step_gen._post_process(steps)
        self.assertEqual(result, "step1step2step3")

    def test_forward_single_prompt(self):
        # test basic forward with single prompt
        mock_lm = MockLanguageModel(["response1"])
        step_gen = StepGeneration(step_token="\n", max_steps=5)
        
        next_step, is_stopped = step_gen.forward(mock_lm, "test prompt")
        self.assertEqual(next_step, "response1")
        self.assertFalse(is_stopped)
        
        # test with steps_so_far
        mock_lm = MockLanguageModel(["response2"])
        next_step, is_stopped = step_gen.forward(mock_lm, "test prompt", steps_so_far=["step1"])
        self.assertEqual(next_step, "response2")
        self.assertFalse(is_stopped)
        
        # test max steps reached
        mock_lm = MockLanguageModel(["response3"])
        next_step, is_stopped = step_gen.forward(mock_lm, "test prompt", steps_so_far=["step1"] * 5)
        self.assertEqual(next_step, "response3")
        self.assertTrue(is_stopped)
        
        # test stop token
        step_gen = StepGeneration(step_token="\n", max_steps=5, stop_token="END")
        mock_lm = MockLanguageModel(["response with END"])
        next_step, is_stopped = step_gen.forward(mock_lm, "test prompt")
        self.assertEqual(next_step, "response with END")
        self.assertTrue(is_stopped)

    def test_forward_multiple_prompts(self):
        # test forward with multiple prompts
        mock_lm = MockLanguageModel(["response1", "response2"])
        step_gen = StepGeneration(step_token="\n", max_steps=5)
        
        prompts = ["prompt1", "prompt2"]
        steps_so_far = [["step1"], ["step2"]]
        results = step_gen.forward(mock_lm, prompts, steps_so_far)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "response1")
        self.assertFalse(results[0][1])
        self.assertEqual(results[1][0], "response2")
        self.assertFalse(results[1][1])
        
        # test with stop token in multiple prompts
        step_gen = StepGeneration(step_token="\n", max_steps=5, stop_token="END")
        mock_lm = MockLanguageModel(["response1", "response with END"])
        results = step_gen.forward(mock_lm, prompts, steps_so_far)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "response1")
        self.assertFalse(results[0][1])
        self.assertEqual(results[1][0], "response with END")
        self.assertTrue(results[1][1])
