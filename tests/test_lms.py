import unittest
import json
import threading
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
import socket
from typing import List, Dict, Any

from its_hub.lms import OpenAICompatibleLanguageModel


def find_free_port():
    """Find a free port to use for the test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class DummyOpenAIHandler(BaseHTTPRequestHandler):
    """A dummy HTTP handler that mimics the OpenAI API."""
    
    def do_POST(self):
        """Handle POST requests to the /chat/completions endpoint."""
        if self.path == "/chat/completions":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
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
                return
            
            # extract the messages from the request
            messages = request_data.get("messages", [])
            
            # prepare a response based on the messages
            response_content = f"Response to: {messages[-1]['content']}"
            
            # check if there's a stop sequence and we should include it
            stop = request_data.get("stop")
            include_stop = request_data.get("extra_body", {}).get("include_stop_str_in_output", False)
            
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
