# ITS Hub - Inference-Time Scaling Commands

# =============================================================================
# Development Setup
# =============================================================================

# Install package in development mode
install:
    uv sync --extra dev

# Setup development environment (general Python setup)
setup:
    make setup

# Setup Envoy gateway environment (includes submodules and proto compilation)
setup-envoy:
    make setup-envoy

# Initialize and update git submodules
submodule-init:
    make submodule-init

# Restore proto submodules to pinned commits from .gitmodules
upgrade-protos:
    make upgrade-protos

# Run all tests
test:
    uv run pytest tests/

# =============================================================================
# Proto Compilation
# =============================================================================

# Compile Envoy external processor proto files to Python
proto-compile:
    make proto-compile

# Clean generated proto files
proto-clean:
    make proto-clean

# =============================================================================
# Service Management
# =============================================================================

# Start IaaS service on localhost:8108
iaas-start:
    uv run its-iaas --host 0.0.0.0 --port 8108

# Check service health
iaas-health:
    curl -v -s http://localhost:8108/v1/models | jq .

# Start Envoy External Processor gRPC service on port 50051
envoy-grpc-start:
    uv run envoy-grpc

# Test Envoy External Processor with sample requests
envoy-grpc-test:
    uv run python scripts/test_envoy_grpc.py

# Start Envoy proxy with ext_proc configuration
envoy-start:
    envoy -c config/envoy/ext_proc.yaml

# Check Envoy cluster health and statistics
envoy-health:
    @echo "=== Cluster Status ==="
    @curl -s http://localhost:9901/clusters | grep -A 10 "ext_proc_cluster" || echo "Envoy not running or admin port not accessible"
    @echo "\n=== ext_proc Statistics ==="
    @curl -s http://localhost:9901/stats | grep ext_proc || echo "No ext_proc stats found"

# =============================================================================
# Algorithm Configuration
# =============================================================================



# Configure tool-vote
config-self-consistency-bedrock:
    #!/bin/bash
    source .env
    curl -X POST http://localhost:8108/configure \
        -H "Content-Type: application/json" \
        -d "{
            \"provider\": \"litellm\",
            \"endpoint\": \"auto\",
            \"api_key\": null,
            \"model\": \"bedrock/$BEDROCK_MODEL\",
            \"alg\": \"self-consistency\",
            \"tool_vote\": \"tool_hierarchical\",
            \"exclude_args\": [\"timestamp\", \"request_id\", \"id\", \"type\", \"stepTitle\", \"stepDescription\", \"threadId\", \"domain_id\"],
            \"extra_args\": {
                \"aws_access_key_id\": \"$AWS_ACCESS_KEY_ID\",
                \"aws_secret_access_key\": \"$AWS_SECRET_ACCESS_KEY\",
                \"aws_region_name\": \"$AWS_REGION\"
            }
        }" \
        -w "\nHTTP Status: %{http_code}\n" -v



config-self-consistency-openai:
    #!/bin/bash
    source .env
    curl -X POST http://localhost:8108/configure \
        -H "Content-Type: application/json" \
        -d "{
            \"provider\": \"litellm\",
            \"endpoint\": \"auto\",
            \"api_key\": \"$OPENAI_API_KEY\",
            \"model\": \"gpt-5\",
            \"alg\": \"self-consistency\",
            \"tool_vote\": \"tool_hierarchical\",
            \"exclude_args\": [\"timestamp\", \"request_id\", \"id\", \"type\", \"stepTitle\", \"stepDescription\", \"threadId\", \"domain_id\"]
        }" \
        -w "\nHTTP Status: %{http_code}\n" -v



# Configure best-of-n with LLM judge (bedrock models)
config-bon-bedrock:
    #!/bin/bash
    source .env
    curl -X POST http://localhost:8108/configure \
        -H "Content-Type: application/json" \
        -d '{
            "provider": "litellm",
            "endpoint": "auto",
            "api_key": null,
            "model": "bedrock/'"$BEDROCK_MODEL"'",
            "alg": "best-of-n",
            "rm_name": "llm-judge",
            "judge_model": "bedrock/'"$BEDROCK_MODEL"'",
            "judge_base_url": "auto",
            "judge_mode": "groupwise",
            "judge_criterion": "multi_step_tool_judge",
            "judge_api_key": null,
            "judge_temperature": 0.7,
            "judge_max_tokens": 2048,
            "extra_args": {
                "aws_access_key_id": "'"$AWS_ACCESS_KEY_ID"'",
                "aws_secret_access_key": "'"$AWS_SECRET_ACCESS_KEY"'",
                "aws_region_name": "'"$AWS_REGION"'"
            }
        }' \
        -w "\nHTTP Status: %{http_code}\n" -v

# Configure best-of-n with LLM judge (openai models)
config-bon-openai:
    #!/bin/bash
    source .env
    curl -X POST http://localhost:8108/configure \
        -H "Content-Type: application/json" \
        -d '{
            "provider": "litellm",
            "endpoint": "auto",
            "api_key": "'"$OPENAI_API_KEY"'",
            "model": "gpt-4.1-mini",
            "alg": "best-of-n",
            "rm_name": "llm-judge",
            "judge_model": "gpt-4.1-mini",
            "judge_base_url": "auto",
            "judge_mode": "groupwise",
            "judge_criterion": "multi_step_tool_judge",
            "judge_api_key": "'"$OPENAI_API_KEY"'",
            "judge_temperature": 0.7,
            "judge_max_tokens": 2048
        }' \
        -w "\nHTTP Status: %{http_code}\n" -v


# Test simple conversation
test-collie:
    source .env
    curl -s -X POST http://localhost:8108/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0", "messages": [{"role": "user", "content": [{"type": "text", "text": "Explain quantum computing in one sentence."}]}], "budget": 2, "return_response_only": false}' | jq .




# Configure self-consistency (regex-based content voting)
config-self-consistency:
    #!/bin/bash
    source .env
    curl -X POST http://localhost:8108/configure \
        -H "Content-Type: application/json" \
        -d "{\"endpoint\": \"https://api.openai.com/v1\", \"api_key\": \"$OPENAI_API_KEY\", \"model\": \"gpt-4.1-mini\", \"alg\": \"self-consistency\", \"regex_patterns\": [\"boxed\\\\{([^}]+)\\\\}\"]}" \
        -w "\nHTTP Status: %{http_code}\n" -v


config-self-elevance:
    #!/bin/bash
    source .env
    curl -X POST http://localhost:8108/configure \
        -H "Content-Type: application/json" \
        -d "{\"endpoint\": \"http://localhost:8100/v1\", \"api_key\": \"nothing\", \"model\": \"meta-llama/Llama-3.3-70B-Instruct\", \"alg\": \"self-consistency\", \"regex_patterns\": [\"elevance\"]}" \
        -w "\nHTTP Status: %{http_code}\n" -v




# Configure self-consistency with tool name voting
config-tool-name-vote:
    #!/bin/bash
    source .env
    curl -X POST http://localhost:8108/configure \
        -H "Content-Type: application/json" \
        -d "{\"endpoint\": \"https://api.openai.com/v1\", \"api_key\": \"$OPENAI_API_KEY\", \"model\": \"gpt-4.1-mini\", \"alg\": \"self-consistency\", \"regex_patterns\": [\"boxed\\\\{([^}]+)\\\\}\"], \"tool_vote\": \"tool_name\"}" \
        -w "\nHTTP Status: %{http_code}\n" -v

# Configure self-consistency with hierarchical tool voting (name + args)
config-tool-hierarchical-vote:
    #!/bin/bash
    source .env
    curl -X POST http://localhost:8108/configure \
        -H "Content-Type: application/json" \
        -d "{\"endpoint\": \"https://api.openai.com/v1\", \"api_key\": \"$OPENAI_API_KEY\", \"model\": \"gpt-4.1-mini\", \"alg\": \"self-consistency\", \"regex_patterns\": [\"boxed\\\\{([^}]+)\\\\}\"], \"tool_vote\": \"tool_hierarchical\", \"exclude_args\": [\"timestamp\", \"request_id\"]}" \
        -w "\nHTTP Status: %{http_code}\n" -v

# Configure best-of-n algorithm (requires reward model)
config-best-of-n:
    curl -X POST http://localhost:8108/configure \
        -H "Content-Type: application/json" \
        -d "{\"endpoint\": \"https://api.openai.com/v1\", \"api_key\": \"$OPENAI_API_KEY\", \"model\": \"gpt-4.1-mini\", \"alg\": \"best-of-n\", \"rm_name\": \"reward-model\"}"

# Configure best-of-n with LLM judge (groupwise ranking)
config-best-of-n-llm-judge:
    #!/bin/bash
    source .env
    curl -X POST http://localhost:8108/configure \
        -H "Content-Type: application/json" \
        -d "{\"endpoint\": \"https://api.openai.com/v1\", \"api_key\": \"$OPENAI_API_KEY\", \"model\": \"gpt-4.1-mini\", \"alg\": \"best-of-n\", \"use_llm_judge\": true, \"judge_type\": \"groupwise\", \"judge_model\": \"gpt-4.1-mini\", \"judge_criterion\": \"tool-judge\", \"judge_api_key\": \"$OPENAI_API_KEY\", \"judge_temperature\": 0.7, \"judge_prompt\": \"tool-judge\", \"judge_max_tokens\": 2048}" \
        -w "\nHTTP Status: %{http_code}\n" -v

# =============================================================================
# Basic Tests
# =============================================================================

# Test simple conversation
test-chat:
    curl -s -X POST http://localhost:8108/v1/chat/completions \
        -H "Content-Type: application/json" \
        -H "X-ITS-Budget: 3" \
        -H "X-ITS-Endpoint: https://api.openai.com/v1" \
        -H "X-ITS-API-Key: $OPENAI_API_KEY" \
        -d '{"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "Choose 1 word to best represent quantum computing and that word will be: "}]}'

# Test conversation with empty assistant response
test-chat-empty:
    curl -s -X POST http://localhost:8108/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "What is artificial intelligence?"}, {"role": "assistant", "content": ""}, {"role": "user", "content": "Can you explain it in simple terms?"}], "budget": 2, "return_response_only": false}' | jq .


# Test simple conversation
test-elevance:
    curl -s -X POST http://localhost:8108/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "meta-llama/Llama-3.3-70B-Instruct", "messages": [{"role": "user", "content": "repeat the following string exactly: task #1 {\n    %1 = members.ref_member\n    %2 = priorAuthorization.filter patient_member_id:%1 qualifier_phrase:\"HEIGHTS SURGERY CENTER\"\n    %3 = priorAuthorization.ref_authorizations filter:%2\n    %4 = priorAuthorization.show_authorizations priorAuthorizationResponse:%3\n    %5 = builtin.abstract_question dependency:%3 question:\"Was my procedure pre-approved?\"\n    %6 = builtin.show data:%5\n}"}], "budget": 8, "return_response_only": false}' | jq .


# Test mathematical reasoning
test-math:
    curl -s -X POST http://localhost:8108/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "gpt-4.1-mini", "messages": [{"role": "system", "content": "Solve step by step and put your final answer in \\boxed{} format."},{"role": "user", "content": "If a train travels 240 miles in 3 hours, what is its average speed in mph?"}], "budget": 2, "return_response_only": false}' | jq .

# Test best-of-n with LLM judge
test-bon-llm-judge:
    curl -s -X POST http://localhost:8108/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "Explain the difference between machine learning and deep learning in simple terms."}], "budget": 4, "return_response_only": false}' | jq .

# Test multi-modal content format
test-multimodal:
    curl -s -X POST http://localhost:8108/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": [{"type": "text", "text": "Explain quantum computing in one sentence."}]}], "budget": 2, "return_response_only": false}' | jq .

# =============================================================================
# Tool Call Tests  
# =============================================================================

# Test calculator tool with simple arithmetic
test-calculator:
    curl -s -X POST http://localhost:8108/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{ \
            "model": "gpt-4.1-mini", \
            "messages": [ \
                { \
                    "role": "system", \
                    "content": "You are a precise calculator. Always use the calculator tool for arithmetic and format your final answer as \\boxed{result}." \
                }, \
                { \
                    "role": "user", \
                    "content": "What is 847 * 293 + 156?" \
                } \
            ], \
            "budget": 16, \
            "return_response_only": false, \
            "tools": [ \
                { \
                    "type": "function", \
                    "function": { \
                        "name": "calculator", \
                        "description": "Perform arithmetic calculations", \
                        "parameters": { \
                            "type": "object", \
                            "properties": { \
                                "expression": { \
                                    "type": "string", \
                                    "description": "Mathematical expression to evaluate" \
                                } \
                            }, \
                            "required": ["expression"] \
                        } \
                    } \
                } \
            ], \
            "tool_choice": "auto" \
        }' | jq .

# Test multiple diverse tools - research assistant scenario
test-research-assistant:
    curl -s -X POST http://localhost:8108/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{ \
            "model": "gpt-4.1-mini", \
            "messages": [ \
                { \
                    "role": "system", \
                    "content": "You are a research assistant with access to various tools. Use appropriate tools based on the user request." \
                }, \
                { \
                    "role": "user", \
                    "content": "I need to research the current weather in Tokyo and calculate how many days until Christmas 2024." \
                } \
            ], \
            "budget": 2, \
            "return_response_only": false, \
            "tools": [ \
                { \
                    "type": "function", \
                    "function": { \
                        "name": "get_weather", \
                        "description": "Get current weather information for a city", \
                        "parameters": { \
                            "type": "object", \
                            "properties": { \
                                "city": { \
                                    "type": "string", \
                                    "description": "City name" \
                                }, \
                                "country": { \
                                    "type": "string", \
                                    "description": "Country code (optional)" \
                                } \
                            }, \
                            "required": ["city"] \
                        } \
                    } \
                }, \
                { \
                    "type": "function", \
                    "function": { \
                        "name": "calculate_date_difference", \
                        "description": "Calculate days between two dates", \
                        "parameters": { \
                            "type": "object", \
                            "properties": { \
                                "start_date": { \
                                    "type": "string", \
                                    "description": "Start date in YYYY-MM-DD format" \
                                }, \
                                "end_date": { \
                                    "type": "string", \
                                    "description": "End date in YYYY-MM-DD format" \
                                } \
                            }, \
                            "required": ["start_date", "end_date"] \
                        } \
                    } \
                } \
            ], \
            "tool_choice": "auto" \
        }' | jq .

# Test file operations scenario
test-file-operations:
    curl -s -X POST http://localhost:8108/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{ \
            "model": "gpt-4.1-mini", \
            "messages": [ \
                { \
                    "role": "system", \
                    "content": "You are a file management assistant. Help users with file operations like reading, writing, and searching files." \
                }, \
                { \
                    "role": "user", \
                    "content": "Create a todo list file called tasks.txt with 3 programming tasks, then read it back to verify." \
                } \
            ], \
            "budget": 2, \
            "return_response_only": false, \
            "tools": [ \
                { \
                    "type": "function", \
                    "function": { \
                        "name": "write_file", \
                        "description": "Write content to a file", \
                        "parameters": { \
                            "type": "object", \
                            "properties": { \
                                "filename": { \
                                    "type": "string", \
                                    "description": "Name of the file to write" \
                                }, \
                                "content": { \
                                    "type": "string", \
                                    "description": "Content to write to the file" \
                                } \
                            }, \
                            "required": ["filename", "content"] \
                        } \
                    } \
                }, \
                { \
                    "type": "function", \
                    "function": { \
                        "name": "read_file", \
                        "description": "Read content from a file", \
                        "parameters": { \
                            "type": "object", \
                            "properties": { \
                                "filename": { \
                                    "type": "string", \
                                    "description": "Name of the file to read" \
                                } \
                            }, \
                            "required": ["filename"] \
                        } \
                    } \
                } \
            ], \
            "tool_choice": "auto" \
        }' | jq .

# Test code analysis tools
test-code-analysis:
    curl -s -X POST http://localhost:8108/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{ \
            "model": "gpt-4.1-mini", \
            "messages": [ \
                { \
                    "role": "system", \
                    "content": "You are a code analysis expert with access to various programming tools." \
                }, \
                { \
                    "role": "user", \
                    "content": "Analyze this Python function for potential issues: def factorial(n): return n * factorial(n-1) if n > 0 else 1" \
                } \
            ], \
            "budget": 2, \
            "return_response_only": false, \
            "tools": [ \
                { \
                    "type": "function", \
                    "function": { \
                        "name": "run_code", \
                        "description": "Execute Python code and return the result", \
                        "parameters": { \
                            "type": "object", \
                            "properties": { \
                                "code": { \
                                    "type": "string", \
                                    "description": "Python code to execute" \
                                }, \
                                "test_inputs": { \
                                    "type": "array", \
                                    "items": {"type": "string"}, \
                                    "description": "Test inputs to try with the code" \
                                } \
                            }, \
                            "required": ["code"] \
                        } \
                    } \
                }, \
                { \
                    "type": "function", \
                    "function": { \
                        "name": "lint_code", \
                        "description": "Check code for style and potential issues", \
                        "parameters": { \
                            "type": "object", \
                            "properties": { \
                                "code": { \
                                    "type": "string", \
                                    "description": "Code to analyze" \
                                }, \
                                "language": { \
                                    "type": "string", \
                                    "description": "Programming language" \
                                } \
                            }, \
                            "required": ["code", "language"] \
                        } \
                    } \
                } \
            ], \
            "tool_choice": "auto" \
        }' | jq .

# Test tool choice constraints - force specific tool usage
test-forced-tool-choice:
    curl -s -X POST http://localhost:8108/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{ \
            "model": "gpt-4.1-mini", \
            "messages": [ \
                { \
                    "role": "system", \
                    "content": "You must use the search tool for any information requests." \
                }, \
                { \
                    "role": "user", \
                    "content": "What is the capital of France?" \
                } \
            ], \
            "budget": 2, \
            "return_response_only": false, \
            "tools": [ \
                { \
                    "type": "function", \
                    "function": { \
                        "name": "web_search", \
                        "description": "Search the web for information", \
                        "parameters": { \
                            "type": "object", \
                            "properties": { \
                                "query": { \
                                    "type": "string", \
                                    "description": "Search query" \
                                } \
                            }, \
                            "required": ["query"] \
                        } \
                    } \
                }, \
                { \
                    "type": "function", \
                    "function": { \
                        "name": "calculator", \
                        "description": "Perform calculations", \
                        "parameters": { \
                            "type": "object", \
                            "properties": { \
                                "expression": { \
                                    "type": "string", \
                                    "description": "Mathematical expression" \
                                } \
                            }, \
                            "required": ["expression"] \
                        } \
                    } \
                } \
            ], \
            "tool_choice": {"type": "function", "function": {"name": "web_search"}} \
        }' | jq .

# =============================================================================
# Advanced Tests
# =============================================================================

# Test high budget for complex reasoning
test-complex-reasoning:
    curl -s -X POST http://localhost:8108/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "gpt-4.1-mini", "messages": [{"role": "system", "content": "You are an expert problem solver. Work through problems step by step and show your reasoning."},{"role": "user", "content": "A rectangular garden is 3 times as long as it is wide. If the perimeter is 56 meters, what are the dimensions? Show all work and put your final answer in \\boxed{} format."}], "budget": 2, "return_response_only": false}' | jq .

# Test edge case: budget of 1  
test-budget-one:
    curl -s -X POST http://localhost:8108/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "What is 5 + 3?"}], "budget": 1, "return_response_only": false}' | jq .

# Test response format comparison
test-response-formats:
    @echo "=== Testing return_response_only=true ==="
    curl -s -X POST http://localhost:8108/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "What is 2+2?"}], "budget": 2, "return_response_only": true}' | jq .
    @echo -e "\n=== Testing return_response_only=false ==="
    curl -s -X POST http://localhost:8108/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "What is 2+2?"}], "budget": 2, "return_response_only": false}' | jq .
