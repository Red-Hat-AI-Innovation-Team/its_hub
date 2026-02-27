# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: inference_time_scaling-dev
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.11.11
# ---

# %% [markdown]
# # Self-Consistency Algorithm Demo
# This notebook demonstrates the Self-Consistency algorithm for mathematical reasoning.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os

import nest_asyncio
from dotenv import load_dotenv

from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT

nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

# Main example: OpenAI API endpoint with gpt-4o-mini
lm = OpenAICompatibleLanguageModel(
    endpoint="https://api.openai.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),  # Load API key from environment
    model_name="gpt-4o-mini",
    system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT,
    is_async=True,
)
# %%
# Alternative: vLLM local endpoint (commented out)
# lm = OpenAICompatibleLanguageModel(
#     endpoint="http://localhost:8000/v1",
#     api_key="NO_API_KEY",
#     model_name="qwen2-math-1.5b-instruct",
#     system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT,
#     is_async=True,
# )

# %%
# Mathematical problem to solve
prompt = r"Let $a$ be a positive real number such that all the roots of \[x^3 + ax^2 + ax + 1 = 0\]are real. Find the smallest possible value of $a.$"

# Generate response using the proper format
from its_hub.types import ChatMessages

chat_messages = ChatMessages.from_prompt_or_messages(prompt)
response = lm.generate(chat_messages.to_batch(1))[0]

print(response)


# %%
def extract_boxed(s: str) -> str:
    import re
    # find all occurrences of \boxed{...}
    boxed_matches = re.findall(r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}', s)
    # return the last match if any were found
    return boxed_matches[-1] if boxed_matches else ""

print(extract_boxed(response['content']))

# %% [markdown]
# ## Self-Consistency Algorithm
# Now we'll use the Self-Consistency algorithm to improve the answer quality.

# %%
from its_hub.algorithms import SelfConsistency

# Set computational budget for scaling
budget = 4

scaling_alg = SelfConsistency(extract_boxed)

scaling_result = scaling_alg.infer(
    lm, prompt, budget, return_response_only=False
)

print("######## Self-Consistency Result ########")
print(scaling_result.the_one)

# %%
print("######## Extracted Response Counts ########")
print(scaling_result.response_counts)

# %%


# %% [markdown]
# ## Self-Consistency Algorithm for Tool Calls
# We have hierarchical tool-voting support in Self-Consistency algorithm
# It first votes on tool names, and then on tool arguments.

# %%
from its_hub.types import ChatMessage, ChatMessages

# Tool schema (OpenAI-style dicts)
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform arithmetic calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# ChatMessages instance with system + user
tool_call_messages = ChatMessages([
    ChatMessage(
        role="system",
        content="You are a precise calculator. Always use the calculator tool for arithmetic and format your final answer as \\boxed{result}."
    ),
    ChatMessage(
        role="user",
        content="What is 847 * 293 + 156?"
    ),
])

# %%
# Use hierarchical tool voting
scaling_alg_tool = SelfConsistency(tool_vote="tool_hierarchical")

budget = 5
scaling_result = scaling_alg_tool.infer(
    lm, tool_call_messages, budget, return_response_only=False, tools=tools, tool_choice="auto"
)

# %%
print("######## Self-Consistency Result ########")
print(scaling_result.the_one)

print("######## Tool Call Response Counts ########")
print(scaling_result.response_counts)

