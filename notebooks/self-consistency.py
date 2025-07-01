# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
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

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from its_hub.utils import SAL_STEP_BY_STEP_SYSTEM_PROMPT
from its_hub.lms import OpenAICompatibleLanguageModel

lm = OpenAICompatibleLanguageModel(
    endpoint="http://localhost:1234/v1", 
    api_key="NO_API_KEY", 
    model_name="qwen2-math-1.5b-instruct:2", 
    system_prompt=SAL_STEP_BY_STEP_SYSTEM_PROMPT, 
)
prompt = r"Let $a$ be a positive real number such that all the roots of \[x^3 + ax^2 + ax + 1 = 0\]are real. Find the smallest possible value of $a.$"

response = lm.generate(prompt)

print(response)


# %%
def extract_boxed(s: str) -> str:
    import re
    # find all occurrences of \boxed{...}
    boxed_matches = re.findall(r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}', s)
    # return the last match if any were found
    return boxed_matches[-1] if boxed_matches else ""
    
extract_boxed(response)

# %%
from its_hub.algorithms import SelfConsistency

budget = 16

scaling_alg = SelfConsistency(extract_boxed)

scaling_result = scaling_alg.infer(
    lm, prompt, budget, show_progress=True, return_response_only=False
)

print(scaling_result.the_one)

# %%
scaling_result.response_counts

# %%
