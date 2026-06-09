import json
import math

# the system prompt for step-by-step reasoning taken from https://github.com/huggingface/search-and-learn
SAL_STEP_BY_STEP_SYSTEM_PROMPT = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."

QWEN_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def extract_content_from_lm_response(message: dict) -> str:
    """
    Extract content from a single LM response message object.

    Args:
        message: A message dict returned by fetch_single_response.

    Returns:
        The content string. If the message contains tool calls, returns the content
        if available, otherwise returns an empty string.
    """
    # TODO: This conversion to text is not ideal as it involves manually formatting
    # tool calls and neglects images in multi-modal content. Consider refactoring
    # to work with structured message objects instead of flattening to strings.

    # Extract text content (handle both string and list[dict] formats)
    raw_content = message.get("content")

    if raw_content is None:
        content = ""
    elif isinstance(raw_content, str):
        content = raw_content
    elif isinstance(raw_content, list):
        # Multi-modal content: extract text parts (images are ignored)
        text_parts = [
            item.get("text", "")
            for item in raw_content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        content = " ".join(text_parts)
    else:
        raise ValueError(
            f"Invalid content type: {type(raw_content)}, expected str, list[dict], or None"
        )

    # If there are tool calls, add tool-calls to the content
    if message.get("tool_calls"):
        tool_calls = message.get("tool_calls", [])
        tool_descriptions = []
        for tc in tool_calls:
            if isinstance(tc, dict) and "function" in tc:
                func = tc["function"]
                func_name = func.get("name", "unknown")
                tool_descriptions.append(
                    f"[Tool call: {func_name} Tool args: {json.dumps(func.get('arguments', {}))}]"
                )
            else:
                raise ValueError(
                    f"Invalid tool call: {tc}, expected a dict with a 'function' key"
                )
        content += " ".join(tool_descriptions)

    return content


def summarize_step_logprobs(logprobs: dict | None) -> dict:
    """Summarize an OpenAI-style ``logprobs`` object for one generated step.

    Used to derive self-certainty particle weights from the *generator* model's
    own token log-probabilities (no separate reward model).

    Args:
        logprobs: The ``choices[0].logprobs`` object returned by an
            OpenAI-compatible API when ``logprobs=true`` was requested. Expected
            shape: ``{"content": [{"token": str, "logprob": float,
            "top_logprobs": [{"token": str, "logprob": float}, ...]}, ...]}``.

    Returns:
        dict with:
          - ``mean_logprob``: mean per-token log-probability of the chosen tokens
            (<= 0; closer to 0 = more confident). 0.0 if unavailable.
          - ``entropy``: mean per-token entropy in nats, approximated over the
            returned ``top_logprobs`` at each position (>= 0; lower = more
            decisive). ``None`` if ``top_logprobs`` were not returned.
          - ``num_tokens``: number of tokens summarized.
    """
    if not logprobs:
        return {"mean_logprob": 0.0, "entropy": None, "num_tokens": 0}

    content = logprobs.get("content") or []
    token_logprobs = [
        t.get("logprob") for t in content if t.get("logprob") is not None
    ]
    num_tokens = len(token_logprobs)
    if num_tokens == 0:
        return {"mean_logprob": 0.0, "entropy": None, "num_tokens": 0}

    mean_logprob = sum(token_logprobs) / num_tokens

    # Entropy is approximated over the returned top-k logprobs at each position.
    entropies = []
    for t in content:
        tops = t.get("top_logprobs") or []
        lps = [e.get("logprob") for e in tops if e.get("logprob") is not None]
        if lps:
            entropies.append(-sum(math.exp(lp) * lp for lp in lps))
    entropy = sum(entropies) / len(entropies) if entropies else None

    return {
        "mean_logprob": mean_logprob,
        "entropy": entropy,
        "num_tokens": num_tokens,
    }
