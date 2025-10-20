import json
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
    # Extract text content (may be empty if message only has tool calls)
    content = message.get("content", "") or ""

    # If there are tool calls, add tool-calls to the content
    if message.get("tool_calls"):
        tool_calls = message.get("tool_calls", [])
        tool_descriptions = []
        for tc in tool_calls[:1]:
            if isinstance(tc, dict) and "function" in tc:
                func = tc["function"]
                func_name = func.get("name", "unknown")
                tool_descriptions.append(f"[Tool call: {func_name} Tool args: {json.dumps(func.get('arguments', {}))}]")
            else:
                raise ValueError(f"Invalid tool call: {tc}, expected a dict with a 'function' key")
        content += " ".join(tool_descriptions)

    return content
