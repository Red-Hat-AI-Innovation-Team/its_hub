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
        content = ""

    # If there are tool calls but no text content, create a text representation
    if message.get("tool_calls") and not content:
        tool_calls = message.get("tool_calls", [])
        tool_descriptions = []
        for tc in tool_calls:
            if isinstance(tc, dict) and "function" in tc:
                func = tc["function"]
                func_name = func.get("name", "unknown")
                tool_descriptions.append(f"[Tool call: {func_name}]")
        content = " ".join(tool_descriptions) if tool_descriptions else ""

    return content
