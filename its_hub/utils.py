"""Utility functions for its_hub."""

import json


def extract_content_from_lm_response(message: dict) -> str:
    """
    Extract content from a language model response message.

    Args:
        message: A message dict with 'content' and optional 'tool_calls'

    Returns:
        The content string. For tool calls, includes formatted tool information.
    """
    # Extract text content (handle both string and list[dict] formats)
    raw_content = message.get("content")

    if raw_content is None:
        content = ""
    elif isinstance(raw_content, str):
        content = raw_content
    elif isinstance(raw_content, list):
        # Multi-modal content: extract text parts
        text_parts = [
            item.get("text", "")
            for item in raw_content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        content = " ".join(text_parts)
    else:
        raise ValueError(
            f"Invalid content type: {type(raw_content)}, "
            f"expected str, list[dict], or None"
        )

    # If there are tool calls, append tool call information
    if message.get("tool_calls"):
        tool_calls = message.get("tool_calls", [])
        tool_descriptions = []
        for tc in tool_calls:
            if isinstance(tc, dict) and "function" in tc:
                func = tc["function"]
                func_name = func.get("name", "unknown")
                func_args = json.dumps(func.get("arguments", {}))
                tool_descriptions.append(
                    f"[Tool call: {func_name} Tool args: {func_args}]"
                )
            else:
                raise ValueError(
                    f"Invalid tool call: {tc}, expected dict with 'function' key"
                )
        content += " ".join(tool_descriptions)

    return content
