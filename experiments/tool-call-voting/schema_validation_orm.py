"""Schema Validation Outcome Reward Model.

Zero-cost, deterministic scorer that validates tool calls against
the function's JSON schema. Plugs into its_hub's BestOfN algorithm
via AbstractOutcomeRewardModel.
"""

from __future__ import annotations

import json
import logging

from its_hub.api.reward_models.orm import AbstractOutcomeRewardModel
from its_hub.api.types import ChatMessage, ChatMessages

logger = logging.getLogger(__name__)


class SchemaValidationORM(AbstractOutcomeRewardModel):
    """Scores tool calls by validating against provided JSON schemas.

    Each check (function name, required params, type correctness, enum
    validity, no extra params) contributes equally to the final score.
    """

    def __init__(self, tools: list[dict]) -> None:
        self._schemas: dict[str, dict] = {}
        for tool in tools:
            func = tool.get("function", tool)
            name = func.get("name", "")
            if name:
                self._schemas[name] = func

    def score(
        self,
        messages: list[ChatMessage] | ChatMessages,
        **kwargs,
    ) -> list[float] | float:
        is_batch = messages and isinstance(messages[0], list)
        conversations = messages if is_batch else [messages]
        scores = [self._score_conversation(conv) for conv in conversations]
        return scores if is_batch else scores[0]

    async def ascore(self, messages, orchestrator=None, **kwargs):
        return self.score(messages, **kwargs)

    def _score_conversation(self, conversation: list) -> float:
        last_msg = conversation[-1] if conversation else None
        if last_msg is None:
            return 0.0

        if isinstance(last_msg, ChatMessage):
            tool_calls = last_msg.tool_calls
        elif isinstance(last_msg, dict):
            tool_calls = last_msg.get("tool_calls")
        else:
            return 0.0

        if not tool_calls:
            return 0.0

        scores = [self._score_tool_call(tc) for tc in tool_calls]
        return sum(scores) / len(scores)

    def _score_tool_call(self, tool_call: dict) -> float:
        func = tool_call.get("function", {})
        name = func.get("name", "")
        args_raw = func.get("arguments", "{}")

        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except (json.JSONDecodeError, TypeError):
                args = {}
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            args = {}

        checks: list[bool] = []

        # Check 1: function name exists in schemas
        name_valid = name in self._schemas
        checks.append(name_valid)

        if not name_valid:
            return 0.0

        schema = self._schemas[name]
        params = schema.get("parameters", {})
        properties = params.get("properties", {})
        required = set(params.get("required", []))

        # Check 2: all required parameters present
        for req in required:
            checks.append(req in args)

        # Check 3: parameter types match schema
        for param_name, param_value in args.items():
            if param_name in properties:
                prop_schema = properties[param_name]
                checks.append(_type_matches(param_value, prop_schema))

        # Check 4: enum values valid
        for param_name, param_value in args.items():
            if param_name in properties:
                prop_schema = properties[param_name]
                if "enum" in prop_schema:
                    checks.append(param_value in prop_schema["enum"])

        # Check 5: no unexpected parameters
        if properties:
            additional = params.get("additionalProperties", True)
            if not additional or properties:
                for param_name in args:
                    checks.append(param_name in properties)

        if not checks:
            return 1.0

        return sum(1 for c in checks if c) / len(checks)


def _type_matches(value, prop_schema: dict) -> bool:
    """Check if a value matches a JSON schema type, with BFCL type normalization."""
    schema_type = prop_schema.get("type", "")

    type_map = {
        "dict": "object",
        "Dict": "object",
        "list": "array",
        "List": "array",
        "float": "number",
        "int": "integer",
        "str": "string",
        "bool": "boolean",
        "tuple": "array",
        "Tuple": "array",
    }
    schema_type = type_map.get(schema_type, schema_type)

    if schema_type == "string":
        return isinstance(value, str)
    elif schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    elif schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    elif schema_type == "boolean":
        return isinstance(value, bool)
    elif schema_type == "array":
        return isinstance(value, list)
    elif schema_type == "object":
        return isinstance(value, dict)
    elif schema_type == "null":
        return value is None

    return True
