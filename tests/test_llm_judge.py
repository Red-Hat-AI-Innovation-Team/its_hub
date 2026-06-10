"""Tests for LLMJudge JSON parsing and structured output support."""

import threading

import pytest

from its_hub.api import ChatMessage
from its_hub.core.orchestrator import LMOrchestrator
from its_hub.core.reward_models.llm_judge import LLMJudge


class DummyLM:
    """Minimal stub — only _parse_score is tested, no LM calls needed."""
    pass


class MockLM:
    """Mock LM that returns configurable responses and records call kwargs."""

    def __init__(self, response_content: str = '{"score": 8, "reasoning": "good"}'):
        self.response_content = response_content
        self.calls: list[dict] = []
        self._lock = threading.Lock()

    async def agenerate_single(self, messages, loop=None, **kwargs):
        with self._lock:
            self.calls.append({"messages": messages, **kwargs})
        return {"role": "assistant", "content": self.response_content}


@pytest.fixture
def judge():
    return LLMJudge(lm=DummyLM(), fallback_score=5.0)


# ===========================================================================
# 1. JSON Parsing
# ===========================================================================

class TestParseScore:
    """Test _parse_score handles common LLM response patterns."""

    def test_raw_json(self, judge):
        assert judge._parse_score('{"score": 8}') == 8.0

    def test_json_with_whitespace(self, judge):
        assert judge._parse_score('  {"score": 7.5}  ') == 7.5

    def test_markdown_code_block(self, judge):
        text = '```json\n{"score": 9}\n```'
        assert judge._parse_score(text) == 9.0

    def test_markdown_code_block_no_lang(self, judge):
        text = '```\n{"score": 6}\n```'
        assert judge._parse_score(text) == 6.0

    def test_json_with_preamble(self, judge):
        text = 'Here is my evaluation:\n{"score": 4}'
        assert judge._parse_score(text) == 4.0

    def test_json_with_surrounding_text(self, judge):
        text = 'Based on analysis, {"score": 3} is my rating.'
        assert judge._parse_score(text) == 3.0

    def test_json_with_reasoning(self, judge):
        text = '{"score": 7, "reasoning": "Good response"}'
        assert judge._parse_score(text) == 7.0

    def test_missing_score_key(self, judge):
        assert judge._parse_score('{"rating": 8}') == 5.0

    def test_no_json_at_all(self, judge):
        assert judge._parse_score("This is a great response") == 5.0

    def test_empty_string(self, judge):
        assert judge._parse_score("") == 5.0

    def test_integer_score(self, judge):
        assert judge._parse_score('{"score": 10}') == 10.0

    def test_markdown_with_extra_text(self, judge):
        text = 'I evaluated the conversation:\n```json\n{"score": 2}\n```\nHope this helps!'
        assert judge._parse_score(text) == 2.0


class TestExtractJson:
    """Test _extract_json directly for edge cases."""

    def test_returns_none_for_garbage(self, judge):
        assert judge._extract_json("no json here") is None

    def test_nested_braces_takes_first_flat_object(self, judge):
        result = judge._extract_json('outer {"score": 5} end')
        assert result == {"score": 5}

    def test_code_block_preferred_over_inline(self, judge):
        text = 'text {"score": 1}\n```json\n{"score": 9}\n```'
        result = judge._extract_json(text)
        assert result["score"] == 9


# ===========================================================================
# 2. Structured Output Configuration
# ===========================================================================

class TestStructuredOutputConfig:
    """Test LLMJudge response_format configuration."""

    def test_default_response_format(self):
        judge = LLMJudge(lm=DummyLM())
        assert judge.response_format is not None
        assert judge.response_format["type"] == "json_schema"
        schema = judge.response_format["json_schema"]["schema"]
        assert "score" in schema["properties"]
        assert "reasoning" in schema["properties"]

    def test_disabled_response_format(self):
        judge = LLMJudge(lm=DummyLM(), response_format=None)
        assert judge.response_format is None

    def test_custom_response_format(self):
        custom = {"type": "json_object"}
        judge = LLMJudge(lm=DummyLM(), response_format=custom)
        assert judge.response_format == {"type": "json_object"}

    def test_score_response_format_schema_is_valid(self):
        """Verify the default schema matches OpenAI structured output requirements."""
        fmt = LLMJudge.SCORE_RESPONSE_FORMAT
        assert fmt["type"] == "json_schema"
        js = fmt["json_schema"]
        assert js["name"] == "judge_score"
        assert js["strict"] is True
        schema = js["schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {"score", "reasoning"}


# ===========================================================================
# 3. Structured Output Forwarding (LLMJudge → Orchestrator → LM)
# ===========================================================================

class TestStructuredOutputForwarding:
    """Test that response_format flows from LLMJudge through orchestrator to LM."""

    @pytest.mark.asyncio
    async def test_default_response_format_forwarded(self):
        lm = MockLM()
        judge = LLMJudge(lm=lm)
        messages = [ChatMessage(role="user", content="hello")]

        await judge.ascore(messages)

        assert len(lm.calls) == 1
        assert lm.calls[0]["response_format"] == LLMJudge.SCORE_RESPONSE_FORMAT

    @pytest.mark.asyncio
    async def test_disabled_response_format_not_forwarded(self):
        lm = MockLM()
        judge = LLMJudge(lm=lm, response_format=None)
        messages = [ChatMessage(role="user", content="hello")]

        await judge.ascore(messages)

        assert len(lm.calls) == 1
        # response_format should be None when disabled
        assert lm.calls[0].get("response_format") is None

    @pytest.mark.asyncio
    async def test_kwargs_override_default_response_format(self):
        lm = MockLM()
        judge = LLMJudge(lm=lm)
        messages = [ChatMessage(role="user", content="hello")]
        custom = {"type": "json_object"}

        await judge.ascore(messages, response_format=custom)

        assert lm.calls[0]["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_batch_scoring_forwards_response_format(self):
        lm = MockLM()
        judge = LLMJudge(lm=lm)
        batch = [
            [ChatMessage(role="user", content="conv1")],
            [ChatMessage(role="user", content="conv2")],
        ]

        scores = await judge.ascore(batch)

        assert len(scores) == 2
        for call in lm.calls:
            assert call["response_format"] == LLMJudge.SCORE_RESPONSE_FORMAT

    @pytest.mark.asyncio
    async def test_score_parsed_from_structured_output(self):
        lm = MockLM(response_content='{"score": 9.5, "reasoning": "excellent"}')
        judge = LLMJudge(lm=lm)
        messages = [ChatMessage(role="user", content="hello")]

        score = await judge.ascore(messages)
        assert score == 9.5


# ===========================================================================
# 4. Orchestrator response_format Forwarding
# ===========================================================================

class TestOrchestratorResponseFormat:
    """Test that LMOrchestrator threads response_format to the LM."""

    @pytest.mark.asyncio
    async def test_response_format_forwarded(self):
        lm = MockLM()
        orch = LMOrchestrator(max_concurrency=4)
        batch = [[ChatMessage(role="user", content="msg")]]
        fmt = {"type": "json_object"}

        await orch.agenerate(lm, batch, response_format=fmt)

        assert lm.calls[0]["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_response_format_none_by_default(self):
        lm = MockLM()
        orch = LMOrchestrator(max_concurrency=4)
        batch = [[ChatMessage(role="user", content="msg")]]

        await orch.agenerate(lm, batch)

        assert lm.calls[0].get("response_format") is None

    @pytest.mark.asyncio
    async def test_response_format_with_other_params(self):
        lm = MockLM()
        orch = LMOrchestrator(max_concurrency=4)
        batch = [[ChatMessage(role="user", content="msg")]]
        fmt = {"type": "json_schema", "json_schema": {"name": "test", "strict": True, "schema": {}}}

        await orch.agenerate(
            lm, batch,
            temperature=0.0,
            max_tokens=100,
            response_format=fmt,
        )

        call = lm.calls[0]
        assert call["temperature"] == 0.0
        assert call["max_tokens"] == 100
        assert call["response_format"] == fmt

    @pytest.mark.asyncio
    async def test_response_format_same_across_batch(self):
        lm = MockLM()
        orch = LMOrchestrator(max_concurrency=4)
        batch = [
            [ChatMessage(role="user", content=f"msg-{i}")]
            for i in range(3)
        ]
        fmt = {"type": "json_object"}

        await orch.agenerate(lm, batch, response_format=fmt)

        assert len(lm.calls) == 3
        for call in lm.calls:
            assert call["response_format"] == fmt

    def test_sync_generate_forwards_response_format(self):
        lm = MockLM()
        orch = LMOrchestrator(max_concurrency=4)
        batch = [[ChatMessage(role="user", content="msg")]]
        fmt = {"type": "json_object"}

        orch.generate(lm, batch, response_format=fmt)

        assert lm.calls[0]["response_format"] == {"type": "json_object"}


# ===========================================================================
# 5. _prepare_request_data
# ===========================================================================

class TestPrepareRequestData:
    """Test that response_format is set in the HTTP request payload."""

    def _make_lm(self):
        from its_hub.core.lms.openai_lm import OpenAICompatibleLanguageModel
        return OpenAICompatibleLanguageModel(
            endpoint="http://localhost:8000/v1",
            api_key="test",
            model_name="test-model",
        )

    def test_response_format_included(self):
        lm = self._make_lm()
        fmt = {"type": "json_object"}
        data = lm._prepare_request_data(
            [ChatMessage(role="user", content="hi")],
            response_format=fmt,
        )
        assert data["response_format"] == {"type": "json_object"}

    def test_response_format_not_included_when_none(self):
        lm = self._make_lm()
        data = lm._prepare_request_data(
            [ChatMessage(role="user", content="hi")],
        )
        assert "response_format" not in data

    def test_response_format_json_schema(self):
        lm = self._make_lm()
        fmt = LLMJudge.SCORE_RESPONSE_FORMAT
        data = lm._prepare_request_data(
            [ChatMessage(role="user", content="hi")],
            response_format=fmt,
        )
        assert data["response_format"]["type"] == "json_schema"
        assert data["response_format"]["json_schema"]["strict"] is True
