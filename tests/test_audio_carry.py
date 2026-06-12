"""Tests for the audio step-path carry: structured (e.g. audio) user content must
survive through PF/EPF down to the model, while the plain-text path stays unchanged.

All offline (no GPU): a mock LM records the exact messages it receives so we can
assert the audio part is delivered verbatim to every particle.
"""

import copy

from its_hub import AbstractLanguageModel, StepGeneration
from its_hub.api.types import ChatMessage, ChatMessages
from its_hub.core.algorithms.particle_filtering import ParticleFiltering


def audio_user_msg(text: str, n_audio: int = 1) -> ChatMessage:
    """A user turn with `n_audio` input_audio parts followed by a text part."""
    content = [
        {"type": "input_audio", "input_audio": {"data": f"BASE64_{i}", "format": "wav"}}
        for i in range(n_audio)
    ]
    content.append({"type": "text", "text": text})
    return ChatMessage(role="user", content=content)


class AudioEchoMockLM(AbstractLanguageModel):
    """Records every conversation it receives; emits a step + fake logprobs."""

    def __init__(self, mean_logprob: float = -0.3):
        self.mean_logprob = mean_logprob
        self.received: list[list[ChatMessage]] = []
        self.n = 0

    def _msg(self, want_lp: bool, want_top: int | None) -> dict:
        m = {"role": "assistant", "content": f"step{self.n}"}
        self.n += 1
        if want_lp:
            tok = {"token": "t", "logprob": self.mean_logprob}
            if want_top is not None:
                tok["top_logprobs"] = [
                    {"logprob": self.mean_logprob},
                    {"logprob": self.mean_logprob - 1.0},
                ]
            m["_logprobs"] = {"content": [tok, dict(tok)]}
        return m

    async def agenerate(self, messages, logprobs=False, top_logprobs=None, **kwargs):
        is_batch = (
            isinstance(messages, list)
            and len(messages) > 0
            and isinstance(messages[0], list)
        )
        if is_batch:
            for conv in messages:
                self.received.append(copy.deepcopy(conv))
            return [self._msg(logprobs, top_logprobs) for _ in messages]
        self.received.append(copy.deepcopy(messages))
        return self._msg(logprobs, top_logprobs)

    async def agenerate_single(self, messages, **kwargs):
        return await self.agenerate(messages, **kwargs)


def _run_pf(lm, messages, budget=3, max_steps=2):
    sg = StepGeneration(step_token="\n", max_steps=max_steps)
    pf = ParticleFiltering(sg=sg)
    return pf.infer(lm, messages, budget=budget, return_response_only=False)


# --------------------------------------------------------------------------- #
# Carry: audio reaches the model for every particle                           #
# --------------------------------------------------------------------------- #


def test_audio_survives_to_model_every_particle():
    lm = AudioEchoMockLM()
    _run_pf(lm, [audio_user_msg("What is playing? A. piano B. violin")], budget=3)
    assert lm.received, "model was never called"
    for conv in lm.received:
        user = conv[0]
        assert user.role == "user"
        assert isinstance(user.content, list)
        audio_parts = [p for p in user.content if p.get("type") == "input_audio"]
        assert len(audio_parts) == 1
        assert audio_parts[0]["input_audio"]["data"] == "BASE64_0"  # verbatim
        assert any(p.get("type") == "text" for p in user.content)  # text preserved


def test_multi_audio_order_preserved():
    lm = AudioEchoMockLM()
    _run_pf(lm, [audio_user_msg("q", n_audio=3)], budget=2, max_steps=1)
    assert lm.received
    for conv in lm.received:
        audio = [p for p in conv[0].content if p.get("type") == "input_audio"]
        assert [a["input_audio"]["data"] for a in audio] == [
            "BASE64_0",
            "BASE64_1",
            "BASE64_2",
        ]


def test_first_step_has_no_assistant_turn_then_continuation():
    lm = AudioEchoMockLM()
    _run_pf(lm, [audio_user_msg("q")], budget=2, max_steps=2)
    lens = [len(c) for c in lm.received]
    assert 1 in lens  # step 1: user only (no steps yet)
    assert 2 in lens  # step 2: user + assistant continuation
    for c in lm.received:
        if len(c) == 2:
            assert c[-1].role == "assistant"
            assert isinstance(c[-1].content, str)  # reasoning steps are text


def test_base_user_turn_identical_across_particles():
    lm = AudioEchoMockLM()
    _run_pf(lm, [audio_user_msg("q")], budget=4, max_steps=1)
    users = [c[0].to_dict() for c in lm.received]
    assert len(users) >= 4
    assert all(u == users[0] for u in users)  # same audio prompt for every particle


# --------------------------------------------------------------------------- #
# Backward compatibility: the plain-text path is unchanged                    #
# --------------------------------------------------------------------------- #


def test_string_prompt_stays_text_only():
    lm = AudioEchoMockLM()
    _run_pf(lm, "plain text question", budget=2, max_steps=1)
    assert lm.received
    for conv in lm.received:
        user = conv[0]
        assert user.role == "user"
        assert isinstance(user.content, str)  # NOT a list — legacy path untouched
        assert "plain text question" in user.content


# --------------------------------------------------------------------------- #
# types.py helpers                                                            #
# --------------------------------------------------------------------------- #


def test_extract_text_content_tolerates_audio():
    m = audio_user_msg("hello there")
    assert m.extract_text_content() == "hello there"  # must not raise


def test_extract_text_content_unknown_type_does_not_raise():
    m = ChatMessage(role="user", content=[{"type": "video_url", "video_url": {}}])
    assert m.extract_text_content() == ""  # unknown type skipped, no ValueError


def test_has_nontext_content():
    assert ChatMessages.from_prompt_or_messages([audio_user_msg("q")]).has_nontext_content()
    assert not ChatMessages.from_prompt_or_messages("just text").has_nontext_content()
    assert not ChatMessages.from_prompt_or_messages(
        [ChatMessage(role="user", content="hi")]
    ).has_nontext_content()
    assert not ChatMessages.from_prompt_or_messages(
        [ChatMessage(role="user", content=[{"type": "text", "text": "x"}])]
    ).has_nontext_content()


def test_base_user_messages_string_case():
    cm = ChatMessages.from_prompt_or_messages("just text")
    assert cm.base_user_messages() == [ChatMessage(role="user", content="just text")]
