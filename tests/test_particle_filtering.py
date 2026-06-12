"""Tests for the ParticleFiltering / EntropicParticleFiltering algorithms."""

import pytest

from its_hub import (
    EntropicParticleFiltering,
    ParticleFiltering,
    ParticleFilteringResult,
    StepGeneration,
)
from its_hub.api.types import ChatMessage
from its_hub.core.algorithms.particle_filtering import (
    Particle,
    ResamplingMethod,
    SelectionMethod,
)
from tests.mocks.language_models import LogprobMockLM


def _pf(**kwargs) -> ParticleFiltering:
    return ParticleFiltering(
        sg=StepGeneration(step_token="\n", max_steps=3), **kwargs
    )


# --------------------------------------------------------------------------- #
# Data structures                                                              #
# --------------------------------------------------------------------------- #


def test_particle_deepcopy():
    p = Particle(steps=["a", "b"], is_stopped=False, partial_log_weights=[0.1, 0.2])
    q = p.deepcopy()
    q.steps.append("c")
    q.partial_log_weights.append(0.3)
    q.is_stopped = True
    assert p.steps == ["a", "b"]
    assert p.partial_log_weights == [0.1, 0.2]
    assert p.is_stopped is False


def test_particle_log_weight_property():
    assert Particle(steps=[], is_stopped=False, partial_log_weights=[]).log_weight == 0.0
    assert (
        Particle(steps=["a"], is_stopped=False, partial_log_weights=[0.4, -1.5]).log_weight
        == -1.5
    )


# --------------------------------------------------------------------------- #
# ParticleFiltering                                                            #
# --------------------------------------------------------------------------- #


def test_particle_filtering_infer_returns_response_dict():
    lm = LogprobMockLM()
    result = _pf().infer(lm, "prompt", budget=4)
    assert isinstance(result, dict)
    assert result["role"] == "assistant"
    assert isinstance(result["content"], str)


def test_particle_filtering_full_result():
    lm = LogprobMockLM()
    result = _pf().infer(lm, "prompt", budget=4, return_response_only=False)
    assert isinstance(result, ParticleFilteringResult)
    assert len(result.responses) == 4
    assert len(result.log_weights_lst) == 4
    assert len(result.steps_used_lst) == 4
    assert 0 <= result.selected_index < 4
    assert result.the_one is result.responses[result.selected_index]


@pytest.mark.asyncio
async def test_particle_filtering_ainfer():
    lm = LogprobMockLM()
    result = await _pf().ainfer(lm, "prompt", budget=2, return_response_only=False)
    assert isinstance(result, ParticleFilteringResult)
    assert len(result.responses) == 2


def test_particle_filtering_with_chat_messages():
    lm = LogprobMockLM()
    messages = [
        ChatMessage(role="user", content="What is 2+2?"),
    ]
    result = _pf().infer(lm, messages, budget=2)
    assert isinstance(result, dict)


@pytest.mark.parametrize("selection", ["argmax", SelectionMethod.SAMPLE])
def test_particle_filtering_selection_methods(selection):
    lm = LogprobMockLM()
    result = _pf(final_response_selection=selection).infer(
        lm, "prompt", budget=4, return_response_only=False
    )
    assert 0 <= result.selected_index < 4


def test_particle_filtering_invalid_budget():
    lm = LogprobMockLM()
    with pytest.raises(AssertionError, match="budget"):
        _pf().infer(lm, "prompt", budget=0)


# --------------------------------------------------------------------------- #
# Resampling configuration                                                     #
# --------------------------------------------------------------------------- #


def test_resampling_method_is_honored(monkeypatch):
    """Regression: resampling_method used to be silently dropped by
    ParticleFiltering.__init__ (always multinomial)."""
    pf = _pf(resampling_method="systematic")
    assert pf.resampling_method == ResamplingMethod.SYSTEMATIC

    calls = {"systematic": 0}
    original = ParticleFiltering._resampling_systematic

    def spy(self, particles, probabilities, num_particles):
        calls["systematic"] += 1
        return original(self, particles, probabilities, num_particles)

    monkeypatch.setattr(ParticleFiltering, "_resampling_systematic", spy)
    pf.infer(LogprobMockLM(), "prompt", budget=4)
    assert calls["systematic"] > 0


def test_epf_default_resampling_is_systematic():
    epf = EntropicParticleFiltering(sg=StepGeneration(step_token="\n", max_steps=3))
    assert epf.resampling_method == ResamplingMethod.SYSTEMATIC
    pf = _pf()
    assert pf.resampling_method == ResamplingMethod.MULTINOMIAL


# --------------------------------------------------------------------------- #
# Propagation                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_apropagate_skips_stopped_particles():
    lm = LogprobMockLM()
    pf = _pf()
    active = Particle(steps=[], is_stopped=False, partial_log_weights=[])
    stopped = Particle(
        steps=["done"], is_stopped=True, partial_log_weights=[1.23]
    )

    particles = await pf._apropagate(lm, [active, stopped], "prompt")

    # the stopped particle is untouched
    assert particles[1].steps == ["done"]
    assert particles[1].partial_log_weights == [1.23]
    assert particles[1].is_stopped is True
    # the active particle grew by one step and one weight
    assert len(particles[0].steps) == 1
    assert len(particles[0].partial_log_weights) == 1


# --------------------------------------------------------------------------- #
# Tools pass-through                                                           #
# --------------------------------------------------------------------------- #


def test_tools_and_tool_choice_pass_through():
    lm = LogprobMockLM()
    tools = [{"type": "function", "function": {"name": "calculate"}}]
    _pf().infer(lm, "prompt", budget=2, tools=tools, tool_choice="auto")
    assert lm.saw_tools == tools
    assert lm.saw_tool_choice == "auto"
