"""Tests for self-certainty particle weights (generator-derived, no external PRM).

These verify the mechanism added for the audio inference-time-scaling experiment:
PF/EPF can weight & resample particles using the *generator* model's own token
logprobs/entropy instead of a separate process reward model. All offline (no GPU).
"""

import math

import pytest

from its_hub import StepGeneration
from its_hub.core.algorithms.particle_filtering import (
    EntropicParticleFiltering,
    ParticleFiltering,
    ParticleFilteringResult,
    _inv_sigmoid,
)
from its_hub.core.utils import summarize_step_logprobs
from tests.mocks.language_models import LogprobMockLM

# --------------------------------------------------------------------------- #
# 1. The logprob-summary helper                                               #
# --------------------------------------------------------------------------- #


def test_summarize_step_logprobs_mean_and_entropy():
    logprobs = {
        "content": [
            {"logprob": -0.2, "top_logprobs": [{"logprob": -0.2}, {"logprob": -1.2}]},
            {"logprob": -0.4, "top_logprobs": [{"logprob": -0.4}, {"logprob": -1.4}]},
        ]
    }
    s = summarize_step_logprobs(logprobs)
    assert s["num_tokens"] == 2
    assert s["mean_logprob"] == pytest.approx(-0.3)
    # entropy ≈ mean over tokens of -Σ p·logp over the returned top-k
    h0 = -(math.exp(-0.2) * -0.2 + math.exp(-1.2) * -1.2)
    h1 = -(math.exp(-0.4) * -0.4 + math.exp(-1.4) * -1.4)
    assert s["entropy"] == pytest.approx((h0 + h1) / 2)


def test_summarize_step_logprobs_missing():
    assert summarize_step_logprobs(None) == {
        "mean_logprob": 0.0,
        "entropy": None,
        "num_tokens": 0,
    }
    # no top_logprobs → entropy is None, mean still computed
    s = summarize_step_logprobs({"content": [{"logprob": -0.5}]})
    assert s["mean_logprob"] == pytest.approx(-0.5)
    assert s["entropy"] is None


# --------------------------------------------------------------------------- #
# 2. The weight transform (signal x style)                                    #
# --------------------------------------------------------------------------- #


def _pf(signal="mean_logprob", style="logit"):
    return ParticleFiltering(
        sg=StepGeneration(step_token="\n", max_steps=3),
        self_certainty_signal=signal,
        self_certainty_style=style,
    )


def test_self_certainty_logweight_mean_logprob():
    raw = _pf("mean_logprob", "raw")._self_certainty_logweight({"mean_logprob": -0.3})
    assert raw == pytest.approx(-0.3)

    logit = _pf("mean_logprob", "logit")._self_certainty_logweight(
        {"mean_logprob": -0.3}
    )
    assert logit == pytest.approx(_inv_sigmoid(math.exp(-0.3)))


def test_self_certainty_logweight_entropy():
    raw = _pf("entropy", "raw")._self_certainty_logweight(
        {"mean_logprob": -0.3, "entropy": 0.5}
    )
    assert raw == pytest.approx(-0.5)
    # entropy missing → falls back to mean_logprob
    fb = _pf("entropy", "raw")._self_certainty_logweight(
        {"mean_logprob": -0.7, "entropy": None}
    )
    assert fb == pytest.approx(-0.7)


# --------------------------------------------------------------------------- #
# 3. End-to-end PF / EPF with self-certainty weights                          #
# --------------------------------------------------------------------------- #


def test_particle_filtering_self_certainty_runs():
    lm = LogprobMockLM()
    pf = _pf("mean_logprob", "logit")
    result = pf.infer(lm, "prompt", budget=4, return_response_only=False)
    assert isinstance(result, ParticleFilteringResult)
    assert len(result.responses) == 4
    assert len(result.log_weights_lst) == 4
    assert all(math.isfinite(float(w)) for w in result.log_weights_lst)
    assert isinstance(result.selected_index, int)
    assert lm.saw_logprobs is True  # the generator was asked for logprobs


def test_epf_self_certainty_entropy_runs():
    lm = LogprobMockLM()
    epf = EntropicParticleFiltering(
        sg=StepGeneration(step_token="\n", max_steps=3),
        self_certainty_signal="entropy",
    )
    result = epf.infer(lm, "prompt", budget=4, return_response_only=False)
    assert isinstance(result, ParticleFilteringResult)
    assert len(result.responses) == 4
    assert lm.saw_logprobs is True
    assert lm.saw_top_logprobs == 20  # entropy auto-requests top_logprobs


def test_invalid_self_certainty_options_raise():
    sg = StepGeneration(step_token="\n", max_steps=3)
    with pytest.raises(ValueError, match="self_certainty_signal"):
        ParticleFiltering(sg=sg, self_certainty_signal="bogus")
    with pytest.raises(ValueError, match="self_certainty_style"):
        ParticleFiltering(sg=sg, self_certainty_style="bogus")
