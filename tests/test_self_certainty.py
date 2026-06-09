"""Tests for self-certainty particle weights (generator-derived, no external PRM).

These verify the mechanism added for the audio inference-time-scaling experiment:
PF/EPF can weight & resample particles using the *generator* model's own token
logprobs/entropy instead of a separate process reward model. All offline (no GPU).
"""

import math

import pytest

from its_hub import AbstractLanguageModel, StepGeneration
from its_hub.core.algorithms.particle_gibbs import (
    EntropicParticleFiltering,
    ParticleFiltering,
    ParticleFilteringResult,
    WeightSource,
    _inv_sigmoid,
)
from its_hub.core.utils import summarize_step_logprobs


class LogprobMockLM(AbstractLanguageModel):
    """Mock LM that emits OpenAI-style `_logprobs` when `logprobs=True`.

    Each generated step gets a (cycled) target mean logprob so different
    particles end up with different self-certainty weights.
    """

    def __init__(self, mean_logprobs=(-0.1, -0.5, -1.0, -0.2), n_tokens=2):
        self.mean_logprobs = list(mean_logprobs)
        self.n_tokens = n_tokens
        self.call_count = 0
        self.saw_logprobs = False
        self.saw_top_logprobs = None

    def _make_message(self, idx, want_logprobs, want_top):
        base = self.mean_logprobs[idx % len(self.mean_logprobs)]
        msg = {"role": "assistant", "content": f"step{idx}"}
        if want_logprobs:
            toks = []
            for t in range(self.n_tokens):
                entry = {"token": f"tok{t}", "logprob": base}
                if want_top is not None:
                    entry["top_logprobs"] = [
                        {"token": f"tok{t}", "logprob": base},
                        {"token": "other", "logprob": base - 1.0},
                    ]
                toks.append(entry)
            msg["_logprobs"] = {"content": toks}
        return msg

    async def agenerate(
        self,
        messages,
        stop=None,
        max_tokens=None,
        temperature=None,
        include_stop_str_in_output=None,
        tools=None,
        tool_choice=None,
        response_format=None,
        logprobs=False,
        top_logprobs=None,
    ):
        self.saw_logprobs = self.saw_logprobs or bool(logprobs)
        if top_logprobs is not None:
            self.saw_top_logprobs = top_logprobs
        is_batch = (
            isinstance(messages, list)
            and len(messages) > 0
            and isinstance(messages[0], list)
        )
        if is_batch:
            out = []
            for _ in messages:
                out.append(self._make_message(self.call_count, logprobs, top_logprobs))
                self.call_count += 1
            return out
        msg = self._make_message(self.call_count, logprobs, top_logprobs)
        self.call_count += 1
        return msg

    async def agenerate_single(self, messages, **kwargs):
        return await self.agenerate(messages, **kwargs)


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
        weight_source="self_certainty",
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
        weight_source="self_certainty",
        self_certainty_signal="entropy",
    )
    result = epf.infer(lm, "prompt", budget=4, return_response_only=False)
    assert isinstance(result, ParticleFilteringResult)
    assert len(result.responses) == 4
    assert lm.saw_logprobs is True
    assert lm.saw_top_logprobs == 20  # entropy auto-requests top_logprobs


def test_weight_source_enum_and_string_equivalent():
    a = _pf()
    assert a.weight_source == WeightSource.SELF_CERTAINTY


def test_prm_required_when_weight_source_is_prm():
    with pytest.raises(ValueError, match="prm must be provided"):
        ParticleFiltering(sg=StepGeneration(step_token="\n", max_steps=3))


def test_invalid_self_certainty_options_raise():
    sg = StepGeneration(step_token="\n", max_steps=3)
    with pytest.raises(ValueError, match="self_certainty_signal"):
        ParticleFiltering(sg=sg, weight_source="self_certainty", self_certainty_signal="bogus")
    with pytest.raises(ValueError, match="self_certainty_style"):
        ParticleFiltering(sg=sg, weight_source="self_certainty", self_certainty_style="bogus")
