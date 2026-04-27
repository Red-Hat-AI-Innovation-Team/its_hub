"""Tests for trajectory aggregators."""

import math
import os
import tempfile

import pytest

from its_hub.aggregators import HardcodedAggregator, LearnedGBDTAggregator, LearnedMLPAggregator
from its_hub.algorithms.particle_gibbs import ParticleFiltering
from its_hub.base import AbstractTrajectoryAggregator
from its_hub.lms import StepGeneration

from tests.mocks.language_models import StepMockLanguageModel
from tests.mocks.reward_models import MockProcessRewardModel


class TestHardcodedAggregatorProd:
    def test_known_values(self):
        agg = HardcodedAggregator("prod")
        result = agg.aggregate([0.9, 0.8, 0.7])
        assert result == pytest.approx(0.9 * 0.8 * 0.7)

    def test_single_step(self):
        agg = HardcodedAggregator("prod")
        assert agg.aggregate([0.5]) == pytest.approx(0.5)

    def test_empty(self):
        agg = HardcodedAggregator("prod")
        assert agg.aggregate([]) == 0.0

    def test_all_ones(self):
        agg = HardcodedAggregator("prod")
        assert agg.aggregate([1.0, 1.0, 1.0]) == pytest.approx(1.0)


class TestHardcodedAggregatorMin:
    def test_known_values(self):
        agg = HardcodedAggregator("min")
        assert agg.aggregate([0.9, 0.8, 0.7]) == pytest.approx(0.7)

    def test_single_step(self):
        agg = HardcodedAggregator("min")
        assert agg.aggregate([0.4]) == pytest.approx(0.4)

    def test_min_at_start(self):
        agg = HardcodedAggregator("min")
        assert agg.aggregate([0.1, 0.9, 0.8]) == pytest.approx(0.1)


class TestHardcodedAggregatorMean:
    def test_known_values(self):
        agg = HardcodedAggregator("mean")
        result = agg.aggregate([0.9, 0.8, 0.7])
        assert result == pytest.approx((0.9 + 0.8 + 0.7) / 3)

    def test_single_step(self):
        agg = HardcodedAggregator("mean")
        assert agg.aggregate([0.6]) == pytest.approx(0.6)

    def test_uniform(self):
        agg = HardcodedAggregator("mean")
        assert agg.aggregate([0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


class TestHardcodedAggregatorInvalid:
    def test_unknown_reduction_raises(self):
        agg = HardcodedAggregator("prod")
        agg.reduction = "invalid"
        with pytest.raises(ValueError, match="Unknown reduction"):
            agg.aggregate([0.5, 0.6])


class TestAbstractAggregatorAsync:
    @pytest.mark.asyncio
    async def test_aaggregate_delegates_to_sync(self):
        agg = HardcodedAggregator("prod")
        sync_result = agg.aggregate([0.9, 0.8, 0.7])
        async_result = await agg.aaggregate([0.9, 0.8, 0.7])
        assert async_result == pytest.approx(sync_result)

    @pytest.mark.asyncio
    async def test_aaggregate_all_reductions(self):
        scores = [0.6, 0.4, 0.9]
        for reduction in ("prod", "min", "mean"):
            agg = HardcodedAggregator(reduction)
            assert await agg.aaggregate(scores) == pytest.approx(agg.aggregate(scores))


class TestLearnedMLPAggregator:
    @pytest.fixture
    def dummy_checkpoint(self, tmp_path):
        """Create a minimal valid MLP checkpoint."""
        torch = pytest.importorskip("torch")
        import torch.nn as nn

        hidden_width = 8
        net = nn.Sequential(
            nn.Linear(10, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 1),
            nn.Sigmoid(),
        )
        checkpoint_path = str(tmp_path / "dummy.pt")
        torch.save({"state_dict": net.state_dict(), "hidden_width": hidden_width}, checkpoint_path)
        return checkpoint_path

    def test_loads_and_runs_forward(self, dummy_checkpoint):
        agg = LearnedMLPAggregator(dummy_checkpoint)
        result = agg.aggregate([0.9, 0.8, 0.7])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_empty_scores(self, dummy_checkpoint):
        agg = LearnedMLPAggregator(dummy_checkpoint)
        result = agg.aggregate([])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_single_step(self, dummy_checkpoint):
        agg = LearnedMLPAggregator(dummy_checkpoint)
        result = agg.aggregate([0.5])
        assert isinstance(result, float)

    def test_is_abstract_aggregator(self, dummy_checkpoint):
        agg = LearnedMLPAggregator(dummy_checkpoint)
        assert isinstance(agg, AbstractTrajectoryAggregator)

    @pytest.mark.asyncio
    async def test_aaggregate_delegates_to_sync(self, dummy_checkpoint):
        agg = LearnedMLPAggregator(dummy_checkpoint)
        scores = [0.9, 0.7, 0.8]
        assert await agg.aaggregate(scores) == pytest.approx(agg.aggregate(scores))


class TestLearnedGBDTAggregator:
    @pytest.fixture
    def dummy_checkpoint(self, tmp_path):
        """Create a minimal valid GBDT checkpoint."""
        joblib = pytest.importorskip("joblib")
        sklearn = pytest.importorskip("sklearn.ensemble")

        from sklearn.ensemble import GradientBoostingClassifier
        import numpy as np

        clf = GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=0)
        X = np.random.default_rng(0).random((20, 10)).astype(np.float32)
        y = (X[:, 0] > 0.5).astype(int)
        clf.fit(X, y)

        checkpoint_path = str(tmp_path / "dummy_gbdt.pkl")
        joblib.dump(clf, checkpoint_path)
        return checkpoint_path

    def test_loads_and_runs_forward(self, dummy_checkpoint):
        agg = LearnedGBDTAggregator(dummy_checkpoint)
        result = agg.aggregate([0.9, 0.8, 0.7])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_empty_scores(self, dummy_checkpoint):
        agg = LearnedGBDTAggregator(dummy_checkpoint)
        result = agg.aggregate([])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_single_step(self, dummy_checkpoint):
        agg = LearnedGBDTAggregator(dummy_checkpoint)
        result = agg.aggregate([0.5])
        assert isinstance(result, float)

    def test_is_abstract_aggregator(self, dummy_checkpoint):
        agg = LearnedGBDTAggregator(dummy_checkpoint)
        assert isinstance(agg, AbstractTrajectoryAggregator)

    @pytest.mark.asyncio
    async def test_aaggregate_delegates_to_sync(self, dummy_checkpoint):
        agg = LearnedGBDTAggregator(dummy_checkpoint)
        scores = [0.9, 0.7, 0.8]
        assert await agg.aaggregate(scores) == pytest.approx(agg.aggregate(scores))


class TestParticleFilteringAggregatorIntegration:
    def test_accepts_aggregator_parameter(self):
        mock_prm = MockProcessRewardModel([0.5, 0.8, 0.3, 0.9])
        sg = StepGeneration(step_token="\n", max_steps=1)
        agg = HardcodedAggregator("min")
        pf = ParticleFiltering(sg=sg, prm=mock_prm, aggregator=agg)
        assert pf.aggregator is agg

    def test_defaults_to_hardcoded_prod(self):
        mock_prm = MockProcessRewardModel([0.5])
        sg = StepGeneration(step_token="\n", max_steps=1)
        pf = ParticleFiltering(sg=sg, prm=mock_prm)
        assert isinstance(pf.aggregator, HardcodedAggregator)
        assert pf.aggregator.reduction == "prod"

    def test_uses_aggregator_for_selection(self):
        """Custom aggregator returning constant 0 should still produce a valid result."""
        class ZeroAggregator(AbstractTrajectoryAggregator):
            def aggregate(self, step_scores):
                return 0.0

        mock_lm = StepMockLanguageModel(["step1", "step2", "step3", "step4"])
        mock_prm = MockProcessRewardModel([0.9, 0.1, 0.8, 0.2])
        sg = StepGeneration(step_token="\n", max_steps=1)
        pf = ParticleFiltering(sg=sg, prm=mock_prm, aggregator=ZeroAggregator())
        result = pf.infer(mock_lm, "test prompt", budget=2, return_response_only=True)
        assert isinstance(result, dict)

    def test_different_aggregators_produce_valid_results(self):
        """Both prod and min aggregators should return valid dicts."""
        sg = StepGeneration(step_token="\n", max_steps=1)

        pf_prod = ParticleFiltering(
            sg=sg, prm=MockProcessRewardModel([0.9, 0.1]),
            aggregator=HardcodedAggregator("prod"),
        )
        pf_min = ParticleFiltering(
            sg=sg, prm=MockProcessRewardModel([0.9, 0.1]),
            aggregator=HardcodedAggregator("min"),
        )

        lm = StepMockLanguageModel(["step1", "step2"] * 4)
        result_prod = pf_prod.infer(lm, "test", budget=2, return_response_only=True)
        result_min = pf_min.infer(
            StepMockLanguageModel(["step1", "step2"] * 4), "test",
            budget=2, return_response_only=True,
        )
        assert isinstance(result_prod, dict)
        assert isinstance(result_min, dict)
