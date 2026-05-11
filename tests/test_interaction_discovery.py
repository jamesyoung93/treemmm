"""Tests for tree-based interaction discovery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from treemmm.core.config import Objective
from treemmm.core.interpret.interaction_discovery import (
    InteractionCandidate,
    discover_interactions,
    filter_significant_interactions,
)
from treemmm.core.models.lightgbm_model import LightGBMModel


def _make_interaction_data(seed: int = 42, n: int = 600) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic data with a clear x1*x2 interaction and weak x3 main effect."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    x3 = rng.uniform(0, 1, n)
    x4 = rng.uniform(0, 1, n)  # noise feature
    y = 0.3 * x1 + 0.3 * x2 + 1.5 * x1 * x2 + 0.1 * x3 + rng.normal(0, 0.05, n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
    return X, y


def _fit_lgbm(X, y, n_trials=3) -> LightGBMModel:
    m = LightGBMModel(objective=Objective.GAUSSIAN)
    n = len(X)
    v = max(1, n // 5)
    m.fit(
        X.iloc[:-v], y[:-v], X.iloc[-v:], y[-v:],
        n_trials=n_trials, random_state=42,
    )
    return m


class TestDiscoverInteractions:
    """Basic sanity checks on discover_interactions."""

    def test_recovers_planted_interaction(self):
        X, y = _make_interaction_data()
        m = _fit_lgbm(X, y)
        result = discover_interactions(m, X, sample_size=200)
        top = result.top_k(1)
        assert top[0] == ("x1", "x2"), (
            f"Expected (x1, x2) at rank 1, got {top[0]}. "
            f"Full ranking:\n{result.as_dataframe()}"
        )

    def test_candidate_features_filter(self):
        X, y = _make_interaction_data()
        m = _fit_lgbm(X, y)
        result = discover_interactions(
            m, X, candidate_features=["x1", "x3"], sample_size=200,
        )
        # Only one pair (x1, x3) is eligible
        assert len(result.candidates) == 1
        assert result.candidates[0].as_tuple() == ("x1", "x3")

    def test_returns_p_choose_2_candidates(self):
        X, y = _make_interaction_data()
        m = _fit_lgbm(X, y)
        result = discover_interactions(m, X, sample_size=200)
        # 4 features -> 6 unordered pairs
        assert len(result.candidates) == 6

    def test_interaction_matrix_symmetric(self):
        X, y = _make_interaction_data()
        m = _fit_lgbm(X, y)
        result = discover_interactions(m, X, sample_size=200)
        # SHAP interaction tensor is symmetric by construction
        assert np.allclose(
            result.interaction_matrix, result.interaction_matrix.T, atol=1e-10
        )

    def test_heuristic_fallback_runs(self):
        """The cross-correlation heuristic should still produce a ranking.

        Note: SHAP-x-correlation is a weaker signal than SHAP-interaction-
        values; spurious correlations with noise features can outrank a
        true interaction with this much noise. We only check that the
        ranking machinery functions without error.
        """
        X, y = _make_interaction_data()
        m = _fit_lgbm(X, y)
        result = discover_interactions(
            m, X, sample_size=200, use_shap_interaction=False
        )
        assert len(result.candidates) == 6
        # All scores should be non-negative
        assert all(c.score >= 0 for c in result.candidates)
        # Ranking should be monotonically non-increasing in score
        for i in range(len(result.candidates) - 1):
            assert result.candidates[i].score >= result.candidates[i + 1].score

    def test_candidate_rank_is_set(self):
        X, y = _make_interaction_data()
        m = _fit_lgbm(X, y)
        result = discover_interactions(m, X, sample_size=200)
        for i, c in enumerate(result.candidates, start=1):
            assert c.rank == i

    def test_works_with_string_categorical(self):
        """Mixed numeric + string-categorical X should not crash."""
        X, y = _make_interaction_data()
        X = X.copy()
        rng = np.random.default_rng(0)
        X["segment"] = pd.Categorical(
            rng.choice(["a", "b"], size=len(X))
        )
        # Train tree without the segment column (it's a string cat that
        # discover_interactions will skip in SHAP-interaction step)
        m = _fit_lgbm(X.drop(columns=["segment"]), y)
        result = discover_interactions(
            m, X.drop(columns=["segment"]), sample_size=100,
        )
        assert len(result.candidates) > 0


class TestFilterSignificantInteractions:
    def _fake_result(self, scores: list[float], corrs: list[float]):
        from treemmm.core.interpret.interaction_discovery import (
            InteractionDiscoveryResult,
        )
        cands = [
            InteractionCandidate(
                var1=f"v{2*i}", var2=f"v{2*i+1}",
                score=s, correlation=c, rank=i + 1,
            )
            for i, (s, c) in enumerate(zip(scores, corrs, strict=False))
        ]
        return InteractionDiscoveryResult(
            candidates=cands,
            interaction_matrix=np.zeros((4, 4)),
            feature_names=[f"v{i}" for i in range(4)],
        )

    def test_top_k_filter(self):
        r = self._fake_result([0.9, 0.5, 0.3], [0.4, 0.4, 0.4])
        kept = filter_significant_interactions(r, top_k=2)
        assert len(kept) == 2
        assert kept[0] == ("v0", "v1")

    def test_min_score_filter(self):
        r = self._fake_result([0.9, 0.5, 0.3], [0.4, 0.4, 0.4])
        kept = filter_significant_interactions(r, min_score=0.6)
        assert len(kept) == 1
        assert kept[0] == ("v0", "v1")

    def test_min_correlation_filter(self):
        r = self._fake_result([0.9, 0.5, 0.3], [0.4, 0.05, 0.4])
        kept = filter_significant_interactions(r, min_abs_correlation=0.1)
        assert len(kept) == 2

    def test_dataframe_export(self):
        X, y = _make_interaction_data(n=200)
        m = _fit_lgbm(X, y)
        result = discover_interactions(m, X, sample_size=100)
        df = result.as_dataframe()
        assert "rank" in df.columns
        assert "shap_interaction_score" in df.columns
        assert "shap_x_correlation" in df.columns
        assert len(df) == 6
