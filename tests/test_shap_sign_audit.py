"""Tests for the SHAP sign audit diagnostic."""

import numpy as np
import pytest

from treemmm.core.diagnostics.shap_sign_audit import shap_sign_audit
from treemmm.core.interpret.shap_engine import SHAPResult


class TestShapSignAudit:
    """Tests for the sign audit function."""

    def test_all_positive_shap(self):
        """All positive SHAP values -> consistency near 1.0."""
        result = SHAPResult(
            values=np.array([[1.0, 2.0], [0.5, 1.5], [0.8, 1.2]]),
            expected_value=0.0,
            feature_names=["x1", "x2"],
            link="identity",
        )
        audit = shap_sign_audit(result)
        assert len(audit.variable_reports) == 2
        for r in audit.variable_reports:
            assert r.sign_consistency == pytest.approx(1.0, abs=0.01)
            assert r.dominant_sign == "positive"
            assert r.frac_negative == 0.0

    def test_mixed_sign_shap(self):
        """Mixed signs -> low consistency."""
        vals = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]])
        result = SHAPResult(
            values=vals,
            expected_value=0.0,
            feature_names=["x1", "x2"],
            link="log",
        )
        audit = shap_sign_audit(result)
        for r in audit.variable_reports:
            assert r.sign_consistency < 0.1
            assert r.dominant_sign == "mixed"

    def test_all_negative_shap(self):
        """All negative -> consistency near 1.0, dominant = negative."""
        result = SHAPResult(
            values=np.array([[-2.0], [-1.0], [-3.0]]),
            expected_value=5.0,
            feature_names=["x1"],
            link="identity",
        )
        audit = shap_sign_audit(result)
        r = audit.variable_reports[0]
        assert r.sign_consistency == pytest.approx(1.0, abs=0.01)
        assert r.dominant_sign == "negative"
        assert r.frac_negative == 1.0

    def test_to_dataframe(self):
        result = SHAPResult(
            values=np.array([[1.0, -0.5], [0.5, -1.0]]),
            expected_value=0.0,
            feature_names=["a", "b"],
            link="identity",
        )
        audit = shap_sign_audit(result)
        df = audit.to_dataframe()
        assert len(df) == 2
        assert "variable" in df.columns
        assert "sign_consistency" in df.columns

    def test_summary_string(self):
        result = SHAPResult(
            values=np.array([[1.0, -0.5], [0.5, -1.0]]),
            expected_value=0.0,
            feature_names=["a", "b"],
            link="identity",
        )
        audit = shap_sign_audit(result)
        summary = audit.summary()
        assert "a" in summary
        assert "b" in summary

    def test_zero_shap(self):
        """All zero SHAP -> consistency = 1.0 (no inconsistency)."""
        result = SHAPResult(
            values=np.zeros((5, 2)),
            expected_value=1.0,
            feature_names=["x1", "x2"],
            link="identity",
        )
        audit = shap_sign_audit(result)
        for r in audit.variable_reports:
            assert r.sign_consistency == 1.0
            assert r.frac_zero == 1.0
