"""Tests for reporting modules (PPTX builder + ZIP packager).

PPTX tests are skipped if python-pptx is not installed.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from treemmm.core.attribution.decomposer import Attribution
from treemmm.core.models.base import FoldResult, ModelResult
from treemmm.core.reporting.zip_packager import package_zip

try:
    from pptx import Presentation
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False


def _make_attribution() -> Attribution:
    """Create a minimal Attribution for testing."""
    n, p = 50, 3
    rng = np.random.default_rng(42)
    values = rng.normal(0, 1, (n, p))
    base = rng.normal(5, 0.5, n)
    preds = base + values.sum(axis=1)
    return Attribution(
        values=values,
        base_values=base,
        predictions=preds,
        feature_names=["x1", "x2", "x3"],
        link="identity",
    )


def _make_model_result() -> ModelResult:
    """Create a minimal ModelResult for testing."""
    rng = np.random.default_rng(42)
    folds = []
    for i in range(3):
        n = 20
        y_true = rng.normal(10, 2, n)
        y_pred = y_true + rng.normal(0, 0.5, n)
        folds.append(FoldResult(
            fold_idx=i,
            train_periods=list(range(1, 7)),
            test_periods=list(range(7, 10)),
            y_true=y_true,
            y_pred=y_pred,
        ))
    mr = ModelResult(model_name="TestModel", fold_results=folds)
    mr.compute_aggregate_metrics()
    return mr


class TestZipPackager:
    """Tests for ZIP bundling."""

    def test_package_empty_dir(self):
        """Empty dir returns the path but no file is created (nothing to bundle)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = package_zip(tmpdir)
            # No files to bundle, so ZIP is not created
            assert not Path(zip_path).exists()

    def test_package_with_csvs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some CSV files
            for name in ["test1.csv", "test2.csv"]:
                pd.DataFrame({"a": [1, 2]}).to_csv(
                    Path(tmpdir) / name, index=False
                )
            zip_path = package_zip(tmpdir)
            assert Path(zip_path).exists()

            import zipfile
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                assert "test1.csv" in names
                assert "test2.csv" in names

    def test_package_custom_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pd.DataFrame({"a": [1]}).to_csv(
                Path(tmpdir) / "data.csv", index=False
            )
            custom_path = Path(tmpdir) / "custom_bundle.zip"
            result = package_zip(tmpdir, zip_path=custom_path)
            assert result == custom_path
            assert custom_path.exists()


@pytest.mark.skipif(not HAS_PPTX, reason="python-pptx not installed")
class TestPPTXBuilder:
    """Tests for PowerPoint report generation."""

    def test_build_pptx_returns_bytes(self):
        from treemmm.core.reporting.pptx_builder import build_pptx

        mr = _make_model_result()
        attr = _make_attribution()
        result = build_pptx(mr, attr, title="Test Report")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_build_pptx_with_time(self):
        from treemmm.core.reporting.pptx_builder import build_pptx

        mr = _make_model_result()
        attr = _make_attribution()
        time_vals = np.repeat(np.arange(1, 6), 10)
        result = build_pptx(mr, attr, time_values=time_vals, title="Test")
        assert isinstance(result, bytes)

    def test_build_pptx_to_file(self):
        from treemmm.core.reporting.pptx_builder import build_pptx

        mr = _make_model_result()
        attr = _make_attribution()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.pptx"
            result = build_pptx(mr, attr, output_path=path)
            assert result is None
            assert path.exists()
            assert path.stat().st_size > 0


class TestChartGeneration:
    """Test individual chart generation functions."""

    def test_attribution_bar(self):
        from treemmm.core.reporting.pptx_builder import _make_attribution_bar
        attr = _make_attribution()
        png_bytes = _make_attribution_bar(attr)
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 100

    def test_feature_importance_chart(self):
        from treemmm.core.reporting.pptx_builder import _make_feature_importance
        attr = _make_attribution()
        png_bytes = _make_feature_importance(attr)
        assert isinstance(png_bytes, bytes)

    def test_performance_chart(self):
        from treemmm.core.reporting.pptx_builder import _make_performance_chart
        mr = _make_model_result()
        png_bytes = _make_performance_chart(mr)
        assert isinstance(png_bytes, bytes)
