"""Tests for the CLI runner."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from treemmm.ui.cli_runner import build_parser, main


class TestCLIParser:
    """Tests for argument parsing."""

    def test_parser_builds(self):
        parser = build_parser()
        assert parser is not None

    def test_version_flag(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_no_command_returns_zero(self):
        result = main([])
        assert result == 0

    def test_demo_parser(self):
        parser = build_parser()
        args = parser.parse_args(["demo", "pharma", "--n-customers", "50"])
        assert args.command == "demo"
        assert args.dataset == "pharma"
        assert args.n_customers == 50

    def test_run_parser(self):
        parser = build_parser()
        args = parser.parse_args([
            "run", "data.csv",
            "--customer-id", "cid",
            "--time-col", "period",
            "--outcome-col", "y",
            "--promo-vars", "x1,x2",
        ])
        assert args.command == "run"
        assert args.data == "data.csv"
        assert args.promo_vars == "x1,x2"

    def test_benchmark_parser(self):
        parser = build_parser()
        args = parser.parse_args(["benchmark", "--n-customers", "30"])
        assert args.command == "benchmark"
        assert args.n_customers == 30


class TestCLIDemo:
    """Tests for the demo dataset generation command."""

    def test_demo_pharma(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "pharma.csv")
            result = main(["demo", "pharma", "--n-customers", "20",
                           "--n-periods", "6", "--output", output])
            assert result == 0
            assert Path(output).exists()
            df = pd.read_csv(output)
            assert len(df) == 20 * 6

    def test_demo_cpg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "cpg.csv")
            result = main(["demo", "cpg", "--n-customers", "20",
                           "--n-periods", "6", "--output", output])
            assert result == 0
            assert Path(output).exists()

    def test_demo_saas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "saas.csv")
            result = main(["demo", "saas", "--n-customers", "20",
                           "--n-periods", "6", "--output", output])
            assert result == 0

    def test_demo_linear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "linear.csv")
            result = main(["demo", "linear", "--n-customers", "20",
                           "--n-periods", "6", "--output", output])
            assert result == 0

    def test_demo_unknown_dataset(self):
        with pytest.raises(SystemExit):
            main(["demo", "unknown"])


class TestCLIRun:
    """Tests for the run command with actual data."""

    def test_run_missing_file(self):
        result = main([
            "run", "nonexistent.csv",
            "--customer-id", "cid",
            "--time-col", "period",
            "--outcome-col", "y",
            "--promo-vars", "x1",
        ])
        assert result == 1
