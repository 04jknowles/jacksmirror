"""Tests for jacksmirror.reporting module."""

import json

import pytest

from jacksmirror.analysis import run_all_analyses
from jacksmirror.reporting import (
    generate_json_report,
    generate_text_report,
    print_summary,
)


@pytest.fixture()
def analysis_results(sample_dataframe):
    """Run all analyses on the sample fixture for reporting tests."""
    return run_all_analyses(sample_dataframe)


# ---------------------------------------------------------------------------
# generate_text_report
# ---------------------------------------------------------------------------


class TestGenerateTextReport:
    """Tests for generate_text_report()."""

    def test_creates_file(self, analysis_results, tmp_output_dir):
        """Should create a text report file."""
        filepath = generate_text_report(analysis_results, tmp_output_dir)
        assert filepath.exists()

    def test_file_is_nonempty(self, analysis_results, tmp_output_dir):
        """Report file should have content."""
        filepath = generate_text_report(analysis_results, tmp_output_dir)
        assert filepath.stat().st_size > 0

    def test_filename_is_correct(self, analysis_results, tmp_output_dir):
        """Report file should have the expected name."""
        filepath = generate_text_report(analysis_results, tmp_output_dir)
        assert filepath.name == "analysis_report.txt"

    def test_contains_descriptive_statistics_header(
        self, analysis_results, tmp_output_dir
    ):
        """Report should contain the Descriptive Statistics section."""
        filepath = generate_text_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        assert "DESCRIPTIVE STATISTICS" in text

    def test_contains_fisher_section(self, analysis_results, tmp_output_dir):
        """Report should contain the Fisher's Exact Test section."""
        filepath = generate_text_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        assert "FISHER" in text.upper()

    def test_contains_odds_ratio_section(self, analysis_results, tmp_output_dir):
        """Report should contain the Odds Ratio section."""
        filepath = generate_text_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        assert "ODDS RATIO" in text

    def test_contains_logistic_regression_section(
        self, analysis_results, tmp_output_dir
    ):
        """Report should contain the Logistic Regression section."""
        filepath = generate_text_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        assert "LOGISTIC REGRESSION" in text

    def test_contains_conclusions_section(self, analysis_results, tmp_output_dir):
        """Report should contain a Conclusions section."""
        filepath = generate_text_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        assert "CONCLUSIONS" in text

    def test_contains_report_header(self, analysis_results, tmp_output_dir):
        """Report should contain the project title."""
        filepath = generate_text_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        assert "JACKSMIRROR" in text

    def test_contains_sample_size(self, analysis_results, tmp_output_dir):
        """Report should mention sample size."""
        filepath = generate_text_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        assert "sample size" in text.lower()


# ---------------------------------------------------------------------------
# generate_json_report
# ---------------------------------------------------------------------------


class TestGenerateJsonReport:
    """Tests for generate_json_report()."""

    def test_creates_file(self, analysis_results, tmp_output_dir):
        """Should create a JSON report file."""
        filepath = generate_json_report(analysis_results, tmp_output_dir)
        assert filepath.exists()

    def test_filename_is_correct(self, analysis_results, tmp_output_dir):
        """JSON report should have the expected name."""
        filepath = generate_json_report(analysis_results, tmp_output_dir)
        assert filepath.name == "analysis_results.json"

    def test_file_is_valid_json(self, analysis_results, tmp_output_dir):
        """Report content should be parseable JSON."""
        filepath = generate_json_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        data = json.loads(text)
        assert isinstance(data, dict)

    def test_json_contains_descriptive_key(self, analysis_results, tmp_output_dir):
        """JSON should have a 'descriptive' top-level key."""
        filepath = generate_json_report(analysis_results, tmp_output_dir)
        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert "descriptive" in data

    def test_json_contains_contingency_test_key(
        self, analysis_results, tmp_output_dir
    ):
        """JSON should have a 'contingency_test' top-level key."""
        filepath = generate_json_report(analysis_results, tmp_output_dir)
        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert "contingency_test" in data

    def test_json_contains_odds_ratio_key(self, analysis_results, tmp_output_dir):
        """JSON should have an 'odds_ratio' top-level key."""
        filepath = generate_json_report(analysis_results, tmp_output_dir)
        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert "odds_ratio" in data

    def test_json_contains_logistic_regression_key(
        self, analysis_results, tmp_output_dir
    ):
        """JSON should have a 'logistic_regression' top-level key."""
        filepath = generate_json_report(analysis_results, tmp_output_dir)
        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert "logistic_regression" in data

    def test_json_contains_n_species(self, analysis_results, tmp_output_dir):
        """JSON should have an 'n_species' field."""
        filepath = generate_json_report(analysis_results, tmp_output_dir)
        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert "n_species" in data
        assert data["n_species"] == 10  # sample fixture has 10 rows

    def test_json_contains_timestamp(self, analysis_results, tmp_output_dir):
        """JSON should have a 'timestamp' field."""
        filepath = generate_json_report(analysis_results, tmp_output_dir)
        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert "timestamp" in data
        assert isinstance(data["timestamp"], str)
        assert len(data["timestamp"]) > 0


# ---------------------------------------------------------------------------
# print_summary
# ---------------------------------------------------------------------------


class TestPrintSummary:
    """Tests for print_summary()."""

    def test_does_not_raise(self, analysis_results):
        """print_summary should execute without raising an exception."""
        print_summary(analysis_results)

    def test_prints_to_stdout(self, analysis_results, capsys):
        """print_summary should produce output on stdout."""
        print_summary(analysis_results)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_output_contains_sample_size(self, analysis_results, capsys):
        """Stdout output should mention sample size."""
        print_summary(analysis_results)
        captured = capsys.readouterr()
        assert "sample size" in captured.out.lower() or "n" in captured.out.lower()

    def test_output_contains_odds_ratio(self, analysis_results, capsys):
        """Stdout output should mention the odds ratio."""
        print_summary(analysis_results)
        captured = capsys.readouterr()
        assert "odds ratio" in captured.out.lower()

    def test_output_contains_conclusion(self, analysis_results, capsys):
        """Stdout output should contain a conclusion statement."""
        print_summary(analysis_results)
        captured = capsys.readouterr()
        assert "conclusion" in captured.out.lower()

    def test_output_contains_p_value(self, analysis_results, capsys):
        """Stdout output should contain a p-value."""
        print_summary(analysis_results)
        captured = capsys.readouterr()
        assert "p =" in captured.out.lower() or "p=" in captured.out.lower()
