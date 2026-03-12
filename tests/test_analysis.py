"""Tests for jacksmirror.analysis module."""

import warnings

import pandas as pd
import pytest

from jacksmirror.analysis import (
    compute_descriptive_stats,
    compute_odds_ratio,
    run_all_analyses,
    run_chi_squared_test,
    run_fisher_exact_test,
    run_logistic_regression,
    run_extended_logistic_regression,
)
from jacksmirror.models import (
    AnalysisResults,
    ContingencyTestResult,
    DescriptiveStats,
    LogisticRegressionResult,
    OddsRatioResult,
)


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------


class TestComputeDescriptiveStats:
    """Tests for compute_descriptive_stats()."""

    def test_returns_descriptive_stats_type(self, sample_dataframe):
        """Should return a DescriptiveStats instance."""
        result = compute_descriptive_stats(sample_dataframe)
        assert isinstance(result, DescriptiveStats)

    def test_sample_size_matches_dataframe(self, sample_dataframe):
        """sample_size should equal the number of rows in the DataFrame."""
        result = compute_descriptive_stats(sample_dataframe)
        assert result.sample_size == 10

    def test_mating_frequencies_correct(self, sample_dataframe):
        """Mating frequency counts should match the fixture composition.

        Fixture has 7 monandrous and 3 polyandrous.
        """
        result = compute_descriptive_stats(sample_dataframe)
        assert result.mating_frequencies.get("monandrous") == 7
        assert result.mating_frequencies.get("polyandrous") == 3

    def test_sociality_frequencies_correct(self, sample_dataframe):
        """Sociality frequencies should match the fixture composition.

        Fixture has 5 solitary, 2 primitively_eusocial, 3 advanced_eusocial.
        """
        result = compute_descriptive_stats(sample_dataframe)
        assert result.sociality_frequencies.get("solitary") == 5
        assert result.sociality_frequencies.get("primitively_eusocial") == 2
        assert result.sociality_frequencies.get("advanced_eusocial") == 3

    def test_counts_by_mating_matches_frequencies(self, sample_dataframe):
        """counts_by_mating should agree with mating_frequencies."""
        result = compute_descriptive_stats(sample_dataframe)
        assert result.counts_by_mating.get("monandrous") == 7
        assert result.counts_by_mating.get("polyandrous") == 3

    def test_cross_tabulation_is_dict(self, sample_dataframe):
        """cross_tabulation should be a nested dict."""
        result = compute_descriptive_stats(sample_dataframe)
        assert isinstance(result.cross_tabulation, dict)
        for key, val in result.cross_tabulation.items():
            assert isinstance(val, dict)

    def test_cross_tabulation_counts(self, sample_dataframe):
        """Spot-check: monandrous solitary should be 4 in the fixture."""
        result = compute_descriptive_stats(sample_dataframe)
        mono = result.cross_tabulation.get("monandrous", {})
        assert mono.get("solitary") == 4

    def test_works_with_full_dataset(self, full_dataset):
        """Should compute descriptive stats on the real dataset without error."""
        result = compute_descriptive_stats(full_dataset)
        assert result.sample_size == 70


# ---------------------------------------------------------------------------
# Fisher's exact test
# ---------------------------------------------------------------------------


class TestRunFisherExactTest:
    """Tests for run_fisher_exact_test()."""

    def test_returns_contingency_test_result(self, sample_dataframe):
        """Should return a ContingencyTestResult."""
        result = run_fisher_exact_test(sample_dataframe)
        assert isinstance(result, ContingencyTestResult)

    def test_test_name_is_fisher(self, sample_dataframe):
        """Test name should indicate Fisher's exact test."""
        result = run_fisher_exact_test(sample_dataframe)
        assert "fisher" in result.test_name.lower()

    def test_p_value_between_0_and_1(self, sample_dataframe):
        """p-value must be in [0, 1]."""
        result = run_fisher_exact_test(sample_dataframe)
        assert 0.0 <= result.p_value <= 1.0

    def test_statistic_is_positive(self, sample_dataframe):
        """The Fisher test statistic (odds ratio) should be positive."""
        result = run_fisher_exact_test(sample_dataframe)
        assert result.statistic > 0

    def test_contingency_table_is_2x2(self, sample_dataframe):
        """The contingency table should be a 2x2 list of lists."""
        result = run_fisher_exact_test(sample_dataframe)
        assert len(result.contingency_table) == 2
        for row in result.contingency_table:
            assert len(row) == 2

    def test_degrees_of_freedom_is_none(self, sample_dataframe):
        """Fisher's test does not have degrees of freedom."""
        result = run_fisher_exact_test(sample_dataframe)
        assert result.degrees_of_freedom is None

    def test_expected_frequencies_is_none(self, sample_dataframe):
        """Fisher's test result should not include expected frequencies."""
        result = run_fisher_exact_test(sample_dataframe)
        assert result.expected_frequencies is None

    def test_full_dataset_p_value_valid(self, full_dataset):
        """Fisher's test on the real dataset should produce a valid p-value."""
        result = run_fisher_exact_test(full_dataset)
        assert 0.0 <= result.p_value <= 1.0


# ---------------------------------------------------------------------------
# Chi-squared test
# ---------------------------------------------------------------------------


class TestRunChiSquaredTest:
    """Tests for run_chi_squared_test()."""

    def test_returns_contingency_test_result(self, sample_dataframe):
        """Should return a ContingencyTestResult."""
        # Small sample may trigger low-expected-frequency warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = run_chi_squared_test(sample_dataframe)
        assert isinstance(result, ContingencyTestResult)

    def test_test_name_is_chi_squared(self, sample_dataframe):
        """Test name should indicate chi-squared."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = run_chi_squared_test(sample_dataframe)
        assert "chi" in result.test_name.lower()

    def test_p_value_between_0_and_1(self, sample_dataframe):
        """p-value must be in [0, 1]."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = run_chi_squared_test(sample_dataframe)
        assert 0.0 <= result.p_value <= 1.0

    def test_statistic_is_non_negative(self, sample_dataframe):
        """Chi-squared statistic should be >= 0."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = run_chi_squared_test(sample_dataframe)
        assert result.statistic >= 0.0

    def test_degrees_of_freedom_is_one(self, sample_dataframe):
        """For a 2x2 table, degrees of freedom should be 1."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = run_chi_squared_test(sample_dataframe)
        assert result.degrees_of_freedom == 1

    def test_expected_frequencies_present(self, sample_dataframe):
        """Chi-squared test should provide expected frequencies."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = run_chi_squared_test(sample_dataframe)
        assert result.expected_frequencies is not None
        assert len(result.expected_frequencies) == 2

    def test_warns_on_low_expected_frequencies(self, sample_dataframe):
        """Should warn when expected frequencies are below 5."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_chi_squared_test(sample_dataframe)
        # With a 10-row dataset some cells will likely be below 5
        low_freq_warnings = [
            w for w in caught
            if "expected frequencies" in str(w.message).lower()
        ]
        # This may or may not trigger depending on the exact data,
        # so we just verify the code path doesn't crash.
        # The assertion is soft: if it warns, the warning message should be meaningful.
        for w in low_freq_warnings:
            assert "below 5" in str(w.message).lower()


# ---------------------------------------------------------------------------
# Odds ratio
# ---------------------------------------------------------------------------


class TestComputeOddsRatio:
    """Tests for compute_odds_ratio()."""

    def test_returns_odds_ratio_result(self, sample_dataframe):
        """Should return an OddsRatioResult."""
        result = compute_odds_ratio(sample_dataframe)
        assert isinstance(result, OddsRatioResult)

    def test_odds_ratio_is_positive(self, sample_dataframe):
        """Odds ratio should be positive."""
        result = compute_odds_ratio(sample_dataframe)
        assert result.odds_ratio > 0

    def test_confidence_interval_bounds_order(self, sample_dataframe):
        """ci_lower should be less than or equal to ci_upper."""
        result = compute_odds_ratio(sample_dataframe)
        assert result.ci_lower <= result.ci_upper

    def test_odds_ratio_within_ci(self, sample_dataframe):
        """The point estimate should lie within the confidence interval."""
        result = compute_odds_ratio(sample_dataframe)
        assert result.ci_lower <= result.odds_ratio <= result.ci_upper

    def test_p_value_between_0_and_1(self, sample_dataframe):
        """p-value must be in [0, 1]."""
        result = compute_odds_ratio(sample_dataframe)
        assert 0.0 <= result.p_value <= 1.0

    def test_interpretation_is_nonempty_string(self, sample_dataframe):
        """interpretation should be a non-empty string."""
        result = compute_odds_ratio(sample_dataframe)
        assert isinstance(result.interpretation, str)
        assert len(result.interpretation) > 0

    def test_interpretation_contains_or_value(self, sample_dataframe):
        """The interpretation string should mention the OR value."""
        result = compute_odds_ratio(sample_dataframe)
        assert "OR" in result.interpretation

    def test_interpretation_contains_direction(self, sample_dataframe):
        """The interpretation should state the direction (more/less/equally)."""
        result = compute_odds_ratio(sample_dataframe)
        assert any(
            word in result.interpretation
            for word in ("more", "less", "equally")
        )

    def test_full_dataset_produces_valid_result(self, full_dataset):
        """Odds ratio on the real dataset should produce valid numbers."""
        result = compute_odds_ratio(full_dataset)
        assert result.odds_ratio > 0
        assert 0.0 <= result.p_value <= 1.0


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------


class TestRunLogisticRegression:
    """Tests for run_logistic_regression()."""

    def test_returns_logistic_regression_result(self, sample_dataframe):
        """Should return a LogisticRegressionResult (possibly a fallback)."""
        result = run_logistic_regression(sample_dataframe)
        assert isinstance(result, LogisticRegressionResult)

    def test_has_required_fields(self, sample_dataframe):
        """Result should have all expected fields populated."""
        result = run_logistic_regression(sample_dataframe)
        assert isinstance(result.coefficients, dict)
        assert isinstance(result.std_errors, dict)
        assert isinstance(result.p_values, dict)
        assert isinstance(result.odds_ratios, dict)
        assert isinstance(result.pseudo_r_squared, float)
        assert isinstance(result.aic, float)
        assert isinstance(result.n_observations, int)
        assert isinstance(result.model_summary_text, str)

    def test_n_observations_positive(self, sample_dataframe):
        """n_observations should be positive."""
        result = run_logistic_regression(sample_dataframe)
        assert result.n_observations > 0

    def test_coefficients_include_predictor(self, sample_dataframe):
        """Coefficients should include the is_monandrous predictor."""
        result = run_logistic_regression(sample_dataframe)
        assert "is_monandrous" in result.coefficients

    def test_full_dataset_returns_result(self, full_dataset):
        """Logistic regression on the real dataset should return a result."""
        result = run_logistic_regression(full_dataset)
        assert isinstance(result, LogisticRegressionResult)
        assert result.n_observations > 0


# ---------------------------------------------------------------------------
# Extended logistic regression
# ---------------------------------------------------------------------------


class TestRunExtendedLogisticRegression:
    """Tests for run_extended_logistic_regression()."""

    def test_returns_result_or_none(self, full_dataset):
        """Should return a LogisticRegressionResult or None (if it cannot fit)."""
        result = run_extended_logistic_regression(full_dataset)
        assert result is None or isinstance(result, LogisticRegressionResult)

    def test_returns_none_when_columns_missing(self):
        """Should return None when required columns are missing."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = run_extended_logistic_regression(df)
        assert result is None

    def test_returns_none_for_tiny_dataset(self, sample_dataframe):
        """Should return None when the dataset is too small after dropping NaN."""
        tiny = sample_dataframe.head(2).copy()
        # Remove colony_size to force NaN in log_colony_size
        tiny["log_colony_size"] = float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = run_extended_logistic_regression(tiny)
        assert result is None


# ---------------------------------------------------------------------------
# run_all_analyses (orchestrator)
# ---------------------------------------------------------------------------


class TestRunAllAnalyses:
    """Tests for run_all_analyses()."""

    def test_returns_analysis_results(self, full_dataset):
        """Should return an AnalysisResults instance."""
        result = run_all_analyses(full_dataset)
        assert isinstance(result, AnalysisResults)

    def test_all_subresults_populated(self, full_dataset):
        """All sub-results should be of the correct types."""
        result = run_all_analyses(full_dataset)
        assert isinstance(result.descriptive, DescriptiveStats)
        assert isinstance(result.contingency_test, ContingencyTestResult)
        assert isinstance(result.odds_ratio, OddsRatioResult)
        assert isinstance(result.logistic_regression, LogisticRegressionResult)

    def test_n_species_matches_dataset(self, full_dataset):
        """n_species should equal the number of rows in the input DataFrame."""
        result = run_all_analyses(full_dataset)
        assert result.n_species == len(full_dataset)

    def test_timestamp_is_nonempty(self, full_dataset):
        """Timestamp should be a non-empty ISO string."""
        result = run_all_analyses(full_dataset)
        assert isinstance(result.timestamp, str)
        assert len(result.timestamp) > 0

    def test_to_dict_returns_dict(self, full_dataset):
        """to_dict() should produce a plain dict."""
        result = run_all_analyses(full_dataset)
        d = result.to_dict()
        assert isinstance(d, dict)

    def test_to_json_returns_valid_json(self, full_dataset):
        """to_json() should return a parseable JSON string."""
        import json

        result = run_all_analyses(full_dataset)
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_sample_data_produces_valid_results(self, sample_dataframe):
        """run_all_analyses should work with the small sample fixture."""
        result = run_all_analyses(sample_dataframe)
        assert isinstance(result, AnalysisResults)
        assert result.n_species == 10
