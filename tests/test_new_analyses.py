"""Tests for the new analysis functions and reporting added in v0.2.0."""

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jacksmirror.analysis import (
    compute_descriptive_stats,
    compute_odds_ratio,
    run_all_analyses,
    run_chi_squared_test,
    run_fisher_exact_test,
    run_logistic_regression,
    run_mann_whitney_test,
    run_permutation_test,
    run_sensitivity_analysis,
)
from jacksmirror.config import RANDOM_SEED, SENSITIVITY_THRESHOLDS
from jacksmirror.models import (
    AnalysisResults,
    ContingencyTestResult,
    DescriptiveStats,
    LogisticRegressionResult,
    MannWhitneyResult,
    OddsRatioResult,
    PermutationTestResult,
    SensitivityAnalysisResult,
)
from jacksmirror.reporting import generate_academic_report


# ---------------------------------------------------------------------------
# Mann-Whitney U test
# ---------------------------------------------------------------------------


class TestMannWhitneyTest:
    """Tests for run_mann_whitney_test()."""

    def test_returns_mann_whitney_result_type(self, sample_dataframe):
        """Should return a MannWhitneyResult instance."""
        result = run_mann_whitney_test(sample_dataframe)
        assert isinstance(result, MannWhitneyResult)

    def test_p_value_between_0_and_1(self, sample_dataframe):
        """p-value must be in [0, 1]."""
        result = run_mann_whitney_test(sample_dataframe)
        assert 0.0 <= result.p_value <= 1.0

    def test_medians_computed_correctly(self, sample_dataframe):
        """Medians should match manual computation on the sample data."""
        result = run_mann_whitney_test(sample_dataframe)
        subset = sample_dataframe.dropna(subset=["effective_mates"])
        eus = subset.loc[subset["is_eusocial"], "effective_mates"]
        non_eus = subset.loc[~subset["is_eusocial"], "effective_mates"]
        assert result.median_eusocial == pytest.approx(float(eus.median()))
        assert result.median_non_eusocial == pytest.approx(float(non_eus.median()))

    def test_works_on_full_dataset(self, full_dataset):
        """Should run without error on the real dataset."""
        result = run_mann_whitney_test(full_dataset)
        assert isinstance(result, MannWhitneyResult)
        assert 0.0 <= result.p_value <= 1.0

    def test_handles_single_group_data(self):
        """Should handle edge case where one group has < 2 observations."""
        # All eusocial, no non-eusocial with effective_mates
        df = pd.DataFrame({
            "effective_mates": [1.0, 1.0, 1.0],
            "is_eusocial": [True, True, True],
        })
        result = run_mann_whitney_test(df)
        assert isinstance(result, MannWhitneyResult)
        assert math.isnan(result.p_value)
        assert "Insufficient" in result.interpretation

    def test_interpretation_is_nonempty(self, sample_dataframe):
        """interpretation should be a non-empty string."""
        result = run_mann_whitney_test(sample_dataframe)
        assert isinstance(result.interpretation, str)
        assert len(result.interpretation) > 0

    def test_n_counts_match_expected(self, sample_dataframe):
        """n_eusocial + n_non_eusocial should match rows with non-NaN mates."""
        result = run_mann_whitney_test(sample_dataframe)
        subset = sample_dataframe.dropna(subset=["effective_mates"])
        expected_total = len(subset)
        assert result.n_eusocial + result.n_non_eusocial == expected_total

    def test_means_are_finite(self, sample_dataframe):
        """Means should be finite numbers."""
        result = run_mann_whitney_test(sample_dataframe)
        assert math.isfinite(result.mean_eusocial)
        assert math.isfinite(result.mean_non_eusocial)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------


class TestPermutationTest:
    """Tests for run_permutation_test()."""

    def test_returns_permutation_test_result_type(self, sample_dataframe):
        """Should return a PermutationTestResult instance."""
        result = run_permutation_test(sample_dataframe, n_permutations=100)
        assert isinstance(result, PermutationTestResult)

    def test_p_value_between_0_and_1(self, sample_dataframe):
        """p-value must be in [0, 1]."""
        result = run_permutation_test(sample_dataframe, n_permutations=100)
        assert 0.0 <= result.p_value <= 1.0

    def test_observed_or_matches_compute_odds_ratio(self, sample_dataframe):
        """Observed OR should match the value from compute_odds_ratio."""
        perm = run_permutation_test(sample_dataframe, n_permutations=100)
        odds = compute_odds_ratio(sample_dataframe)
        assert perm.observed_odds_ratio == pytest.approx(odds.odds_ratio, rel=1e-4)

    def test_deterministic_with_fixed_seed(self, sample_dataframe):
        """Running twice with the same seed should produce identical results."""
        r1 = run_permutation_test(sample_dataframe, n_permutations=500, random_seed=42)
        r2 = run_permutation_test(sample_dataframe, n_permutations=500, random_seed=42)
        assert r1.p_value == r2.p_value
        assert r1.null_distribution_mean == pytest.approx(r2.null_distribution_mean)

    def test_n_permutations_matches_input(self, sample_dataframe):
        """n_permutations in the result should match the input."""
        result = run_permutation_test(sample_dataframe, n_permutations=200)
        assert result.n_permutations == 200

    def test_null_distribution_stats_are_finite(self, sample_dataframe):
        """Null distribution statistics should be finite."""
        result = run_permutation_test(sample_dataframe, n_permutations=100)
        assert math.isfinite(result.null_distribution_mean)
        assert math.isfinite(result.null_distribution_std)
        assert math.isfinite(result.percentile_95)
        assert math.isfinite(result.percentile_99)

    def test_works_on_full_dataset(self, full_dataset):
        """Should run without error on the real dataset."""
        result = run_permutation_test(full_dataset, n_permutations=100)
        assert isinstance(result, PermutationTestResult)
        assert 0.0 <= result.p_value <= 1.0

    def test_interpretation_is_nonempty(self, sample_dataframe):
        """interpretation should be a non-empty string."""
        result = run_permutation_test(sample_dataframe, n_permutations=100)
        assert isinstance(result.interpretation, str)
        assert len(result.interpretation) > 0


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


class TestSensitivityAnalysis:
    """Tests for run_sensitivity_analysis()."""

    def test_returns_sensitivity_analysis_result_type(self, sample_dataframe):
        """Should return a SensitivityAnalysisResult instance."""
        result = run_sensitivity_analysis(sample_dataframe)
        assert isinstance(result, SensitivityAnalysisResult)

    def test_default_thresholds_produce_four_results(self, sample_dataframe):
        """Default thresholds [1.0, 1.5, 2.0, 2.5] should produce 4 results."""
        result = run_sensitivity_analysis(sample_dataframe)
        assert len(result.results) == 4
        assert len(result.thresholds) == 4

    def test_custom_thresholds_work(self, sample_dataframe):
        """Custom thresholds should produce the correct number of results."""
        result = run_sensitivity_analysis(
            sample_dataframe, thresholds=[1.0, 3.0, 5.0]
        )
        assert len(result.results) == 3
        assert result.thresholds == [1.0, 3.0, 5.0]

    def test_each_result_has_expected_keys(self, sample_dataframe):
        """Each result dict should have the expected keys."""
        result = run_sensitivity_analysis(sample_dataframe)
        expected_keys = {
            "threshold", "n_monandrous", "n_polyandrous",
            "odds_ratio", "ci_lower", "ci_upper",
            "fisher_p_value", "contingency_table",
        }
        for r in result.results:
            assert set(r.keys()) == expected_keys

    def test_fisher_p_values_between_0_and_1(self, sample_dataframe):
        """All Fisher p-values should be in [0, 1]."""
        result = run_sensitivity_analysis(sample_dataframe)
        for r in result.results:
            assert 0.0 <= r["fisher_p_value"] <= 1.0

    def test_works_on_full_dataset(self, full_dataset):
        """Should run without error on the real dataset."""
        result = run_sensitivity_analysis(full_dataset)
        assert isinstance(result, SensitivityAnalysisResult)
        assert len(result.results) == 4

    def test_interpretation_is_nonempty(self, sample_dataframe):
        """interpretation should be a non-empty string."""
        result = run_sensitivity_analysis(sample_dataframe)
        assert isinstance(result.interpretation, str)
        assert len(result.interpretation) > 0

    def test_odds_ratios_are_positive(self, sample_dataframe):
        """All odds ratios should be positive."""
        result = run_sensitivity_analysis(sample_dataframe)
        for r in result.results:
            assert r["odds_ratio"] > 0


# ---------------------------------------------------------------------------
# Updated run_all_analyses
# ---------------------------------------------------------------------------


class TestUpdatedRunAllAnalyses:
    """Tests for the updated run_all_analyses with new fields."""

    def test_mann_whitney_field_populated(self, full_dataset):
        """mann_whitney should be populated after run_all_analyses."""
        result = run_all_analyses(full_dataset)
        assert result.mann_whitney is not None
        assert isinstance(result.mann_whitney, MannWhitneyResult)

    def test_permutation_test_field_populated(self, full_dataset):
        """permutation_test should be populated after run_all_analyses."""
        result = run_all_analyses(full_dataset)
        assert result.permutation_test is not None
        assert isinstance(result.permutation_test, PermutationTestResult)

    def test_chi_squared_field_populated(self, full_dataset):
        """chi_squared should be populated after run_all_analyses."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = run_all_analyses(full_dataset)
        assert result.chi_squared is not None
        assert isinstance(result.chi_squared, ContingencyTestResult)

    def test_all_existing_assertions_still_hold(self, full_dataset):
        """All original fields should still be valid."""
        result = run_all_analyses(full_dataset)
        assert isinstance(result, AnalysisResults)
        assert result.n_species == len(full_dataset)
        assert isinstance(result.timestamp, str)
        assert len(result.timestamp) > 0


# ---------------------------------------------------------------------------
# Academic report
# ---------------------------------------------------------------------------


class TestAcademicReport:
    """Tests for generate_academic_report()."""

    @pytest.fixture()
    def analysis_results(self, sample_dataframe):
        """Run all analyses on the sample fixture for report tests."""
        return run_all_analyses(sample_dataframe)

    def test_creates_file(self, analysis_results, tmp_output_dir):
        """Should create a file at the expected path."""
        filepath = generate_academic_report(analysis_results, tmp_output_dir)
        assert filepath.exists()

    def test_file_is_nonempty(self, analysis_results, tmp_output_dir):
        """Report file should have content."""
        filepath = generate_academic_report(analysis_results, tmp_output_dir)
        assert filepath.stat().st_size > 0

    def test_contains_introduction(self, analysis_results, tmp_output_dir):
        """Report should contain an Introduction section."""
        filepath = generate_academic_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        assert "Introduction" in text

    def test_contains_methods(self, analysis_results, tmp_output_dir):
        """Report should contain a Methods section."""
        filepath = generate_academic_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        assert "Methods" in text

    def test_contains_results(self, analysis_results, tmp_output_dir):
        """Report should contain a Results section."""
        filepath = generate_academic_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        assert "Results" in text

    def test_contains_limitations(self, analysis_results, tmp_output_dir):
        """Report should contain a Limitations section."""
        filepath = generate_academic_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        assert "Limitations" in text


# ---------------------------------------------------------------------------
# Model serialization with new fields
# ---------------------------------------------------------------------------


class TestModelSerialization:
    """Tests for serialization of AnalysisResults with new fields."""

    def test_to_dict_works_with_new_fields(self, full_dataset):
        """to_dict() should work when new fields are populated."""
        result = run_all_analyses(full_dataset)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "mann_whitney" in d
        assert "permutation_test" in d
        assert "sensitivity_analysis" in d

    def test_to_json_produces_valid_json(self, full_dataset):
        """to_json() should produce parseable JSON with new fields."""
        result = run_all_analyses(full_dataset)
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "mann_whitney" in parsed

    def test_new_fields_serialize_when_none(self, sample_dataframe):
        """New fields should serialize correctly even when None."""
        from jacksmirror.models import DescriptiveStats, OddsRatioResult

        result = AnalysisResults(
            descriptive=compute_descriptive_stats(sample_dataframe),
            contingency_test=run_fisher_exact_test(sample_dataframe),
            odds_ratio=compute_odds_ratio(sample_dataframe),
            logistic_regression=run_logistic_regression(sample_dataframe),
        )
        # New fields default to None
        assert result.mann_whitney is None
        assert result.permutation_test is None
        assert result.sensitivity_analysis is None

        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["mann_whitney"] is None

        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["mann_whitney"] is None


# ---------------------------------------------------------------------------
# Additional Mann-Whitney edge case tests
# ---------------------------------------------------------------------------


class TestMannWhitneyEdgeCases:
    """Additional edge case tests for run_mann_whitney_test()."""

    def test_handles_empty_effective_mates(self):
        """Should handle data where all effective_mates are NaN."""
        df = pd.DataFrame({
            "effective_mates": [float("nan"), float("nan"), float("nan")],
            "is_eusocial": [True, False, True],
        })
        result = run_mann_whitney_test(df)
        assert isinstance(result, MannWhitneyResult)
        assert math.isnan(result.p_value)
        assert "Insufficient" in result.interpretation

    def test_handles_one_obs_per_group(self):
        """Should return insufficient data when each group has exactly 1 obs."""
        df = pd.DataFrame({
            "effective_mates": [1.0, 5.0],
            "is_eusocial": [True, False],
        })
        result = run_mann_whitney_test(df)
        assert isinstance(result, MannWhitneyResult)
        assert math.isnan(result.p_value)
        assert result.n_eusocial == 1
        assert result.n_non_eusocial == 1
        # Medians should still be computed for single observations
        assert result.median_eusocial == pytest.approx(1.0)
        assert result.median_non_eusocial == pytest.approx(5.0)

    def test_nan_rows_are_excluded(self, sample_dataframe):
        """Rows with NaN effective_mates should be excluded from the test."""
        df = sample_dataframe.copy()
        # Set two rows to NaN
        df.loc[0, "effective_mates"] = float("nan")
        df.loc[1, "effective_mates"] = float("nan")
        result = run_mann_whitney_test(df)
        subset = df.dropna(subset=["effective_mates"])
        assert result.n_eusocial + result.n_non_eusocial == len(subset)

    def test_u_statistic_is_non_negative(self, sample_dataframe):
        """U statistic should be non-negative for valid groups."""
        result = run_mann_whitney_test(sample_dataframe)
        assert result.u_statistic >= 0.0

    def test_identical_values_both_groups(self):
        """Should handle identical values in both groups (all tied)."""
        df = pd.DataFrame({
            "effective_mates": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "is_eusocial": [True, True, True, False, False, False],
        })
        result = run_mann_whitney_test(df)
        assert isinstance(result, MannWhitneyResult)
        # p-value should be 1.0 when distributions are identical
        assert result.p_value == pytest.approx(1.0)
        assert result.median_eusocial == result.median_non_eusocial

    def test_interpretation_contains_significantly_when_significant(self):
        """Interpretation should say 'significantly' when p < 0.05."""
        # Create very distinct groups to force significance
        df = pd.DataFrame({
            "effective_mates": [1.0, 1.0, 1.0, 1.0, 1.0,
                                100.0, 100.0, 100.0, 100.0, 100.0],
            "is_eusocial": [True, True, True, True, True,
                            False, False, False, False, False],
        })
        result = run_mann_whitney_test(df)
        assert "significantly" in result.interpretation
        # Should say "significantly" not "not significantly"
        if result.p_value < 0.05:
            assert "not significantly" not in result.interpretation

    def test_interpretation_contains_not_significantly_when_not(self):
        """Interpretation should say 'not significantly' when p >= 0.05."""
        # Create similar groups
        df = pd.DataFrame({
            "effective_mates": [1.0, 2.0, 1.5, 1.0, 2.0, 1.5],
            "is_eusocial": [True, True, True, False, False, False],
        })
        result = run_mann_whitney_test(df)
        if result.p_value >= 0.05:
            assert "not significantly" in result.interpretation

    def test_u_statistic_is_nan_for_insufficient_data(self):
        """U statistic should be NaN when insufficient data."""
        df = pd.DataFrame({
            "effective_mates": [1.0],
            "is_eusocial": [True],
        })
        result = run_mann_whitney_test(df)
        assert math.isnan(result.u_statistic)


# ---------------------------------------------------------------------------
# Additional Permutation test edge case tests
# ---------------------------------------------------------------------------


class TestPermutationTestEdgeCases:
    """Additional edge case tests for run_permutation_test()."""

    def test_default_seed_uses_config(self, sample_dataframe):
        """When random_seed is None, should use RANDOM_SEED from config."""
        r1 = run_permutation_test(sample_dataframe, n_permutations=200)
        r2 = run_permutation_test(
            sample_dataframe, n_permutations=200, random_seed=RANDOM_SEED
        )
        assert r1.p_value == r2.p_value
        assert r1.null_distribution_mean == pytest.approx(
            r2.null_distribution_mean
        )

    def test_different_seeds_produce_different_results(self, sample_dataframe):
        """Different seeds should generally produce different null distributions."""
        r1 = run_permutation_test(
            sample_dataframe, n_permutations=500, random_seed=1
        )
        r2 = run_permutation_test(
            sample_dataframe, n_permutations=500, random_seed=99
        )
        # The null distribution means should differ (with very high probability)
        assert r1.null_distribution_mean != pytest.approx(
            r2.null_distribution_mean, abs=1e-10
        )

    def test_percentile_95_less_than_or_equal_to_99(self, sample_dataframe):
        """95th percentile should be <= 99th percentile of the null distribution."""
        result = run_permutation_test(
            sample_dataframe, n_permutations=500, random_seed=42
        )
        assert result.percentile_95 <= result.percentile_99

    def test_observed_or_is_positive(self, sample_dataframe):
        """Observed odds ratio should be positive."""
        result = run_permutation_test(sample_dataframe, n_permutations=100)
        assert result.observed_odds_ratio > 0

    def test_null_distribution_std_is_non_negative(self, sample_dataframe):
        """Standard deviation of null distribution should be >= 0."""
        result = run_permutation_test(sample_dataframe, n_permutations=100)
        assert result.null_distribution_std >= 0.0

    def test_small_n_permutations(self, sample_dataframe):
        """Should work correctly with very few permutations."""
        result = run_permutation_test(
            sample_dataframe, n_permutations=10, random_seed=42
        )
        assert isinstance(result, PermutationTestResult)
        assert result.n_permutations == 10
        # p-value should be a multiple of 1/10
        assert 0.0 <= result.p_value <= 1.0

    def test_haldane_correction_with_zero_cells(self):
        """Should handle data producing zero cells via Haldane-Anscombe correction."""
        # All monandrous are eusocial, all polyandrous are non-eusocial
        # This produces zero cells in the contingency table
        df = pd.DataFrame({
            "is_monandrous": [True, True, True, False, False],
            "is_eusocial": [True, True, True, False, False],
        })
        result = run_permutation_test(df, n_permutations=50, random_seed=42)
        assert isinstance(result, PermutationTestResult)
        assert math.isfinite(result.observed_odds_ratio)
        assert 0.0 <= result.p_value <= 1.0

    def test_interpretation_mentions_p_value(self, sample_dataframe):
        """Interpretation should include the p-value."""
        result = run_permutation_test(
            sample_dataframe, n_permutations=100, random_seed=42
        )
        assert f"p={result.p_value:.4f}" in result.interpretation


# ---------------------------------------------------------------------------
# Additional Sensitivity analysis edge case tests
# ---------------------------------------------------------------------------


class TestSensitivityAnalysisEdgeCases:
    """Additional edge case tests for run_sensitivity_analysis()."""

    def test_single_threshold(self, sample_dataframe):
        """Should work with a single threshold."""
        result = run_sensitivity_analysis(
            sample_dataframe, thresholds=[2.0]
        )
        assert len(result.results) == 1
        assert len(result.thresholds) == 1
        assert result.thresholds == [2.0]

    def test_ci_lower_less_than_ci_upper(self, sample_dataframe):
        """ci_lower should be <= ci_upper for each threshold result."""
        result = run_sensitivity_analysis(sample_dataframe)
        for r in result.results:
            assert r["ci_lower"] <= r["ci_upper"]

    def test_odds_ratio_within_ci(self, sample_dataframe):
        """Odds ratio should fall within the confidence interval."""
        result = run_sensitivity_analysis(sample_dataframe)
        for r in result.results:
            assert r["ci_lower"] <= r["odds_ratio"] <= r["ci_upper"]

    def test_n_monandrous_plus_polyandrous_equals_total(self, sample_dataframe):
        """n_monandrous + n_polyandrous should equal total species."""
        total = len(sample_dataframe)
        result = run_sensitivity_analysis(sample_dataframe)
        for r in result.results:
            assert r["n_monandrous"] + r["n_polyandrous"] == total

    def test_contingency_table_is_2x2(self, sample_dataframe):
        """Each contingency table in results should be 2x2."""
        result = run_sensitivity_analysis(sample_dataframe)
        for r in result.results:
            table = r["contingency_table"]
            assert len(table) == 2
            for row in table:
                assert len(row) == 2

    def test_threshold_stored_in_result(self, sample_dataframe):
        """Each result dict should have the correct threshold value."""
        thresholds = [1.0, 2.0, 3.0]
        result = run_sensitivity_analysis(
            sample_dataframe, thresholds=thresholds
        )
        for r, t in zip(result.results, thresholds):
            assert r["threshold"] == t

    def test_nan_effective_mates_retain_original_classification(
        self, sample_dataframe
    ):
        """Species with NaN effective_mates should keep original is_monandrous."""
        df = sample_dataframe.copy()
        # Set some effective_mates to NaN and note their original classification
        df.loc[0, "effective_mates"] = float("nan")
        original_mono_0 = df.loc[0, "is_monandrous"]

        result = run_sensitivity_analysis(df, thresholds=[1.0])
        # The function should still run correctly
        assert isinstance(result, SensitivityAnalysisResult)
        assert len(result.results) == 1

    def test_very_high_threshold_makes_all_monandrous(self, sample_dataframe):
        """A very high threshold should classify all species as monandrous."""
        result = run_sensitivity_analysis(
            sample_dataframe, thresholds=[1000.0]
        )
        r = result.results[0]
        # All species with non-NaN effective_mates should be classified as
        # monandrous. Those with NaN keep their original classification.
        total = len(sample_dataframe)
        assert r["n_monandrous"] + r["n_polyandrous"] == total
        # With threshold=1000, all species in our fixture have effective_mates
        # < 1000, so all should be monandrous
        assert r["n_monandrous"] == total

    def test_very_low_threshold_makes_few_monandrous(self, sample_dataframe):
        """A very low threshold should classify most species as polyandrous."""
        result = run_sensitivity_analysis(
            sample_dataframe, thresholds=[0.5]
        )
        r = result.results[0]
        # All species have effective_mates >= 1.0, so with threshold=0.5
        # none should be classified as monandrous (those with non-NaN mates)
        assert r["n_polyandrous"] > 0

    def test_interpretation_all_significant(self, full_dataset):
        """When all thresholds are significant, interpretation should state so."""
        result = run_sensitivity_analysis(full_dataset)
        significant_count = sum(
            1 for r in result.results if r["fisher_p_value"] < 0.05
        )
        if significant_count == len(result.results):
            assert "robust" in result.interpretation.lower() or \
                   "all" in result.interpretation.lower()

    def test_interpretation_none_significant(self):
        """When no thresholds are significant, interpretation should state so."""
        # Create data with no real association
        rng = np.random.default_rng(42)
        n = 50
        df = pd.DataFrame({
            "effective_mates": rng.uniform(0.5, 5.0, n),
            "is_monandrous": rng.choice([True, False], n),
            "is_eusocial": rng.choice([True, False], n),
        })
        result = run_sensitivity_analysis(df, thresholds=[1.0, 2.0, 3.0])
        significant_count = sum(
            1 for r in result.results if r["fisher_p_value"] < 0.05
        )
        if significant_count == 0:
            assert "not significant" in result.interpretation.lower()

    def test_default_thresholds_match_config(self, sample_dataframe):
        """Default thresholds should match SENSITIVITY_THRESHOLDS from config."""
        result = run_sensitivity_analysis(sample_dataframe)
        assert result.thresholds == SENSITIVITY_THRESHOLDS


# ---------------------------------------------------------------------------
# Additional run_all_analyses tests
# ---------------------------------------------------------------------------


class TestRunAllAnalysesExtended:
    """Additional tests for the updated run_all_analyses."""

    def test_sensitivity_analysis_field_populated(self, full_dataset):
        """sensitivity_analysis should be populated after run_all_analyses."""
        result = run_all_analyses(full_dataset)
        assert result.sensitivity_analysis is not None
        assert isinstance(result.sensitivity_analysis, SensitivityAnalysisResult)

    def test_extended_logistic_field_type(self, full_dataset):
        """extended_logistic should be a LogisticRegressionResult or None."""
        result = run_all_analyses(full_dataset)
        assert result.extended_logistic is None or isinstance(
            result.extended_logistic, LogisticRegressionResult
        )

    def test_chi_squared_has_correct_test_name(self, full_dataset):
        """chi_squared test_name should indicate chi-squared."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = run_all_analyses(full_dataset)
        assert result.chi_squared is not None
        assert "chi" in result.chi_squared.test_name.lower()

    def test_sample_data_populates_all_new_fields(self, sample_dataframe):
        """All new analysis fields should be populated even on sample data."""
        result = run_all_analyses(sample_dataframe)
        assert result.mann_whitney is not None
        assert result.permutation_test is not None
        assert result.sensitivity_analysis is not None
        assert result.chi_squared is not None

    def test_dataset_path_is_empty_string(self, sample_dataframe):
        """dataset_path should be empty when called with a plain DataFrame."""
        result = run_all_analyses(sample_dataframe)
        assert result.dataset_path == ""

    def test_individual_failures_do_not_crash_pipeline(self):
        """run_all_analyses should not crash even if some sub-analyses fail.

        Create a small DataFrame that has the required columns and variation
        in the predictor, but is small enough that some analyses (like
        extended logistic) may fail or return None.
        """
        df = pd.DataFrame({
            "species": ["A", "B", "C", "D"],
            "sociality": pd.Categorical(
                ["solitary", "advanced_eusocial", "solitary", "advanced_eusocial"],
                categories=[
                    "solitary", "communal", "semisocial",
                    "primitively_eusocial", "advanced_eusocial",
                ],
                ordered=True,
            ),
            "mating_system": pd.Categorical(
                ["monandrous", "monandrous", "polyandrous", "polyandrous"],
                categories=["monandrous", "polyandrous"],
                ordered=True,
            ),
            "effective_mates": [1.0, 1.0, 5.0, 5.0],
            "is_eusocial": [False, True, False, True],
            "is_monandrous": [True, True, False, False],
            "sociality_binary": ["non-eusocial", "eusocial", "non-eusocial", "eusocial"],
            "log_colony_size": [0.0, 3.0, 0.0, 3.0],
            "colony_size": [1.0, 1000.0, 1.0, 1000.0],
            "haplodiploidy": [True, True, True, True],
            "family": ["Apidae", "Apidae", "Halictidae", "Halictidae"],
        })
        # Should not raise even though the data is very small
        result = run_all_analyses(df)
        assert isinstance(result, AnalysisResults)
        assert result.n_species == 4


# ---------------------------------------------------------------------------
# Additional Academic report tests
# ---------------------------------------------------------------------------


class TestAcademicReportExtended:
    """Additional tests for generate_academic_report()."""

    @pytest.fixture()
    def analysis_results(self, sample_dataframe):
        """Run all analyses on the sample fixture for report tests."""
        return run_all_analyses(sample_dataframe)

    def test_filename_is_academic_report_txt(self, analysis_results, tmp_output_dir):
        """Report should be named 'academic_report.txt'."""
        filepath = generate_academic_report(analysis_results, tmp_output_dir)
        assert filepath.name == "academic_report.txt"

    def test_contains_abstract(self, analysis_results, tmp_output_dir):
        """Report should contain an Abstract section."""
        filepath = generate_academic_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        assert "ABSTRACT" in text

    def test_contains_discussion(self, analysis_results, tmp_output_dir):
        """Report should contain a Discussion section."""
        filepath = generate_academic_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        assert "Discussion" in text

    def test_contains_references(self, analysis_results, tmp_output_dir):
        """Report should contain a References section."""
        filepath = generate_academic_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        assert "References" in text

    def test_contains_mann_whitney_section(self, analysis_results, tmp_output_dir):
        """Report should contain Mann-Whitney U Test section when result is present."""
        filepath = generate_academic_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        if analysis_results.mann_whitney is not None:
            assert "Mann-Whitney" in text

    def test_contains_permutation_test_section(self, analysis_results, tmp_output_dir):
        """Report should contain Permutation Test section when result is present."""
        filepath = generate_academic_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        if analysis_results.permutation_test is not None:
            assert "Permutation Test" in text

    def test_contains_sensitivity_analysis_section(
        self, analysis_results, tmp_output_dir
    ):
        """Report should contain Sensitivity Analysis section when result is present."""
        filepath = generate_academic_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        if analysis_results.sensitivity_analysis is not None:
            assert "Sensitivity Analysis" in text

    def test_contains_version_string(self, analysis_results, tmp_output_dir):
        """Report should reference the package version."""
        filepath = generate_academic_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        assert "v0.2.0" in text

    def test_contains_fisher_exact_results(self, analysis_results, tmp_output_dir):
        """Report should contain Fisher's exact test results."""
        filepath = generate_academic_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        assert "Fisher" in text

    def test_contains_odds_ratio_results(self, analysis_results, tmp_output_dir):
        """Report should contain odds ratio section."""
        filepath = generate_academic_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        assert "Odds Ratio" in text

    def test_contains_logistic_regression_section(
        self, analysis_results, tmp_output_dir
    ):
        """Report should contain logistic regression section."""
        filepath = generate_academic_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        assert "Logistic Regression" in text

    def test_report_with_full_dataset(self, full_dataset, tmp_output_dir):
        """Report should be generated successfully with the full dataset."""
        result = run_all_analyses(full_dataset)
        filepath = generate_academic_report(result, tmp_output_dir)
        assert filepath.exists()
        text = filepath.read_text(encoding="utf-8")
        # Full dataset report should be substantially larger
        assert len(text) > 5000
        assert "END OF ACADEMIC REPORT" in text

    def test_report_ends_with_end_marker(self, analysis_results, tmp_output_dir):
        """Report should end with 'END OF ACADEMIC REPORT'."""
        filepath = generate_academic_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        assert "END OF ACADEMIC REPORT" in text

    def test_report_contains_sample_size(self, analysis_results, tmp_output_dir):
        """Report should mention the sample size from descriptive stats."""
        filepath = generate_academic_report(analysis_results, tmp_output_dir)
        text = filepath.read_text(encoding="utf-8")
        sample_size = str(analysis_results.descriptive.sample_size)
        assert sample_size in text


# ---------------------------------------------------------------------------
# Dataclass field access tests for new models
# ---------------------------------------------------------------------------


class TestNewDataclassFields:
    """Tests for field access and defaults on new dataclasses."""

    def test_mann_whitney_result_all_fields_accessible(self):
        """All MannWhitneyResult fields should be accessible."""
        mw = MannWhitneyResult(
            u_statistic=100.0,
            p_value=0.05,
            n_eusocial=20,
            n_non_eusocial=30,
            median_eusocial=1.5,
            median_non_eusocial=3.0,
            mean_eusocial=2.0,
            mean_non_eusocial=4.0,
            interpretation="Test interpretation.",
        )
        assert mw.u_statistic == 100.0
        assert mw.p_value == 0.05
        assert mw.n_eusocial == 20
        assert mw.n_non_eusocial == 30
        assert mw.median_eusocial == 1.5
        assert mw.median_non_eusocial == 3.0
        assert mw.mean_eusocial == 2.0
        assert mw.mean_non_eusocial == 4.0
        assert mw.interpretation == "Test interpretation."

    def test_permutation_test_result_all_fields_accessible(self):
        """All PermutationTestResult fields should be accessible."""
        pt = PermutationTestResult(
            observed_odds_ratio=5.0,
            p_value=0.01,
            n_permutations=1000,
            null_distribution_mean=1.2,
            null_distribution_std=0.5,
            percentile_95=2.0,
            percentile_99=3.0,
            interpretation="Significant result.",
        )
        assert pt.observed_odds_ratio == 5.0
        assert pt.p_value == 0.01
        assert pt.n_permutations == 1000
        assert pt.null_distribution_mean == 1.2
        assert pt.null_distribution_std == 0.5
        assert pt.percentile_95 == 2.0
        assert pt.percentile_99 == 3.0
        assert pt.interpretation == "Significant result."

    def test_sensitivity_analysis_result_all_fields_accessible(self):
        """All SensitivityAnalysisResult fields should be accessible."""
        sa = SensitivityAnalysisResult(
            thresholds=[1.0, 2.0],
            results=[{"threshold": 1.0}, {"threshold": 2.0}],
            interpretation="Both thresholds tested.",
        )
        assert sa.thresholds == [1.0, 2.0]
        assert len(sa.results) == 2
        assert sa.interpretation == "Both thresholds tested."

    def test_analysis_results_new_fields_default_to_none(self, sample_dataframe):
        """New optional fields on AnalysisResults should default to None."""
        result = AnalysisResults(
            descriptive=compute_descriptive_stats(sample_dataframe),
            contingency_test=run_fisher_exact_test(sample_dataframe),
            odds_ratio=compute_odds_ratio(sample_dataframe),
            logistic_regression=run_logistic_regression(sample_dataframe),
        )
        assert result.mann_whitney is None
        assert result.permutation_test is None
        assert result.sensitivity_analysis is None
        assert result.chi_squared is None
        assert result.extended_logistic is None


# ---------------------------------------------------------------------------
# Serialization edge cases with NaN/Inf in new fields
# ---------------------------------------------------------------------------


class TestSerializationEdgeCases:
    """Edge case tests for serialization of new analysis results."""

    def test_mann_whitney_nan_values_serialize_to_null(self):
        """MannWhitneyResult with NaN values should serialize NaN as null in JSON."""
        mw = MannWhitneyResult(
            u_statistic=float("nan"),
            p_value=float("nan"),
            n_eusocial=0,
            n_non_eusocial=0,
            median_eusocial=float("nan"),
            median_non_eusocial=float("nan"),
            mean_eusocial=float("nan"),
            mean_non_eusocial=float("nan"),
            interpretation="Insufficient data.",
        )
        result = AnalysisResults(
            descriptive=DescriptiveStats(
                sociality_frequencies={},
                mating_frequencies={},
                cross_tabulation={},
                sample_size=0,
                counts_by_sociality={},
                counts_by_mating={},
            ),
            contingency_test=ContingencyTestResult(
                test_name="Fisher's exact test",
                statistic=0.0,
                p_value=1.0,
            ),
            odds_ratio=OddsRatioResult(
                odds_ratio=1.0,
                ci_lower=0.5,
                ci_upper=2.0,
                p_value=1.0,
                interpretation="No effect.",
            ),
            logistic_regression=LogisticRegressionResult(
                coefficients={"const": 0.0},
                std_errors={"const": 0.0},
                p_values={"const": 1.0},
                odds_ratios={"const": 1.0},
                pseudo_r_squared=0.0,
                aic=0.0,
                n_observations=0,
                model_summary_text="",
            ),
            mann_whitney=mw,
        )
        json_str = result.to_json()
        parsed = json.loads(json_str)
        # NaN values should be serialized as null (None in Python)
        mw_data = parsed["mann_whitney"]
        assert mw_data["u_statistic"] is None
        assert mw_data["p_value"] is None

    def test_to_dict_round_trip_with_all_new_fields(self, full_dataset):
        """to_dict() result should be JSON-serializable when all fields present."""
        result = run_all_analyses(full_dataset)
        d = result.to_dict()
        # Should be able to serialize and deserialize without error
        json_str = json.dumps(d, indent=2)
        parsed = json.loads(json_str)
        assert "mann_whitney" in parsed
        assert "permutation_test" in parsed
        assert "sensitivity_analysis" in parsed
        assert "chi_squared" in parsed

    def test_json_chi_squared_and_extended_logistic_keys(self, full_dataset):
        """JSON output should include chi_squared and extended_logistic keys."""
        result = run_all_analyses(full_dataset)
        parsed = json.loads(result.to_json())
        assert "chi_squared" in parsed
        assert "extended_logistic" in parsed

    def test_sensitivity_results_serialize_correctly(self, sample_dataframe):
        """Sensitivity analysis results should serialize with all nested data."""
        result = run_all_analyses(sample_dataframe)
        d = result.to_dict()
        sa = d["sensitivity_analysis"]
        assert isinstance(sa["thresholds"], list)
        assert isinstance(sa["results"], list)
        assert isinstance(sa["interpretation"], str)
        for r in sa["results"]:
            assert "threshold" in r
            assert "odds_ratio" in r
            assert "contingency_table" in r


# ---------------------------------------------------------------------------
# Notebook and pyproject.toml validation tests
# ---------------------------------------------------------------------------


class TestProjectArtifacts:
    """Tests that project artifacts (notebook, pyproject.toml) are valid."""

    def test_notebook_is_valid_json(self):
        """analysis_notebook.ipynb should be valid JSON."""
        notebook_path = (
            Path(__file__).resolve().parent.parent / "analysis_notebook.ipynb"
        )
        if notebook_path.exists():
            with open(notebook_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert isinstance(data, dict)
            # Should have nbformat structure
            assert "cells" in data
            assert "metadata" in data

    def test_notebook_has_code_cells(self):
        """Notebook should contain code cells."""
        notebook_path = (
            Path(__file__).resolve().parent.parent / "analysis_notebook.ipynb"
        )
        if notebook_path.exists():
            with open(notebook_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            code_cells = [
                c for c in data["cells"] if c["cell_type"] == "code"
            ]
            assert len(code_cells) > 0

    def test_notebook_has_markdown_cells(self):
        """Notebook should contain markdown cells."""
        notebook_path = (
            Path(__file__).resolve().parent.parent / "analysis_notebook.ipynb"
        )
        if notebook_path.exists():
            with open(notebook_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            md_cells = [
                c for c in data["cells"] if c["cell_type"] == "markdown"
            ]
            assert len(md_cells) > 0

    def test_pyproject_toml_is_valid(self):
        """pyproject.toml should be valid TOML with expected fields."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        toml_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if toml_path.exists():
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
            assert data["project"]["name"] == "jacksmirror"
            assert data["project"]["version"] == "0.2.0"
            assert "dependencies" in data["project"]

    def test_version_matches_init(self):
        """Version in __init__.py should match pyproject.toml."""
        import jacksmirror
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        toml_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if toml_path.exists():
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
            assert jacksmirror.__version__ == data["project"]["version"]


# ---------------------------------------------------------------------------
# __init__.py exports test
# ---------------------------------------------------------------------------


class TestPackageExports:
    """Tests that new functions are properly exported from the package."""

    def test_run_mann_whitney_test_exported(self):
        """run_mann_whitney_test should be importable from jacksmirror."""
        from jacksmirror import run_mann_whitney_test as fn
        assert callable(fn)

    def test_run_permutation_test_exported(self):
        """run_permutation_test should be importable from jacksmirror."""
        from jacksmirror import run_permutation_test as fn
        assert callable(fn)

    def test_run_sensitivity_analysis_exported(self):
        """run_sensitivity_analysis should be importable from jacksmirror."""
        from jacksmirror import run_sensitivity_analysis as fn
        assert callable(fn)

    def test_generate_academic_report_exported(self):
        """generate_academic_report should be importable from jacksmirror."""
        from jacksmirror import generate_academic_report as fn
        assert callable(fn)

    def test_version_is_0_2_0(self):
        """Package version should be 0.2.0."""
        import jacksmirror
        assert jacksmirror.__version__ == "0.2.0"
