"""Dataclasses for structured analysis results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass
class DescriptiveStats:
    """Frequency tables and summary counts."""

    sociality_frequencies: dict[str, int]
    mating_frequencies: dict[str, int]
    cross_tabulation: dict[str, dict[str, int]]
    sample_size: int
    counts_by_sociality: dict[str, int]
    counts_by_mating: dict[str, int]


@dataclass
class ContingencyTestResult:
    """Result of a chi-square or Fisher's exact test."""

    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[int] = None
    contingency_table: list[list[int]] = field(default_factory=list)
    expected_frequencies: Optional[list[list[float]]] = None


@dataclass
class OddsRatioResult:
    """Odds-ratio estimate with confidence interval."""

    odds_ratio: float
    ci_lower: float
    ci_upper: float
    p_value: float
    interpretation: str


@dataclass
class LogisticRegressionResult:
    """Summary of a logistic regression model."""

    coefficients: dict[str, float]
    std_errors: dict[str, float]
    p_values: dict[str, float]
    odds_ratios: dict[str, float]
    pseudo_r_squared: float
    aic: float
    n_observations: int
    model_summary_text: str


@dataclass
class MannWhitneyResult:
    """Result of a Mann-Whitney U test comparing effective mates."""

    u_statistic: float
    p_value: float
    n_eusocial: int
    n_non_eusocial: int
    median_eusocial: float
    median_non_eusocial: float
    mean_eusocial: float
    mean_non_eusocial: float
    interpretation: str


@dataclass
class PermutationTestResult:
    """Result of a permutation test for the monandry-eusociality association."""

    observed_odds_ratio: float
    p_value: float
    n_permutations: int
    null_distribution_mean: float
    null_distribution_std: float
    percentile_95: float
    percentile_99: float
    interpretation: str


@dataclass
class SensitivityAnalysisResult:
    """Result of a sensitivity analysis across monandry thresholds."""

    thresholds: list[float]
    results: list[dict]
    interpretation: str


@dataclass
class AnalysisResults:
    """Aggregated container for all analysis outputs."""

    descriptive: DescriptiveStats
    contingency_test: ContingencyTestResult
    odds_ratio: OddsRatioResult
    logistic_regression: LogisticRegressionResult
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    dataset_path: str = ""
    n_species: int = 0
    mann_whitney: MannWhitneyResult | None = None
    permutation_test: PermutationTestResult | None = None
    sensitivity_analysis: SensitivityAnalysisResult | None = None
    chi_squared: ContingencyTestResult | None = None
    extended_logistic: LogisticRegressionResult | None = None

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary of all results."""
        data = asdict(self)
        # Convert any remaining non-serializable types
        return _make_serializable(data)

    def to_json(self, indent: int = 2) -> str:
        """Return a JSON string representation of all results."""
        return json.dumps(self.to_dict(), indent=indent)


def _make_serializable(obj: object) -> object:
    """Recursively convert non-serializable types for JSON output."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, set):
        return sorted(_make_serializable(item) for item in obj)
    if isinstance(obj, float):
        # Handle NaN / Inf which are not valid JSON
        if obj != obj:  # NaN check
            return None
        if obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    return obj
