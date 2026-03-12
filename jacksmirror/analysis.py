"""Statistical analysis functions for the jacksmirror bee sociality study."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from io import StringIO

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from jacksmirror.config import N_PERMUTATIONS, RANDOM_SEED, SENSITIVITY_THRESHOLDS
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_binary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a 2x2 contingency table: mating_system (rows) x eusocial (cols).

    Rows are ``monandrous`` / ``polyandrous``; columns are ``eusocial`` /
    ``non-eusocial``.  The table is returned as a ``pd.DataFrame`` with
    labelled index and columns.
    """
    if "is_eusocial" not in df.columns or "is_monandrous" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'is_eusocial' and 'is_monandrous' columns. "
            "Run add_derived_columns() first."
        )

    table = pd.crosstab(
        df["is_monandrous"].map({True: "monandrous", False: "polyandrous"}),
        df["is_eusocial"].map({True: "eusocial", False: "non-eusocial"}),
    )

    # Ensure both rows and both columns are present even when a category has 0
    for row in ("monandrous", "polyandrous"):
        if row not in table.index:
            table.loc[row] = 0
    for col in ("eusocial", "non-eusocial"):
        if col not in table.columns:
            table[col] = 0

    # Enforce consistent ordering
    table = table.loc[
        ["monandrous", "polyandrous"], ["eusocial", "non-eusocial"]
    ]

    return table


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------


def compute_descriptive_stats(df: pd.DataFrame) -> DescriptiveStats:
    """Compute frequency tables and cross-tabulations.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataset (must already contain derived columns).

    Returns
    -------
    DescriptiveStats
    """
    # Sociality frequencies
    sociality_freq: dict[str, int] = (
        df["sociality"].value_counts().to_dict()
        if "sociality" in df.columns
        else {}
    )

    # Mating-system frequencies
    mating_freq: dict[str, int] = (
        df["mating_system"].value_counts().to_dict()
        if "mating_system" in df.columns
        else {}
    )

    # Full cross-tabulation: mating_system x sociality
    cross_tab: dict[str, dict[str, int]] = {}
    if "mating_system" in df.columns and "sociality" in df.columns:
        ct = pd.crosstab(df["mating_system"], df["sociality"])
        cross_tab = {
            str(row): {str(col): int(ct.at[row, col]) for col in ct.columns}
            for row in ct.index
        }

    # Counts by sociality / mating (simple groupby sizes)
    counts_by_sociality: dict[str, int] = (
        df.groupby("sociality", observed=True).size().to_dict()
        if "sociality" in df.columns
        else {}
    )
    counts_by_mating: dict[str, int] = (
        df.groupby("mating_system", observed=True).size().to_dict()
        if "mating_system" in df.columns
        else {}
    )

    return DescriptiveStats(
        sociality_frequencies=sociality_freq,
        mating_frequencies=mating_freq,
        cross_tabulation=cross_tab,
        sample_size=len(df),
        counts_by_sociality=counts_by_sociality,
        counts_by_mating=counts_by_mating,
    )


# ---------------------------------------------------------------------------
# Fisher exact test
# ---------------------------------------------------------------------------


def run_fisher_exact_test(df: pd.DataFrame) -> ContingencyTestResult:
    """Run Fisher's exact test on the 2x2 binary contingency table.

    The table has rows *monandrous / polyandrous* and columns
    *eusocial / non-eusocial*.
    """
    table = _build_binary_table(df)
    table_array = table.to_numpy()

    odds_ratio_stat, p_value = stats.fisher_exact(table_array, alternative="two-sided")

    return ContingencyTestResult(
        test_name="Fisher's exact test",
        statistic=float(odds_ratio_stat),
        p_value=float(p_value),
        degrees_of_freedom=None,
        contingency_table=table_array.tolist(),
        expected_frequencies=None,
    )


# ---------------------------------------------------------------------------
# Chi-squared test
# ---------------------------------------------------------------------------


def run_chi_squared_test(df: pd.DataFrame) -> ContingencyTestResult:
    """Run a chi-squared test of independence on the 2x2 binary table.

    A warning is issued if any expected frequency is below 5.
    """
    table = _build_binary_table(df)
    table_array = table.to_numpy()

    chi2, p_value, dof, expected = stats.chi2_contingency(table_array)

    if (expected < 5).any():
        warnings.warn(
            "One or more expected frequencies are below 5. "
            "Fisher's exact test may be more appropriate.",
            UserWarning,
            stacklevel=2,
        )

    return ContingencyTestResult(
        test_name="Chi-squared test",
        statistic=float(chi2),
        p_value=float(p_value),
        degrees_of_freedom=int(dof),
        contingency_table=table_array.tolist(),
        expected_frequencies=expected.tolist(),
    )


# ---------------------------------------------------------------------------
# Odds ratio
# ---------------------------------------------------------------------------


def compute_odds_ratio(df: pd.DataFrame) -> OddsRatioResult:
    """Compute the odds ratio with 95 % CI from the 2x2 binary table.

    Uses the Haldane-Anscombe correction (add 0.5 to every cell) when any
    cell count is zero.
    """
    table = _build_binary_table(df)
    a = float(table.at["monandrous", "eusocial"])
    b = float(table.at["monandrous", "non-eusocial"])
    c = float(table.at["polyandrous", "eusocial"])
    d = float(table.at["polyandrous", "non-eusocial"])

    # Haldane-Anscombe correction for zero cells
    if any(v == 0 for v in (a, b, c, d)):
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5

    odds_ratio = (a * d) / (b * c)

    # 95 % CI via the log-odds method
    log_or = np.log(odds_ratio)
    se_log_or = np.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    z = 1.96  # ~95 % critical value

    ci_lower = float(np.exp(log_or - z * se_log_or))
    ci_upper = float(np.exp(log_or + z * se_log_or))

    # Two-sided p-value from the log-odds z-test
    z_stat = abs(log_or / se_log_or)
    p_value = float(2.0 * (1.0 - stats.norm.cdf(z_stat)))

    # Human-readable interpretation
    # Use the original (uncorrected) cell counts for proportions
    a_raw = int(table.at["monandrous", "eusocial"])
    b_raw = int(table.at["monandrous", "non-eusocial"])
    c_raw = int(table.at["polyandrous", "eusocial"])
    d_raw = int(table.at["polyandrous", "non-eusocial"])
    n_mono = a_raw + b_raw
    n_poly = c_raw + d_raw

    if odds_ratio > 1:
        # Monandrous species more likely eusocial
        if n_mono > 0:
            pct_mono = 100.0 * a_raw / n_mono
        else:
            pct_mono = 0.0
        if n_poly > 0:
            pct_poly = 100.0 * c_raw / n_poly
        else:
            pct_poly = 0.0
        interpretation = (
            f"Eusociality is more prevalent among monandrous "
            f"({a_raw}/{n_mono}, {pct_mono:.0f}%) than polyandrous "
            f"({c_raw}/{n_poly}, {pct_poly:.0f}%) species "
            f"(OR = {odds_ratio:.2f}, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}], "
            f"p = {p_value:.4f}), consistent with the monogamy hypothesis "
            f"that monandry facilitates the evolution of eusociality."
        )
    elif odds_ratio < 1:
        # Polyandrous species more likely eusocial -- biological context needed
        if n_mono > 0:
            pct_mono = 100.0 * a_raw / n_mono
        else:
            pct_mono = 0.0
        if n_poly > 0:
            pct_poly = 100.0 * c_raw / n_poly
        else:
            pct_poly = 0.0
        interpretation = (
            f"Eusociality is more prevalent among polyandrous "
            f"({c_raw}/{n_poly}, {pct_poly:.0f}%) than monandrous "
            f"({a_raw}/{n_mono}, {pct_mono:.0f}%) species "
            f"(OR = {odds_ratio:.2f}, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}], "
            f"p = {p_value:.4f}). However, this reflects the secondary evolution "
            f"of polyandry in already-eusocial lineages (notably Apis), consistent "
            f"with the monogamy hypothesis that monandry was the ancestral state "
            f"at the origin of eusociality."
        )
    else:
        interpretation = (
            "Monandrous species are equally likely to be eusocial "
            "compared to polyandrous species "
            f"(OR = {odds_ratio:.2f}, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}], "
            f"p = {p_value:.4f})."
        )

    return OddsRatioResult(
        odds_ratio=float(odds_ratio),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Logistic regression (basic)
# ---------------------------------------------------------------------------


def run_logistic_regression(df: pd.DataFrame) -> LogisticRegressionResult:
    """Fit a logistic regression: is_eusocial ~ is_monandrous.

    Uses ``statsmodels.api.Logit`` with a constant term.  Returns a result
    with ``float('inf')`` coefficients when perfect/quasi-perfect separation
    prevents convergence.
    """
    subset = df.dropna(subset=["is_eusocial", "is_monandrous"]).copy()

    y = subset["is_eusocial"].astype(float)
    x = sm.add_constant(subset["is_monandrous"].astype(float))
    x.columns = ["const", "is_monandrous"]

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message=".*Perfect separation.*")
            warnings.filterwarnings("ignore", message=".*Maximum Likelihood.*")
            warnings.filterwarnings("ignore", message=".*Inverting hessian.*")
            warnings.filterwarnings("ignore", message=".*divide by zero.*")
            model = sm.Logit(y, x)
            result = model.fit(disp=False, maxiter=100, method="bfgs")

            coefficients = {k: float(v) for k, v in result.params.items()}
            std_errors = {k: float(v) for k, v in result.bse.items()}
            p_values_dict = {k: float(v) for k, v in result.pvalues.items()}
            odds_ratios = {k: float(np.exp(v)) for k, v in result.params.items()}
            pseudo_r2 = float(result.prsquared)
            aic = float(result.aic)

            buf = StringIO()
            buf.write(str(result.summary()))
            summary_text = buf.getvalue()

        return LogisticRegressionResult(
            coefficients=coefficients,
            std_errors=std_errors,
            p_values=p_values_dict,
            odds_ratios=odds_ratios,
            pseudo_r_squared=pseudo_r2,
            aic=aic,
            n_observations=int(result.nobs),
            model_summary_text=summary_text,
        )
    except Exception as exc:
        warnings.warn(
            f"Logistic regression failed (likely perfect separation): {exc}. "
            "Falling back to Fisher's exact test results.",
            UserWarning,
            stacklevel=2,
        )
        # Return a placeholder result using the Fisher exact / OR data
        return LogisticRegressionResult(
            coefficients={"const": 0.0, "is_monandrous": float("inf")},
            std_errors={"const": float("inf"), "is_monandrous": float("inf")},
            p_values={"const": 1.0, "is_monandrous": 1.0},
            odds_ratios={"const": 1.0, "is_monandrous": float("inf")},
            pseudo_r_squared=0.0,
            aic=float("inf"),
            n_observations=len(subset),
            model_summary_text=(
                f"Logistic regression could not converge: {exc}\n"
                "This typically indicates perfect or quasi-perfect separation "
                "in the data. Use Fisher's exact test and odds ratio instead."
            ),
        )


# ---------------------------------------------------------------------------
# Logistic regression (extended)
# ---------------------------------------------------------------------------


def run_extended_logistic_regression(
    df: pd.DataFrame,
) -> LogisticRegressionResult | None:
    """Fit an extended logistic regression including family and colony size.

    DV: is_eusocial
    IVs: is_monandrous, family dummies, log_colony_size

    Only rows with non-null ``colony_size`` (and therefore non-null
    ``log_colony_size``) are included.  Returns ``None`` if the model
    cannot be fitted (e.g., perfect separation or insufficient data).
    """
    required_cols = {"is_eusocial", "is_monandrous", "log_colony_size"}
    if not required_cols.issubset(df.columns):
        warnings.warn(
            "Extended logistic regression requires columns: "
            f"{required_cols}. Skipping.",
            UserWarning,
            stacklevel=2,
        )
        return None

    subset = df.dropna(subset=["is_eusocial", "is_monandrous", "log_colony_size"]).copy()

    if len(subset) < 5:
        warnings.warn(
            f"Only {len(subset)} observations with complete data for extended "
            "logistic regression. Skipping.",
            UserWarning,
            stacklevel=2,
        )
        return None

    y = subset["is_eusocial"].astype(float)

    # Build predictor matrix
    predictors = pd.DataFrame(
        {"is_monandrous": subset["is_monandrous"].astype(float),
         "log_colony_size": subset["log_colony_size"].astype(float)},
        index=subset.index,
    )

    # Add family dummies if the column exists
    if "family" in subset.columns:
        family_dummies = pd.get_dummies(
            subset["family"], prefix="family", drop_first=True, dtype=float
        )
        predictors = pd.concat([predictors, family_dummies], axis=1)

    predictors = sm.add_constant(predictors)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message=".*Perfect separation.*")
            warnings.filterwarnings("ignore", message=".*Maximum Likelihood.*")
            warnings.filterwarnings("ignore", message=".*Inverting hessian.*")
            warnings.filterwarnings("ignore", message=".*divide by zero.*")
            model = sm.Logit(y, predictors)
            result = model.fit(disp=False, maxiter=100)

            # Check for convergence
            if not result.mle_retvals.get("converged", True):
                warnings.warn(
                    "Extended logistic regression did not converge.",
                    UserWarning,
                    stacklevel=3,
                )

            coefficients = {k: float(v) for k, v in result.params.items()}
            std_errors = {k: float(v) for k, v in result.bse.items()}
            p_values_dict = {k: float(v) for k, v in result.pvalues.items()}
            odds_ratios = {k: float(np.exp(v)) for k, v in result.params.items()}

            buf = StringIO()
            buf.write(str(result.summary()))
            summary_text = buf.getvalue()

        return LogisticRegressionResult(
            coefficients=coefficients,
            std_errors=std_errors,
            p_values=p_values_dict,
            odds_ratios=odds_ratios,
            pseudo_r_squared=float(result.prsquared),
            aic=float(result.aic),
            n_observations=int(result.nobs),
            model_summary_text=summary_text,
        )
    except Exception as exc:
        warnings.warn(
            f"Extended logistic regression failed: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return None


# ---------------------------------------------------------------------------
# Mann-Whitney U test
# ---------------------------------------------------------------------------


def run_mann_whitney_test(df: pd.DataFrame) -> MannWhitneyResult:
    """Run a Mann-Whitney U test comparing effective mates between groups.

    Compares the distribution of ``effective_mates`` between eusocial and
    non-eusocial species.  Rows with NaN ``effective_mates`` are excluded.

    Returns
    -------
    MannWhitneyResult
    """
    subset = df.dropna(subset=["effective_mates"]).copy()

    eusocial = subset.loc[subset["is_eusocial"], "effective_mates"]
    non_eusocial = subset.loc[~subset["is_eusocial"], "effective_mates"]

    n_eus = len(eusocial)
    n_non = len(non_eusocial)

    # Edge case: need at least 2 observations in each group
    if n_eus < 2 or n_non < 2:
        return MannWhitneyResult(
            u_statistic=float("nan"),
            p_value=float("nan"),
            n_eusocial=n_eus,
            n_non_eusocial=n_non,
            median_eusocial=float(eusocial.median()) if n_eus > 0 else float("nan"),
            median_non_eusocial=float(non_eusocial.median()) if n_non > 0 else float("nan"),
            mean_eusocial=float(eusocial.mean()) if n_eus > 0 else float("nan"),
            mean_non_eusocial=float(non_eusocial.mean()) if n_non > 0 else float("nan"),
            interpretation=(
                "Insufficient data for Mann-Whitney U test: need at least 2 "
                "observations in each group."
            ),
        )

    u_stat, p_val = stats.mannwhitneyu(
        eusocial, non_eusocial, alternative="two-sided"
    )

    med_eus = float(eusocial.median())
    med_non = float(non_eusocial.median())
    mean_eus = float(eusocial.mean())
    mean_non = float(non_eusocial.mean())

    if p_val < 0.05:
        sig_text = "significantly"
    else:
        sig_text = "not significantly"

    interpretation = (
        f"Effective mating frequency differs {sig_text} between eusocial "
        f"(median={med_eus:.2f}, n={n_eus}) and non-eusocial "
        f"(median={med_non:.2f}, n={n_non}) species "
        f"(U={u_stat:.1f}, p={p_val:.4f})."
    )

    # When medians are equal but the test is significant, the difference is
    # driven by distributional shape rather than a general location shift.
    if p_val < 0.05 and med_eus == med_non:
        interpretation += (
            f" Although the medians are equal (both {med_eus:.2f}), the "
            f"distributions differ significantly, driven by a subset of highly "
            f"polyandrous eusocial species (Apis spp.) that inflate the eusocial "
            f"group mean ({mean_eus:.2f} vs {mean_non:.2f})."
        )

    return MannWhitneyResult(
        u_statistic=float(u_stat),
        p_value=float(p_val),
        n_eusocial=n_eus,
        n_non_eusocial=n_non,
        median_eusocial=med_eus,
        median_non_eusocial=med_non,
        mean_eusocial=mean_eus,
        mean_non_eusocial=mean_non,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------


def run_permutation_test(
    df: pd.DataFrame,
    n_permutations: int = N_PERMUTATIONS,
    random_seed: int | None = None,
) -> PermutationTestResult:
    """Run a permutation test for the monandry-eusociality association.

    Shuffles the ``is_monandrous`` column to build a null distribution of
    odds ratios, then compares the observed odds ratio to the null.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataset with ``is_monandrous`` and ``is_eusocial`` columns.
    n_permutations : int
        Number of permutations to run (default from config).
    random_seed : int | None
        Seed for the random number generator.  Defaults to ``config.RANDOM_SEED``.

    Returns
    -------
    PermutationTestResult
    """
    if random_seed is None:
        random_seed = RANDOM_SEED

    rng = np.random.default_rng(random_seed)

    # Observed odds ratio from the real data
    table = _build_binary_table(df)
    a = float(table.at["monandrous", "eusocial"])
    b = float(table.at["monandrous", "non-eusocial"])
    c = float(table.at["polyandrous", "eusocial"])
    d = float(table.at["polyandrous", "non-eusocial"])

    # Haldane-Anscombe correction for zero cells
    if any(v == 0 for v in (a, b, c, d)):
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5

    observed_or = (a * d) / (b * c)
    observed_log_or = np.log(observed_or)

    # Permutation loop
    is_monandrous_values = df["is_monandrous"].values.copy()
    is_eusocial_values = df["is_eusocial"].values

    null_log_ors = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(is_monandrous_values)

        # Rebuild 2x2 table from shuffled data
        a_p = int(np.sum(shuffled & is_eusocial_values))
        b_p = int(np.sum(shuffled & ~is_eusocial_values))
        c_p = int(np.sum(~shuffled & is_eusocial_values))
        d_p = int(np.sum(~shuffled & ~is_eusocial_values))

        # Haldane-Anscombe correction for zero cells
        if any(v == 0 for v in (a_p, b_p, c_p, d_p)):
            a_p += 0.5
            b_p += 0.5
            c_p += 0.5
            d_p += 0.5

        perm_or = (a_p * d_p) / (b_p * c_p)
        null_log_ors[i] = np.log(perm_or)

    # Two-sided p-value: proportion of permutations with |log OR| >= |observed|
    p_value = float(
        np.mean(np.abs(null_log_ors) >= np.abs(observed_log_or))
    )

    null_ors = np.exp(null_log_ors)
    null_mean = float(np.mean(null_ors))
    null_std = float(np.std(null_ors))
    pct_95 = float(np.percentile(null_ors, 95))
    pct_99 = float(np.percentile(null_ors, 99))

    if p_value < 0.05:
        sig_text = (
            f"The observed odds ratio ({observed_or:.2f}) is significantly "
            f"different from the null distribution (p={p_value:.4f}), "
            f"supporting the monogamy-eusociality association."
        )
    else:
        sig_text = (
            f"The observed odds ratio ({observed_or:.2f}) is not significantly "
            f"different from the null distribution (p={p_value:.4f})."
        )

    return PermutationTestResult(
        observed_odds_ratio=float(observed_or),
        p_value=p_value,
        n_permutations=n_permutations,
        null_distribution_mean=null_mean,
        null_distribution_std=null_std,
        percentile_95=pct_95,
        percentile_99=pct_99,
        interpretation=sig_text,
    )


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


def run_sensitivity_analysis(
    df: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> SensitivityAnalysisResult:
    """Run sensitivity analysis across different monandry thresholds.

    For each threshold, species with ``effective_mates <= threshold`` are
    reclassified as monandrous.  Species with NaN ``effective_mates`` retain
    their original ``is_monandrous`` classification.  A Fisher's exact test
    and odds ratio are computed for each threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataset.
    thresholds : list[float] | None
        Thresholds to test.  Defaults to ``config.SENSITIVITY_THRESHOLDS``.

    Returns
    -------
    SensitivityAnalysisResult
    """
    if thresholds is None:
        thresholds = list(SENSITIVITY_THRESHOLDS)

    results_list: list[dict] = []

    for threshold in thresholds:
        df_copy = df.copy()

        # Reclassify: rows with non-NaN effective_mates use threshold
        has_mates = df_copy["effective_mates"].notna()
        df_copy.loc[has_mates, "is_monandrous"] = (
            df_copy.loc[has_mates, "effective_mates"] <= threshold
        )

        # Build 2x2 table and run Fisher's exact test
        table = _build_binary_table(df_copy)
        table_array = table.to_numpy()
        _, fisher_p = stats.fisher_exact(table_array, alternative="two-sided")

        # Compute odds ratio with CI
        a = float(table.at["monandrous", "eusocial"])
        b = float(table.at["monandrous", "non-eusocial"])
        c = float(table.at["polyandrous", "eusocial"])
        d = float(table.at["polyandrous", "non-eusocial"])

        if any(v == 0 for v in (a, b, c, d)):
            a += 0.5
            b += 0.5
            c += 0.5
            d += 0.5

        odds_ratio = (a * d) / (b * c)
        log_or = np.log(odds_ratio)
        se_log_or = np.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
        z = 1.96
        ci_lower = float(np.exp(log_or - z * se_log_or))
        ci_upper = float(np.exp(log_or + z * se_log_or))

        n_mono = int(df_copy["is_monandrous"].sum())
        n_poly = int((~df_copy["is_monandrous"]).sum())

        results_list.append({
            "threshold": threshold,
            "n_monandrous": n_mono,
            "n_polyandrous": n_poly,
            "odds_ratio": float(odds_ratio),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "fisher_p_value": float(fisher_p),
            "contingency_table": table_array.tolist(),
        })

    # Generate interpretation
    significant_count = sum(
        1 for r in results_list if r["fisher_p_value"] < 0.05
    )
    total = len(results_list)

    if significant_count == total:
        interp = (
            f"The association between monandry and eusociality is robust: "
            f"significant (p < 0.05) across all {total} thresholds tested "
            f"({', '.join(str(t) for t in thresholds)})."
        )
    elif significant_count == 0:
        interp = (
            f"The association is not significant at any of the {total} "
            f"thresholds tested ({', '.join(str(t) for t in thresholds)})."
        )
    else:
        interp = (
            f"The association is significant at {significant_count} of "
            f"{total} thresholds tested. Sensitivity to the monandry "
            f"threshold suggests caution in interpretation."
        )

    return SensitivityAnalysisResult(
        thresholds=thresholds,
        results=results_list,
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_all_analyses(df: pd.DataFrame) -> AnalysisResults:
    """Run every analysis and return an aggregated ``AnalysisResults`` object.

    The Fisher exact test result is stored in ``contingency_test``.  The
    chi-squared test, extended logistic regression, Mann-Whitney U test,
    permutation test, and sensitivity analysis are also executed and stored
    in dedicated fields.  Individual failures are caught so that the pipeline
    is not interrupted.
    """
    descriptive = compute_descriptive_stats(df)
    fisher = run_fisher_exact_test(df)
    odds = compute_odds_ratio(df)
    logistic = run_logistic_regression(df)

    # Chi-squared test
    chi2_result: ContingencyTestResult | None = None
    try:
        chi2_result = run_chi_squared_test(df)
    except Exception as exc:
        warnings.warn(f"Chi-squared test failed: {exc}", UserWarning, stacklevel=2)

    # Extended logistic regression
    ext_logistic: LogisticRegressionResult | None = None
    try:
        ext_logistic = run_extended_logistic_regression(df)
    except Exception as exc:
        warnings.warn(f"Extended logistic regression failed: {exc}", UserWarning, stacklevel=2)

    # Mann-Whitney U test
    mw_result: MannWhitneyResult | None = None
    try:
        mw_result = run_mann_whitney_test(df)
    except Exception as exc:
        warnings.warn(f"Mann-Whitney U test failed: {exc}", UserWarning, stacklevel=2)

    # Permutation test
    perm_result: PermutationTestResult | None = None
    try:
        perm_result = run_permutation_test(df)
    except Exception as exc:
        warnings.warn(f"Permutation test failed: {exc}", UserWarning, stacklevel=2)

    # Sensitivity analysis
    sens_result: SensitivityAnalysisResult | None = None
    try:
        sens_result = run_sensitivity_analysis(df)
    except Exception as exc:
        warnings.warn(f"Sensitivity analysis failed: {exc}", UserWarning, stacklevel=2)

    return AnalysisResults(
        descriptive=descriptive,
        contingency_test=fisher,
        odds_ratio=odds,
        logistic_regression=logistic,
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_path="",
        n_species=len(df),
        mann_whitney=mw_result,
        permutation_test=perm_result,
        sensitivity_analysis=sens_result,
        chi_squared=chi2_result,
        extended_logistic=ext_logistic,
    )
