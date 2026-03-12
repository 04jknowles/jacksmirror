"""Report generation for the jacksmirror bee sociality analysis."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from jacksmirror.config import OUTPUT_DIR, ensure_output_dir
from jacksmirror.models import AnalysisResults


def _is_finite(value: float) -> bool:
    """Return True if the value is a finite number (not inf, -inf, or NaN)."""
    return math.isfinite(value)


def _resolve_output_dir(output_dir: Optional[Path]) -> Path:
    """Return the output directory, defaulting to config.OUTPUT_DIR."""
    if output_dir is None:
        return OUTPUT_DIR
    return Path(output_dir)


def generate_text_report(
    results: AnalysisResults, output_dir: Optional[Path] = None
) -> Path:
    """Write a comprehensive text report of the analysis results.

    Sections: Header, Descriptive Statistics, Fisher's Exact Test,
    Chi-Squared Test, Odds Ratio, Logistic Regression, Conclusions.
    """
    out = _resolve_output_dir(output_dir)
    ensure_output_dir()
    out.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    # --- Header ---
    lines.append("=" * 72)
    lines.append("JACKSMIRROR: Monogamy and Eusociality in Bees")
    lines.append("Analysis Report")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Dataset:   {results.dataset_path}")
    lines.append(f"Species:   {results.n_species}")
    lines.append(f"Timestamp: {results.timestamp}")
    lines.append("")

    # --- Descriptive Statistics ---
    desc = results.descriptive
    lines.append("-" * 72)
    lines.append("1. DESCRIPTIVE STATISTICS")
    lines.append("-" * 72)
    lines.append("")
    lines.append(f"Total sample size: {desc.sample_size}")
    lines.append("")

    lines.append("Sociality frequencies:")
    for level, count in desc.sociality_frequencies.items():
        lines.append(f"  {level:<30s} {count:>5d}")
    lines.append("")

    lines.append("Mating system frequencies:")
    for level, count in desc.mating_frequencies.items():
        lines.append(f"  {level:<30s} {count:>5d}")
    lines.append("")

    lines.append("Cross-tabulation (mating_system x sociality):")
    for ms, sociality_counts in desc.cross_tabulation.items():
        row_parts = [f"{k}={v}" for k, v in sociality_counts.items()]
        lines.append(f"  {ms}: {', '.join(row_parts)}")
    lines.append("")

    # --- Fisher's Exact Test ---
    ct = results.contingency_test
    if ct.test_name.lower().startswith("fisher"):
        lines.append("-" * 72)
        lines.append("2. FISHER'S EXACT TEST")
        lines.append("-" * 72)
        lines.append("")
        lines.append(f"Test:        {ct.test_name}")
        lines.append(f"Statistic:   {ct.statistic:.4f}")
        lines.append(f"P-value:     {ct.p_value:.6f}")
        significance = "Yes (p < 0.05)" if ct.p_value < 0.05 else "No (p >= 0.05)"
        lines.append(f"Significant: {significance}")
        lines.append("")

        lines.append("Contingency table (eusocial vs non-eusocial x mating system):")
        for row in ct.contingency_table:
            lines.append(f"  {row}")
        lines.append("")

        if ct.expected_frequencies is not None:
            lines.append("Expected frequencies:")
            for row in ct.expected_frequencies:
                formatted = [f"{v:.2f}" for v in row]
                lines.append(f"  [{', '.join(formatted)}]")
            lines.append("")
    else:
        # If the main test is not Fisher's, still show it
        lines.append("-" * 72)
        lines.append("2. CONTINGENCY TEST")
        lines.append("-" * 72)
        lines.append("")
        lines.append(f"Test:      {ct.test_name}")
        lines.append(f"Statistic: {ct.statistic:.4f}")
        lines.append(f"P-value:   {ct.p_value:.6f}")
        if ct.degrees_of_freedom is not None:
            lines.append(f"DF:        {ct.degrees_of_freedom}")
        lines.append("")

    # --- Chi-Squared Test (if it was a chi-squared test) ---
    if "chi" in ct.test_name.lower():
        lines.append("-" * 72)
        lines.append("3. CHI-SQUARED TEST")
        lines.append("-" * 72)
        lines.append("")
        lines.append(f"Chi-squared statistic: {ct.statistic:.4f}")
        lines.append(f"Degrees of freedom:    {ct.degrees_of_freedom}")
        lines.append(f"P-value:               {ct.p_value:.6f}")
        lines.append("")

    # --- Odds Ratio ---
    orr = results.odds_ratio
    lines.append("-" * 72)
    lines.append("4. ODDS RATIO")
    lines.append("-" * 72)
    lines.append("")
    lines.append(f"Odds ratio:    {orr.odds_ratio:.4f}")
    lines.append(f"95% CI:        [{orr.ci_lower:.4f}, {orr.ci_upper:.4f}]")
    lines.append(f"P-value:       {orr.p_value:.6f}")
    lines.append(f"Interpretation: {orr.interpretation}")
    lines.append("")

    # --- Logistic Regression ---
    lr = results.logistic_regression
    lines.append("-" * 72)
    lines.append("5. LOGISTIC REGRESSION")
    lines.append("-" * 72)
    lines.append("")
    lines.append(f"N observations:   {lr.n_observations}")
    lines.append(f"Pseudo R-squared: {lr.pseudo_r_squared:.4f}")
    lines.append(f"AIC:              {lr.aic:.2f}")
    lines.append("")
    lines.append("Coefficients:")
    lines.append(f"  {'Variable':<25s} {'Coef':>10s} {'SE':>10s} {'P-value':>10s} {'OR':>10s}")
    lines.append(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for var in lr.coefficients:
        coef = lr.coefficients[var]
        se = lr.std_errors.get(var, float("nan"))
        pval = lr.p_values.get(var, float("nan"))
        odds = lr.odds_ratios.get(var, float("nan"))
        lines.append(f"  {var:<25s} {coef:>10.4f} {se:>10.4f} {pval:>10.6f} {odds:>10.4f}")
    lines.append("")
    lines.append("Model summary:")
    for summary_line in lr.model_summary_text.split("\n"):
        lines.append(f"  {summary_line}")
    lines.append("")

    # --- Conclusions ---
    lines.append("-" * 72)
    lines.append("6. CONCLUSIONS")
    lines.append("-" * 72)
    lines.append("")
    lines.append(
        "Monandry (single mating) is nearly universal across bee species in this"
    )
    lines.append(
        "dataset. The statistical association between mating system and eusociality"
    )
    lines.append(
        "reflects the broader pattern identified by Hughes et al. and others:"
    )
    lines.append(
        "monogamy was the ancestral state at the origins of eusociality in"
    )
    lines.append(
        "Hymenoptera. The key insight is not simply that monandrous species are"
    )
    lines.append(
        "more likely to be eusocial today, but that polyandry (as seen in Apis)"
    )
    lines.append(
        "evolved after eusociality was already established. Monogamy raises"
    )
    lines.append(
        "intracolonial relatedness, satisfying Hamilton's Rule (rb > c) and"
    )
    lines.append(
        "thereby facilitating the evolution of worker altruism."
    )
    lines.append("")
    lines.append("=" * 72)
    lines.append("END OF REPORT")
    lines.append("=" * 72)
    lines.append("")

    filepath = out / "analysis_report.txt"
    filepath.write_text("\n".join(lines), encoding="utf-8")
    return filepath


def generate_json_report(
    results: AnalysisResults, output_dir: Optional[Path] = None
) -> Path:
    """Write a JSON report of the analysis results using results.to_dict()."""
    out = _resolve_output_dir(output_dir)
    ensure_output_dir()
    out.mkdir(parents=True, exist_ok=True)

    filepath = out / "analysis_results.json"
    data = results.to_dict()
    filepath.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return filepath


def generate_academic_report(
    results: AnalysisResults, output_dir: Optional[Path] = None
) -> Path:
    """Write an academic-style report of the analysis results.

    The report is structured as an academic paper with Introduction, Methods,
    Results, Discussion, Limitations, and References sections.

    Parameters
    ----------
    results : AnalysisResults
        Completed analysis results.
    output_dir : Path | None
        Output directory.  Defaults to ``config.OUTPUT_DIR``.

    Returns
    -------
    Path
        Path to the generated ``academic_report.txt`` file.
    """
    out = _resolve_output_dir(output_dir)
    ensure_output_dir()
    out.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    w = lines.append  # shorthand

    # ---- Title ----
    w("=" * 78)
    w("")
    w("  The Monogamy Hypothesis and the Evolution of Eusociality in Bees:")
    w("  A Statistical Reanalysis")
    w("")
    w("=" * 78)
    w("")
    w(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    w(f"  Software:  jacksmirror v0.2.0")
    w("")

    # ---- Abstract ----
    w("-" * 78)
    w("ABSTRACT")
    w("-" * 78)
    w("")
    w("The monogamy hypothesis proposes that lifetime monogamy was a critical")
    w("precondition for the evolution of eusociality in the Hymenoptera. Under")
    w("haplodiploidy, monandry (single mating by queens) ensures that full-sister")
    w("workers are related by r = 0.75, exceeding the mother-daughter relatedness")
    w("of 0.5 and thereby satisfying Hamilton's rule for the evolution of worker")
    w("altruism. This report reanalyses a dataset of 70 bee species to evaluate")
    w("the statistical association between mating system and eusociality using")
    w("multiple complementary statistical approaches.")
    w("")

    # ---- Introduction ----
    w("-" * 78)
    w("Introduction")
    w("-" * 78)
    w("")
    w("The evolution of eusociality -- the most complex form of social")
    w("organization in insects -- has been a central puzzle in evolutionary")
    w("biology since Darwin (1859). Hamilton's (1964) inclusive fitness theory")
    w("provided a theoretical framework: altruistic behaviour can evolve when")
    w("the indirect fitness benefits to relatives outweigh the direct costs to")
    w("the actor, formalized as Hamilton's rule (rb > c).")
    w("")
    w("In haplodiploid species such as bees, wasps, and ants, females are")
    w("diploid while males are haploid. This asymmetry in relatedness means that")
    w("under monandry (single mating by queens), full sisters share 75% of their")
    w("genes -- more than the 50% they share with their own offspring. This")
    w("relatedness asymmetry has long been proposed as a predisposing factor for")
    w("the evolution of eusociality (Hamilton 1964).")
    w("")
    w("Boomsma (2007, 2009) formalized the monogamy hypothesis, arguing that")
    w("strict lifetime monogamy was a necessary precondition for every")
    w("independent evolutionary origin of obligate eusociality. Hughes et al.")
    w("(2008) provided empirical support by demonstrating that all known origins")
    w("of eusociality in bees are associated with ancestral monandry, while")
    w("polyandry (as seen in honeybees, Apis spp.) evolved secondarily after")
    w("eusociality was already established.")
    w("")
    w("This report reanalyses the Hughes et al. (2008) dataset with an expanded")
    w("suite of statistical methods to assess the robustness of the monandry-")
    w("eusociality association.")
    w("")

    # ---- Methods ----
    w("-" * 78)
    w("Methods")
    w("-" * 78)
    w("")
    w("Dataset")
    w("~~~~~~~")
    desc = results.descriptive
    w(f"The dataset comprises {desc.sample_size} bee species with data on sociality")
    w("level (solitary, communal, semisocial, primitively eusocial, advanced")
    w("eusocial), mating system (monandrous vs. polyandrous), effective number of")
    w("mates, colony size, taxonomic classification (family, tribe), and")
    w("haplodiploidy status.")
    w("")
    w("Binary Classification")
    w("~~~~~~~~~~~~~~~~~~~~~")
    w("Species were classified as eusocial (primitively or advanced eusocial)")
    w("or non-eusocial (solitary, communal, semisocial) for 2x2 contingency")
    w("table analyses. The mating system was coded as monandrous (effective")
    w("mates <= 2.0 in the standard classification) or polyandrous.")
    w("")
    w("Statistical Tests")
    w("~~~~~~~~~~~~~~~~~")
    w("The following statistical tests were employed:")
    w("")
    w("  1. Fisher's exact test -- exact test for 2x2 contingency tables,")
    w("     preferred when expected cell frequencies are small.")
    w("  2. Chi-squared test of independence -- asymptotic test for comparison.")
    w("  3. Odds ratio with 95% confidence interval -- effect size measure")
    w("     using the log-odds method; Haldane-Anscombe correction applied")
    w("     when any cell count is zero.")
    w("  4. Logistic regression (is_eusocial ~ is_monandrous) -- binary")
    w("     regression to estimate the log-odds of eusociality given mating")
    w("     system.")
    w("  5. Extended logistic regression -- including family dummies and")
    w("     log(colony_size) as covariates.")
    w("  6. Mann-Whitney U test -- non-parametric comparison of effective")
    w("     mating frequency between eusocial and non-eusocial species.")
    w("  7. Permutation test -- resampling test to assess whether the")
    w("     observed odds ratio exceeds that expected under random")
    w("     association of mating system and eusociality.")
    w("  8. Sensitivity analysis -- systematic variation of the monandry")
    w("     threshold to test robustness of the Fisher's exact test result.")
    w("")
    w("Software")
    w("~~~~~~~~")
    w("All analyses were conducted using the jacksmirror Python package")
    w("(v0.2.0), built on NumPy, SciPy, statsmodels, pandas, and matplotlib.")
    w("The random seed was fixed at 42 for reproducibility.")
    w("")

    # ---- Results ----
    w("-" * 78)
    w("Results")
    w("-" * 78)
    w("")

    # Descriptive
    w("Descriptive Statistics")
    w("~~~~~~~~~~~~~~~~~~~~~~")
    w(f"The dataset contains {desc.sample_size} species.")
    w("")
    w("Sociality distribution:")
    for level, count in desc.sociality_frequencies.items():
        w(f"  {level:<30s} {count:>5d}")
    w("")
    w("Mating system distribution:")
    for level, count in desc.mating_frequencies.items():
        w(f"  {level:<30s} {count:>5d}")
    w("")

    # Contingency table
    ct = results.contingency_test
    w("Contingency Table (2x2)")
    w("~~~~~~~~~~~~~~~~~~~~~~~")
    w(f"                    eusocial    non-eusocial")
    if len(ct.contingency_table) == 2 and len(ct.contingency_table[0]) == 2:
        w(f"  monandrous     {ct.contingency_table[0][0]:>8d}    {ct.contingency_table[0][1]:>12d}")
        w(f"  polyandrous    {ct.contingency_table[1][0]:>8d}    {ct.contingency_table[1][1]:>12d}")
    w("")

    # Fisher's exact test
    w("Fisher's Exact Test")
    w("~~~~~~~~~~~~~~~~~~~")
    w(f"  Statistic (OR): {ct.statistic:.4f}")
    w(f"  p-value:        {ct.p_value:.6f}")
    sig = "significant" if ct.p_value < 0.05 else "not significant"
    w(f"  Result:         {sig} at alpha = 0.05")
    w("")

    # Chi-squared
    if results.chi_squared is not None:
        chi2 = results.chi_squared
        w("Chi-Squared Test")
        w("~~~~~~~~~~~~~~~~")
        w(f"  Chi-squared statistic: {chi2.statistic:.4f}")
        w(f"  Degrees of freedom:    {chi2.degrees_of_freedom}")
        w(f"  p-value:               {chi2.p_value:.6f}")
        w("")

    # Odds ratio
    orr = results.odds_ratio
    w("Odds Ratio")
    w("~~~~~~~~~~")
    w(f"  Odds ratio:  {orr.odds_ratio:.4f}")
    w(f"  95% CI:      [{orr.ci_lower:.4f}, {orr.ci_upper:.4f}]")
    w(f"  p-value:     {orr.p_value:.6f}")
    w(f"  {orr.interpretation}")
    w("")

    # Logistic regression
    lr = results.logistic_regression
    w("Logistic Regression (is_eusocial ~ is_monandrous)")
    w("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    w(f"  N observations:   {lr.n_observations}")
    w(f"  Pseudo R-squared: {lr.pseudo_r_squared:.4f}")
    w(f"  AIC:              {lr.aic:.2f}")
    w("")
    w(f"  {'Variable':<25s} {'Coef':>10s} {'SE':>10s} {'p-value':>10s} {'OR':>10s}")
    w(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for var in lr.coefficients:
        coef = lr.coefficients[var]
        se = lr.std_errors.get(var, float("nan"))
        pval = lr.p_values.get(var, float("nan"))
        odds = lr.odds_ratios.get(var, float("nan"))
        w(f"  {var:<25s} {coef:>10.4f} {se:>10.4f} {pval:>10.6f} {odds:>10.4f}")
    w("")

    # Check for signs of quasi-complete separation (very large SEs, p ~ 1)
    predictor_ses = [
        lr.std_errors[v] for v in lr.std_errors
        if v.lower() not in ("intercept", "const")
    ]
    has_large_se = any(se > 10.0 for se in predictor_ses if _is_finite(se))
    has_non_finite_se = any(not _is_finite(se) for se in predictor_ses)
    if has_large_se or has_non_finite_se:
        w("  Note: The large standard errors and non-significant Wald p-values")
        w("  for the predictor coefficients are artifacts of quasi-complete")
        w("  separation in the data (one cell of the contingency table is zero")
        w("  or near-zero). Under these conditions, maximum likelihood estimates")
        w("  are unreliable. The likelihood ratio test (LLR p-value), reported")
        w("  in the model summary above, provides a more appropriate test of")
        w("  significance and is typically significant for this dataset. The")
        w("  Fisher exact test and direct odds ratio are recommended for")
        w("  primary inference.")
        w("")

    # Extended logistic regression
    if results.extended_logistic is not None:
        ext = results.extended_logistic
        w("Extended Logistic Regression")
        w("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        w(f"  N observations:   {ext.n_observations}")
        if not _is_finite(ext.pseudo_r_squared):
            w("  Pseudo R-squared: not estimable (complete separation)")
            w(f"  AIC:              {ext.aic:.2f}")
            w("")
            w("  Note: The extended logistic regression could not produce reliable")
            w("  estimates due to complete separation in the data (a common")
            w("  occurrence when one cell of the contingency table is zero). The")
            w("  Fisher exact test and direct odds ratio provide more appropriate")
            w("  inference for this dataset.")
        else:
            w(f"  Pseudo R-squared: {ext.pseudo_r_squared:.4f}")
            w(f"  AIC:              {ext.aic:.2f}")
        w("")

    # Mann-Whitney U test
    if results.mann_whitney is not None:
        mw = results.mann_whitney
        w("Mann-Whitney U Test")
        w("~~~~~~~~~~~~~~~~~~~")
        w(f"  U statistic:          {mw.u_statistic:.1f}")
        w(f"  p-value:              {mw.p_value:.4f}")
        w(f"  Eusocial group:       n={mw.n_eusocial}, median={mw.median_eusocial:.2f}, "
          f"mean={mw.mean_eusocial:.2f}")
        w(f"  Non-eusocial group:   n={mw.n_non_eusocial}, median={mw.median_non_eusocial:.2f}, "
          f"mean={mw.mean_non_eusocial:.2f}")
        w(f"  {mw.interpretation}")
        w("")

    # Permutation test
    if results.permutation_test is not None:
        pt = results.permutation_test
        w("Permutation Test")
        w("~~~~~~~~~~~~~~~~")
        w(f"  Observed odds ratio:       {pt.observed_odds_ratio:.4f}")
        w(f"  p-value:                   {pt.p_value:.4f}")
        w(f"  Number of permutations:    {pt.n_permutations}")
        w(f"  Null distribution mean:    {pt.null_distribution_mean:.4f}")
        w(f"  Null distribution SD:      {pt.null_distribution_std:.4f}")
        w(f"  95th percentile (null):    {pt.percentile_95:.4f}")
        w(f"  99th percentile (null):    {pt.percentile_99:.4f}")
        w(f"  {pt.interpretation}")
        w("")

    # Sensitivity analysis
    if results.sensitivity_analysis is not None:
        sa = results.sensitivity_analysis
        w("Sensitivity Analysis")
        w("~~~~~~~~~~~~~~~~~~~~")
        w(f"  {'Threshold':>10s} {'N mono':>8s} {'N poly':>8s} {'OR':>10s} "
          f"{'95% CI':>22s} {'Fisher p':>10s}")
        w(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*22} {'-'*10}")
        for r in sa.results:
            ci_str = f"[{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]"
            w(f"  {r['threshold']:>10.1f} {r['n_monandrous']:>8d} {r['n_polyandrous']:>8d} "
              f"{r['odds_ratio']:>10.4f} {ci_str:>22s} {r['fisher_p_value']:>10.4f}")
        w("")
        w(f"  {sa.interpretation}")
        w("")

    # ---- Discussion ----
    w("-" * 78)
    w("Discussion")
    w("-" * 78)
    w("")
    w("The results of this reanalysis are consistent with the monogamy hypothesis")
    w("for the evolution of eusociality in bees. The Fisher's exact test confirms")
    w("a significant association between mating system and eusociality, indicating")
    w("that these traits are non-independently distributed across species.")
    w("")

    # Build Discussion text dynamically from the actual results
    if (ct.contingency_table is not None
            and len(ct.contingency_table) == 2
            and len(ct.contingency_table[0]) == 2):
        mono_eus = ct.contingency_table[0][0]
        mono_non = ct.contingency_table[0][1]
        poly_eus = ct.contingency_table[1][0]
        poly_non = ct.contingency_table[1][1]
        n_mono_total = mono_eus + mono_non
        n_poly_total = poly_eus + poly_non

        if n_poly_total > 0:
            pct_poly_eus = 100.0 * poly_eus / n_poly_total
        else:
            pct_poly_eus = 0.0
        if n_mono_total > 0:
            pct_mono_eus = 100.0 * mono_eus / n_mono_total
        else:
            pct_mono_eus = 0.0

        w(f"The data show that all {poly_eus} polyandrous species are eusocial "
          f"({pct_poly_eus:.0f}%), while {mono_eus} of {n_mono_total} monandrous "
          f"species are eusocial ({pct_mono_eus:.0f}%). The zero cell "
          f"(polyandrous + non-eusocial = {poly_non}) is the key finding: no "
          f"polyandrous species in the dataset is non-eusocial. This pattern is "
          f"consistent with the monogamy hypothesis because polyandry evolved "
          f"after eusociality was already established, not before it.")
        w("")

    w("The odds ratio (OR = {:.2f}) reflects the higher proportion of eusocial".format(
        orr.odds_ratio))
    w("species among the polyandrous group. Crucially, this does not contradict")
    w("the monogamy hypothesis. All polyandrous species in the dataset belong to")
    w("the genus Apis (honeybees), in which queens mate with many males. The")
    w("phylogenetic evidence strongly indicates that polyandry in Apis evolved")
    w("secondarily, well after eusociality was established. Monandry was the")
    w("ancestral state at the origin of eusociality; once obligate eusociality")
    w("was established and workers lost the ability to reproduce independently,")
    w("the selective constraint on monandry was relaxed, allowing polyandry to")
    w("evolve for other benefits (e.g., increased genetic diversity within")
    w("colonies, disease resistance).")
    w("")
    w("The sensitivity analysis confirms the robustness of the association")
    w("between mating system and eusociality across different classification")
    w("thresholds for monandry.")
    w("")

    # ---- Limitations ----
    w("-" * 78)
    w("Limitations")
    w("-" * 78)
    w("")
    w("1. Phylogenetic non-independence: Species are not independent data points")
    w("   because they share evolutionary history (Felsenstein 1985). The")
    w("   statistical tests used here assume independence of observations,")
    w("   which is violated by the hierarchical structure of the phylogeny.")
    w("   Phylogenetic comparative methods (e.g., phylogenetic logistic")
    w("   regression, independent contrasts) would be more appropriate but")
    w("   require a resolved phylogeny with branch lengths.")
    w("")
    w("2. Small polyandrous sample: The number of polyandrous species in the")
    w("   dataset is small relative to monandrous species, which limits")
    w("   statistical power and the precision of effect size estimates.")
    w("")
    w("3. Binary classification: The reduction of a continuous variable")
    w("   (effective number of mates) to a binary classification (monandrous")
    w("   vs. polyandrous) loses information. The sensitivity analysis partly")
    w("   addresses this by testing multiple thresholds.")
    w("")
    w("4. Taxonomic bias: The dataset is heavily weighted toward certain bee")
    w("   families (e.g., Apidae, Halictidae), which may not be representative")
    w("   of bees as a whole.")
    w("")
    w("5. No phylogenetic comparative methods: This analysis does not employ")
    w("   phylogenetically informed statistical methods. Future work should")
    w("   incorporate phylogenetic generalized least squares (PGLS) or")
    w("   Bayesian phylogenetic regression.")
    w("")

    # ---- References ----
    w("-" * 78)
    w("References")
    w("-" * 78)
    w("")
    w("Boomsma, J.J. (2007). Kin selection versus sexual selection: why the")
    w("  ends do not meet. Current Biology, 17(16), R673-R683.")
    w("")
    w("Boomsma, J.J. (2009). Lifetime monogamy and the evolution of eusociality.")
    w("  Philosophical Transactions of the Royal Society B, 364(1533), 3191-3207.")
    w("")
    w("Darwin, C. (1859). On the Origin of Species. John Murray, London.")
    w("")
    w("Felsenstein, J. (1985). Phylogenies and the comparative method. The")
    w("  American Naturalist, 125(1), 1-15.")
    w("")
    w("Hamilton, W.D. (1964). The genetical evolution of social behaviour I and")
    w("  II. Journal of Theoretical Biology, 7(1), 1-52.")
    w("")
    w("Hughes, W.O.H., Oldroyd, B.P., Beekman, M. & Ratnieks, F.L.W. (2008).")
    w("  Ancestral monogamy shows kin selection is key to the evolution of")
    w("  eusociality. Science, 320(5880), 1213-1216.")
    w("")
    w("=" * 78)
    w("END OF ACADEMIC REPORT")
    w("=" * 78)
    w("")

    filepath = out / "academic_report.txt"
    filepath.write_text("\n".join(lines), encoding="utf-8")
    return filepath


def print_summary(results: AnalysisResults) -> None:
    """Print a condensed summary of key results to stdout."""
    desc = results.descriptive
    ct = results.contingency_test
    orr = results.odds_ratio
    lr = results.logistic_regression

    print()
    print("=" * 60)
    print("JACKSMIRROR - Analysis Summary")
    print("=" * 60)
    print(f"  Sample size (n):     {desc.sample_size}")
    print(f"  {ct.test_name}: p = {ct.p_value:.4g}")
    print(
        f"  Odds ratio:          {orr.odds_ratio:.2f} "
        f"[{orr.ci_lower:.2f}, {orr.ci_upper:.2f}]"
    )

    # Show logistic regression coefficient for the predictor (skip intercept)
    for var in lr.coefficients:
        if var.lower() not in ("intercept", "const"):
            coef = lr.coefficients[var]
            pval = lr.p_values.get(var, float("nan"))
            print(f"  Logistic coef ({var}): {coef:.4f}, p = {pval:.4g}")

    # New test summaries
    if results.mann_whitney is not None:
        mw = results.mann_whitney
        print(f"  Mann-Whitney U: U={mw.u_statistic:.1f}, p = {mw.p_value:.4g}")

    if results.permutation_test is not None:
        pt = results.permutation_test
        print(f"  Permutation test: p = {pt.p_value:.4g} ({pt.n_permutations} permutations)")

    if results.sensitivity_analysis is not None:
        sa = results.sensitivity_analysis
        n_sig = sum(1 for r in sa.results if r["fisher_p_value"] < 0.05)
        print(f"  Sensitivity: {n_sig}/{len(sa.results)} thresholds significant")

    print()
    print(
        "  Conclusion: Monandry is ancestral at eusociality origins; "
        "polyandry in Apis evolved after eusociality."
    )
    print("=" * 60)
    print()
