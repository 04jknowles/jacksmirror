"""Visualization functions for the jacksmirror bee sociality analysis."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Optional

import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from jacksmirror.config import (
    FIGURE_DPI,
    FIGURE_FORMAT,
    OUTPUT_DIR,
    SOCIALITY_ORDER,
)
from jacksmirror.models import AnalysisResults

# --- Styling ---
plt.style.use("seaborn-v0_8-whitegrid")

# Consistent 5-color palette for sociality levels (ordered solitary -> advanced_eusocial)
SOCIALITY_PALETTE: dict[str, str] = {
    "solitary": "#4c72b0",
    "communal": "#55a868",
    "semisocial": "#c44e52",
    "primitively_eusocial": "#8172b2",
    "advanced_eusocial": "#ccb974",
}

EUSOCIAL_PALETTE: dict[str, str] = {
    "Non-eusocial": "#4c72b0",
    "Eusocial": "#c44e52",
}


def _resolve_output_dir(output_dir: Optional[Path]) -> Path:
    """Return the output directory, defaulting to config.OUTPUT_DIR."""
    if output_dir is None:
        return OUTPUT_DIR
    return Path(output_dir)


def plot_sociality_by_mating_system(
    df: pd.DataFrame, output_dir: Optional[Path] = None
) -> Path:
    """Create a stacked bar chart of sociality distribution by mating system.

    X-axis: mating system (2 bars). Y-axis: proportion (0-1).
    Segments colored by sociality level with count annotations.
    """
    out = _resolve_output_dir(output_dir)

    # Build cross-tabulation and normalize to proportions
    ct = pd.crosstab(df["mating_system"], df["sociality"])
    # Reindex to ensure consistent ordering
    present_sociality = [s for s in SOCIALITY_ORDER if s in ct.columns]
    ct = ct.reindex(columns=present_sociality, fill_value=0)
    counts = ct.copy()
    proportions = ct.div(ct.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))

    mating_systems = proportions.index.tolist()
    x = np.arange(len(mating_systems))
    bar_width = 0.5
    bottoms = np.zeros(len(mating_systems))

    for sociality in present_sociality:
        heights = proportions[sociality].values
        color = SOCIALITY_PALETTE.get(sociality, "#999999")
        bars = ax.bar(
            x, heights, bar_width, bottom=bottoms, label=sociality, color=color
        )

        # Annotate with counts in the center of each segment
        for i, (bar, count_val) in enumerate(
            zip(bars, counts[sociality].values)
        ):
            if count_val > 0:
                segment_center = bottoms[i] + heights[i] / 2
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    segment_center,
                    str(count_val),
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                )

        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels(mating_systems, fontsize=11)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Distribution of Social Organization by Mating System", fontsize=13
    )
    ax.legend(
        title="Sociality",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9,
    )

    fig.tight_layout()
    filepath = out / f"sociality_by_mating_system.{FIGURE_FORMAT}"
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_eusociality_proportion(
    df: pd.DataFrame, output_dir: Optional[Path] = None
) -> Path:
    """Create a grouped bar chart of eusocial vs non-eusocial proportions.

    Includes Wilson confidence interval error bars and annotations.
    """
    out = _resolve_output_dir(output_dir)

    groups: list[dict] = []
    for ms in df["mating_system"].unique():
        subset = df[df["mating_system"] == ms]
        n = len(subset)
        n_eusocial = int(subset["is_eusocial"].sum())
        n_non = n - n_eusocial
        prop_eu = n_eusocial / n if n > 0 else 0.0
        prop_non = n_non / n if n > 0 else 0.0

        ci_low_eu, ci_high_eu = proportion_confint(
            n_eusocial, n, alpha=0.05, method="wilson"
        )
        ci_low_non, ci_high_non = proportion_confint(
            n_non, n, alpha=0.05, method="wilson"
        )

        groups.append(
            {
                "mating_system": ms,
                "Eusocial": prop_eu,
                "Non-eusocial": prop_non,
                "eu_ci_low": ci_low_eu,
                "eu_ci_high": ci_high_eu,
                "non_ci_low": ci_low_non,
                "non_ci_high": ci_high_non,
                "n": n,
                "n_eusocial": n_eusocial,
                "n_non": n_non,
            }
        )

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(groups))
    bar_width = 0.35

    # Non-eusocial bars
    non_props = [g["Non-eusocial"] for g in groups]
    non_err_low = [g["Non-eusocial"] - g["non_ci_low"] for g in groups]
    non_err_high = [g["non_ci_high"] - g["Non-eusocial"] for g in groups]
    bars_non = ax.bar(
        x - bar_width / 2,
        non_props,
        bar_width,
        label="Non-eusocial",
        color=EUSOCIAL_PALETTE["Non-eusocial"],
        yerr=[non_err_low, non_err_high],
        capsize=4,
    )

    # Eusocial bars
    eu_props = [g["Eusocial"] for g in groups]
    eu_err_low = [g["Eusocial"] - g["eu_ci_low"] for g in groups]
    eu_err_high = [g["eu_ci_high"] - g["Eusocial"] for g in groups]
    bars_eu = ax.bar(
        x + bar_width / 2,
        eu_props,
        bar_width,
        label="Eusocial",
        color=EUSOCIAL_PALETTE["Eusocial"],
        yerr=[eu_err_low, eu_err_high],
        capsize=4,
    )

    # Annotate bars with proportion and n
    for i, g in enumerate(groups):
        ax.text(
            i - bar_width / 2,
            g["Non-eusocial"] + (g["non_ci_high"] - g["Non-eusocial"]) + 0.02,
            f'{g["Non-eusocial"]:.2f}\n(n={g["n_non"]})',
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            i + bar_width / 2,
            g["Eusocial"] + (g["eu_ci_high"] - g["Eusocial"]) + 0.02,
            f'{g["Eusocial"]:.2f}\n(n={g["n_eusocial"]})',
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([g["mating_system"] for g in groups], fontsize=11)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Proportion of Eusocial Species by Mating System", fontsize=13
    )
    ax.legend(fontsize=10)

    fig.tight_layout()
    filepath = out / f"eusociality_proportion.{FIGURE_FORMAT}"
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_odds_ratio_forest(
    results: AnalysisResults, output_dir: Optional[Path] = None
) -> Path:
    """Create a forest plot for odds ratio estimates.

    Shows the point estimate (diamond) with confidence interval (horizontal line).
    Vertical dashed reference line at OR=1.0.
    """
    out = _resolve_output_dir(output_dir)

    # Collect odds ratio entries to plot
    entries: list[dict] = []

    or_result = results.odds_ratio
    entries.append(
        {
            "label": "Cross-tabulation OR",
            "or": or_result.odds_ratio,
            "ci_low": or_result.ci_lower,
            "ci_high": or_result.ci_upper,
        }
    )

    # If logistic regression has an odds ratio for the mating variable, include it
    lr = results.logistic_regression
    for key in lr.odds_ratios:
        if key.lower() != "intercept" and key.lower() != "const":
            entries.append(
                {
                    "label": f"Logistic Regression ({key})",
                    "or": lr.odds_ratios[key],
                    "ci_low": None,
                    "ci_high": None,
                }
            )

    fig, ax = plt.subplots(figsize=(8, max(3, len(entries) * 1.2 + 1)))

    y_positions = list(range(len(entries)))

    for i, entry in enumerate(entries):
        or_val = entry["or"]
        ci_low = entry["ci_low"]
        ci_high = entry["ci_high"]

        # Plot CI line if available
        if ci_low is not None and ci_high is not None:
            ax.plot(
                [ci_low, ci_high],
                [i, i],
                color="#333333",
                linewidth=2,
                solid_capstyle="butt",
            )
            # Annotate with values
            ax.annotate(
                f"OR={or_val:.2f} [{ci_low:.2f}, {ci_high:.2f}]",
                xy=(ci_high, i),
                xytext=(10, 0),
                textcoords="offset points",
                fontsize=9,
                va="center",
            )
        else:
            ax.annotate(
                f"OR={or_val:.2f}",
                xy=(or_val, i),
                xytext=(10, 0),
                textcoords="offset points",
                fontsize=9,
                va="center",
            )

        # Plot point estimate as diamond
        ax.plot(
            or_val,
            i,
            marker="D",
            markersize=9,
            color="#c44e52",
            zorder=5,
        )

    # Reference line at OR = 1
    ax.axvline(x=1.0, color="grey", linestyle="--", linewidth=1, zorder=0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([e["label"] for e in entries], fontsize=10)
    ax.set_xlabel("Odds Ratio", fontsize=12)
    ax.set_title("Odds Ratio: Monandry and Eusociality", fontsize=13)
    ax.invert_yaxis()

    fig.tight_layout()
    filepath = out / f"odds_ratio_forest.{FIGURE_FORMAT}"
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_heatmap(
    df: pd.DataFrame, output_dir: Optional[Path] = None
) -> Path:
    """Create a heatmap of the mating_system x sociality cross-tabulation."""
    out = _resolve_output_dir(output_dir)

    ct = pd.crosstab(df["mating_system"], df["sociality"])
    present_sociality = [s for s in SOCIALITY_ORDER if s in ct.columns]
    ct = ct.reindex(columns=present_sociality, fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        ct,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(
        "Cross-tabulation: Mating System x Social Organization",
        fontsize=13,
    )
    ax.set_xlabel("Social Organization", fontsize=11)
    ax.set_ylabel("Mating System", fontsize=11)

    fig.tight_layout()
    filepath = out / f"cross_tabulation_heatmap.{FIGURE_FORMAT}"
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_effective_mates_by_sociality(
    df: pd.DataFrame, output_dir: Optional[Path] = None
) -> Path:
    """Create a strip plot of effective number of mates by sociality level.

    Only includes rows where effective_mates is not NA.
    """
    out = _resolve_output_dir(output_dir)

    plot_df = df.dropna(subset=["effective_mates"]).copy()
    # Restrict to sociality levels present in the data, maintaining order
    present_order = [s for s in SOCIALITY_ORDER if s in plot_df["sociality"].unique()]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.stripplot(
        data=plot_df,
        x="sociality",
        y="effective_mates",
        hue="sociality",
        order=present_order,
        hue_order=present_order,
        jitter=0.25,
        alpha=0.7,
        size=6,
        palette={s: SOCIALITY_PALETTE.get(s, "#999999") for s in present_order},
        legend=False,
        ax=ax,
    )

    ax.set_title(
        "Effective Number of Mates by Social Organization", fontsize=13
    )
    ax.set_xlabel("Social Organization", fontsize=11)
    ax.set_ylabel("Effective Number of Mates", fontsize=11)
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    filepath = out / f"effective_mates_by_sociality.{FIGURE_FORMAT}"
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return filepath


def generate_all_plots(
    df: pd.DataFrame,
    results: AnalysisResults,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """Generate all analysis plots and return the list of saved file paths."""
    out = _resolve_output_dir(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = [
        plot_sociality_by_mating_system(df, out),
        plot_eusociality_proportion(df, out),
        plot_odds_ratio_forest(results, out),
        plot_heatmap(df, out),
        plot_effective_mates_by_sociality(df, out),
    ]
    return paths
