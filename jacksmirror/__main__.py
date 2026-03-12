"""CLI entry point for the jacksmirror bee sociality analysis.

Usage:
    python -m jacksmirror [--data-path PATH] [--output-dir DIR]
                          [--no-plots] [--json-only] [--verbose]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from jacksmirror.config import DATASET_PATH, OUTPUT_DIR
from jacksmirror.data_loader import get_prepared_dataset
from jacksmirror.analysis import run_all_analyses
from jacksmirror.reporting import (
    generate_text_report,
    generate_json_report,
    generate_academic_report,
    print_summary,
)
from jacksmirror.visualization import generate_all_plots


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="jacksmirror",
        description=(
            "Analyse the relationship between mating system and eusociality "
            "in bees, replicating and extending the monogamy hypothesis."
        ),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=f"Path to input CSV file (default: built-in at {DATASET_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Directory for output files (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Skip plot generation",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        default=False,
        help="Generate only the JSON report (no text report, no plots)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed output during analysis",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the full jacksmirror analysis pipeline."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    data_path: Path | None = args.data_path
    output_dir: Path = args.output_dir if args.output_dir is not None else OUTPUT_DIR
    verbose: bool = args.verbose

    # --- Step 1: Load and prepare data ---
    if verbose:
        print("Loading and preparing dataset...")
    df = get_prepared_dataset(data_path)
    if verbose:
        print(f"  Loaded {len(df)} species records.")

    # --- Step 2: Run analyses ---
    if verbose:
        print("Running statistical analyses...")
    results = run_all_analyses(df)
    results.dataset_path = str(data_path or DATASET_PATH)
    if verbose:
        print("  Analyses complete.")

    # --- Step 3: Print summary ---
    print_summary(results)

    # --- Step 4: Generate reports ---
    generated_files: list[Path] = []

    if not args.json_only:
        if verbose:
            print("Generating text report...")
        text_path = generate_text_report(results, output_dir)
        generated_files.append(text_path)
        if verbose:
            print(f"  Text report: {text_path}")

        if verbose:
            print("Generating academic report...")
        academic_path = generate_academic_report(results, output_dir)
        generated_files.append(academic_path)
        if verbose:
            print(f"  Academic report: {academic_path}")

    if verbose:
        print("Generating JSON report...")
    json_path = generate_json_report(results, output_dir)
    generated_files.append(json_path)
    if verbose:
        print(f"  JSON report: {json_path}")

    # --- Step 5: Generate plots ---
    if not args.no_plots and not args.json_only:
        if verbose:
            print("Generating plots...")
        plot_paths = generate_all_plots(df, results, output_dir)
        generated_files.extend(plot_paths)
        if verbose:
            for p in plot_paths:
                print(f"  Plot: {p}")

    # --- Summary of generated files ---
    print(f"\nGenerated {len(generated_files)} output file(s):")
    for filepath in generated_files:
        print(f"  {filepath}")


if __name__ == "__main__":
    main()
