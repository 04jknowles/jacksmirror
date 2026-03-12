"""Data loading, validation, and preparation for the jacksmirror analysis."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from jacksmirror.config import (
    DATASET_PATH,
    EUSOCIAL_CATEGORIES,
    MATING_CATEGORIES,
    SOCIALITY_ORDER,
)


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Read the bee species CSV and apply appropriate column types.

    Parameters
    ----------
    path : Path | None
        Path to the CSV file.  Defaults to ``config.DATASET_PATH``.

    Returns
    -------
    pd.DataFrame
        The loaded dataset with typed columns.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    """
    csv_path = path if path is not None else DATASET_PATH

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # --- Categorical types with explicit ordering ---
    sociality_cat = pd.CategoricalDtype(categories=SOCIALITY_ORDER, ordered=True)
    mating_cat = pd.CategoricalDtype(categories=MATING_CATEGORIES, ordered=True)

    if "sociality" in df.columns:
        df["sociality"] = df["sociality"].astype(sociality_cat)

    if "mating_system" in df.columns:
        df["mating_system"] = df["mating_system"].astype(mating_cat)

    # --- Boolean ---
    if "haplodiploidy" in df.columns:
        df["haplodiploidy"] = df["haplodiploidy"].astype(bool)

    # --- Nullable floats ---
    for col in ("effective_mates", "colony_size"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    return df


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable validation warnings for *df*.

    The checks performed are:
    1. No null values in the ``species`` column.
    2. All ``sociality`` values belong to the allowed set.
    3. All ``mating_system`` values are either *monandrous* or *polyandrous*.
    4. ``effective_mates`` is >= 1.0 wherever it is not null.
    5. No duplicate species names.
    """
    warnings_list: list[str] = []

    # 1. Null species
    if "species" in df.columns:
        null_count = df["species"].isna().sum()
        if null_count > 0:
            warnings_list.append(
                f"Found {null_count} row(s) with null species."
            )
    else:
        warnings_list.append("Column 'species' is missing from the dataset.")

    # 2. Sociality values
    if "sociality" in df.columns:
        allowed = set(SOCIALITY_ORDER)
        # Use codes == -1 for unknown categories (or check raw strings)
        invalid = df["sociality"].dropna()
        invalid_vals = {v for v in invalid if v not in allowed}
        if invalid_vals:
            warnings_list.append(
                f"Invalid sociality values: {sorted(invalid_vals)}"
            )
    else:
        warnings_list.append("Column 'sociality' is missing from the dataset.")

    # 3. Mating system values
    if "mating_system" in df.columns:
        allowed_mating = set(MATING_CATEGORIES)
        invalid_mating = df["mating_system"].dropna()
        invalid_vals = {v for v in invalid_mating if v not in allowed_mating}
        if invalid_vals:
            warnings_list.append(
                f"Invalid mating_system values: {sorted(invalid_vals)}"
            )
    else:
        warnings_list.append(
            "Column 'mating_system' is missing from the dataset."
        )

    # 4. effective_mates >= 1.0
    if "effective_mates" in df.columns:
        non_null = df["effective_mates"].dropna()
        bad_rows = non_null[non_null < 1.0]
        if len(bad_rows) > 0:
            warnings_list.append(
                f"Found {len(bad_rows)} row(s) with effective_mates < 1.0."
            )

    # 5. Duplicate species
    if "species" in df.columns:
        dupes = df["species"].dropna()
        dup_names = dupes[dupes.duplicated()].unique().tolist()
        if dup_names:
            warnings_list.append(
                f"Duplicate species found: {dup_names}"
            )

    return warnings_list


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed columns used by downstream analyses.

    New columns:
    - ``is_eusocial``     (bool) – True when sociality is in EUSOCIAL_CATEGORIES.
    - ``is_monandrous``   (bool) – True when mating_system == "monandrous".
    - ``sociality_binary`` (str) – "eusocial" or "non-eusocial".
    - ``log_colony_size`` (float) – log10 of colony_size (NaN where missing).

    Returns a copy; the original DataFrame is not modified.
    """
    df = df.copy()

    # is_eusocial
    if "sociality" in df.columns:
        df["is_eusocial"] = df["sociality"].isin(EUSOCIAL_CATEGORIES)
    else:
        df["is_eusocial"] = False

    # is_monandrous
    if "mating_system" in df.columns:
        df["is_monandrous"] = df["mating_system"] == "monandrous"
    else:
        df["is_monandrous"] = False

    # sociality_binary
    df["sociality_binary"] = df["is_eusocial"].map(
        {True: "eusocial", False: "non-eusocial"}
    )

    # log_colony_size (NaN-safe)
    if "colony_size" in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df["log_colony_size"] = np.where(
                df["colony_size"].notna() & (df["colony_size"] > 0),
                np.log10(df["colony_size"]),
                np.nan,
            )
    else:
        df["log_colony_size"] = np.nan

    return df


def get_prepared_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load, validate, and enrich the dataset in a single call.

    Validation warnings are printed to stdout so they are visible during
    interactive use but do not interrupt the pipeline.
    """
    df = load_dataset(path)

    validation_warnings = validate_dataset(df)
    if validation_warnings:
        for warning_msg in validation_warnings:
            print(f"[WARNING] {warning_msg}")

    df = add_derived_columns(df)
    return df
