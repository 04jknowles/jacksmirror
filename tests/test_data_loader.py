"""Tests for jacksmirror.data_loader module."""

import pandas as pd
import pytest

from jacksmirror.config import (
    DATASET_PATH,
    EUSOCIAL_CATEGORIES,
    MATING_CATEGORIES,
    SOCIALITY_ORDER,
)
from jacksmirror.data_loader import (
    add_derived_columns,
    load_dataset,
    validate_dataset,
    get_prepared_dataset,
)


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


class TestLoadDataset:
    """Tests for load_dataset()."""

    def test_loads_csv_without_error(self):
        """CSV file loads successfully from the default path."""
        df = load_dataset()
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self):
        """Dataset should contain approximately 70 species rows."""
        df = load_dataset()
        assert 60 <= len(df) <= 80, f"Expected ~70 rows, got {len(df)}"

    def test_exact_row_count(self):
        """Dataset should contain exactly 70 species rows."""
        df = load_dataset()
        assert len(df) == 70

    def test_all_required_columns_present(self):
        """All expected columns exist in the loaded DataFrame."""
        df = load_dataset()
        required = {
            "species", "family", "tribe", "sociality",
            "mating_system", "effective_mates", "haplodiploidy",
            "colony_size", "source",
        }
        assert required.issubset(set(df.columns)), (
            f"Missing columns: {required - set(df.columns)}"
        )

    def test_sociality_is_categorical(self):
        """The sociality column should have an ordered CategoricalDtype."""
        df = load_dataset()
        assert isinstance(df["sociality"].dtype, pd.CategoricalDtype)
        assert df["sociality"].cat.ordered is True
        assert list(df["sociality"].cat.categories) == SOCIALITY_ORDER

    def test_mating_system_is_categorical(self):
        """The mating_system column should have an ordered CategoricalDtype."""
        df = load_dataset()
        assert isinstance(df["mating_system"].dtype, pd.CategoricalDtype)
        assert df["mating_system"].cat.ordered is True
        assert list(df["mating_system"].cat.categories) == MATING_CATEGORIES

    def test_haplodiploidy_is_bool(self):
        """The haplodiploidy column should be boolean dtype."""
        df = load_dataset()
        assert df["haplodiploidy"].dtype == bool

    def test_effective_mates_is_float(self):
        """The effective_mates column should be float64."""
        df = load_dataset()
        assert df["effective_mates"].dtype == "float64"

    def test_colony_size_is_float(self):
        """The colony_size column should be float64."""
        df = load_dataset()
        assert df["colony_size"].dtype == "float64"

    def test_raises_file_not_found_for_missing_path(self, tmp_path):
        """Loading from a non-existent path should raise FileNotFoundError."""
        bad_path = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            load_dataset(bad_path)


# ---------------------------------------------------------------------------
# validate_dataset
# ---------------------------------------------------------------------------


class TestValidateDataset:
    """Tests for validate_dataset()."""

    def test_real_dataset_has_no_warnings(self):
        """The real bee_species.csv should pass validation cleanly."""
        df = load_dataset()
        warnings_list = validate_dataset(df)
        assert warnings_list == [], f"Unexpected warnings: {warnings_list}"

    def test_catches_null_species(self):
        """Validation should flag null values in the species column."""
        df = pd.DataFrame({
            "species": [None, "Bee_A"],
            "sociality": pd.Categorical(
                ["solitary", "solitary"], categories=SOCIALITY_ORDER, ordered=True
            ),
            "mating_system": pd.Categorical(
                ["monandrous", "monandrous"], categories=MATING_CATEGORIES, ordered=True
            ),
            "effective_mates": [1.0, 1.0],
        })
        warnings_list = validate_dataset(df)
        assert any("null species" in w.lower() for w in warnings_list)

    def test_catches_duplicate_species(self):
        """Validation should flag duplicate species names."""
        df = pd.DataFrame({
            "species": ["Bee_A", "Bee_A"],
            "sociality": pd.Categorical(
                ["solitary", "solitary"], categories=SOCIALITY_ORDER, ordered=True
            ),
            "mating_system": pd.Categorical(
                ["monandrous", "monandrous"], categories=MATING_CATEGORIES, ordered=True
            ),
            "effective_mates": [1.0, 1.0],
        })
        warnings_list = validate_dataset(df)
        assert any("duplicate" in w.lower() for w in warnings_list)

    def test_catches_low_effective_mates(self):
        """Validation should flag effective_mates < 1.0."""
        df = pd.DataFrame({
            "species": ["Bee_A"],
            "sociality": pd.Categorical(
                ["solitary"], categories=SOCIALITY_ORDER, ordered=True
            ),
            "mating_system": pd.Categorical(
                ["monandrous"], categories=MATING_CATEGORIES, ordered=True
            ),
            "effective_mates": [0.5],
        })
        warnings_list = validate_dataset(df)
        assert any("effective_mates" in w for w in warnings_list)

    def test_catches_missing_species_column(self):
        """Validation should flag when the species column is entirely absent."""
        df = pd.DataFrame({
            "sociality": pd.Categorical(
                ["solitary"], categories=SOCIALITY_ORDER, ordered=True
            ),
            "mating_system": pd.Categorical(
                ["monandrous"], categories=MATING_CATEGORIES, ordered=True
            ),
        })
        warnings_list = validate_dataset(df)
        assert any("species" in w.lower() and "missing" in w.lower() for w in warnings_list)

    def test_catches_missing_sociality_column(self):
        """Validation should flag when the sociality column is absent."""
        df = pd.DataFrame({
            "species": ["Bee_A"],
            "mating_system": pd.Categorical(
                ["monandrous"], categories=MATING_CATEGORIES, ordered=True
            ),
        })
        warnings_list = validate_dataset(df)
        assert any("sociality" in w.lower() and "missing" in w.lower() for w in warnings_list)

    def test_catches_missing_mating_system_column(self):
        """Validation should flag when the mating_system column is absent."""
        df = pd.DataFrame({
            "species": ["Bee_A"],
            "sociality": pd.Categorical(
                ["solitary"], categories=SOCIALITY_ORDER, ordered=True
            ),
        })
        warnings_list = validate_dataset(df)
        assert any("mating_system" in w.lower() and "missing" in w.lower() for w in warnings_list)


# ---------------------------------------------------------------------------
# add_derived_columns
# ---------------------------------------------------------------------------


class TestAddDerivedColumns:
    """Tests for add_derived_columns()."""

    def test_is_eusocial_true_for_eusocial_species(self):
        """is_eusocial should be True for primitively_ and advanced_eusocial."""
        df = pd.DataFrame({
            "sociality": pd.Categorical(
                ["primitively_eusocial", "advanced_eusocial", "solitary"],
                categories=SOCIALITY_ORDER,
                ordered=True,
            ),
            "mating_system": pd.Categorical(
                ["monandrous", "monandrous", "monandrous"],
                categories=MATING_CATEGORIES,
                ordered=True,
            ),
            "colony_size": [500.0, 50000.0, 1.0],
        })
        result = add_derived_columns(df)
        assert result["is_eusocial"].tolist() == [True, True, False]

    def test_is_eusocial_false_for_non_eusocial(self):
        """is_eusocial should be False for solitary, communal, semisocial."""
        df = pd.DataFrame({
            "sociality": pd.Categorical(
                ["solitary", "communal", "semisocial"],
                categories=SOCIALITY_ORDER,
                ordered=True,
            ),
            "mating_system": pd.Categorical(
                ["monandrous", "monandrous", "monandrous"],
                categories=MATING_CATEGORIES,
                ordered=True,
            ),
            "colony_size": [1.0, 10.0, 20.0],
        })
        result = add_derived_columns(df)
        assert all(val is False or val == False for val in result["is_eusocial"])

    def test_is_monandrous_correct(self):
        """is_monandrous should be True only when mating_system == monandrous."""
        df = pd.DataFrame({
            "sociality": pd.Categorical(
                ["solitary", "solitary"],
                categories=SOCIALITY_ORDER,
                ordered=True,
            ),
            "mating_system": pd.Categorical(
                ["monandrous", "polyandrous"],
                categories=MATING_CATEGORIES,
                ordered=True,
            ),
            "colony_size": [1.0, 1.0],
        })
        result = add_derived_columns(df)
        assert result["is_monandrous"].tolist() == [True, False]

    def test_sociality_binary_values(self):
        """sociality_binary should be 'eusocial' or 'non-eusocial'."""
        df = pd.DataFrame({
            "sociality": pd.Categorical(
                ["advanced_eusocial", "solitary"],
                categories=SOCIALITY_ORDER,
                ordered=True,
            ),
            "mating_system": pd.Categorical(
                ["monandrous", "monandrous"],
                categories=MATING_CATEGORIES,
                ordered=True,
            ),
            "colony_size": [5000.0, 1.0],
        })
        result = add_derived_columns(df)
        assert result["sociality_binary"].tolist() == ["eusocial", "non-eusocial"]

    def test_log_colony_size_computed_correctly(self):
        """log_colony_size should be log10 of colony_size."""
        import numpy as np

        df = pd.DataFrame({
            "sociality": pd.Categorical(
                ["solitary", "solitary"],
                categories=SOCIALITY_ORDER,
                ordered=True,
            ),
            "mating_system": pd.Categorical(
                ["monandrous", "monandrous"],
                categories=MATING_CATEGORIES,
                ordered=True,
            ),
            "colony_size": [100.0, 10000.0],
        })
        result = add_derived_columns(df)
        assert abs(result["log_colony_size"].iloc[0] - 2.0) < 1e-10
        assert abs(result["log_colony_size"].iloc[1] - 4.0) < 1e-10

    def test_log_colony_size_nan_for_missing(self):
        """log_colony_size should be NaN when colony_size is NaN."""
        import numpy as np

        df = pd.DataFrame({
            "sociality": pd.Categorical(
                ["solitary"],
                categories=SOCIALITY_ORDER,
                ordered=True,
            ),
            "mating_system": pd.Categorical(
                ["monandrous"],
                categories=MATING_CATEGORIES,
                ordered=True,
            ),
            "colony_size": [float("nan")],
        })
        result = add_derived_columns(df)
        assert np.isnan(result["log_colony_size"].iloc[0])

    def test_does_not_modify_original_dataframe(self):
        """add_derived_columns should return a copy, not mutate the input."""
        df = pd.DataFrame({
            "sociality": pd.Categorical(
                ["solitary"],
                categories=SOCIALITY_ORDER,
                ordered=True,
            ),
            "mating_system": pd.Categorical(
                ["monandrous"],
                categories=MATING_CATEGORIES,
                ordered=True,
            ),
            "colony_size": [1.0],
        })
        original_cols = set(df.columns)
        _ = add_derived_columns(df)
        assert set(df.columns) == original_cols

    def test_all_derived_columns_present(self):
        """All four derived columns should be added."""
        df = pd.DataFrame({
            "sociality": pd.Categorical(
                ["solitary"],
                categories=SOCIALITY_ORDER,
                ordered=True,
            ),
            "mating_system": pd.Categorical(
                ["monandrous"],
                categories=MATING_CATEGORIES,
                ordered=True,
            ),
            "colony_size": [1.0],
        })
        result = add_derived_columns(df)
        expected = {"is_eusocial", "is_monandrous", "sociality_binary", "log_colony_size"}
        assert expected.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# get_prepared_dataset
# ---------------------------------------------------------------------------


class TestGetPreparedDataset:
    """Tests for get_prepared_dataset()."""

    def test_returns_dataframe_with_derived_columns(self):
        """get_prepared_dataset should return a DataFrame with derived columns."""
        df = get_prepared_dataset()
        derived = {"is_eusocial", "is_monandrous", "sociality_binary", "log_colony_size"}
        assert derived.issubset(set(df.columns))

    def test_row_count_matches_load_dataset(self):
        """Row count should match what load_dataset returns."""
        df_raw = load_dataset()
        df_prepared = get_prepared_dataset()
        assert len(df_prepared) == len(df_raw)
