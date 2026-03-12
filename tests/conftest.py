"""Shared fixtures for jacksmirror test suite."""

import numpy as np
import pandas as pd
import pytest

from jacksmirror.config import DATASET_PATH, SOCIALITY_ORDER, MATING_CATEGORIES
from jacksmirror.data_loader import get_prepared_dataset


@pytest.fixture()
def sample_dataframe() -> pd.DataFrame:
    """A small 10-row DataFrame with known values for deterministic testing.

    Composition:
        3 monandrous eusocial  (2 primitively, 1 advanced)
        2 polyandrous eusocial (Apis-like, advanced)
        4 monandrous solitary
        1 polyandrous solitary

    All derived columns (is_eusocial, is_monandrous, sociality_binary,
    log_colony_size) are pre-computed so the fixture can be used directly
    by analysis and visualization functions.
    """
    data = {
        "species": [
            "Bee_mono_eusoc_1",
            "Bee_mono_eusoc_2",
            "Bee_mono_eusoc_3",
            "Bee_poly_eusoc_1",
            "Bee_poly_eusoc_2",
            "Bee_mono_solit_1",
            "Bee_mono_solit_2",
            "Bee_mono_solit_3",
            "Bee_mono_solit_4",
            "Bee_poly_solit_1",
        ],
        "family": [
            "Apidae", "Apidae", "Apidae", "Apidae", "Apidae",
            "Halictidae", "Halictidae", "Halictidae", "Halictidae", "Halictidae",
        ],
        "tribe": [
            "Meliponini", "Meliponini", "Bombini", "Apini", "Apini",
            "Halictini", "Halictini", "Halictini", "Halictini", "Halictini",
        ],
        "sociality": pd.Categorical(
            [
                "primitively_eusocial", "primitively_eusocial", "advanced_eusocial",
                "advanced_eusocial", "advanced_eusocial",
                "solitary", "solitary", "solitary", "solitary", "solitary",
            ],
            categories=SOCIALITY_ORDER,
            ordered=True,
        ),
        "mating_system": pd.Categorical(
            [
                "monandrous", "monandrous", "monandrous",
                "polyandrous", "polyandrous",
                "monandrous", "monandrous", "monandrous", "monandrous",
                "polyandrous",
            ],
            categories=MATING_CATEGORIES,
            ordered=True,
        ),
        "effective_mates": [1.0, 1.0, 1.0, 12.0, 10.0, 1.0, 1.0, 1.0, 1.0, 3.0],
        "haplodiploidy": [True] * 10,
        "colony_size": [
            2000.0, 1500.0, 50000.0, 60000.0, 30000.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
        ],
        "source": ["Test data"] * 10,
    }

    df = pd.DataFrame(data)

    # Derived columns
    df["is_eusocial"] = df["sociality"].isin(
        {"primitively_eusocial", "advanced_eusocial"}
    )
    df["is_monandrous"] = df["mating_system"] == "monandrous"
    df["sociality_binary"] = df["is_eusocial"].map(
        {True: "eusocial", False: "non-eusocial"}
    )
    df["log_colony_size"] = np.log10(df["colony_size"])

    return df


@pytest.fixture()
def full_dataset() -> pd.DataFrame:
    """Load the real bee_species.csv via get_prepared_dataset()."""
    return get_prepared_dataset(DATASET_PATH)


@pytest.fixture()
def tmp_output_dir(tmp_path):
    """Provide a temporary directory for file-writing tests."""
    return tmp_path
