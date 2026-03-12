"""Tests for jacksmirror.visualization module."""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from jacksmirror.analysis import run_all_analyses
from jacksmirror.visualization import (
    generate_all_plots,
    plot_effective_mates_by_sociality,
    plot_eusociality_proportion,
    plot_heatmap,
    plot_odds_ratio_forest,
    plot_sociality_by_mating_system,
)

PNG_HEADER = b"\x89PNG"
MIN_FILE_SIZE = 1024  # 1 KB


@pytest.fixture()
def analysis_results(sample_dataframe):
    """Run all analyses on the sample dataframe for use by plot tests."""
    return run_all_analyses(sample_dataframe)


@pytest.fixture(autouse=True)
def close_all_figures():
    """Close all matplotlib figures after every test to prevent resource leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_sociality_by_mating_system
# ---------------------------------------------------------------------------


class TestPlotSocialityByMatingSystem:
    """Tests for plot_sociality_by_mating_system()."""

    def test_creates_file(self, sample_dataframe, tmp_output_dir):
        """Should create a file at the expected path."""
        filepath = plot_sociality_by_mating_system(sample_dataframe, tmp_output_dir)
        assert filepath.exists()

    def test_file_is_valid_png(self, sample_dataframe, tmp_output_dir):
        """Output file should have a PNG header."""
        filepath = plot_sociality_by_mating_system(sample_dataframe, tmp_output_dir)
        with open(filepath, "rb") as f:
            header = f.read(4)
        assert header == PNG_HEADER

    def test_file_size_nontrivial(self, sample_dataframe, tmp_output_dir):
        """Output file should be larger than 1 KB."""
        filepath = plot_sociality_by_mating_system(sample_dataframe, tmp_output_dir)
        assert filepath.stat().st_size > MIN_FILE_SIZE

    def test_filename_is_correct(self, sample_dataframe, tmp_output_dir):
        """Filename should match the expected convention."""
        filepath = plot_sociality_by_mating_system(sample_dataframe, tmp_output_dir)
        assert filepath.name == "sociality_by_mating_system.png"


# ---------------------------------------------------------------------------
# plot_eusociality_proportion
# ---------------------------------------------------------------------------


class TestPlotEusocialityProportion:
    """Tests for plot_eusociality_proportion()."""

    def test_creates_file(self, sample_dataframe, tmp_output_dir):
        """Should create a file at the expected path."""
        filepath = plot_eusociality_proportion(sample_dataframe, tmp_output_dir)
        assert filepath.exists()

    def test_file_is_valid_png(self, sample_dataframe, tmp_output_dir):
        """Output file should have a PNG header."""
        filepath = plot_eusociality_proportion(sample_dataframe, tmp_output_dir)
        with open(filepath, "rb") as f:
            header = f.read(4)
        assert header == PNG_HEADER

    def test_file_size_nontrivial(self, sample_dataframe, tmp_output_dir):
        """Output file should be larger than 1 KB."""
        filepath = plot_eusociality_proportion(sample_dataframe, tmp_output_dir)
        assert filepath.stat().st_size > MIN_FILE_SIZE

    def test_filename_is_correct(self, sample_dataframe, tmp_output_dir):
        """Filename should match the expected convention."""
        filepath = plot_eusociality_proportion(sample_dataframe, tmp_output_dir)
        assert filepath.name == "eusociality_proportion.png"


# ---------------------------------------------------------------------------
# plot_odds_ratio_forest
# ---------------------------------------------------------------------------


class TestPlotOddsRatioForest:
    """Tests for plot_odds_ratio_forest()."""

    def test_creates_file(self, analysis_results, tmp_output_dir):
        """Should create a file at the expected path."""
        filepath = plot_odds_ratio_forest(analysis_results, tmp_output_dir)
        assert filepath.exists()

    def test_file_is_valid_png(self, analysis_results, tmp_output_dir):
        """Output file should have a PNG header."""
        filepath = plot_odds_ratio_forest(analysis_results, tmp_output_dir)
        with open(filepath, "rb") as f:
            header = f.read(4)
        assert header == PNG_HEADER

    def test_file_size_nontrivial(self, analysis_results, tmp_output_dir):
        """Output file should be larger than 1 KB."""
        filepath = plot_odds_ratio_forest(analysis_results, tmp_output_dir)
        assert filepath.stat().st_size > MIN_FILE_SIZE

    def test_filename_is_correct(self, analysis_results, tmp_output_dir):
        """Filename should match the expected convention."""
        filepath = plot_odds_ratio_forest(analysis_results, tmp_output_dir)
        assert filepath.name == "odds_ratio_forest.png"


# ---------------------------------------------------------------------------
# plot_heatmap
# ---------------------------------------------------------------------------


class TestPlotHeatmap:
    """Tests for plot_heatmap()."""

    def test_creates_file(self, sample_dataframe, tmp_output_dir):
        """Should create a file at the expected path."""
        filepath = plot_heatmap(sample_dataframe, tmp_output_dir)
        assert filepath.exists()

    def test_file_is_valid_png(self, sample_dataframe, tmp_output_dir):
        """Output file should have a PNG header."""
        filepath = plot_heatmap(sample_dataframe, tmp_output_dir)
        with open(filepath, "rb") as f:
            header = f.read(4)
        assert header == PNG_HEADER

    def test_file_size_nontrivial(self, sample_dataframe, tmp_output_dir):
        """Output file should be larger than 1 KB."""
        filepath = plot_heatmap(sample_dataframe, tmp_output_dir)
        assert filepath.stat().st_size > MIN_FILE_SIZE

    def test_filename_is_correct(self, sample_dataframe, tmp_output_dir):
        """Filename should match the expected convention."""
        filepath = plot_heatmap(sample_dataframe, tmp_output_dir)
        assert filepath.name == "cross_tabulation_heatmap.png"


# ---------------------------------------------------------------------------
# plot_effective_mates_by_sociality
# ---------------------------------------------------------------------------


class TestPlotEffectiveMatesBySociality:
    """Tests for plot_effective_mates_by_sociality()."""

    def test_creates_file(self, sample_dataframe, tmp_output_dir):
        """Should create a file at the expected path."""
        filepath = plot_effective_mates_by_sociality(sample_dataframe, tmp_output_dir)
        assert filepath.exists()

    def test_file_is_valid_png(self, sample_dataframe, tmp_output_dir):
        """Output file should have a PNG header."""
        filepath = plot_effective_mates_by_sociality(sample_dataframe, tmp_output_dir)
        with open(filepath, "rb") as f:
            header = f.read(4)
        assert header == PNG_HEADER

    def test_file_size_nontrivial(self, sample_dataframe, tmp_output_dir):
        """Output file should be larger than 1 KB."""
        filepath = plot_effective_mates_by_sociality(sample_dataframe, tmp_output_dir)
        assert filepath.stat().st_size > MIN_FILE_SIZE

    def test_filename_is_correct(self, sample_dataframe, tmp_output_dir):
        """Filename should match the expected convention."""
        filepath = plot_effective_mates_by_sociality(sample_dataframe, tmp_output_dir)
        assert filepath.name == "effective_mates_by_sociality.png"


# ---------------------------------------------------------------------------
# generate_all_plots
# ---------------------------------------------------------------------------


class TestGenerateAllPlots:
    """Tests for generate_all_plots()."""

    def test_returns_five_paths(self, sample_dataframe, analysis_results, tmp_output_dir):
        """Should return exactly 5 file paths."""
        paths = generate_all_plots(sample_dataframe, analysis_results, tmp_output_dir)
        assert len(paths) == 5

    def test_all_files_exist(self, sample_dataframe, analysis_results, tmp_output_dir):
        """All returned paths should point to existing files."""
        paths = generate_all_plots(sample_dataframe, analysis_results, tmp_output_dir)
        for p in paths:
            assert p.exists(), f"Expected file not found: {p}"

    def test_all_files_are_valid_pngs(
        self, sample_dataframe, analysis_results, tmp_output_dir
    ):
        """Every generated file should start with the PNG header."""
        paths = generate_all_plots(sample_dataframe, analysis_results, tmp_output_dir)
        for p in paths:
            with open(p, "rb") as f:
                header = f.read(4)
            assert header == PNG_HEADER, f"Not a valid PNG: {p}"

    def test_all_files_nontrivial_size(
        self, sample_dataframe, analysis_results, tmp_output_dir
    ):
        """Every generated file should be larger than 1 KB."""
        paths = generate_all_plots(sample_dataframe, analysis_results, tmp_output_dir)
        for p in paths:
            assert p.stat().st_size > MIN_FILE_SIZE, (
                f"File too small ({p.stat().st_size} bytes): {p}"
            )
