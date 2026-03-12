# jacksmirror

**A statistical reanalysis of the monogamy hypothesis and the evolution of eusociality in bees.**

This project tests whether monandry (single mating by queens) is associated with the evolution of eusociality in bees, replicating and extending the landmark study by Hughes et al. (2008). It analyses a dataset of 70 bee species using multiple statistical approaches and produces a complete set of results, figures, and an academic-style report.

The entire analysis -- from raw data to publication-quality output -- was built by Claude (Anthropic's AI) in a single session as a demonstration of AI-assisted scientific computing.

---

## What's inside

| File/Folder | What it is |
|------------|-----------|
| `analysis_notebook.ipynb` | **Start here.** Jupyter notebook walking through the full analysis with biological context, code, figures, and interpretation. |
| `DATA_SOURCES.md` | Full data provenance: how the datasets were identified, obtained, extracted, and curated. |
| `data/bee_species.csv` | The 70-species dataset (species, family, sociality, mating system, effective mates, colony size) |
| `data/hughes_2008_table_s1.csv` | The full 266-species Hughes et al. (2008) Table S1, extracted from the original PDF |
| `jacksmirror/` | Python package with all statistical analysis code |
| `tests/` | 226 automated tests verifying the analysis is correct |
| `output/` | Generated after running (reports, figures, JSON) |

## Quick start

### Option A: Just read the notebook (easiest)

If you just want to see the analysis and results, open `analysis_notebook.ipynb` directly on GitHub -- it renders in the browser with no setup required.

### Option B: Run it yourself

You'll need Python 3.11+ installed. On macOS, open Terminal and run:

```bash
# 1. Check you have Python
python3 --version
# Should show 3.11 or higher. If not, install from https://www.python.org/downloads/

# 2. Navigate to wherever you saved this project
cd path/to/jacksmirror

# 3. Create a virtual environment (keeps things tidy)
python3 -m venv .venv
source .venv/bin/activate

# 4. Install the package and its dependencies
pip install -e ".[dev,notebook]"

# 5. Run the full analysis (takes ~10 seconds)
python -m jacksmirror --verbose

# 6. Look at the results
ls output/
```

This generates:
- `output/academic_report.txt` -- The full report (Introduction, Methods, Results, Discussion, Limitations, References)
- `output/analysis_results.json` -- All results in machine-readable format
- Five publication-quality PNG figures (300 DPI)

### Option C: Run the Jupyter notebook interactively

```bash
# After steps 1-4 above:
jupyter lab analysis_notebook.ipynb
```

This opens the notebook in your browser. Click "Run All" (or Shift+Enter through each cell) to execute the analysis step by step with inline figures and commentary.

---

## The biological question

Hamilton's (1964) inclusive fitness theory predicts that altruistic behaviour evolves when the indirect fitness benefits to relatives outweigh the direct costs to the actor (rb > c). In haplodiploid species like bees, monandry ensures that full sisters share 75% of their genes -- more than the 50% they share with their own offspring -- creating conditions that favour the evolution of sterile worker castes.

Boomsma (2007, 2009) formalised this as the **monogamy hypothesis**: strict lifetime monogamy was a necessary precondition for every independent origin of eusociality. Hughes et al. (2008) demonstrated empirically that all known origins of eusociality in Hymenoptera are associated with ancestral monandry, while polyandry (as in *Apis*) evolved secondarily.

This project reanalyses the data with an expanded suite of statistical methods.

## Statistical tests

| Test | What it does | Result |
|------|-------------|--------|
| Fisher's exact test | Tests whether mating system and eusociality are independent | p = 0.008 |
| Chi-squared test | Asymptotic test of independence | p = 0.018 |
| Odds ratio (95% CI) | Effect size with Haldane-Anscombe correction | OR = 0.05 [0.00, 0.95] |
| Logistic regression | Binary regression: eusociality ~ monandry | Quasi-complete separation (see report) |
| Mann-Whitney U test | Compares effective mate counts: eusocial vs non-eusocial | p = 0.0001 |
| Permutation test | 10,000 resamples to validate the odds ratio significance | p = 0.007 |
| Sensitivity analysis | Varies the monandry threshold to test robustness | Significant at 3/4 thresholds |

## Key findings

1. All 5 polyandrous species in the dataset are eusocial (100%), while only 23/65 monandrous species are eusocial (35%)
2. The zero cell (polyandrous + non-eusocial = 0) confirms that polyandry only occurs within already-eusocial lineages
3. This is consistent with the monogamy hypothesis: monandry was ancestral at the origin of eusociality, and polyandry evolved secondarily in *Apis*
4. Results are robust across multiple statistical methods and classification thresholds

## Limitations (acknowledged in the report)

- **Phylogenetic non-independence**: Species share evolutionary history, violating the independence assumption. Phylogenetic comparative methods (PGLS, independent contrasts) would be more rigorous but require a resolved phylogeny.
- **Small polyandrous sample**: Only 5 polyandrous species limits statistical power.
- **Binary classification**: Reduces a continuous variable (effective mates) to a binary one. The sensitivity analysis partially addresses this.
- **Taxonomic bias**: Dataset weighted toward well-studied families (Apidae, Halictidae).

## Running the tests

```bash
# Run all 226 tests
python -m pytest tests/ -v
```

## References

- Boomsma, J.J. (2007). Kin selection versus sexual selection: why the ends do not meet. *Current Biology*, 17(16), R673-R683.
- Boomsma, J.J. (2009). Lifetime monogamy and the evolution of eusociality. *Philosophical Transactions of the Royal Society B*, 364(1533), 3191-3207.
- Hamilton, W.D. (1964). The genetical evolution of social behaviour I and II. *Journal of Theoretical Biology*, 7(1), 1-52.
- Hughes, W.O.H., Oldroyd, B.P., Beekman, M. & Ratnieks, F.L.W. (2008). Ancestral monogamy shows kin selection is key to the evolution of eusociality. *Science*, 320(5880), 1213-1216.

## License

MIT
