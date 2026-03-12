"""jacksmirror: Monogamy hypothesis analysis for bee eusociality."""

__version__ = "0.2.0"

from jacksmirror.data_loader import get_prepared_dataset
from jacksmirror.analysis import (
    run_all_analyses,
    run_mann_whitney_test,
    run_permutation_test,
    run_sensitivity_analysis,
)
from jacksmirror.visualization import generate_all_plots
from jacksmirror.reporting import (
    generate_json_report,
    generate_text_report,
    generate_academic_report,
)
