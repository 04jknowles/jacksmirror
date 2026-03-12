"""Configuration constants and paths for the jacksmirror analysis."""

from pathlib import Path


# --- Paths ---
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
DATASET_PATH: Path = DATA_DIR / "bee_species.csv"

# --- Figure settings ---
FIGURE_DPI: int = 300
FIGURE_FORMAT: str = "png"

# --- Analysis categories ---
SOCIALITY_ORDER: list[str] = [
    "solitary",
    "communal",
    "semisocial",
    "primitively_eusocial",
    "advanced_eusocial",
]

EUSOCIAL_CATEGORIES: set[str] = {"primitively_eusocial", "advanced_eusocial"}

MATING_CATEGORIES: list[str] = ["monandrous", "polyandrous"]

# --- Reproducibility ---
RANDOM_SEED: int = 42

# --- Permutation / sensitivity settings ---
N_PERMUTATIONS: int = 10000
SENSITIVITY_THRESHOLDS: list[float] = [1.0, 2.0, 5.0, 10.0]


def ensure_output_dir() -> Path:
    """Create the output directory if it does not exist and return its path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
