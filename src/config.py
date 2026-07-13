"""Project paths, random seed, and configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

DEFAULT_CONSTRAINTS_PATH = CONFIG_DIR / "default_constraints.yaml"
DEFAULT_PRICES_PATH = CONFIG_DIR / "default_prices.yaml"

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "concrete_processed.csv"

RANDOM_SEED = 42

SAFETY_NOTICE = (
    "This application is a demonstration decision-support tool. It is not "
    "approved for production use. Every proposed concrete mixture must be "
    "reviewed by qualified materials engineers and validated through "
    "laboratory and production testing before use."
)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary with a clear error on failure."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ValueError(f"Configuration file {path} did not parse to a mapping.")
    return content


def ensure_directories() -> None:
    """Create the writable project directories if they do not exist."""
    for directory in (
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
