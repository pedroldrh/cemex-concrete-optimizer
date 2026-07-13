"""Dataset loading and column normalization for the UCI concrete dataset."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW_DIR

logger = logging.getLogger(__name__)

# Normalized internal column names (units preserved from the source data).
COLUMNS = [
    "cement",              # kg/m3
    "slag",                # kg/m3 (blast furnace slag)
    "fly_ash",             # kg/m3
    "water",               # kg/m3
    "superplasticizer",    # kg/m3
    "coarse_aggregate",    # kg/m3
    "fine_aggregate",      # kg/m3
    "age_days",            # days
    "strength_mpa",        # MPa (concrete compressive strength)
]

UNITS = {
    "cement": "kg/m3",
    "slag": "kg/m3",
    "fly_ash": "kg/m3",
    "water": "kg/m3",
    "superplasticizer": "kg/m3",
    "coarse_aggregate": "kg/m3",
    "fine_aggregate": "kg/m3",
    "age_days": "days",
    "strength_mpa": "MPa",
}

# Keyword fragments used to map raw UCI headers (which include unit text such
# as "Cement (component 1)(kg in a m^3 mixture)") to internal names. Order
# matters: more specific fragments are matched first.
_KEYWORD_MAP: list[tuple[str, str]] = [
    ("superplastic", "superplasticizer"),
    ("coarse", "coarse_aggregate"),
    ("fine", "fine_aggregate"),
    ("slag", "slag"),
    ("fly", "fly_ash"),
    ("ash", "fly_ash"),
    ("cement", "cement"),
    ("water", "water"),
    ("age", "age_days"),
    ("strength", "strength_mpa"),
]

DATASET_INSTRUCTIONS = f"""
Dataset not found.

This prototype uses the public UCI Concrete Compressive Strength dataset
(~1,030 rows). Place it at ONE of these locations:

  {DATA_RAW_DIR / 'concrete_data.xlsx'}
  {DATA_RAW_DIR / 'concrete_data.csv'}

How to obtain it:
  1. Visit https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
  2. Download and unzip the archive (it contains 'Concrete_Data.xls').
  3. Open the .xls file and save it as 'concrete_data.xlsx' (or export to
     'concrete_data.csv'), then place it in the data/raw/ folder above.

The application will NOT generate synthetic data as a substitute; a small
synthetic dataset is used only inside the automated tests.
""".strip()


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the concrete dataset is missing, with setup instructions."""


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw dataset columns to the normalized internal names.

    Raises:
        ValueError: if any expected column cannot be identified or a target
            column would be mapped twice.
    """
    rename: dict[str, str] = {}
    for raw_name in df.columns:
        lowered = str(raw_name).lower()
        for fragment, target in _KEYWORD_MAP:
            if fragment in lowered:
                rename[raw_name] = target
                break
    mapped = list(rename.values())
    duplicates = {name for name in mapped if mapped.count(name) > 1}
    if duplicates:
        raise ValueError(
            f"Multiple raw columns mapped to the same internal name(s) "
            f"{sorted(duplicates)}. Raw columns: {list(df.columns)}"
        )
    out = df.rename(columns=rename)
    missing = [c for c in COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(
            f"Could not identify required column(s) {missing} in the dataset. "
            f"Raw columns found: {list(df.columns)}"
        )
    return out[COLUMNS]


def find_dataset_path(data_dir: Path | None = None) -> Path:
    """Locate the raw dataset file, preferring xlsx over csv."""
    directory = data_dir or DATA_RAW_DIR
    for name in ("concrete_data.xlsx", "concrete_data.csv"):
        candidate = directory / name
        if candidate.exists():
            return candidate
    raise DatasetNotFoundError(DATASET_INSTRUCTIONS)


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load and normalize the concrete dataset from xlsx or csv.

    Args:
        path: Explicit file path. If None, searches data/raw/.

    Returns:
        DataFrame with the nine normalized columns.
    """
    resolved = Path(path) if path is not None else find_dataset_path()
    if not resolved.exists():
        raise DatasetNotFoundError(DATASET_INSTRUCTIONS)
    logger.info("Loading dataset from %s", resolved)
    if resolved.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(resolved)
    elif resolved.suffix.lower() == ".csv":
        raw = pd.read_csv(resolved)
    else:
        raise ValueError(
            f"Unsupported dataset format '{resolved.suffix}'. Use .xlsx or .csv."
        )
    df = normalize_columns(raw)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
    return df


def dataset_hash(path: Path) -> str:
    """SHA-256 hash of the raw dataset file, for reproducibility metadata."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
