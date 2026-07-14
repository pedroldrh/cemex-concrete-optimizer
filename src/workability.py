"""Advisory workability (slump) prediction — prototype grade.

Trained on the UCI Concrete Slump Test dataset (I-Cheng Yeh, 2007,
https://doi.org/10.24432/C5FG7D): 103 laboratory mixes with measured slump
and flow. 103 rows is a very small dataset from materials unrelated to the
strength dataset, so this model is ADVISORY ONLY: it flags recommendations
whose predicted slump looks unpumpable or soupy; it must never be treated as
a hard guarantee of workability. Real deployments should retrain on plant
slump records, which the industry measures on nearly every load.

Run with:  python -m src.workability
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

from src import config
from src.schemas import INGREDIENTS

logger = logging.getLogger(__name__)

SLUMP_DATA_PATH = config.DATA_RAW_DIR / "concrete_slump.csv"
SLUMP_MODEL_PATH = config.MODELS_DIR / "slump_model.joblib"
SLUMP_METADATA_PATH = config.MODELS_DIR / "slump_model_metadata.json"

# Raw UCI header fragments -> our ingredient names (order matters).
_COLUMN_MAP = {
    "cement": "cement",
    "slag": "slag",
    "fly ash": "fly_ash",
    "water": "water",
    "sp": "superplasticizer",
    "coarse": "coarse_aggregate",
    "fine": "fine_aggregate",
    "slump": "slump_cm",
}

SLUMP_INSTRUCTIONS = f"""
Slump dataset not found at {SLUMP_DATA_PATH}.

Download the UCI Concrete Slump Test dataset (103 rows):
  https://archive.ics.uci.edu/dataset/182/concrete+slump+test
and save 'slump_test.data' as 'concrete_slump.csv' in data/raw/.

Workability prediction is optional: the strength optimizer works without it,
it just cannot flag slump-implausible recommendations.
""".strip()


def load_slump_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load and normalize the UCI slump dataset (ingredients + slump_cm)."""
    resolved = path or SLUMP_DATA_PATH
    if not resolved.exists():
        raise FileNotFoundError(SLUMP_INSTRUCTIONS)
    raw = pd.read_csv(resolved)
    rename: dict[str, str] = {}
    for raw_name in raw.columns:
        lowered = str(raw_name).lower()
        for fragment, target in _COLUMN_MAP.items():
            if lowered.startswith(fragment):
                rename[raw_name] = target
                break
    df = raw.rename(columns=rename)
    needed = list(INGREDIENTS) + ["slump_cm"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Slump dataset is missing column(s) {missing}; found {list(raw.columns)}"
        )
    return df[needed]


def train_slump_model(seed: int = config.RANDOM_SEED) -> dict:
    """Train and persist the advisory slump model; returns metadata."""
    config.ensure_directories()
    df = load_slump_dataset()
    X, y = df[list(INGREDIENTS)], df["slump_cm"]
    model = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    cv_mae = -cross_val_score(
        model, X, y,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=seed),
    )
    model.fit(X, y)
    metadata = {
        "purpose": "ADVISORY workability screen only — not a guarantee",
        "dataset": "UCI Concrete Slump Test (Yeh 2007), 103 rows",
        "target": "slump_cm",
        "features": list(INGREDIENTS),
        "cv_mae_cm_mean": float(np.mean(cv_mae)),
        "cv_mae_cm_std": float(np.std(cv_mae)),
        "slump_observed_range_cm": [float(y.min()), float(y.max())],
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "caveat": (
            "103 lab mixes with materials unrelated to the strength dataset. "
            "Predictions flag implausible workability; they must not be used "
            "as acceptance criteria. Retrain on plant slump records for real use."
        ),
    }
    joblib.dump(model, SLUMP_MODEL_PATH)
    with open(SLUMP_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(
        "Slump model saved: CV MAE %.1f ± %.1f cm",
        metadata["cv_mae_cm_mean"], metadata["cv_mae_cm_std"],
    )
    return metadata


def load_slump_predictor() -> tuple[object, dict] | None:
    """Return (model, metadata) if a trained slump model exists, else None."""
    if not (SLUMP_MODEL_PATH.exists() and SLUMP_METADATA_PATH.exists()):
        return None
    with open(SLUMP_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return joblib.load(SLUMP_MODEL_PATH), metadata


def predict_slump(model: object, mixes: pd.DataFrame) -> np.ndarray:
    """Predict slump (cm) for a frame containing the seven ingredient columns."""
    return np.asarray(model.predict(mixes[list(INGREDIENTS)]), dtype=float)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    meta = train_slump_model()
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
