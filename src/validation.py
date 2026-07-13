"""Data validation and data-quality reporting.

Fatal problems (missing columns, non-numeric data, negative quantities,
non-positive strength or age) raise DataValidationError. Everything else
(missing values, duplicates, outliers, implausible ratios) is *reported*,
never silently deleted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import COLUMNS, UNITS

INGREDIENT_COLUMNS = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
]

# Water-to-binder ratios outside this window are physically implausible for
# real concrete and are flagged (not deleted) in the quality report.
PLAUSIBLE_WB_RANGE = (0.20, 2.0)


class DataValidationError(ValueError):
    """Raised when the dataset fails a fatal validation check."""


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run fatal validation checks and return the validated frame unchanged."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Dataset is missing required columns: {missing}")

    non_numeric = [
        c for c in COLUMNS if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise DataValidationError(
            f"Non-numeric values found in columns: {non_numeric}. "
            "All measurements must be numeric."
        )

    for col in INGREDIENT_COLUMNS:
        negatives = int((df[col] < 0).sum())
        if negatives:
            raise DataValidationError(
                f"Column '{col}' contains {negatives} negative value(s). "
                "Ingredient quantities must be >= 0 kg/m3."
            )

    if int((df["strength_mpa"] <= 0).sum()):
        raise DataValidationError("Strength values must be strictly positive (MPa).")
    if int((df["age_days"] <= 0).sum()):
        raise DataValidationError("Curing age must be strictly positive (days).")
    return df


def compute_mix_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """Add binder, water-to-binder ratio, and total mix mass columns."""
    out = df.copy()
    out["binder"] = out["cement"] + out["slag"] + out["fly_ash"]
    out["water_binder_ratio"] = np.where(
        out["binder"] > 0, out["water"] / out["binder"], np.nan
    )
    out["total_mass"] = out[INGREDIENT_COLUMNS].sum(axis=1)
    return out


def _iqr_outlier_count(series: pd.Series) -> int:
    """Count values outside 1.5*IQR fences (standard Tukey rule)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


def build_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Build the data-quality report described in the project specification."""
    enriched = compute_mix_quantities(df)
    wb = enriched["water_binder_ratio"]

    field_stats: dict[str, dict[str, float]] = {}
    outlier_counts: dict[str, int] = {}
    for col in COLUMNS:
        series = df[col].dropna()
        field_stats[col] = {
            "unit": UNITS[col],
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
        }
        outlier_counts[col] = _iqr_outlier_count(series)

    implausible_wb = int(
        ((wb < PLAUSIBLE_WB_RANGE[0]) | (wb > PLAUSIBLE_WB_RANGE[1]) | wb.isna()).sum()
    )

    return {
        "n_rows": int(len(df)),
        "missing_values": {c: int(df[c].isna().sum()) for c in COLUMNS},
        "duplicate_rows": int(df.duplicated().sum()),
        "field_stats": field_stats,
        "outlier_counts_iqr": outlier_counts,
        "water_binder_ratio": {
            "min": float(wb.min()),
            "max": float(wb.max()),
            "mean": float(wb.mean()),
            "std": float(wb.std()),
            "implausible_count": implausible_wb,
            "plausible_range": list(PLAUSIBLE_WB_RANGE),
        },
        "total_mass_kg_m3": {
            "min": float(enriched["total_mass"].min()),
            "max": float(enriched["total_mass"].max()),
            "mean": float(enriched["total_mass"].mean()),
            "std": float(enriched["total_mass"].std()),
        },
        "units": UNITS,
        "note": (
            "Outliers and implausible ratios are reported, not deleted. "
            "Any row removal must be an explicit, documented decision."
        ),
    }
