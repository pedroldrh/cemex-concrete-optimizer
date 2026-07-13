"""Feature engineering and leakage-safe grouping keys.

All derived features are functions of the mixture ingredients and age only.
The target (strength_mpa) is never used to build features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURES = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age_days",
]

DERIVED_FEATURES = [
    "binder_total",
    "water_binder_ratio",
    "aggregate_total",
    "coarse_fine_ratio",
    "cement_fraction",
    "scm_fraction",
    "total_mix_mass",
]

FEATURE_COLUMNS = BASE_FEATURES + DERIVED_FEATURES
TARGET = "strength_mpa"

# Recipe grouping: ingredient quantities rounded to the nearest kg identify a
# physical recipe regardless of curing age, so the same recipe tested at
# multiple ages never straddles a train/test boundary.
_GROUP_COLUMNS = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append derived mix-design features to a frame with the base columns.

    Division guards return NaN-free values by construction only when binder
    and aggregate totals are positive; zero-binder rows produce 0 ratios and
    are expected to be rejected by validation or optimizer penalties upstream.
    """
    out = df.copy()
    binder = out["cement"] + out["slag"] + out["fly_ash"]
    aggregate = out["coarse_aggregate"] + out["fine_aggregate"]
    total = binder + out["water"] + out["superplasticizer"] + aggregate

    out["binder_total"] = binder
    out["water_binder_ratio"] = np.where(binder > 0, out["water"] / binder, 0.0)
    out["aggregate_total"] = aggregate
    out["coarse_fine_ratio"] = np.where(
        out["fine_aggregate"] > 0,
        out["coarse_aggregate"] / out["fine_aggregate"],
        0.0,
    )
    out["cement_fraction"] = np.where(binder > 0, out["cement"] / binder, 0.0)
    out["scm_fraction"] = np.where(
        binder > 0, (out["slag"] + out["fly_ash"]) / binder, 0.0
    )
    out["total_mix_mass"] = total
    return out


def make_group_key(df: pd.DataFrame) -> pd.Series:
    """Build a recipe-identity key from rounded ingredient quantities.

    Age and strength are deliberately excluded so that repeated tests of one
    recipe at different curing ages share a group and stay in the same split.
    """
    rounded = df[_GROUP_COLUMNS].round(0).astype(int).astype(str)
    return rounded.agg("|".join, axis=1)
