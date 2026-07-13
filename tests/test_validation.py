"""Tests for fatal validation checks and derived mix quantities."""

from __future__ import annotations

import numpy as np
import pytest

from src.features import add_derived_features, make_group_key
from src.validation import (
    DataValidationError,
    build_quality_report,
    compute_mix_quantities,
    validate_dataframe,
)


def test_valid_dataset_passes(synthetic_df):
    assert validate_dataframe(synthetic_df) is synthetic_df


def test_missing_column_rejected(synthetic_df):
    broken = synthetic_df.drop(columns=["water"])
    with pytest.raises(DataValidationError, match="water"):
        validate_dataframe(broken)


def test_negative_ingredient_rejected(synthetic_df):
    broken = synthetic_df.copy()
    broken.loc[0, "cement"] = -1.0
    with pytest.raises(DataValidationError, match="negative"):
        validate_dataframe(broken)


def test_non_positive_strength_rejected(synthetic_df):
    broken = synthetic_df.copy()
    broken.loc[0, "strength_mpa"] = 0.0
    with pytest.raises(DataValidationError, match="Strength"):
        validate_dataframe(broken)


def test_non_numeric_rejected(synthetic_df):
    broken = synthetic_df.copy()
    broken["cement"] = broken["cement"].astype(object)
    broken.loc[0, "cement"] = "a lot"
    with pytest.raises(DataValidationError, match="Non-numeric"):
        validate_dataframe(broken)


def test_water_binder_calculation(synthetic_df):
    enriched = compute_mix_quantities(synthetic_df)
    row = enriched.iloc[0]
    binder = row["cement"] + row["slag"] + row["fly_ash"]
    assert row["binder"] == pytest.approx(binder)
    assert row["water_binder_ratio"] == pytest.approx(row["water"] / binder)


def test_quality_report_structure(synthetic_df):
    report = build_quality_report(synthetic_df)
    assert report["n_rows"] == len(synthetic_df)
    assert "water_binder_ratio" in report
    assert "total_mass_kg_m3" in report
    assert set(report["missing_values"]) == set(synthetic_df.columns)


def test_derived_features_do_not_use_target(synthetic_df):
    shuffled_target = synthetic_df.copy()
    shuffled_target["strength_mpa"] = (
        shuffled_target["strength_mpa"].sample(frac=1, random_state=1).to_numpy()
    )
    a = add_derived_features(synthetic_df).drop(columns=["strength_mpa"])
    b = add_derived_features(shuffled_target).drop(columns=["strength_mpa"])
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_group_key_ignores_age(synthetic_df):
    df = synthetic_df.head(2).copy()
    df.iloc[1] = df.iloc[0]
    df.loc[df.index[1], "age_days"] = 90.0
    keys = make_group_key(df)
    assert keys.iloc[0] == keys.iloc[1]
