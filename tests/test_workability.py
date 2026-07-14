"""Tests for the advisory slump screen and the durability binder floor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.features import FEATURE_COLUMNS, TARGET, add_derived_features
from src.extrapolation import Extrapolator
from src.optimizer import ADVISORY_SLUMP_RANGE_CM, optimize_mixes
from src.schemas import INGREDIENTS
from src.uncertainty import UncertaintyEstimator
from src.workability import predict_slump


@pytest.fixture(scope="module")
def slump_model(synthetic_df):
    """Tiny stand-in slump model trained on synthetic quantities.

    Real slump values are irrelevant here; the tests only verify wiring
    (columns, flags), not slump physics.
    """
    rng = np.random.default_rng(1)
    X = synthetic_df[list(INGREDIENTS)]
    # Wetter, more plasticized mixes slump more — crude but directionally sane.
    y = (
        0.15 * synthetic_df["water"]
        + 0.4 * synthetic_df["superplasticizer"]
        - 0.02 * synthetic_df["cement"]
        + rng.normal(0, 1, len(X))
    )
    return RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)


def test_predict_slump_shape(slump_model, synthetic_df):
    out = predict_slump(slump_model, synthetic_df.head(7))
    assert out.shape == (7,)


def test_optimizer_appends_advisory_slump_columns(
    slump_model, synthetic_df, prices, constraints
):
    data = add_derived_features(synthetic_df)
    model = RandomForestRegressor(n_estimators=30, random_state=0).fit(
        data[FEATURE_COLUMNS], data[TARGET]
    )
    estimator = UncertaintyEstimator(model, validation_mae=2.0, validation_rmse=2.8)
    extrapolator = Extrapolator().fit(data)
    result = optimize_mixes(
        estimator, prices, constraints, extrapolator,
        n_recommendations=1, max_runs=3, maxiter=25, popsize=8, seed=11,
        slump_model=slump_model,
    )
    assert result.success, result.message
    recs = result.recommendations
    assert "predicted_slump_cm" in recs.columns
    assert set(recs["workability_flag"]).issubset({"ok", "check"})
    low, high = ADVISORY_SLUMP_RANGE_CM
    inside = recs["predicted_slump_cm"].between(low, high)
    assert (recs.loc[inside, "workability_flag"] == "ok").all()


def test_binder_floor_is_enforced(synthetic_df, prices, constraints):
    data = add_derived_features(synthetic_df)
    model = RandomForestRegressor(n_estimators=30, random_state=0).fit(
        data[FEATURE_COLUMNS], data[TARGET]
    )
    estimator = UncertaintyEstimator(model, validation_mae=2.0, validation_rmse=2.8)
    extrapolator = Extrapolator().fit(data)
    floored = constraints.model_copy(update={"min_binder_total": 450.0})
    result = optimize_mixes(
        estimator, prices, floored, extrapolator,
        n_recommendations=1, max_runs=3, maxiter=25, popsize=8, seed=11,
    )
    assert result.success, result.message
    assert (result.recommendations["binder_total"] >= 450.0 - 1e-6).all()
