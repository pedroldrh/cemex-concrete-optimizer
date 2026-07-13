"""Tests for constraint checking, extrapolation, optimization, and model I/O."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.extrapolation import Extrapolator, classify_warning
from src.features import FEATURE_COLUMNS, TARGET, add_derived_features
from src.optimizer import RESULT_COLUMNS, check_candidate, optimize_mixes
from src.uncertainty import UncertaintyEstimator


@pytest.fixture(scope="module")
def trained(synthetic_df):
    """Small RF trained on synthetic data, plus a fitted extrapolator."""
    data = add_derived_features(synthetic_df)
    model = RandomForestRegressor(n_estimators=40, random_state=0, n_jobs=-1)
    model.fit(data[FEATURE_COLUMNS], data[TARGET])
    estimator = UncertaintyEstimator(
        model=model, validation_mae=2.0, validation_rmse=2.8
    )
    extrapolator = Extrapolator().fit(data)
    return estimator, extrapolator, data


def test_uncertainty_predictions_are_conservative(trained):
    estimator, _, data = trained
    preds = estimator.predict(data[FEATURE_COLUMNS].head(10))
    assert (preds["conservative_strength"] < preds["predicted_strength"]).all()
    assert (preds["uncertainty_mpa"] >= estimator.validation_rmse - 1e-9).all()


def test_extrapolation_flags_far_candidates(trained):
    _, extrapolator, data = trained
    inside = data.head(1).copy()
    outside = inside.copy()
    outside["cement"] = 5000.0  # absurdly far outside the training range
    scores = extrapolator.assess(pd.concat([inside, outside], ignore_index=True))
    assert scores["extrapolation_score"].iloc[1] > scores["extrapolation_score"].iloc[0]
    assert scores["warning_level"].iloc[1] == "high"


def test_classify_warning_levels():
    assert classify_warning(0.4) == "low"
    assert classify_warning(1.1) == "medium"
    assert classify_warning(2.0) == "high"


def test_check_candidate_rejects_weak_mix(trained, prices, constraints):
    estimator, extrapolator, _ = trained
    weak = {
        "cement": 150.0,
        "slag": 0.0,
        "fly_ash": 0.0,
        "water": 230.0,  # very high w/b -> low predicted strength
        "superplasticizer": 0.0,
        "coarse_aggregate": 1000.0,
        "fine_aggregate": 800.0,
    }
    strong_requirement = constraints.model_copy(
        update={"required_strength_mpa": 60.0}
    )
    row = check_candidate(weak, estimator, prices, strong_requirement, extrapolator)
    assert not row["valid"]


def test_optimizer_output_structure_and_validity(trained, prices, constraints):
    estimator, extrapolator, _ = trained
    result = optimize_mixes(
        estimator,
        prices,
        constraints,
        extrapolator,
        n_recommendations=3,
        max_runs=5,
        maxiter=40,
        popsize=10,
        seed=7,
    )
    assert result.success, result.message
    recs = result.recommendations
    assert list(recs.columns) == RESULT_COLUMNS
    assert len(recs) >= 1
    # Every returned mix satisfies the strength and extrapolation constraints.
    assert (recs["conservative_strength"] >= constraints.required_strength_mpa).all()
    assert (recs["extrapolation_score"] <= constraints.max_extrapolation_score).all()
    assert (recs[["cement", "water", "coarse_aggregate"]] > 0).all().all()
    # Ranked by ascending cost.
    assert recs["cost_per_m3"].is_monotonic_increasing
    # Distinct recommendations, not near-copies.
    if len(recs) > 1:
        first = recs.iloc[0][["cement", "slag", "fly_ash", "water"]].to_numpy(dtype=float)
        second = recs.iloc[1][["cement", "slag", "fly_ash", "water"]].to_numpy(dtype=float)
        assert not np.allclose(first, second, atol=1.0)


def test_optimizer_reports_infeasible_constraints(trained, prices, constraints):
    estimator, extrapolator, _ = trained
    impossible = constraints.model_copy(update={"required_strength_mpa": 500.0})
    result = optimize_mixes(
        estimator,
        prices,
        impossible,
        extrapolator,
        n_recommendations=2,
        max_runs=2,
        maxiter=15,
        popsize=8,
        seed=3,
    )
    assert not result.success
    assert result.recommendations.empty
    assert "strength" in result.message or result.violation_counts["strength"] > 0


def test_model_save_and_load_roundtrip(trained, tmp_path):
    estimator, _, data = trained
    path = tmp_path / "model.joblib"
    joblib.dump(estimator.model, path)
    loaded = joblib.load(path)
    X = data[FEATURE_COLUMNS].head(5)
    assert np.allclose(loaded.predict(X), estimator.model.predict(X))
