"""Prototype prediction-uncertainty estimation.

This is a heuristic uncertainty estimate, NOT a formally calibrated safety
guarantee. For bagged tree ensembles (RandomForest, ExtraTrees) the spread of
individual tree predictions is used; for all other models the hold-out
validation RMSE serves as a constant uncertainty proxy. A validation-error
buffer (validation MAE) is always subtracted on top.

conservative_strength = predicted_strength
                        - uncertainty_multiplier * uncertainty_mpa
                        - validation_error_buffer
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline


def _final_estimator(model: object) -> object:
    """Unwrap a Pipeline to its final estimator (identity otherwise)."""
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def tree_prediction_std(model: object, X: pd.DataFrame) -> np.ndarray | None:
    """Std of individual tree predictions, or None if the model has no bagged trees."""
    estimator = _final_estimator(model)
    if not isinstance(estimator, (RandomForestRegressor, ExtraTreesRegressor)):
        return None
    features = X
    if isinstance(model, Pipeline) and len(model.steps) > 1:
        features = model[:-1].transform(X)
    features = np.asarray(features)
    per_tree = np.stack([tree.predict(features) for tree in estimator.estimators_])
    return per_tree.std(axis=0)


@dataclass
class UncertaintyEstimator:
    """Bundles a fitted model with its validation-error statistics."""

    model: object
    validation_mae: float
    validation_rmse: float
    uncertainty_multiplier: float = 1.0

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return predicted, uncertainty, and conservative strength per row."""
        predicted = np.asarray(self.model.predict(X), dtype=float)
        tree_std = tree_prediction_std(self.model, X)
        if tree_std is not None:
            # Tree spread alone underestimates total error, so never let the
            # uncertainty fall below the model's validation RMSE.
            uncertainty = np.maximum(tree_std, self.validation_rmse)
        else:
            uncertainty = np.full_like(predicted, self.validation_rmse)
        conservative = (
            predicted
            - self.uncertainty_multiplier * uncertainty
            - self.validation_mae
        )
        return pd.DataFrame(
            {
                "predicted_strength": predicted,
                "uncertainty_mpa": uncertainty,
                "conservative_strength": conservative,
            },
            index=X.index if hasattr(X, "index") else None,
        )
