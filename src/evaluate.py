"""Evaluation plots and grouped error reports for the selected model.

All figures are written to reports/figures/ and tables to reports/metrics/.
The test set is used only here, once, for the final selected model.
"""

from __future__ import annotations

import json
import logging
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src import config
from src.features import FEATURE_COLUMNS
from src.train import regression_metrics

logger = logging.getLogger(__name__)

STRENGTH_BINS = [0, 25, 40, 60, np.inf]
STRENGTH_LABELS = ["<25 MPa", "25-40 MPa", "40-60 MPa", ">60 MPa"]
AGE_BINS = [0, 3, 7, 14, 28, np.inf]
AGE_LABELS = ["1-3 d", "4-7 d", "8-14 d", "15-28 d", ">28 d"]


def _save(fig: plt.Figure, name: str) -> None:
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Saved figure %s", path)


def error_by_group(
    y_true: pd.Series,
    y_pred: np.ndarray,
    grouping: pd.Series,
    bins: Sequence[float],
    labels: Sequence[str],
) -> pd.DataFrame:
    """MAE/RMSE/count per bin of `grouping` (strength or age)."""
    frame = pd.DataFrame(
        {
            "group": pd.cut(grouping, bins=list(bins), labels=list(labels)),
            "y": y_true.to_numpy(),
            "pred": y_pred,
        }
    )
    rows = []
    for label, sub in frame.groupby("group", observed=False):
        if len(sub) == 0:
            rows.append({"group": str(label), "n": 0, "mae": np.nan, "rmse": np.nan})
            continue
        m = regression_metrics(sub["y"].to_numpy(), sub["pred"].to_numpy())
        rows.append({"group": str(label), "n": len(sub), "mae": m["mae"], "rmse": m["rmse"]})
    return pd.DataFrame(rows)


def evaluate_final_model(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    cv_maes: Sequence[float],
    model_name: str,
    test_df: pd.DataFrame,
) -> dict[str, object]:
    """Produce all evaluation figures and grouped error tables."""
    config.ensure_directories()
    y_pred = np.asarray(model.predict(X_test), dtype=float)
    residuals = y_test.to_numpy() - y_pred

    # Actual vs predicted
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, s=14, alpha=0.6)
    lims = [0, max(float(y_test.max()), float(y_pred.max())) * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual strength (MPa)")
    ax.set_ylabel("Predicted strength (MPa)")
    ax.set_title(f"{model_name}: actual vs predicted (test set)")
    ax.legend()
    _save(fig, "actual_vs_predicted.png")

    # Residuals vs predicted
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(y_pred, residuals, s=14, alpha=0.6)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted strength (MPa)")
    ax.set_ylabel("Residual (actual − predicted, MPa)")
    ax.set_title("Residuals vs predicted (test set)")
    _save(fig, "residuals_vs_predicted.png")

    # Residual distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(residuals, bins=30, edgecolor="white")
    ax.set_xlabel("Residual (MPa)")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution (test set)")
    _save(fig, "residual_distribution.png")

    # Cross-validation score distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([list(cv_maes)], tick_labels=[model_name])
    ax.scatter(np.ones(len(cv_maes)), cv_maes, color="tab:blue", zorder=3)
    ax.set_ylabel("Fold MAE (MPa)")
    ax.set_title("Grouped 5-fold CV MAE distribution (training set)")
    _save(fig, "cv_mae_distribution.png")

    # Error by strength range and by age
    by_strength = error_by_group(y_test, y_pred, y_test, STRENGTH_BINS, STRENGTH_LABELS)
    by_age = error_by_group(y_test, y_pred, test_df["age_days"], AGE_BINS, AGE_LABELS)
    by_strength.to_csv(config.METRICS_DIR / "error_by_strength.csv", index=False)
    by_age.to_csv(config.METRICS_DIR / "error_by_age.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(by_strength["group"], by_strength["mae"])
    axes[0].set_title("Test MAE by strength range")
    axes[0].set_ylabel("MAE (MPa)")
    axes[1].bar(by_age["group"], by_age["mae"], color="tab:orange")
    axes[1].set_title("Test MAE by curing age")
    _save(fig, "error_by_group.png")

    # Native feature importance (when supported)
    importances = getattr(model, "feature_importances_", None)
    if importances is None and hasattr(model, "steps"):
        importances = getattr(model.steps[-1][1], "feature_importances_", None)
    if importances is not None:
        order = np.argsort(importances)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(np.array(FEATURE_COLUMNS)[order], np.asarray(importances)[order])
        ax.set_title(f"{model_name}: native feature importance")
        _save(fig, "feature_importance.png")

    # Permutation importance (model-independent), computed on the test set.
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=config.RANDOM_SEED,
        scoring="neg_mean_absolute_error",
    )
    order = np.argsort(perm.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(
        np.array(FEATURE_COLUMNS)[order],
        perm.importances_mean[order],
        xerr=perm.importances_std[order],
    )
    ax.set_xlabel("Increase in MAE when permuted (MPa)")
    ax.set_title("Permutation importance (test set)")
    _save(fig, "permutation_importance.png")

    report = {
        "model": model_name,
        "test_metrics": regression_metrics(y_test.to_numpy(), y_pred),
        "error_by_strength": by_strength.to_dict(orient="records"),
        "error_by_age": by_age.to_dict(orient="records"),
        "cv_mae_folds": list(map(float, cv_maes)),
        "permutation_importance": {
            feat: float(val)
            for feat, val in zip(FEATURE_COLUMNS, perm.importances_mean)
        },
    }
    with open(config.METRICS_DIR / "final_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
