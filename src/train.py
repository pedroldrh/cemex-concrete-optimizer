"""Model training pipeline.

Splitting strategy (documented per specification):
  The UCI dataset contains the same physical recipe tested at several curing
  ages. Random row-level splits would place near-identical rows in both train
  and test, inflating scores. We therefore build a recipe group key from the
  seven ingredient quantities rounded to the nearest kg (age and strength
  excluded) and use GroupShuffleSplit so every group lands in exactly one of
  train (~70%), validation (~15%), or test (~15%). Cross-validation on the
  training set uses GroupKFold with the same groups.

Model selection uses validation MAE (RMSE as tiebreaker) plus a check on
cross-validation stability. The winner is refit on train+validation and
evaluated exactly once on the untouched test set.

Run with:  python -m src.train
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data_loader import DatasetNotFoundError, dataset_hash, find_dataset_path, load_dataset
from src.features import FEATURE_COLUMNS, TARGET, add_derived_features, make_group_key
from src.validation import build_quality_report, validate_dataframe

logger = logging.getLogger(__name__)


def build_model_candidates(seed: int) -> dict[str, dict[str, Any]]:
    """Model zoo with deliberately small hyperparameter grids."""
    return {
        "DummyRegressor": {
            "estimator": DummyRegressor(strategy="mean"),
            "param_grid": None,
        },
        "LinearRegression": {
            "estimator": Pipeline(
                [("scaler", StandardScaler()), ("model", LinearRegression())]
            ),
            "param_grid": None,
        },
        "Ridge": {
            "estimator": Pipeline(
                [("scaler", StandardScaler()), ("model", Ridge(random_state=seed))]
            ),
            "param_grid": {"model__alpha": [0.1, 1.0, 10.0]},
        },
        "RandomForestRegressor": {
            "estimator": RandomForestRegressor(
                n_estimators=300, random_state=seed, n_jobs=-1
            ),
            "param_grid": {"max_features": [0.5, 1.0]},
        },
        "ExtraTreesRegressor": {
            "estimator": ExtraTreesRegressor(
                n_estimators=300, random_state=seed, n_jobs=-1
            ),
            "param_grid": {"max_features": [0.5, 1.0]},
        },
        "HistGradientBoostingRegressor": {
            "estimator": HistGradientBoostingRegressor(random_state=seed),
            "param_grid": {
                "learning_rate": [0.05, 0.1],
                "max_leaf_nodes": [31, 63],
            },
        },
        "GradientBoostingRegressor": {
            "estimator": GradientBoostingRegressor(random_state=seed),
            "param_grid": {"learning_rate": [0.05, 0.1], "n_estimators": [200, 400]},
        },
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, and R2 (never called 'accuracy')."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def group_split(
    df: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group-aware 70/15/15 split keeping each recipe in a single partition."""
    groups = make_group_key(df)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, holdout_idx = next(outer.split(df, groups=groups))
    holdout = df.iloc[holdout_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel, test_rel = next(inner.split(holdout, groups=groups.iloc[holdout_idx]))
    return (
        df.iloc[train_idx].reset_index(drop=True),
        holdout.iloc[val_rel].reset_index(drop=True),
        holdout.iloc[test_rel].reset_index(drop=True),
    )


def _fit_and_evaluate(
    name: str,
    spec: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, Any]:
    """Tune (small grid) with grouped CV, then score on the validation set."""
    cv = GroupKFold(n_splits=5)
    estimator = spec["estimator"]
    if spec["param_grid"]:
        search = GridSearchCV(
            estimator,
            spec["param_grid"],
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
        )
        search.fit(X_train, y_train, groups=groups_train)
        fitted = search.best_estimator_
        best_params = search.best_params_
    else:
        fitted = estimator.fit(X_train, y_train)
        best_params = {}

    # Grouped CV score distribution for the chosen configuration.
    cv_maes: list[float] = []
    for fold_train, fold_test in cv.split(X_train, y_train, groups_train):
        fold_model = clone_estimator(fitted)
        fold_model.fit(X_train.iloc[fold_train], y_train.iloc[fold_train])
        pred = fold_model.predict(X_train.iloc[fold_test])
        cv_maes.append(float(mean_absolute_error(y_train.iloc[fold_test], pred)))

    val_pred = fitted.predict(X_val)
    val_metrics = regression_metrics(y_val.to_numpy(), val_pred)
    logger.info(
        "%s: val MAE=%.3f RMSE=%.3f R2=%.3f | CV MAE=%.3f±%.3f %s",
        name,
        val_metrics["mae"],
        val_metrics["rmse"],
        val_metrics["r2"],
        np.mean(cv_maes),
        np.std(cv_maes),
        best_params,
    )
    return {
        "name": name,
        "model": fitted,
        "best_params": best_params,
        "val_metrics": val_metrics,
        "cv_mae_mean": float(np.mean(cv_maes)),
        "cv_mae_std": float(np.std(cv_maes)),
        "cv_mae_folds": cv_maes,
    }


def clone_estimator(estimator: Any) -> Any:
    """sklearn.clone wrapper kept separate for testability."""
    from sklearn.base import clone

    return clone(estimator)


def train_all(seed: int = config.RANDOM_SEED) -> dict[str, Any]:
    """Full training pipeline. Returns a summary dictionary."""
    config.ensure_directories()

    raw_path = find_dataset_path()
    df = load_dataset(raw_path)
    validate_dataframe(df)

    quality_report = build_quality_report(df)
    with open(config.METRICS_DIR / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)
    logger.info(
        "Data quality: %d rows, %d duplicates reported (kept)",
        quality_report["n_rows"],
        quality_report["duplicate_rows"],
    )

    data = add_derived_features(df)
    data.to_csv(config.PROCESSED_DATA_PATH, index=False)

    train_df, val_df, test_df = group_split(data, seed)
    logger.info(
        "Group-aware split: train=%d, val=%d, test=%d rows",
        len(train_df), len(val_df), len(test_df),
    )

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[TARGET]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET]
    groups_train = make_group_key(train_df)

    results = [
        _fit_and_evaluate(name, spec, X_train, y_train, groups_train, X_val, y_val)
        for name, spec in build_model_candidates(seed).items()
    ]

    # Select by validation MAE, RMSE tiebreaker. CV stability is reported and
    # should be reviewed; a model with wildly unstable folds deserves scrutiny.
    ranked = sorted(
        results, key=lambda r: (r["val_metrics"]["mae"], r["val_metrics"]["rmse"])
    )
    best = ranked[0]
    logger.info("Selected model: %s", best["name"])

    # Refit winner on train + validation, then evaluate ONCE on the test set.
    X_fit = pd.concat([X_train, X_val], ignore_index=True)
    y_fit = pd.concat([y_train, y_val], ignore_index=True)
    final_model = clone_estimator(best["model"])
    final_model.fit(X_fit, y_fit)
    test_pred = final_model.predict(X_test)
    test_metrics = regression_metrics(y_test.to_numpy(), test_pred)
    logger.info(
        "Final test (single evaluation): MAE=%.3f RMSE=%.3f R2=%.3f",
        test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"],
    )

    comparison = pd.DataFrame(
        [
            {
                "model": r["name"],
                "val_mae": r["val_metrics"]["mae"],
                "val_rmse": r["val_metrics"]["rmse"],
                "val_r2": r["val_metrics"]["r2"],
                "cv_mae_mean": r["cv_mae_mean"],
                "cv_mae_std": r["cv_mae_std"],
                "best_params": json.dumps(r["best_params"]),
            }
            for r in ranked
        ]
    )
    comparison.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)

    metadata = {
        "model_type": best["name"],
        "best_params": best["best_params"],
        "features": FEATURE_COLUMNS,
        "target": TARGET,
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "data_rows": int(len(df)),
        "split_rows": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "split_strategy": (
            "GroupShuffleSplit 70/15/15 grouped by ingredient quantities "
            "rounded to the nearest kg (age excluded) to keep repeated tests "
            "of one recipe in a single partition."
        ),
        "validation_metrics": best["val_metrics"],
        "cv_mae_mean": best["cv_mae_mean"],
        "cv_mae_std": best["cv_mae_std"],
        "cv_mae_folds": best["cv_mae_folds"],
        "test_metrics": test_metrics,
        "training_feature_ranges": {
            col: {
                "min": float(X_fit[col].min()),
                "max": float(X_fit[col].max()),
            }
            for col in FEATURE_COLUMNS
        },
        "random_seed": seed,
        "dataset_file": raw_path.name,
        "dataset_sha256": dataset_hash(raw_path),
        "uncertainty_note": (
            "Prototype uncertainty estimate (tree spread / validation error), "
            "not a formally calibrated safety guarantee."
        ),
    }
    joblib.dump(final_model, config.BEST_MODEL_PATH)
    with open(config.MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved %s and %s", config.BEST_MODEL_PATH, config.MODEL_METADATA_PATH)

    # Evaluation artifacts (plots + grouped error tables) for the final model.
    from src.evaluate import evaluate_final_model

    evaluate_final_model(
        final_model,
        X_test,
        y_test,
        X_fit,
        y_fit,
        cv_maes=best["cv_mae_folds"],
        model_name=best["name"],
        test_df=test_df,
    )

    return {
        "best_model_name": best["name"],
        "comparison": comparison,
        "test_metrics": test_metrics,
        "metadata": metadata,
    }


def main() -> None:
    """CLI entry point: python -m src.train"""
    config.ensure_directories()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.METRICS_DIR / "training_log.txt", mode="w"),
        ],
    )
    try:
        summary = train_all()
    except DatasetNotFoundError as exc:
        print(f"\n{exc}\n")
        raise SystemExit(1)
    print("\nModel comparison (sorted by validation MAE):")
    print(summary["comparison"].to_string(index=False))
    print(f"\nBest model: {summary['best_model_name']}")
    print(f"Test metrics (single evaluation): {summary['test_metrics']}")


if __name__ == "__main__":
    main()
