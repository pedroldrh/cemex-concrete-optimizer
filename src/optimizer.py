"""Constrained cost minimization over mixture designs.

Uses scipy differential_evolution (vectorized) with penalty terms for every
constraint. The optimizer is run several times with different seeds and a
diversity penalty against already-accepted solutions, so it returns multiple
distinct valid recipes instead of five near-copies of one optimum.

If no valid solution exists, the result explains which constraints were most
frequently violated instead of silently returning an invalid mixture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from src.economics import cost_array
from src.extrapolation import Extrapolator, classify_warning
from src.features import FEATURE_COLUMNS, add_derived_features
from src.schemas import INGREDIENTS, Constraints, MaterialPrices
from src.uncertainty import UncertaintyEstimator

logger = logging.getLogger(__name__)

# Penalty scale: large relative to any realistic cost per m3 (~30-150) so the
# optimizer always prefers a feasible mix over a cheaper infeasible one.
PENALTY_SCALE = 1e4
# Minimum normalized distance between two accepted recommendations.
MIN_DIVERSITY_DISTANCE = 0.08

RESULT_COLUMNS = [
    "rank", "cement", "slag", "fly_ash", "water", "superplasticizer",
    "coarse_aggregate", "fine_aggregate", "age_days", "binder_total",
    "water_binder_ratio", "predicted_strength", "conservative_strength",
    "uncertainty_mpa", "cost_per_m3", "extrapolation_score",
    "nearest_historical_distance", "warning_level",
]

CONSTRAINT_KEYS = [
    "strength", "water_binder_min", "water_binder_max",
    "total_mass_min", "total_mass_max", "extrapolation", "binder_positive",
    "binder_min",
]

# Advisory slump window (cm): outside this, a recommendation gets a
# workability "check" flag. Typical pumpable targets sit inside it.
ADVISORY_SLUMP_RANGE_CM = (5.0, 25.0)


@dataclass
class OptimizationResult:
    """Recommendations plus diagnostics about constraint feasibility."""

    recommendations: pd.DataFrame
    success: bool
    message: str
    violation_counts: dict[str, int] = field(default_factory=dict)


def _candidate_frame(matrix: np.ndarray, age_days: float) -> pd.DataFrame:
    """Build a full feature frame from an (n, 7) ingredient matrix."""
    frame = pd.DataFrame(matrix, columns=list(INGREDIENTS))
    frame["age_days"] = age_days
    return add_derived_features(frame)


def _evaluate_candidates(
    matrix: np.ndarray,
    estimator: UncertaintyEstimator,
    prices: MaterialPrices,
    constraints: Constraints,
    extrapolator: Extrapolator,
) -> pd.DataFrame:
    """Predict, cost, and score an (n, 7) batch of candidate mixes."""
    frame = _candidate_frame(matrix, constraints.age_days)
    predictions = estimator.predict(frame[FEATURE_COLUMNS])
    extrapolation = extrapolator.assess(frame)
    out = frame.copy()
    out[["predicted_strength", "uncertainty_mpa", "conservative_strength"]] = (
        predictions[["predicted_strength", "uncertainty_mpa", "conservative_strength"]]
    )
    out["cost_per_m3"] = cost_array(matrix, prices)
    out["extrapolation_score"] = extrapolation["extrapolation_score"].to_numpy()
    out["nearest_historical_distance"] = extrapolation[
        "nearest_historical_distance"
    ].to_numpy()
    return out


def _violations(evaluated: pd.DataFrame, constraints: Constraints) -> pd.DataFrame:
    """Per-candidate, per-constraint violation magnitudes (0 when satisfied)."""
    wb = constraints.water_binder_ratio
    tm = constraints.total_mass
    return pd.DataFrame(
        {
            "strength": np.maximum(
                0.0,
                constraints.required_strength_mpa
                - evaluated["conservative_strength"],
            ),
            "water_binder_min": np.maximum(
                0.0, wb.min - evaluated["water_binder_ratio"]
            ),
            "water_binder_max": np.maximum(
                0.0, evaluated["water_binder_ratio"] - wb.max
            ),
            "total_mass_min": np.maximum(0.0, tm.min - evaluated["total_mix_mass"]),
            "total_mass_max": np.maximum(0.0, evaluated["total_mix_mass"] - tm.max),
            "extrapolation": np.maximum(
                0.0,
                evaluated["extrapolation_score"]
                - constraints.max_extrapolation_score,
            ),
            "binder_positive": np.where(evaluated["binder_total"] <= 0, 1.0, 0.0),
            "binder_min": np.maximum(
                0.0, constraints.min_binder_total - evaluated["binder_total"]
            ),
        }
    )


def _is_valid(evaluated: pd.DataFrame, constraints: Constraints) -> pd.Series:
    return _violations(evaluated, constraints).sum(axis=1) <= 1e-12


def check_candidate(
    quantities: dict[str, float],
    estimator: UncertaintyEstimator,
    prices: MaterialPrices,
    constraints: Constraints,
    extrapolator: Extrapolator,
) -> pd.Series:
    """Evaluate a single mix (dict of the seven ingredients) with validity flag."""
    matrix = np.array([[quantities[name] for name in INGREDIENTS]], dtype=float)
    evaluated = _evaluate_candidates(matrix, estimator, prices, constraints, extrapolator)
    row = evaluated.iloc[0].copy()
    row["valid"] = bool(_is_valid(evaluated, constraints).iloc[0])
    row["warning_level"] = classify_warning(float(row["extrapolation_score"]))
    return row


def optimize_mixes(
    estimator: UncertaintyEstimator,
    prices: MaterialPrices,
    constraints: Constraints,
    extrapolator: Extrapolator,
    n_recommendations: int = 5,
    max_runs: int = 12,
    maxiter: int = 120,
    popsize: int = 16,
    seed: int = 42,
    slump_model: object | None = None,
) -> OptimizationResult:
    """Generate up to n_recommendations distinct valid low-cost mixtures.

    Runs differential evolution repeatedly with different seeds; each run adds
    a diversity penalty against already-accepted solutions so results differ
    by at least MIN_DIVERSITY_DISTANCE in bound-normalized space.
    """
    bounds = [
        (constraints.ingredients[name].min, constraints.ingredients[name].max)
        for name in INGREDIENTS
    ]
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    span = np.maximum(upper - lower, 1e-9)

    accepted: list[np.ndarray] = []
    violation_totals: dict[str, int] = {key: 0 for key in CONSTRAINT_KEYS}

    def objective(x: np.ndarray) -> np.ndarray:
        # vectorized=True: x has shape (7, S). Also handle a single vector.
        single = x.ndim == 1
        matrix = x[None, :] if single else x.T
        evaluated = _evaluate_candidates(
            matrix, estimator, prices, constraints, extrapolator
        )
        violations = _violations(evaluated, constraints)
        penalty = PENALTY_SCALE * violations.to_numpy().sum(axis=1)
        cost = evaluated["cost_per_m3"].to_numpy()

        # Diversity penalty: discourage re-finding accepted solutions.
        if accepted:
            normalized = (matrix - lower) / span
            for prev in accepted:
                prev_norm = (prev - lower) / span
                dist = np.linalg.norm(normalized - prev_norm, axis=1) / np.sqrt(
                    len(INGREDIENTS)
                )
                penalty += PENALTY_SCALE * np.maximum(
                    0.0, MIN_DIVERSITY_DISTANCE - dist
                )
        total = cost + penalty
        return total[0] if single else total

    rng_seeds = [seed + 1000 * i for i in range(max_runs)]
    for run_index, run_seed in enumerate(rng_seeds):
        if len(accepted) >= n_recommendations:
            break
        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=run_seed,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-6,
            polish=False,
            vectorized=True,
            updating="deferred",
            init="sobol",
        )
        candidate = np.asarray(result.x, dtype=float)
        evaluated = _evaluate_candidates(
            candidate[None, :], estimator, prices, constraints, extrapolator
        )
        violations = _violations(evaluated, constraints)
        for key in CONSTRAINT_KEYS:
            if violations[key].iloc[0] > 1e-12:
                violation_totals[key] += 1
        if not bool(_is_valid(evaluated, constraints).iloc[0]):
            logger.info("Run %d: best candidate infeasible, skipping", run_index)
            continue
        # Enforce distinctness against accepted solutions.
        normalized = (candidate - lower) / span
        too_close = any(
            np.linalg.norm(normalized - (prev - lower) / span)
            / np.sqrt(len(INGREDIENTS))
            < MIN_DIVERSITY_DISTANCE
            for prev in accepted
        )
        if too_close:
            logger.info("Run %d: duplicate of an accepted solution, skipping", run_index)
            continue
        accepted.append(candidate)
        logger.info(
            "Run %d: accepted mix, cost=%.2f",
            run_index,
            float(evaluated["cost_per_m3"].iloc[0]),
        )

    if not accepted:
        blockers = {k: v for k, v in violation_totals.items() if v > 0}
        message = (
            "No valid mixture found. Constraints most frequently violated by "
            f"the best candidates: {blockers or 'none recorded'}. Consider "
            "relaxing the required strength, safety multiplier, ingredient "
            "limits, or the maximum extrapolation score."
        )
        return OptimizationResult(
            recommendations=pd.DataFrame(columns=RESULT_COLUMNS),
            success=False,
            message=message,
            violation_counts=violation_totals,
        )

    matrix = np.vstack(accepted)
    evaluated = _evaluate_candidates(matrix, estimator, prices, constraints, extrapolator)
    evaluated["warning_level"] = [
        classify_warning(s) for s in evaluated["extrapolation_score"]
    ]
    evaluated = evaluated.sort_values("cost_per_m3").reset_index(drop=True)
    evaluated.insert(0, "rank", np.arange(1, len(evaluated) + 1))

    result_columns = list(RESULT_COLUMNS)
    if slump_model is not None:
        # ADVISORY only: slump model is trained on 103 unrelated lab mixes.
        from src.workability import predict_slump

        slump = predict_slump(slump_model, evaluated)
        evaluated["predicted_slump_cm"] = slump
        low, high = ADVISORY_SLUMP_RANGE_CM
        evaluated["workability_flag"] = np.where(
            (slump >= low) & (slump <= high), "ok", "check"
        )
        result_columns += ["predicted_slump_cm", "workability_flag"]

    message = f"Found {len(evaluated)} distinct valid mixture(s)."
    if len(evaluated) < n_recommendations:
        message += (
            f" Requested {n_recommendations}; the feasible region under the "
            "current constraints did not yield more distinct optima."
        )
    return OptimizationResult(
        recommendations=evaluated[result_columns],
        success=True,
        message=message,
        violation_counts=violation_totals,
    )
