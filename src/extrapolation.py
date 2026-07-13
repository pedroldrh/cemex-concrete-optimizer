"""Extrapolation detection: how far is a candidate mix from historical data?

Three complementary checks:
  1. Feature-range check: any ingredient outside the observed training range.
  2. Standardized distance from the training-data center (z-space Euclidean,
     normalized by feature count).
  3. Nearest-neighbor distance in standardized feature space.

Distances are normalized by the 95th percentile of the same statistic on the
training data itself, so a score of 1.0 means "as far from the center / from
its neighbors as the outer 5% of historical mixes". Scores above the
configured maximum are rejected by the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.features import BASE_FEATURES

WARNING_LOW = "low"
WARNING_MEDIUM = "medium"
WARNING_HIGH = "high"

# Score thresholds for warning levels.
MEDIUM_THRESHOLD = 1.0
HIGH_THRESHOLD = 1.5
# Score assigned to any candidate with an ingredient outside training ranges.
RANGE_VIOLATION_SCORE = 2.0


def classify_warning(score: float) -> str:
    """Map an extrapolation score to a low/medium/high warning level."""
    if score >= HIGH_THRESHOLD:
        return WARNING_HIGH
    if score >= MEDIUM_THRESHOLD:
        return WARNING_MEDIUM
    return WARNING_LOW


@dataclass
class Extrapolator:
    """Fitted on training base features; scores new candidate mixtures."""

    feature_columns: list[str] = field(default_factory=lambda: list(BASE_FEATURES))
    _scaler: StandardScaler = field(init=False, repr=False)
    _nn: NearestNeighbors = field(init=False, repr=False)
    _train_scaled: np.ndarray = field(init=False, repr=False)
    _train_raw: pd.DataFrame = field(init=False, repr=False)
    _z_ref: float = field(init=False)
    _nn_ref: float = field(init=False)
    _range_min: np.ndarray = field(init=False)
    _range_max: np.ndarray = field(init=False)

    def fit(self, train: pd.DataFrame) -> "Extrapolator":
        """Fit scaler, neighbor index, and reference percentiles on training data."""
        X = train[self.feature_columns].to_numpy(dtype=float)
        self._train_raw = train[self.feature_columns].reset_index(drop=True)
        self._scaler = StandardScaler().fit(X)
        self._train_scaled = self._scaler.transform(X)
        self._range_min = X.min(axis=0)
        self._range_max = X.max(axis=0)

        n_features = len(self.feature_columns)
        train_z = np.linalg.norm(self._train_scaled, axis=1) / np.sqrt(n_features)
        self._z_ref = float(np.percentile(train_z, 95))

        self._nn = NearestNeighbors(n_neighbors=2).fit(self._train_scaled)
        # k=2 because each training point's nearest neighbor is itself.
        self_dist, _ = self._nn.kneighbors(self._train_scaled)
        self._nn_ref = float(max(np.percentile(self_dist[:, 1], 95), 1e-9))
        return self

    def assess(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Score candidates; returns extrapolation columns aligned to input rows."""
        X = candidates[self.feature_columns].to_numpy(dtype=float)
        scaled = self._scaler.transform(X)
        n_features = len(self.feature_columns)

        z_score = (
            np.linalg.norm(scaled, axis=1) / np.sqrt(n_features)
        ) / self._z_ref
        nn_dist, nn_idx = self._nn.kneighbors(scaled, n_neighbors=1)
        nn_score = nn_dist[:, 0] / self._nn_ref
        outside = (X < self._range_min - 1e-9) | (X > self._range_max + 1e-9)
        range_score = np.where(outside.any(axis=1), RANGE_VIOLATION_SCORE, 0.0)

        score = np.maximum.reduce([z_score, nn_score, range_score])
        return pd.DataFrame(
            {
                "extrapolation_score": score,
                "nearest_historical_distance": nn_dist[:, 0],
                "nearest_historical_index": nn_idx[:, 0],
                "warning_level": [classify_warning(s) for s in score],
            },
            index=candidates.index,
        )

    def nearest_recipes(self, candidate: pd.Series | pd.DataFrame, k: int = 3) -> pd.DataFrame:
        """Return the k nearest historical recipes for side-by-side comparison."""
        if isinstance(candidate, pd.Series):
            candidate = candidate.to_frame().T
        X = candidate[self.feature_columns].to_numpy(dtype=float)
        scaled = self._scaler.transform(X)
        dist, idx = self._nn.kneighbors(scaled, n_neighbors=k)
        neighbors = self._train_raw.iloc[idx[0]].copy()
        neighbors.insert(0, "standardized_distance", dist[0])
        return neighbors.reset_index(drop=True)
