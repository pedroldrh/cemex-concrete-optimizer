"""Shared fixtures. A small SYNTHETIC dataset is used ONLY for tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.schemas import Bounds, Constraints, MaterialPrices


def make_synthetic_dataset(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Physically plausible synthetic mixes with a known strength relationship.

    Used exclusively in automated tests; the application itself refuses to
    train on anything but the real dataset file.
    """
    rng = np.random.default_rng(seed)
    cement = rng.uniform(150, 500, n)
    slag = rng.uniform(0, 300, n)
    fly_ash = rng.uniform(0, 180, n)
    water = rng.uniform(140, 230, n)
    superplasticizer = rng.uniform(0, 25, n)
    coarse = rng.uniform(820, 1120, n)
    fine = rng.uniform(610, 970, n)
    age = rng.choice([3, 7, 14, 28, 56, 90], n).astype(float)

    binder = cement + slag + fly_ash
    wb = water / binder
    strength = (
        18
        + 0.055 * cement
        + 0.032 * slag
        + 0.02 * fly_ash
        - 42 * wb
        + 0.55 * superplasticizer
        + 9 * np.log(age)
        + rng.normal(0, 2.5, n)
    )
    return pd.DataFrame(
        {
            "cement": cement,
            "slag": slag,
            "fly_ash": fly_ash,
            "water": water,
            "superplasticizer": superplasticizer,
            "coarse_aggregate": coarse,
            "fine_aggregate": fine,
            "age_days": age,
            "strength_mpa": np.clip(strength, 5, None),
        }
    )


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_dataset()


@pytest.fixture(scope="session")
def prices() -> MaterialPrices:
    return MaterialPrices(
        cement=0.13,
        slag=0.055,
        fly_ash=0.04,
        water=0.0015,
        superplasticizer=2.5,
        coarse_aggregate=0.014,
        fine_aggregate=0.012,
        additional_cost_per_m3=0.0,
        currency="USD",
    )


@pytest.fixture(scope="session")
def constraints() -> Constraints:
    return Constraints(
        ingredients={
            "cement": Bounds(min=150, max=500),
            "slag": Bounds(min=0, max=300),
            "fly_ash": Bounds(min=0, max=180),
            "water": Bounds(min=140, max=230),
            "superplasticizer": Bounds(min=0, max=25),
            "coarse_aggregate": Bounds(min=820, max=1120),
            "fine_aggregate": Bounds(min=610, max=970),
        },
        water_binder_ratio=Bounds(min=0.3, max=0.8),
        total_mass=Bounds(min=2100, max=2600),
        required_strength_mpa=30.0,
        age_days=28,
        uncertainty_multiplier=1.0,
        max_extrapolation_score=1.5,
    )
