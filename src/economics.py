"""Material cost and savings calculations.

Only material costs are modeled. Transportation, energy, labor, plant
overhead and margin are excluded unless the user supplies them through
`additional_cost_per_m3`. All savings figures are illustrative, not verified
CEMEX forecasts.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.schemas import INGREDIENTS, MaterialPrices


def material_cost_per_m3(
    quantities: Mapping[str, float] | pd.DataFrame,
    prices: MaterialPrices,
) -> float | np.ndarray:
    """Material cost per cubic metre for one mix (mapping) or many (DataFrame).

    cost = sum(quantity_kg_per_m3 * price_per_kg) + additional_cost_per_m3
    """
    per_kg = prices.per_kg()
    if isinstance(quantities, pd.DataFrame):
        missing = [name for name in INGREDIENTS if name not in quantities.columns]
        if missing:
            raise KeyError(f"Quantities frame is missing ingredient(s): {missing}")
        cost = sum(
            quantities[name].to_numpy(dtype=float) * per_kg[name]
            for name in INGREDIENTS
        )
        return cost + prices.additional_cost_per_m3

    missing = [name for name in INGREDIENTS if name not in quantities]
    if missing:
        raise KeyError(f"Quantities mapping is missing ingredient(s): {missing}")
    total = sum(float(quantities[name]) * per_kg[name] for name in INGREDIENTS)
    return total + prices.additional_cost_per_m3


def cost_array(matrix: np.ndarray, prices: MaterialPrices) -> np.ndarray:
    """Vectorized cost for an (n_samples, 7) ingredient matrix ordered as INGREDIENTS."""
    price_vector = np.array([prices.per_kg()[name] for name in INGREDIENTS])
    return matrix @ price_vector + prices.additional_cost_per_m3


def savings_summary(
    baseline: Mapping[str, float],
    recommended: Mapping[str, float],
    prices: MaterialPrices,
    baseline_strength: float | None = None,
    recommended_strength: float | None = None,
) -> dict[str, Any]:
    """Compare a recommended mix against a baseline mix (both per m3)."""
    baseline_cost = float(material_cost_per_m3(baseline, prices))
    recommended_cost = float(material_cost_per_m3(recommended, prices))
    saving = baseline_cost - recommended_cost

    baseline_binder = baseline["cement"] + baseline["slag"] + baseline["fly_ash"]
    recommended_binder = (
        recommended["cement"] + recommended["slag"] + recommended["fly_ash"]
    )

    summary: dict[str, Any] = {
        "baseline_cost_per_m3": baseline_cost,
        "recommended_cost_per_m3": recommended_cost,
        "saving_per_m3": saving,
        "saving_percent": (saving / baseline_cost * 100.0) if baseline_cost > 0 else 0.0,
        "cement_reduction_kg_m3": float(baseline["cement"] - recommended["cement"]),
        "binder_reduction_kg_m3": float(baseline_binder - recommended_binder),
        "currency": prices.currency,
    }
    if baseline_strength is not None and recommended_strength is not None:
        summary["predicted_strength_difference_mpa"] = float(
            recommended_strength - baseline_strength
        )
    return summary


def annual_saving(
    saving_per_m3: float,
    annual_production_volume_m3: float,
    adoption_rate: float,
) -> float:
    """Illustrative annual saving. Not a verified CEMEX forecast.

    annual_saving = saving_per_m3 * annual_production_volume_m3 * adoption_rate
    """
    if annual_production_volume_m3 < 0:
        raise ValueError("Annual production volume must be >= 0.")
    if not 0.0 <= adoption_rate <= 1.0:
        raise ValueError("Adoption rate must be between 0 and 1.")
    return saving_per_m3 * annual_production_volume_m3 * adoption_rate
