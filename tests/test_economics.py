"""Tests for cost and savings calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.economics import (
    annual_saving,
    cost_array,
    material_cost_per_m3,
    savings_summary,
)
from src.schemas import INGREDIENTS

MIX = {
    "cement": 300.0,
    "slag": 100.0,
    "fly_ash": 50.0,
    "water": 180.0,
    "superplasticizer": 5.0,
    "coarse_aggregate": 1000.0,
    "fine_aggregate": 800.0,
}


def expected_cost(prices):
    per_kg = prices.per_kg()
    return sum(MIX[name] * per_kg[name] for name in INGREDIENTS)


def test_material_cost_mapping(prices):
    assert material_cost_per_m3(MIX, prices) == pytest.approx(expected_cost(prices))


def test_material_cost_dataframe(prices):
    frame = pd.DataFrame([MIX, MIX])
    costs = material_cost_per_m3(frame, prices)
    assert np.allclose(costs, expected_cost(prices))


def test_cost_array_matches_mapping(prices):
    matrix = np.array([[MIX[name] for name in INGREDIENTS]])
    assert cost_array(matrix, prices)[0] == pytest.approx(expected_cost(prices))


def test_missing_ingredient_raises(prices):
    incomplete = {k: v for k, v in MIX.items() if k != "water"}
    with pytest.raises(KeyError, match="water"):
        material_cost_per_m3(incomplete, prices)


def test_additional_cost_included(prices):
    with_extra = prices.model_copy(update={"additional_cost_per_m3": 12.5})
    assert material_cost_per_m3(MIX, with_extra) == pytest.approx(
        expected_cost(prices) + 12.5
    )


def test_savings_summary(prices):
    cheaper = dict(MIX, cement=250.0)
    summary = savings_summary(
        MIX, cheaper, prices, baseline_strength=45.0, recommended_strength=42.0
    )
    assert summary["saving_per_m3"] == pytest.approx(50.0 * prices.cement)
    assert summary["cement_reduction_kg_m3"] == pytest.approx(50.0)
    assert summary["binder_reduction_kg_m3"] == pytest.approx(50.0)
    assert summary["predicted_strength_difference_mpa"] == pytest.approx(-3.0)
    assert summary["saving_percent"] > 0


def test_annual_saving():
    assert annual_saving(3.0, 100_000, 0.5) == pytest.approx(150_000.0)


def test_annual_saving_validates_inputs():
    with pytest.raises(ValueError, match="Adoption rate"):
        annual_saving(3.0, 100_000, 1.5)
    with pytest.raises(ValueError, match="volume"):
        annual_saving(3.0, -1, 0.5)
