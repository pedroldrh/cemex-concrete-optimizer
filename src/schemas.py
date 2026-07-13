"""Pydantic models for prices, constraints, and optimizer outputs."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from src.config import load_yaml

INGREDIENTS: tuple[str, ...] = (
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
)


class MaterialPrices(BaseModel):
    """Material prices in currency units per kilogram (illustrative defaults)."""

    cement: float = Field(ge=0)
    slag: float = Field(ge=0)
    fly_ash: float = Field(ge=0)
    water: float = Field(ge=0)
    superplasticizer: float = Field(ge=0)
    coarse_aggregate: float = Field(ge=0)
    fine_aggregate: float = Field(ge=0)
    additional_cost_per_m3: float = Field(default=0.0, ge=0)
    currency: str = "USD"

    def per_kg(self) -> dict[str, float]:
        """Return the price per kg for each ingredient."""
        return {name: getattr(self, name) for name in INGREDIENTS}

    @classmethod
    def from_yaml(cls, path: Path) -> "MaterialPrices":
        raw = load_yaml(path)
        prices = raw.get("prices_per_kg", {})
        return cls(
            **prices,
            additional_cost_per_m3=raw.get("additional_cost_per_m3", 0.0),
            currency=raw.get("currency", "USD"),
        )


class Bounds(BaseModel):
    """Inclusive [min, max] bounds for one quantity."""

    min: float
    max: float

    @model_validator(mode="after")
    def check_order(self) -> "Bounds":
        if self.max < self.min:
            raise ValueError(f"max ({self.max}) must be >= min ({self.min})")
        return self


class Constraints(BaseModel):
    """Prototype engineering constraints.

    These require review by a concrete materials engineer and must never be
    assumed to represent CEMEX production rules.
    """

    ingredients: dict[str, Bounds]
    water_binder_ratio: Bounds
    total_mass: Bounds
    required_strength_mpa: float = Field(gt=0)
    age_days: float = Field(gt=0)
    uncertainty_multiplier: float = Field(default=1.0, ge=0)
    max_extrapolation_score: float = Field(default=1.2, gt=0)

    @model_validator(mode="after")
    def check_ingredients(self) -> "Constraints":
        missing = [name for name in INGREDIENTS if name not in self.ingredients]
        if missing:
            raise ValueError(f"Missing ingredient bounds for: {missing}")
        negative = [
            name for name, b in self.ingredients.items() if b.min < 0
        ]
        if negative:
            raise ValueError(f"Ingredient bounds must be non-negative: {negative}")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "Constraints":
        return cls(**load_yaml(path))


class Recommendation(BaseModel):
    """One optimizer recommendation (a candidate mixture, per m3)."""

    rank: int
    cement: float
    slag: float
    fly_ash: float
    water: float
    superplasticizer: float
    coarse_aggregate: float
    fine_aggregate: float
    age_days: float
    binder_total: float
    water_binder_ratio: float
    predicted_strength: float
    conservative_strength: float
    uncertainty_mpa: float
    cost_per_m3: float
    extrapolation_score: float
    nearest_historical_distance: float
    warning_level: str
