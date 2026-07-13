"""Tests for column normalization and dataset loading errors."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import (
    COLUMNS,
    DatasetNotFoundError,
    find_dataset_path,
    load_dataset,
    normalize_columns,
)

UCI_HEADERS = {
    "Cement (component 1)(kg in a m^3 mixture)": "cement",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "slag",
    "Fly Ash (component 3)(kg in a m^3 mixture)": "fly_ash",
    "Water  (component 4)(kg in a m^3 mixture)": "water",
    "Superplasticizer (component 5)(kg in a m^3 mixture)": "superplasticizer",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "coarse_aggregate",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)": "fine_aggregate",
    "Age (day)": "age_days",
    "Concrete compressive strength(MPa, megapascals) ": "strength_mpa",
}


def test_normalize_columns_maps_uci_headers():
    raw = pd.DataFrame({name: [1.0] for name in UCI_HEADERS})
    out = normalize_columns(raw)
    assert list(out.columns) == COLUMNS


def test_normalize_columns_missing_column_raises():
    raw = pd.DataFrame({name: [1.0] for name in list(UCI_HEADERS)[:-1]})
    with pytest.raises(ValueError, match="strength_mpa"):
        normalize_columns(raw)


def test_find_dataset_missing_gives_instructions(tmp_path):
    with pytest.raises(DatasetNotFoundError, match="data/raw"):
        find_dataset_path(tmp_path)


def test_load_dataset_rejects_unknown_format(tmp_path):
    bogus = tmp_path / "concrete_data.json"
    bogus.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported dataset format"):
        load_dataset(bogus)


def test_load_dataset_csv_roundtrip(tmp_path, synthetic_df):
    path = tmp_path / "concrete_data.csv"
    renamed = synthetic_df.rename(
        columns={v: k for k, v in UCI_HEADERS.items()}
    )
    renamed.to_csv(path, index=False)
    loaded = load_dataset(path)
    assert list(loaded.columns) == COLUMNS
    assert len(loaded) == len(synthetic_df)
