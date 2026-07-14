"""Tests for the EN 206 overdesign audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.audit import (
    EN206_K,
    AuditInputError,
    make_demo_data,
    normalize_audit_columns,
    render_report,
    run_audit,
)


def test_normalizes_spanish_headers():
    df = make_demo_data()
    out = normalize_audit_columns(df)
    for col in ("family", "fck_mpa", "strength_mpa", "annual_volume_m3"):
        assert col in out.columns


def test_missing_columns_rejected():
    with pytest.raises(AuditInputError, match="fck"):
        normalize_audit_columns(pd.DataFrame({"Familia": ["A"], "otro": [1]}))


def test_required_mean_formula():
    rng = np.random.default_rng(0)
    strengths = rng.normal(36.0, 3.0, 60)
    df = pd.DataFrame({
        "Familia": "HA-30",
        "fck (MPa)": 30.0,
        "Resistencia 28d (MPa)": strengths,
    })
    summary = run_audit(df, cement_price_eur_t=110.0, annual_volume_m3=10_000)
    row = summary.iloc[0]
    sigma = pd.Series(strengths).std(ddof=1)
    assert row["sigma_mpa"] == pytest.approx(sigma)
    assert row["required_mean_mpa"] == pytest.approx(30.0 + EN206_K * sigma)
    assert row["excess_mpa"] == pytest.approx(strengths.mean() - row["required_mean_mpa"])
    # 36 vs required ~34.4 -> positive excess -> positive saving
    assert row["excess_mpa"] > 0
    assert row["annual_saving_eur"] > 0


def test_underperforming_family_flagged_not_trimmed():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "Familia": "HA-40",
        "fck (MPa)": 40.0,
        "Resistencia 28d (MPa)": rng.normal(43.0, 4.0, 50),  # below 40+1.48*4
    })
    summary = run_audit(df, cement_price_eur_t=110.0, annual_volume_m3=5_000)
    row = summary.iloc[0]
    assert row["excess_mpa"] < 0
    assert row["annual_saving_eur"] == 0.0
    assert "no conformidad" in row["note"].lower()


def test_mixed_fck_in_family_rejected():
    df = pd.DataFrame({
        "Familia": ["A", "A"],
        "fck (MPa)": [25.0, 30.0],
        "Resistencia 28d (MPa)": [30.0, 35.0],
    })
    with pytest.raises(AuditInputError, match="varios f_ck"):
        run_audit(df)


def test_small_samples_marked_unreliable():
    df = pd.DataFrame({
        "Familia": "B",
        "fck (MPa)": 25.0,
        "Resistencia 28d (MPa)": [30.0, 31.0, 29.5, 32.0, 30.5],
    })
    summary = run_audit(df)
    assert not summary.iloc[0]["reliable"]


def test_report_renders_all_families():
    summary = run_audit(make_demo_data(), cement_price_eur_t=110.0)
    report = render_report(summary, cement_price_eur_t=110.0)
    for family in summary["family"]:
        assert family in report
    assert "EN 206" in report
    assert "control de calidad" in report.lower()
