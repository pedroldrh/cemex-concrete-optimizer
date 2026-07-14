"""Overdesign audit: the EN 206 conformity math applied to plant records.

Feed it an export of strength test results (one row per sample/batch) and it
answers, per mix family:

  1. What standard deviation does the plant actually have?
  2. What mean strength does EN 206 require?   (f_ck + K * sigma)
  3. What mean strength are they actually producing?
  4. The difference, translated into cement kg/m3, EUR/year and tCO2/year.

This is deliberately model-free: mean, standard deviation, the code's own
formula, and a price. The ML layer of the project only enters later; the
audit must be arguable with nothing but the plant's own numbers.

CLI:
  python -m src.audit path/to/export.csv --price 110 --volume 15000
  python -m src.audit --demo          # synthetic example, no real data

The generated report is written in Spanish because its audience is the
plant's quality team.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# EN 206 continuous-production mean criterion: mean >= f_ck + K * sigma.
EN206_K = 1.48
# Individual result criterion: every result >= f_ck - 4 MPa.
EN206_INDIVIDUAL_MARGIN = 4.0
# Rule of thumb: cement needed per MPa of mean strength near 25-50 MPa.
KG_CEMENT_PER_MPA = 6.0
# Embodied CO2 per kg of average cement (blended; pure clinker is ~0.9).
CO2_KG_PER_KG_CEMENT = 0.7

MIN_SAMPLES = 15          # below this, EN 206 continuous criteria don't apply
LOW_SAMPLES_WARNING = 35  # sigma is noisy below ~35 results

AUDIT_DIR = config.REPORTS_DIR / "audit"

# Flexible header matching for real-world exports (Spanish or English).
_COLUMN_FRAGMENTS: dict[str, list[str]] = {
    "family": ["familia", "family", "formula", "fórmula", "mezcla", "mix", "producto"],
    "fck_mpa": ["fck", "f_ck", "especificada", "specified", "caracteristica",
                 "característica", "proyecto"],
    "strength_mpa": ["rotura", "resistencia 28", "r28", "f28", "28d", "28 d",
                      "measured", "medida", "resistencia", "strength"],
    "date": ["fecha", "date"],
    "cement_kg_m3": ["cemento", "cement"],
    "annual_volume_m3": ["volumen anual", "annual volume"],
}


class AuditInputError(ValueError):
    """Raised when the export lacks the minimum columns for the audit."""


def normalize_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map arbitrary export headers onto the audit's canonical columns.

    Required: family, fck_mpa, strength_mpa. Everything else is optional.
    'fck' fragments are matched before generic 'resistencia'/'strength' so a
    file with both specified and measured strength maps correctly.
    """
    rename: dict[str, str] = {}
    taken: set[str] = set()
    for canonical, fragments in _COLUMN_FRAGMENTS.items():
        for raw in df.columns:
            if raw in rename:
                continue
            lowered = str(raw).lower()
            if any(fragment in lowered for fragment in fragments) and canonical not in taken:
                rename[raw] = canonical
                taken.add(canonical)
                break
    out = df.rename(columns=rename)
    missing = [c for c in ("family", "fck_mpa", "strength_mpa") if c not in out.columns]
    if missing:
        raise AuditInputError(
            f"No pude identificar las columnas {missing} en el export. "
            f"Columnas encontradas: {list(df.columns)}. Renombra las columnas "
            "o pasa un archivo con: familia, fck (especificada), resistencia a 28 días."
        )
    return out


@dataclass
class FamilyAudit:
    """Audit result for one mix family."""

    family: str
    n_samples: int
    fck_mpa: float
    sigma_mpa: float
    required_mean_mpa: float
    actual_mean_mpa: float
    excess_mpa: float
    excess_cement_kg_m3: float
    pct_below_fck: float
    individual_failures: int
    annual_volume_m3: float | None
    annual_saving_eur: float | None
    annual_co2_tonnes: float | None
    reliable: bool
    note: str


def audit_family(
    family: str,
    group: pd.DataFrame,
    cement_price_eur_t: float,
    annual_volume_m3: float | None,
    k_factor: float = EN206_K,
    kg_per_mpa: float = KG_CEMENT_PER_MPA,
) -> FamilyAudit:
    """Run the conformity math for one family's results."""
    strengths = group["strength_mpa"].dropna().astype(float)
    fck_values = group["fck_mpa"].dropna().astype(float).unique()
    if len(fck_values) != 1:
        raise AuditInputError(
            f"La familia '{family}' tiene varios f_ck distintos "
            f"({sorted(fck_values)}). Sepáralas por especificación antes de auditar."
        )
    fck = float(fck_values[0])
    n = int(len(strengths))
    sigma = float(strengths.std(ddof=1)) if n > 1 else float("nan")
    required_mean = fck + k_factor * sigma
    actual_mean = float(strengths.mean())
    excess = actual_mean - required_mean
    excess_cement = excess * kg_per_mpa
    pct_below = float((strengths < fck).mean() * 100)
    individual_failures = int((strengths < fck - EN206_INDIVIDUAL_MARGIN).sum())

    saving = co2 = None
    if annual_volume_m3 is not None and excess > 0:
        saving = excess_cement * annual_volume_m3 * cement_price_eur_t / 1000.0
        co2 = excess_cement * annual_volume_m3 * CO2_KG_PER_KG_CEMENT / 1000.0
    elif annual_volume_m3 is not None:
        saving = 0.0
        co2 = 0.0

    reliable = n >= MIN_SAMPLES
    if not reliable:
        note = f"Solo {n} resultados: por debajo del mínimo EN 206 (15). No concluyente."
    elif n < LOW_SAMPLES_WARNING:
        note = f"{n} resultados: sigma aún ruidosa (ideal ≥ {LOW_SAMPLES_WARNING})."
    elif excess < 0:
        note = (
            "ATENCIÓN: la media está POR DEBAJO de la exigida por la norma — "
            "riesgo de no conformidad, no hay nada que recortar aquí."
        )
    elif excess < 1.0:
        note = "Familia afinada: el margen real está cerca del exigido."
    else:
        note = "Exceso relevante sobre el margen exigido por la norma."

    return FamilyAudit(
        family=str(family),
        n_samples=n,
        fck_mpa=fck,
        sigma_mpa=sigma,
        required_mean_mpa=required_mean,
        actual_mean_mpa=actual_mean,
        excess_mpa=excess,
        excess_cement_kg_m3=excess_cement,
        pct_below_fck=pct_below,
        individual_failures=individual_failures,
        annual_volume_m3=annual_volume_m3,
        annual_saving_eur=saving,
        annual_co2_tonnes=co2,
        reliable=reliable,
        note=note,
    )


def run_audit(
    df: pd.DataFrame,
    cement_price_eur_t: float = 110.0,
    annual_volume_m3: float | None = None,
    k_factor: float = EN206_K,
) -> pd.DataFrame:
    """Audit every family in a normalized export; returns a summary frame."""
    normalized = normalize_audit_columns(df)
    volume_col = "annual_volume_m3" in normalized.columns
    results: list[FamilyAudit] = []
    for family, group in normalized.groupby("family", sort=True):
        volume = (
            float(group["annual_volume_m3"].dropna().iloc[0])
            if volume_col and group["annual_volume_m3"].notna().any()
            else annual_volume_m3
        )
        results.append(
            audit_family(str(family), group, cement_price_eur_t, volume,
                         k_factor=k_factor)
        )
    return pd.DataFrame([r.__dict__ for r in results])


def render_report(
    summary: pd.DataFrame,
    cement_price_eur_t: float,
    k_factor: float = EN206_K,
) -> str:
    """One-page Spanish markdown report — the deliverable for the plant."""
    lines = [
        "# Auditoría de sobrediseño — resultados",
        "",
        "**Método (sin modelos, sin IA):** para cada familia, la media exigida "
        f"por la norma EN 206 es `f_ck + {k_factor:.2f} × σ` con la desviación "
        "estándar σ calculada de los propios resultados de rotura a 28 días. "
        "El *exceso* es la media real menos esa exigencia. Cada MPa de exceso "
        f"se traduce a ~{KG_CEMENT_PER_MPA:.0f} kg de cemento por m³ "
        f"(precio usado: {cement_price_eur_t:.0f} €/t; CO₂: "
        f"{CO2_KG_PER_KG_CEMENT:.1f} kg por kg de cemento).",
        "",
        "| Familia | n | f_ck | σ | Media exigida | Media real | Exceso (MPa) "
        "| Cemento de más (kg/m³) | Ahorro (€/año) | CO₂ (t/año) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        saving = f"{r['annual_saving_eur']:,.0f}" if pd.notna(r["annual_saving_eur"]) else "s/volumen"
        co2 = f"{r['annual_co2_tonnes']:,.1f}" if pd.notna(r["annual_co2_tonnes"]) else "—"
        lines.append(
            f"| {r['family']} | {r['n_samples']} | {r['fck_mpa']:.0f} "
            f"| {r['sigma_mpa']:.2f} | {r['required_mean_mpa']:.1f} "
            f"| {r['actual_mean_mpa']:.1f} | {r['excess_mpa']:+.1f} "
            f"| {max(r['excess_cement_kg_m3'], 0):.0f} | {saving} | {co2} |"
        )
    lines += ["", "**Notas por familia:**", ""]
    for _, r in summary.iterrows():
        lines.append(f"- **{r['family']}**: {r['note']}")
    lines += [
        "",
        "---",
        "**Límites de este análisis.** (1) El recorte real de cualquier "
        "familia exige respetar los suelos de durabilidad (cemento mínimo y "
        "a/c máxima de su clase de exposición) y validarse con amasadas de "
        "prueba; esta hoja no autoriza ningún cambio. (2) Parte de σ puede "
        "ser ruido del propio ensayo (curado, prensa, técnico): reducirlo "
        "baja la exigencia de la norma sin tocar el hormigón. (3) La "
        "conversión MPa→cemento es una regla aproximada; la curva real de "
        "cada familia la conoce el equipo de calidad. Decisión final: "
        "control de calidad.",
    ]
    return "\n".join(lines)


def plot_family(
    normalized: pd.DataFrame, result: FamilyAudit, out_dir: Path
) -> Path:
    """Histogram of one family's results with the three key lines."""
    strengths = normalized.loc[
        normalized["family"] == result.family, "strength_mpa"
    ].dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(strengths, bins=20, edgecolor="white", alpha=0.85)
    ax.axvline(result.fck_mpa, color="black", linestyle="-", linewidth=2,
               label=f"f_ck pedida = {result.fck_mpa:.0f}")
    ax.axvline(result.required_mean_mpa, color="tab:green", linestyle="--",
               linewidth=2,
               label=f"Media exigida (norma) = {result.required_mean_mpa:.1f}")
    ax.axvline(result.actual_mean_mpa, color="tab:red", linestyle="--",
               linewidth=2, label=f"Media real = {result.actual_mean_mpa:.1f}")
    ax.set_xlabel("Resistencia a 28 días (MPa)")
    ax.set_ylabel("Nº de resultados")
    ax.set_title(
        f"Familia {result.family}: exceso {result.excess_mpa:+.1f} MPa "
        f"sobre lo exigido"
    )
    ax.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"audit_{str(result.family).replace('/', '-')}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def make_demo_data(seed: int = 7) -> pd.DataFrame:
    """SYNTHETIC demo export (three families) for demos and tests only."""
    rng = np.random.default_rng(seed)
    frames = []
    # (family, fck, true mean, sigma, n, annual volume)
    scenarios = [
        ("HA-25/B/20/IIa", 25, 33.5, 3.2, 90, 18000),   # fat cushion
        ("HA-30/B/20/IIa", 30, 35.6, 3.4, 80, 15000),   # near-tight
        ("HA-40/F/12/IIIa", 40, 44.1, 4.1, 40, 6000),   # slightly below requirement
    ]
    for family, fck, mean, sigma, n, volume in scenarios:
        frames.append(pd.DataFrame({
            "Fecha": pd.date_range("2025-06-01", periods=n, freq="4D").strftime("%Y-%m-%d"),
            "Familia": family,
            "fck proyecto (MPa)": fck,
            "Resistencia 28 días (MPa)": np.round(rng.normal(mean, sigma, n), 1),
            "Cemento (kg/m3)": np.round(rng.normal(280 + (fck - 25) * 5, 6, n), 0),
            "Volumen anual (m3)": volume,
        }))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", nargs="?", help="CSV/Excel export from the plant")
    parser.add_argument("--demo", action="store_true",
                        help="run on synthetic demo data (no real records needed)")
    parser.add_argument("--price", type=float, default=110.0,
                        help="cement price EUR per tonne (default 110)")
    parser.add_argument("--volume", type=float, default=None,
                        help="annual volume m3 applied to families lacking a volume column")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.demo:
        raw = make_demo_data()
        print("Usando datos SINTÉTICOS de demostración (no son de ninguna planta).\n")
    elif args.export:
        path = Path(args.export)
        raw = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    else:
        parser.error("Pasa un archivo de export o usa --demo")

    summary = run_audit(raw, cement_price_eur_t=args.price,
                        annual_volume_m3=args.volume)
    report = render_report(summary, cement_price_eur_t=args.price)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = AUDIT_DIR / "auditoria_sobrediseno.md"
    report_path.write_text(report, encoding="utf-8")
    normalized = normalize_audit_columns(raw)
    for _, row in summary.iterrows():
        result = FamilyAudit(**row.to_dict())
        plot_family(normalized, result, AUDIT_DIR)

    print(report)
    print(f"\nInforme guardado en {report_path} (figuras en {AUDIT_DIR}/)")


if __name__ == "__main__":
    main()
