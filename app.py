"""CEMEX Concrete Mix Optimizer — Streamlit prototype.

Launch with:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.data_loader import DATASET_INSTRUCTIONS, DatasetNotFoundError, load_dataset
from src.economics import annual_saving, material_cost_per_m3, savings_summary
from src.extrapolation import Extrapolator
from src.features import FEATURE_COLUMNS, add_derived_features
from src.optimizer import check_candidate, optimize_mixes
from src.schemas import INGREDIENTS, Bounds, Constraints, MaterialPrices
from src.uncertainty import UncertaintyEstimator
from src.validation import build_quality_report, validate_dataframe

st.set_page_config(page_title="CEMEX Concrete Mix Optimizer", layout="wide")

INGREDIENT_LABELS = {
    "cement": "Cement",
    "slag": "Slag",
    "fly_ash": "Fly ash",
    "water": "Water",
    "superplasticizer": "Superplasticizer",
    "coarse_aggregate": "Coarse aggregate",
    "fine_aggregate": "Fine aggregate",
}

EXPERT_QUESTIONS = [
    "Are we optimizing ready-mix concrete, cement composition or both?",
    "What properties matter besides compressive strength?",
    "What are the minimum and maximum quantities for each material?",
    "What water-to-binder limits are allowed?",
    "What durability standards apply?",
    "Is slump or workability required?",
    "Are chloride, sulfate, alkali or exposure-class constraints required?",
    "Are recipes specific to each plant?",
    "Are supplier and quarry properties recorded?",
    "How is aggregate moisture handled?",
    "How are material costs stored?",
    "What safety margins are used?",
    "Which curing ages matter?",
    "How are failed batches recorded?",
    "What laboratory validation process is required?",
    "How much recipe variation is permitted?",
    "What data systems contain historical recipes and test results?",
    "Should recommendations optimize cost, carbon emissions or both?",
    "What regulations and internal approvals apply?",
    "How should uncertainty be communicated to engineers?",
]


# ---------------------------------------------------------------- data access
@st.cache_data(show_spinner="Loading dataset...")
def get_dataset(dataset_path: str) -> pd.DataFrame:
    path = Path(dataset_path) if dataset_path else None
    df = load_dataset(path)
    validate_dataframe(df)
    return add_derived_features(df)


@st.cache_resource(show_spinner="Loading trained model...")
def get_model_bundle() -> tuple[object, dict]:
    if not config.BEST_MODEL_PATH.exists() or not config.MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run `python -m src.train` first."
        )
    model = joblib.load(config.BEST_MODEL_PATH)
    with open(config.MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


@st.cache_resource(show_spinner="Fitting extrapolation detector...")
def get_extrapolator(dataset_path: str) -> Extrapolator:
    return Extrapolator().fit(get_dataset(dataset_path))


def build_estimator(model, metadata, multiplier: float) -> UncertaintyEstimator:
    return UncertaintyEstimator(
        model=model,
        validation_mae=metadata["validation_metrics"]["mae"],
        validation_rmse=metadata["validation_metrics"]["rmse"],
        uncertainty_multiplier=multiplier,
    )


def default_constraints() -> Constraints:
    return Constraints.from_yaml(config.DEFAULT_CONSTRAINTS_PATH)


def default_prices() -> MaterialPrices:
    return MaterialPrices.from_yaml(config.DEFAULT_PRICES_PATH)


def show_figure_if_exists(name: str, caption: str) -> None:
    path = config.FIGURES_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


# -------------------------------------------------------------------- sidebar
st.sidebar.title("CEMEX Concrete Mix Optimizer")
st.sidebar.error(config.SAFETY_NOTICE, icon="⚠️")

dataset_path_input = st.sidebar.text_input(
    "Dataset path (blank = data/raw/ auto-detect)", value=""
)
currency = st.sidebar.text_input("Currency", value=default_prices().currency)
required_strength = st.sidebar.number_input(
    "Required strength (MPa)", min_value=5.0, max_value=120.0, value=40.0, step=1.0
)
age_days = st.sidebar.selectbox(
    "Curing age (days)", options=[3, 7, 14, 28, 56, 90], index=3
)
uncertainty_multiplier = st.sidebar.slider(
    "Safety setting (uncertainty multiplier)", 0.0, 3.0, 1.0, 0.25,
    help=(
        "conservative_strength = predicted − multiplier × uncertainty − "
        "validation error buffer. Higher = safer, more expensive mixes. "
        "Prototype heuristic, not a calibrated safety factor."
    ),
)
annual_volume = st.sidebar.number_input(
    "Annual production volume (m³)", min_value=0.0, value=100_000.0, step=10_000.0
)
adoption_rate = st.sidebar.slider("Expected adoption rate", 0.0, 1.0, 0.5, 0.05)

page = st.sidebar.radio(
    "Page",
    [
        "1 · Executive overview",
        "2 · Data explorer",
        "3 · Model performance",
        "4 · Mix optimizer",
        "5 · Scenario analysis",
        "6 · Assumptions & limitations",
    ],
)

# Shared loading with graceful failures.
dataset_error = model_error = None
df = model = metadata = None
try:
    df = get_dataset(dataset_path_input)
except DatasetNotFoundError:
    dataset_error = DATASET_INSTRUCTIONS
except Exception as exc:  # surface, never swallow
    dataset_error = f"Failed to load dataset: {exc}"
try:
    model, metadata = get_model_bundle()
except Exception as exc:
    model_error = str(exc)

st.title(page.split("·", 1)[1].strip())
st.warning(config.SAFETY_NOTICE, icon="⚠️")


def render_prices_editor(key_prefix: str) -> MaterialPrices:
    """Editable prices, clearly marked illustrative."""
    base = default_prices()
    st.caption(
        "Default prices are **illustrative placeholders**, not CEMEX prices. "
        "Edit freely. Units: currency per kg. Transportation, energy, labor, "
        "overhead and margin are excluded unless entered as additional cost."
    )
    cols = st.columns(4)
    values = {}
    for i, name in enumerate(INGREDIENTS):
        with cols[i % 4]:
            values[name] = st.number_input(
                f"{INGREDIENT_LABELS[name]} ({currency}/kg)",
                min_value=0.0,
                value=float(getattr(base, name)),
                step=0.001,
                format="%.4f",
                key=f"{key_prefix}_price_{name}",
            )
    with cols[3]:
        extra = st.number_input(
            f"Additional cost per m³ ({currency})",
            min_value=0.0,
            value=float(base.additional_cost_per_m3),
            step=1.0,
            key=f"{key_prefix}_extra",
        )
    return MaterialPrices(
        **values, additional_cost_per_m3=extra, currency=currency
    )


def render_constraint_editor(key_prefix: str) -> Constraints:
    """Editable constraints derived from dataset ranges."""
    base = default_constraints()
    st.caption(
        "**Prototype constraints requiring review by a concrete materials "
        "engineer.** Defaults are the observed ranges of the public dataset, "
        "NOT CEMEX production rules."
    )
    ingredient_bounds = {}
    for name in INGREDIENTS:
        b = base.ingredients[name]
        lo, hi = st.slider(
            f"{INGREDIENT_LABELS[name]} (kg/m³)",
            min_value=0.0,
            max_value=float(b.max * 1.2),
            value=(float(b.min), float(b.max)),
            key=f"{key_prefix}_limit_{name}",
        )
        ingredient_bounds[name] = Bounds(min=lo, max=hi)
    wb_lo, wb_hi = st.slider(
        "Water-to-binder ratio",
        0.2, 1.2,
        (float(base.water_binder_ratio.min), float(base.water_binder_ratio.max)),
        key=f"{key_prefix}_wb",
    )
    tm_lo, tm_hi = st.slider(
        "Total mix mass (kg/m³)",
        2000.0, 2800.0,
        (float(base.total_mass.min), float(base.total_mass.max)),
        key=f"{key_prefix}_tm",
    )
    max_extrap = st.slider(
        "Maximum extrapolation score (1.0 ≈ edge of historical envelope)",
        0.5, 2.0, float(base.max_extrapolation_score), 0.1,
        key=f"{key_prefix}_extrap",
    )
    return Constraints(
        ingredients=ingredient_bounds,
        water_binder_ratio=Bounds(min=wb_lo, max=wb_hi),
        total_mass=Bounds(min=tm_lo, max=tm_hi),
        required_strength_mpa=required_strength,
        age_days=float(age_days),
        uncertainty_multiplier=uncertainty_multiplier,
        max_extrapolation_score=max_extrap,
    )


def run_optimizer(prices: MaterialPrices, constraints: Constraints, n_recs: int):
    estimator = build_estimator(model, metadata, uncertainty_multiplier)
    extrapolator = get_extrapolator(dataset_path_input)
    return optimize_mixes(
        estimator, prices, constraints, extrapolator,
        n_recommendations=n_recs, seed=config.RANDOM_SEED,
    ), extrapolator, estimator


# ================================================================ page 1
if page.startswith("1"):
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset", "loaded" if df is not None else "missing",
                f"{len(df)} rows" if df is not None else "see instructions below")
    col2.metric("Best model", metadata["model_type"] if metadata else "not trained")
    col3.metric(
        "Test MAE",
        f"{metadata['test_metrics']['mae']:.2f} MPa" if metadata else "—",
    )
    if dataset_error:
        st.error(dataset_error)
    if model_error:
        st.error(model_error)

    st.subheader("Process")
    st.markdown(
        "```\nHistorical mixtures → ML strength prediction → constrained cost "
        "optimization → engineer review → laboratory validation\n```"
    )
    st.info(
        "The model and optimizer only *propose* candidates. Engineer review "
        "and laboratory validation are mandatory steps, not optional ones."
    )

    if df is not None and metadata is not None:
        st.subheader(f"Quick recommendation for {required_strength:.0f} MPa @ {age_days} d")
        if st.button("Compute recommendation", type="primary"):
            with st.spinner("Optimizing (multiple restarts)..."):
                result, extrapolator, estimator = run_optimizer(
                    default_prices().model_copy(update={"currency": currency}),
                    default_constraints().model_copy(update={
                        "required_strength_mpa": required_strength,
                        "age_days": float(age_days),
                        "uncertainty_multiplier": uncertainty_multiplier,
                    }),
                    n_recs=5,
                )
            st.session_state["overview_result"] = result
        result = st.session_state.get("overview_result")
        if result is not None:
            if not result.success:
                st.error(result.message)
            else:
                best = result.recommendations.iloc[0]
                prices_now = default_prices().model_copy(update={"currency": currency})
                # Baseline: cheapest historical mix meeting the requirement at this age.
                candidates = df[
                    (df["strength_mpa"] >= required_strength)
                    & (df["age_days"] == float(age_days))
                ]
                if candidates.empty:
                    candidates = df[df["strength_mpa"] >= required_strength]
                baseline_row = None
                if not candidates.empty:
                    costs = material_cost_per_m3(candidates[list(INGREDIENTS)], prices_now)
                    baseline_row = candidates.iloc[int(np.argmin(costs))]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Recommended cost / m³",
                          f"{best['cost_per_m3']:.2f} {currency}")
                c2.metric("Conservative strength",
                          f"{best['conservative_strength']:.1f} MPa",
                          f"predicted {best['predicted_strength']:.1f} MPa")
                if baseline_row is not None:
                    summary = savings_summary(
                        baseline_row[list(INGREDIENTS)].to_dict(),
                        best[list(INGREDIENTS)].to_dict(),
                        prices_now,
                    )
                    c3.metric("Baseline cost / m³ (historical)",
                              f"{summary['baseline_cost_per_m3']:.2f} {currency}")
                    saving = summary["saving_per_m3"]
                    c4.metric("Saving / m³", f"{saving:.2f} {currency}",
                              f"{summary['saving_percent']:.1f}%")
                    yearly = annual_saving(saving, annual_volume, adoption_rate)
                    st.metric(
                        "Potential annual saving (illustrative, NOT a verified forecast)",
                        f"{yearly:,.0f} {currency}",
                        f"{annual_volume:,.0f} m³ × {adoption_rate:.0%} adoption",
                    )
                st.dataframe(result.recommendations.round(2), width="stretch")
                st.warning(
                    f"Risk: extrapolation warning level of top mix is "
                    f"**{best['warning_level']}**. All mixes require engineer "
                    "review and lab validation.", icon="⚠️",
                )

# ================================================================ page 2
elif page.startswith("2"):
    if df is None:
        st.error(dataset_error or "Dataset unavailable.")
        st.stop()
    report = build_quality_report(df[[c for c in df.columns if c in
                                      list(INGREDIENTS) + ["age_days", "strength_mpa"]]])
    st.subheader("Dataset summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", report["n_rows"])
    c2.metric("Duplicate rows (reported, kept)", report["duplicate_rows"])
    c3.metric("Missing values", sum(report["missing_values"].values()))
    st.dataframe(pd.DataFrame(report["field_stats"]).T, width="stretch")

    st.subheader("Filters")
    f1, f2 = st.columns(2)
    age_range = f1.slider("Age (days)", 1, int(df["age_days"].max()),
                          (1, int(df["age_days"].max())))
    strength_range = f2.slider(
        "Strength (MPa)", 0.0, float(df["strength_mpa"].max()),
        (0.0, float(df["strength_mpa"].max())),
    )
    view = df[
        df["age_days"].between(*age_range)
        & df["strength_mpa"].between(*strength_range)
    ]
    st.caption(f"{len(view)} rows after filtering")

    st.subheader("Distributions")
    plot_cols = list(INGREDIENTS) + ["age_days", "strength_mpa", "water_binder_ratio"]
    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    for ax, col in zip(axes.flat, plot_cols):
        ax.hist(view[col].dropna(), bins=30)
        ax.set_title(col, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation matrix")
    corr = view[plot_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(plot_cols)), plot_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(plot_cols)), plot_cols, fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Data quality flags")
    st.json({
        "outlier_counts_iqr": report["outlier_counts_iqr"],
        "water_binder_ratio": report["water_binder_ratio"],
        "total_mass_kg_m3": report["total_mass_kg_m3"],
    })

# ================================================================ page 3
elif page.startswith("3"):
    if metadata is None:
        st.error(model_error or "Model not trained. Run `python -m src.train`.")
        st.stop()
    comparison_path = config.METRICS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        st.subheader("Model comparison (validation set, sorted by MAE)")
        st.dataframe(pd.read_csv(comparison_path).round(3), width="stretch")
    st.subheader(f"Selected model: {metadata['model_type']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test MAE", f"{metadata['test_metrics']['mae']:.2f} MPa")
    c2.metric("Test RMSE", f"{metadata['test_metrics']['rmse']:.2f} MPa")
    c3.metric("Test R²", f"{metadata['test_metrics']['r2']:.3f}")
    st.caption(
        f"Cross-validation MAE: {metadata['cv_mae_mean']:.2f} ± "
        f"{metadata['cv_mae_std']:.2f} MPa (grouped 5-fold). "
        f"Split strategy: {metadata['split_strategy']}"
    )
    col_a, col_b = st.columns(2)
    with col_a:
        show_figure_if_exists("actual_vs_predicted.png", "Actual vs predicted (test)")
        show_figure_if_exists("residual_distribution.png", "Residual distribution")
        show_figure_if_exists("feature_importance.png", "Native feature importance")
    with col_b:
        show_figure_if_exists("residuals_vs_predicted.png", "Residuals vs predicted")
        show_figure_if_exists("cv_mae_distribution.png", "CV MAE distribution")
        show_figure_if_exists("permutation_importance.png", "Permutation importance")
    show_figure_if_exists("error_by_group.png", "Error by strength range and curing age")
    for name, title in [
        ("error_by_strength.csv", "Error by strength range"),
        ("error_by_age.csv", "Error by curing age"),
    ]:
        path = config.METRICS_DIR / name
        if path.exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(path).round(3), width="stretch")

# ================================================================ page 4
elif page.startswith("4"):
    if df is None or metadata is None:
        st.error(dataset_error or model_error or "Dataset/model unavailable.")
        st.stop()
    st.subheader("Inputs")
    with st.expander("Material prices", expanded=False):
        prices = render_prices_editor("opt")
    with st.expander("Engineering constraints", expanded=False):
        constraints = render_constraint_editor("opt")
    n_recs = st.number_input("Number of recommendations", 1, 10, 5)

    st.subheader("Baseline")
    baseline_mode = st.radio(
        "Baseline method",
        ["Nearest historical recipe meeting required strength",
         "Selected historical recipe", "User-entered recipe"],
        horizontal=True,
    )
    baseline_mix = None
    baseline_strength = None
    if baseline_mode == "Nearest historical recipe meeting required strength":
        candidates = df[
            (df["strength_mpa"] >= required_strength)
            & (df["age_days"] == float(age_days))
        ]
        if candidates.empty:
            candidates = df[df["strength_mpa"] >= required_strength]
        if candidates.empty:
            st.warning("No historical recipe meets the required strength.")
        else:
            costs = material_cost_per_m3(candidates[list(INGREDIENTS)], prices)
            row = candidates.iloc[int(np.argmin(costs))]
            baseline_mix = row[list(INGREDIENTS)].to_dict()
            baseline_strength = float(row["strength_mpa"])
            st.caption(
                f"Cheapest historical mix meeting {required_strength:.0f} MPa: "
                f"measured {baseline_strength:.1f} MPa @ {row['age_days']:.0f} d"
            )
    elif baseline_mode == "Selected historical recipe":
        idx = st.number_input("Dataset row index", 0, len(df) - 1, 0)
        row = df.iloc[int(idx)]
        baseline_mix = row[list(INGREDIENTS)].to_dict()
        baseline_strength = float(row["strength_mpa"])
        st.dataframe(row[list(INGREDIENTS) + ["age_days", "strength_mpa"]].to_frame().T)
    else:
        cols = st.columns(4)
        baseline_mix = {}
        defaults = df[list(INGREDIENTS)].median()
        for i, name in enumerate(INGREDIENTS):
            with cols[i % 4]:
                baseline_mix[name] = st.number_input(
                    f"{INGREDIENT_LABELS[name]} (kg/m³)", 0.0,
                    value=float(defaults[name]), key=f"base_{name}",
                )

    if st.button("Run optimizer", type="primary"):
        with st.spinner("Running differential evolution with multiple restarts..."):
            result, extrapolator, estimator = run_optimizer(prices, constraints, int(n_recs))
        st.session_state["opt_result"] = (result, prices)
    stored = st.session_state.get("opt_result")
    if stored:
        result, used_prices = stored
        extrapolator = get_extrapolator(dataset_path_input)
        if not result.success:
            st.error(result.message)
            st.json(result.violation_counts)
        else:
            st.success(result.message)
            recs = result.recommendations
            st.subheader("Ranked recommendations")
            st.dataframe(recs.round(3), width="stretch")
            st.download_button(
                "Download recommendations (CSV)",
                recs.to_csv(index=False).encode(),
                file_name="mix_recommendations.csv",
                mime="text/csv",
            )
            high_risk = recs[recs["warning_level"] != "low"]
            if not high_risk.empty:
                st.warning(
                    f"{len(high_risk)} recommendation(s) have medium/high "
                    "extrapolation warnings — they sit near or beyond the edge "
                    "of the historical data. Treat with extra skepticism.",
                    icon="⚠️",
                )

            best = recs.iloc[0]
            st.subheader("Top recommendation vs baseline")
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            labels = [INGREDIENT_LABELS[n] for n in INGREDIENTS]
            x = np.arange(len(INGREDIENTS))
            axes[0].bar(x - 0.2, [best[n] for n in INGREDIENTS], 0.4, label="Recommended")
            if baseline_mix:
                axes[0].bar(x + 0.2, [baseline_mix[n] for n in INGREDIENTS], 0.4,
                            label="Baseline")
            axes[0].set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
            axes[0].set_ylabel("kg/m³")
            axes[0].set_title("Recipe composition")
            axes[0].legend()
            axes[1].bar(["Predicted", "Conservative", "Required"],
                        [best["predicted_strength"], best["conservative_strength"],
                         required_strength],
                        color=["tab:blue", "tab:orange", "tab:red"])
            axes[1].set_ylabel("MPa")
            axes[1].set_title("Strength comparison")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            if baseline_mix:
                summary = savings_summary(
                    baseline_mix, best[list(INGREDIENTS)].to_dict(), used_prices,
                    baseline_strength=baseline_strength,
                    recommended_strength=float(best["predicted_strength"]),
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Baseline cost / m³",
                          f"{summary['baseline_cost_per_m3']:.2f} {currency}")
                c2.metric("Recommended cost / m³",
                          f"{summary['recommended_cost_per_m3']:.2f} {currency}")
                c3.metric("Saving / m³",
                          f"{summary['saving_per_m3']:.2f} {currency}",
                          f"{summary['saving_percent']:.1f}%")
                c4.metric("Cement reduction",
                          f"{summary['cement_reduction_kg_m3']:.0f} kg/m³",
                          f"binder −{summary['binder_reduction_kg_m3']:.0f} kg/m³")
                yearly = annual_saving(summary["saving_per_m3"], annual_volume,
                                       adoption_rate)
                st.metric(
                    "Potential annual saving (illustrative)",
                    f"{yearly:,.0f} {currency}",
                )

            st.subheader("Nearest historical recipes (top recommendation)")
            neighbors = extrapolator.nearest_recipes(best, k=3)
            st.dataframe(neighbors.round(2), width="stretch")
            st.caption(
                "Compare the recommendation against these real historical mixes "
                "before taking it to an engineer."
            )

# ================================================================ page 5
elif page.startswith("5"):
    if df is None or metadata is None:
        st.error(dataset_error or model_error or "Dataset/model unavailable.")
        st.stop()
    st.caption(
        "Vary one assumption and re-run the optimizer to see how the "
        "recommended recipe and its cost respond."
    )
    base_prices = default_prices().model_copy(update={"currency": currency})
    scenario_var = st.selectbox(
        "Variable to sweep",
        ["Cement price", "Slag price", "Fly ash price",
         "Required strength", "Safety margin (uncertainty multiplier)"],
    )
    n_points = st.slider("Scenario points", 2, 5, 3)
    if scenario_var == "Cement price":
        values = np.linspace(base_prices.cement * 0.5, base_prices.cement * 1.5, n_points)
    elif scenario_var == "Slag price":
        values = np.linspace(base_prices.slag * 0.5, base_prices.slag * 1.5, n_points)
    elif scenario_var == "Fly ash price":
        values = np.linspace(base_prices.fly_ash * 0.5, base_prices.fly_ash * 1.5, n_points)
    elif scenario_var == "Required strength":
        values = np.linspace(max(required_strength - 10, 10), required_strength + 10, n_points)
    else:
        values = np.linspace(0.5, 2.0, n_points)

    if st.button("Run scenario sweep", type="primary"):
        rows = []
        progress = st.progress(0.0)
        for i, value in enumerate(values):
            prices = base_prices
            cons = default_constraints().model_copy(update={
                "required_strength_mpa": required_strength,
                "age_days": float(age_days),
                "uncertainty_multiplier": uncertainty_multiplier,
            })
            if scenario_var == "Cement price":
                prices = base_prices.model_copy(update={"cement": float(value)})
            elif scenario_var == "Slag price":
                prices = base_prices.model_copy(update={"slag": float(value)})
            elif scenario_var == "Fly ash price":
                prices = base_prices.model_copy(update={"fly_ash": float(value)})
            elif scenario_var == "Required strength":
                cons = cons.model_copy(update={"required_strength_mpa": float(value)})
            else:
                cons = cons.model_copy(update={"uncertainty_multiplier": float(value)})
            estimator = build_estimator(
                model, metadata,
                cons.uncertainty_multiplier,
            )
            result = optimize_mixes(
                estimator, prices, cons, get_extrapolator(dataset_path_input),
                n_recommendations=1, max_runs=4, seed=config.RANDOM_SEED,
            )
            if result.success:
                best = result.recommendations.iloc[0]
                rows.append({
                    scenario_var: float(value),
                    "cost_per_m3": best["cost_per_m3"],
                    "cement": best["cement"], "slag": best["slag"],
                    "fly_ash": best["fly_ash"], "water": best["water"],
                    "conservative_strength": best["conservative_strength"],
                    "warning_level": best["warning_level"],
                })
            else:
                rows.append({scenario_var: float(value), "cost_per_m3": np.nan,
                             "warning_level": "infeasible"})
            progress.progress((i + 1) / len(values))
        scenario_df = pd.DataFrame(rows)
        st.session_state["scenario_df"] = (scenario_var, scenario_df)
    stored = st.session_state.get("scenario_df")
    if stored:
        scenario_var_used, scenario_df = stored
        st.dataframe(scenario_df.round(3), width="stretch")
        valid = scenario_df.dropna(subset=["cost_per_m3"])
        if not valid.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(valid[scenario_var_used], valid["cost_per_m3"], "o-")
            axes[0].set_xlabel(scenario_var_used)
            axes[0].set_ylabel(f"Optimal cost ({currency}/m³)")
            axes[0].set_title("Cost response")
            for col in ["cement", "slag", "fly_ash", "water"]:
                if col in valid:
                    axes[1].plot(valid[scenario_var_used], valid[col], "o-", label=col)
            axes[1].set_xlabel(scenario_var_used)
            axes[1].set_ylabel("kg/m³")
            axes[1].set_title("Recipe response")
            axes[1].legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ================================================================ page 6
else:
    st.subheader("What this prototype is — and is not")
    st.markdown(
        """
**Public dataset limitations.** The UCI dataset (~1,030 lab specimens) lacks
local raw-material chemistry, aggregate grading and moisture, admixture brands,
plant-specific process data, and environmental conditions. Its mixes may not
resemble CEMEX production mixes.

**Missing performance dimensions.** The model predicts compressive strength
only. It knows nothing about durability targets, workability or slump,
setting time, air content, shrinkage, exposure classes, or pumpability — all
of which can make a "cheap" mix unusable in practice.

**Model uncertainty.** The uncertainty estimate (tree-ensemble spread plus
validation error buffer) is a prototype heuristic, **not** a calibrated safety
guarantee. Prediction ≠ causation: the model learns correlations in historical
lab data, not physical laws.

**Extrapolation danger.** Cost optimizers are drawn to the edges of the data.
The extrapolation score guards against this, but any recommendation near the
historical boundary deserves extra skepticism.

**Mandatory human steps.** Every recommendation requires review by qualified
materials engineers and laboratory + production validation. The tool cannot
approve a recipe, and nothing in this app should be read as implying it can.
        """
    )
    st.subheader("Questions requiring a concrete materials expert")
    for i, q in enumerate(EXPERT_QUESTIONS, 1):
        st.markdown(f"{i}. {q}")

    st.subheader("Future CEMEX data design (sample schema)")
    st.caption(
        "What a stronger internal dataset could contain. NOT required for "
        "this public-data prototype."
    )
    st.json({
        "identifiers": ["batch_id", "recipe_id", "plant_id", "product_id",
                         "production_timestamp"],
        "ingredient_quantities": [
            "cement_kg_m3", "slag_kg_m3", "fly_ash_kg_m3",
            "limestone_filler_kg_m3", "silica_fume_kg_m3", "water_kg_m3",
            "fine_aggregate_kg_m3", "coarse_aggregate_kg_m3",
            "admixture_1_kg_m3", "admixture_2_kg_m3"],
        "material_properties": [
            "cement_type", "cement_blaine", "cement_chemistry",
            "aggregate_source", "aggregate_grading", "aggregate_moisture",
            "aggregate_absorption", "supplementary_material_properties"],
        "production_conditions": [
            "mixer_type", "mixing_time", "ambient_temperature",
            "concrete_temperature", "transport_time", "delivery_distance"],
        "performance_targets": [
            "target_strength_mpa", "slump_target", "exposure_class",
            "durability_class", "maximum_water_binder_ratio"],
        "measured_outcomes": [
            "slump_actual", "air_content", "density", "strength_1_day",
            "strength_7_day", "strength_28_day", "setting_time",
            "rejected_batch", "rejection_reason"],
        "economics": [
            "material_price_at_production", "delivered_material_cost",
            "energy_cost", "total_variable_cost"],
        "environmental": ["embodied_co2_per_material", "total_embodied_co2"],
    })
