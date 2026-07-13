# CEMEX Concrete Mix Optimizer (Prototype)

> ⚠️ **This application is a demonstration decision-support tool. It is not
> approved for production use. Every proposed concrete mixture must be
> reviewed by qualified materials engineers and validated through laboratory
> and production testing before use.**

## Objective

Demonstrate how machine learning plus constrained optimization can answer:

> *"What is the lowest-cost concrete mixture that is predicted to achieve a
> required compressive strength, while respecting engineering and material
> constraints?"*

Built on the public UCI Concrete Compressive Strength dataset and designed so
CEMEX data and real engineering constraints can replace the public assumptions
later.

## Screenshot

*(screenshot placeholder — add a capture of the Streamlit executive overview here)*

## Architecture

```
Historical mixtures → ML strength prediction → constrained cost optimization
                    → engineer review → laboratory validation
```

```
cemex-concrete-optimizer/
├── app.py                  # Streamlit UI (6 pages)
├── config/                 # Editable prices & constraints (illustrative)
├── data/raw/               # Put concrete_data.xlsx / .csv here
├── data/processed/         # Written by training
├── models/                 # best_model.joblib + model_metadata.json
├── reports/figures|metrics # Evaluation artifacts, data-quality report
├── src/
│   ├── data_loader.py      # Load + normalize columns
│   ├── validation.py       # Fatal checks + data-quality report
│   ├── features.py         # Derived features, recipe group key
│   ├── train.py            # Group-aware split, model zoo, selection
│   ├── evaluate.py         # Plots, grouped error tables
│   ├── uncertainty.py      # Conservative strength estimate
│   ├── extrapolation.py    # Distance-to-history checks
│   ├── optimizer.py        # Differential evolution + penalties + diversity
│   ├── economics.py        # Costs, savings, annual projection
│   └── schemas.py          # Pydantic prices/constraints/recommendation
└── tests/                  # pytest suite (synthetic data ONLY in tests)
```

## Installation

```bash
python -m venv .venv
```

Mac/Linux:

```bash
source .venv/bin/activate
```

Windows:

```
.venv\Scripts\activate
```

Then:

```bash
pip install -r requirements.txt
```

## Dataset setup

1. Download the UCI Concrete Compressive Strength dataset (~1,030 rows):
   <https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength>
2. The archive contains `Concrete_Data.xls`. Save/export it as either
   `concrete_data.xlsx` or `concrete_data.csv`.
3. Place the file at `data/raw/concrete_data.xlsx` (or `.csv`).

If the file is missing, training and the app stop with these instructions —
the tool never substitutes fake data (a small synthetic dataset exists only
inside the automated tests).

**Dataset attribution:** I-Cheng Yeh, *Concrete Compressive Strength*, UCI
Machine Learning Repository (1998), <https://doi.org/10.24432/C5PK67>,
licensed CC BY 4.0. This repository includes a copy at
`data/raw/concrete_data.xlsx` so the deployed demo works out of the box.

## Commands

```bash
python -m src.train      # train, compare, select, evaluate, save model
streamlit run app.py     # launch the application
pytest                   # run the test suite
```

## Model metrics (what they mean)

* **MAE** — mean absolute error in MPa: on average, how far predictions miss.
* **RMSE** — root mean squared error in MPa: like MAE but punishes large misses.
* **R²** — fraction of strength variance explained (1.0 = perfect).

Regression performance is never called "accuracy". Models are compared on a
group-aware validation set; the winner is evaluated **once** on an untouched
test set. Because the same recipe appears at several curing ages, splits are
grouped by ingredient quantities (rounded to the nearest kg, age excluded) so
related rows never straddle a train/test boundary.

## Uncertainty (prototype heuristic — not a calibrated guarantee)

```
conservative_strength = predicted_strength
                        − uncertainty_multiplier × uncertainty_mpa
                        − validation_error_buffer
```

* `uncertainty_mpa`: spread of individual tree predictions for bagged
  ensembles (floored at validation RMSE); validation RMSE otherwise.
* `validation_error_buffer`: the model's validation MAE.
* A candidate is valid only if `conservative_strength ≥ required_strength`.

## Optimization

`scipy.optimize.differential_evolution` (vectorized) minimizes material cost
per m³ over the seven ingredient quantities with strong penalties for:
strength shortfall, water-to-binder limits, ingredient limits, total-mass
limits, extrapolation score, zero binder, and negative values. The optimizer
restarts with multiple seeds and applies a minimum-distance diversity penalty,
so it returns several *distinct* valid recipes. If nothing is feasible, it
reports which constraints blocked the search instead of returning an invalid
mix.

Extrapolation is scored three ways — training-range check, standardized
distance from the data center, and nearest-neighbor distance — normalized so
1.0 ≈ the edge of the historical envelope. High-risk candidates are rejected
by default and nearest historical recipes are shown for comparison.

## Limitations

* Public lab data: no local material chemistry, aggregate grading/moisture,
  plant process data, or environmental conditions.
* Predicts compressive strength only — no durability, slump/workability,
  setting time, air content, or exposure-class awareness.
* Prices are illustrative placeholders; only material cost is modeled.
* Uncertainty estimate is heuristic; prediction is correlation, not causation.
* Cost optimizers gravitate to data edges; extrapolation checks mitigate but
  do not remove this risk.
* Annual savings figures are illustrative arithmetic, not verified forecasts.

## How CEMEX internal data could replace the public data

Swap `data/raw/concrete_data.*` for internal batch records following the
sample schema on the app's *Assumptions & limitations* page (batch/plant IDs,
material properties, production conditions, slump and durability outcomes,
real delivered costs, embodied CO₂). Replace `config/default_constraints.yaml`
with plant-specific specification limits, and `config/default_prices.yaml`
with live procurement prices. The training, uncertainty, extrapolation, and
optimization layers are agnostic to the data source.

## Questions requiring a concrete materials expert

1. Are we optimizing ready-mix concrete, cement composition or both?
2. What properties matter besides compressive strength?
3. What are the minimum and maximum quantities for each material?
4. What water-to-binder limits are allowed?
5. What durability standards apply?
6. Is slump or workability required?
7. Are chloride, sulfate, alkali or exposure-class constraints required?
8. Are recipes specific to each plant?
9. Are supplier and quarry properties recorded?
10. How is aggregate moisture handled?
11. How are material costs stored?
12. What safety margins are used?
13. Which curing ages matter?
14. How are failed batches recorded?
15. What laboratory validation process is required?
16. How much recipe variation is permitted?
17. What data systems contain historical recipes and test results?
18. Should recommendations optimize cost, carbon emissions or both?
19. What regulations and internal approvals apply?
20. How should uncertainty be communicated to engineers?
