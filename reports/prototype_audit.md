# Prototype Audit — CEMEX Concrete Mix Optimizer

> **Addendum (2026-07-14):** since this audit was written, the project added
> its primary practical tool — the model-free EN 206 overdesign audit
> (`src/audit.py`), an advisory slump screen (`src/workability.py`), and
> exposure-class durability constraints. This document reviews the original
> ML optimizer prototype; the overdesign audit deliberately sidesteps most
> of the model-trust risks listed below by using only the code's conformity
> formula on the plant's own records.

Reviewed from four perspectives: machine-learning engineer, concrete materials
engineer, CEMEX operations manager, and financial analyst.

Audit date: 2026-07-13. Model under audit: ExtraTreesRegressor
(validation MAE 4.12 MPa, single test evaluation MAE 3.30 MPa, R² 0.90,
grouped 5-fold CV MAE 4.17 ± 0.56 MPa on the UCI dataset, 1,030 rows).

---

## What the prototype proves

- A standard ML pipeline on 1,030 public lab records predicts 28-day-class
  compressive strength with a test MAE of ~3.3 MPa — accurate enough to rank
  candidate mixes, on this dataset.
- Constrained global optimization (differential evolution with penalty terms)
  can search the ingredient space and return multiple *distinct* mixtures that
  satisfy a conservative strength requirement at materially different costs.
- Guardrails are implementable and cheap: conservative lower-bound strength,
  three extrapolation checks, hard rejection of high-risk candidates, and
  transparent failure messages when constraints are infeasible.
- The whole loop (data validation → training → uncertainty → optimization →
  economics → UI) runs locally in minutes and is structured so internal data
  can be dropped in without rewriting the pipeline.

## What it does not prove

- **That any recommended mix works in practice.** The model predicts one
  property (compressive strength) from lab specimens. It knows nothing about
  slump/workability, setting time, durability, air content, pumpability,
  shrinkage, or exposure classes — any of which can disqualify a "cheap" mix.
- **That the savings are real.** Prices are illustrative placeholders; only
  material cost is modeled. Baseline mixes come from the same public dataset,
  not from CEMEX production recipes.
- **That the uncertainty bound is safe.** The conservative-strength formula is
  a heuristic (tree spread + validation-error buffer), not a calibrated
  prediction interval, and carries no statistical guarantee.
- **Transferability.** UCI lab mixes with unknown material sources need not
  behave like any CEMEX plant's materials. A model trained on this data says
  nothing quantitative about production concrete in a specific market.

## Technical risks (ML engineer)

- Optimization pressure drives candidates toward the boundary of the training
  distribution — observed in practice: top recommendations sit at
  extrapolation scores 1.1–1.2 (just under the 1.2 cutoff). The guardrail
  works, but predictions at the boundary are the least reliable ones.
- The dataset is small (1,030 rows, ~425 unique recipes after grouping);
  validation metrics carry meaningful variance (CV std ±0.56 MPa).
- Tree-ensemble spread underestimates total predictive uncertainty; the
  validation-RMSE floor mitigates but does not calibrate it. Proper conformal
  or quantile methods are the next step.
- Duplicated rows in the UCI source are reported but retained; group-aware
  splitting prevents the worst leakage, yet subtle batch effects are unknowable
  from the public data.

## Engineering risks (concrete materials engineer)

- Constraint defaults are dataset ranges, not specifications. No standard
  (EN 206, ACI 318, local norms) is encoded. Water-binder limits, minimum
  binder contents per exposure class, and admixture compatibility are absent.
- Superplasticizer dosage interacts with cement chemistry and temperature;
  the model treats it as a free numeric knob.
- Aggregate moisture correction, grading, and yield (m³ check via absolute
  volume method) are not modeled — recommended masses may not batch to 1 m³.
- The tool could anchor engineers to model output ("automation bias"); the UI
  repeats the mandatory-review notice on every page to counter this.

## Financial risks (financial analyst)

- Annual savings = saving/m³ × volume × adoption is deliberately naive: no
  price volatility, no supply constraints on slag/fly ash (whose markets are
  tightening), no requalification or lab-testing costs, no phased rollout.
- Savings versus the *cheapest historical* baseline overstates realizable
  savings versus an already-optimized plant recipe.
- Fly-ash and slag availability and logistics costs vary by region; the
  optimizer will happily recommend SCM-heavy mixes wherever their listed price
  is low.

## Operational risks (operations manager)

- No integration with batching systems, QC LIMS, or dispatch; a real rollout
  needs recipe versioning, approval workflow, and audit trails.
- One global model; real operations need per-plant (at least per-region)
  models with local materials.
- No monitoring/retraining loop: material drift (new clinker source, new
  quarry face) silently degrades predictions.

## Data needed from CEMEX

See the sample schema in the app (page 6) and README. Highest-value items:
batch-level ingredient masses with plant and timestamp, measured strengths at
multiple ages, slump/air/density, aggregate moisture and grading, cement type
and fineness, rejected-batch records, and delivered material costs at
production time. Even 12 months of one plant's batch records would exceed the
public dataset in relevance.

## Recommended next steps

1. Workshop with a concrete materials engineer using the 20-question checklist
   (app page 6) to define real constraints and objectives (cost vs CO₂).
2. Pilot on one plant's historical batch data; re-benchmark models per plant.
3. Replace the heuristic bound with calibrated intervals (conformal prediction
   or quantile gradient boosting) and validate empirical coverage.
4. Add workability/slump as a second predicted target and durability rules as
   hard constraints; add absolute-volume yield check so mixes batch to 1 m³.
5. Define a lab-validation protocol: every accepted recommendation gets trial
   batches before any production use.
6. Add cost-versus-CO₂ multi-objective mode (embodied CO₂ per material is a
   natural extension of the economics module).

## Misleading-claims audit

Checked the UI, README and code for overclaiming:

- The safety notice appears in the sidebar and on every page; the app never
  claims approval authority.
- Regression quality is reported as MAE/RMSE/R², never "accuracy".
- Prices and constraints are labeled illustrative / requiring engineer review
  at every point of entry.
- Annual savings are labeled "illustrative, NOT a verified forecast" where
  displayed.
- Uncertainty is labeled a prototype heuristic, not a calibrated guarantee, in
  the code, the README, and the UI.
- The dataset loader refuses to fabricate data; synthetic data exists only in
  the test suite.

No misleading claims found after this pass; residual risk is a reader skimming
past the labels, which the repetition of the notice is designed to mitigate.
