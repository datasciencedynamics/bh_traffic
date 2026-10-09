<p align="left">
  <img src="https://raw.githubusercontent.com/datasciencedynamics/datasciencedynamics.github.io/refs/heads/main/Astro/public/data_science_dynamics_logo.svg" alt="Data Science Dynamics" width="250"/>
</p>

# Beverly Hills Collision Injury or Fatality Prediction

Machine learning pipeline built on the Beverly Hills Police Department (BHPD)
Traffic Bureau collision records published on the City of Beverly Hills Open
Data portal (records since January 1, 2015). Two analyses share one
preprocessing stage:

1. **Severity classifier.** Whether a reported collision results in injury or
   death. Four classifiers (logistic regression, random forest, XGBoost,
   CatBoost) under a common protocol, with an audit for vulnerable road users
   (pedestrians, bicyclists, motorcyclists and scooter riders) and an ablation
   that removes every "who was involved" feature.
2. **Hotspot screening.** Where injury-or-fatal crashes concentrate. A
   location x month panel, Poisson count models on a temporal split, and
   Empirical Bayes network screening with a backtest, producing a ranked
   table of intersections for all injury crashes and for crashes involving
   vulnerable road users. See [Hotspot screening](#hotspot-screening).

---

## Setup

After cloning, run once:

    pip install -r requirements-dev.txt
    git config core.hooksPath .githooks

This strips notebook outputs automatically on commit. CI rejects
notebooks with outputs, so skipping this means your PR will fail.

## Data

| Item | Value |
|---|---|
| Source | City of Beverly Hills Open Data, "Historic Collision Records of Beverly Hills (New)" |
| Publisher | BHPD Traffic Bureau |
| Records | 8,551 rows; 48 duplicate accident numbers dropped, leaving 8,503 |
| Coverage | January 2015 onward |
| Raw file | `data/raw/bh_collisions.csv` |
| Outcome | Any injury or fatality (`injury_or_fatal`), 19 fatal |

Data is not redistributed in this repository. Download the CSV from the portal
and place it at `data/raw/bh_collisions.csv` before running the pipeline.

### Data quality notes handled in `preprocessing.py`

- **Shuffled fields.** From late 2025 onward, 421 rows arrive with
  `Day of Week`, `Time`, and `Date` permuted (two different permutations).
  They are detected by a weekday name in `Time` and realigned by pattern.
- **Mixed formats.** Dates appear as `8/27/2015 0:00` and `6/12/2025`; times as
  `16:08:00`, `3:15:00 PM`, and military `1458`.
- **Outcome definition.** The injured/killed counts and the free-text
  `Accident Type` disagree on roughly 100 rows; a collision is positive if
  either source indicates injury or death. Both are dropped from `X`.
- **Duplicates.** 48 accident numbers appear twice, and each pair shares the
  same outcome. The first row is kept so twins cannot land on opposite sides
  of the train/test split. The index is `ObjectId`; `accident_number` is
  retained (string, excluded from `X`) for linking to the state CCRS / SWITRS
  records.

---

## Leakage and what the target means

BHPD files injury reports as a matter of course, but non-injury reports mostly
in three circumstances: hit and run, DUI, and city property damage (CPD).
Untagged rows are about 96% injury; tagged rows are about 74% non-injury. Most
ordinary property-damage crashes never enter the data.

The target is therefore **injury or death among reported collisions**, not
among all collisions, and several fields encode the reporting rule or the
label itself. These are listed in `core/constants.py: leak_cols` and dropped
in `feat_gen.py` before `X_columns_list` is saved, so inference inherits the
same exclusion:

| Columns | Why excluded |
|---|---|
| `hit_run`, `dui`, `cpd` | Report tags parsed from `Accident Type`, the same field as the label |
| `pcf_section_23152`, `other_imp_drive_dui`, `other_imp_drive_driving_under_influence_of_alcohol_42104` | Duplicate the DUI tag |
| `pcf_section_23153` | DUI causing bodily injury: injury by legal definition |
| `pcf_section_20001` | Hit and run with injury or death: injury by legal definition |
| `pcf_section_20002` | Hit and run, property damage only: non-injury by legal definition |

Effect on discrimination (logistic regression, 5-fold CV ROC AUC, deduplicated
data):

| Feature set | AUC |
|---|---|
| Report tags only (3 features) | 0.892 |
| All features including leakage | 0.945 |
| `leak_cols` removed (158 features) | 0.857 |

A temporal holdout (train 2015 to 2022, test 2024 onward) matched the random
split, so drift over time is not inflating results.

### Intended use

Cause fields (`other_imp_drive`, `pcf_section`) and collision geometry are
officer judgments recorded after the crash. They are appropriate for severity
triage of a filed report. For prospective risk (where and when injury
collisions occur), only place and time are known in advance, and
discrimination falls to roughly 0.68 AUC.

---

## Hotspot screening

### Panel

`preprocessing.py` builds a `location_key` for every crash: the two streets of
an intersection in alphabetical order (`LA CIENEGA BLVD / WILSHIRE BLVD`), or,
for a mid-block address, the street and its recorded nearest cross street. A
mid-block address with no cross street becomes `<STREET> (MIDBLOCK)`. House
numbers, suffix variants (`BL`, `BLVD`), and N/S prefixes are normalized, with
one exception: North and South Santa Monica Blvd are separate parallel roads,
so their direction is kept.

`panel_gen.py` keeps locations with at least 5 crashes through 2022 (252
locations, 83.5% of injury-or-fatal crashes) and counts, per location and
month, injury-or-fatal crashes (`inj_count`) and those involving a vulnerable
road user (`vru_inj_count`).

Every feature for month *t* uses crashes through *t* - 12 only: the same month
last year, 12- and 24-month rolling sums, the long-run monthly rate, the
citywide 12-month trend, calendar terms, and mid-block and corridor flags. A
forecast for any of the next 12 months is therefore built from data already
on record, and the test period is scored at the horizon the forecast is used
at.

| Split | Months |
|---|---|
| Train | 2017-01 to 2022-12 |
| Valid | 2023 |
| Test | 2024-01 to the last month on record |
| Forecast | The next 12 months |

### Count models

`train_counts.py` fits a Poisson GLM (`pr`), XGBoost (`count:poisson`), and
CatBoost (`Poisson`) through `model_tuner` in regression mode, passing the
temporal split as `custom_splits` and tuning on `neg_mean_poisson_deviance`.
`evaluation_counts.py` scores each against two baselines computed from the
same lagged history: the long-run monthly rate and the last 12 months.

Test period, `inj_count`:

| Model | Poisson deviance | MAE | Predicted / observed |
|---|---|---|---|
| CatBoost | 0.493 | 0.201 | 0.96 |
| XGBoost | 0.494 | 0.201 | 0.98 |
| Poisson GLM | 0.501 | 0.211 | 1.06 |
| History rate | 0.499 | 0.213 | 1.14 |

The models are well calibrated, where the history rate overpredicts by 14%
because crash counts have declined. Their gain in deviance over the history
rate is small.

Each evaluation also writes, to `models/eval/<count outcome>/` and to MLflow:

| File | Shows |
|---|---|
| `<model>_orig_<outcome>_test_location_calibration.png` | Predicted vs observed test-period crashes per location |
| `<model>_orig_<outcome>_test_lorenz.png` | Lorenz curves on the test period: the model, both baselines, and perfect foresight |
| `<model>_orig_<outcome>_test_shap_beeswarm.png` | SHAP beeswarm, top 15 features |
| `<model>_orig_shap_importance.csv` | Mean absolute SHAP for every feature |

SHAP values explain the log of the expected count (the Poisson link), so
they add on the log scale and exponentiate to multiplicative effects: +0.4
means about 1.5 times the expected crashes. XGBoost uses its native TreeSHAP
(`pred_contribs`), because shap's `TreeExplainer` cannot read the
`base_score` format of XGBoost 2.1 and later; CatBoost uses `TreeExplainer`
and the Poisson GLM `LinearExplainer`. For all three, the long-run injury and
crash rates at the location dominate, followed by corridor (Olympic, Santa
Monica, Canon, Rodeo), mid-block (lower), and the citywide trend.

The test-period Lorenz curve sums one prediction per month over the whole
test period, which turns the last-12-months baseline into a moving multi-year
average; it ties the models there. The cutoff backtest below is the test of
ranking as the city would use it.

### Empirical Bayes ranking

`hotspots.py` ranks locations by the Highway Safety Manual Empirical Bayes
(EB) estimate. A safety performance function (Poisson GLM on mid-block and
corridor flags) gives the expected count for a location of its type; EB blends
it with the location's own last 60 months:

    w  = 1 / (1 + alpha * SPF)
    EB = w * SPF + (1 - w) * observed

with negative binomial overdispersion *alpha* estimated by moments. EB pulls a
location with one unusual year back toward its type. `excess_inj_per_year`
(EB minus SPF) is the HSM potential for safety improvement.

### Backtest

At each cutoff, locations are ranked with data through the cutoff only and
scored on the next 12 months (`inj_count`, Spearman correlation between the
ranking and observed counts):

| Ranking | 2022-12 | 2023-12 | 2024-12 |
|---|---|---|---|
| Observed, last 12 months | 0.44 | 0.51 | 0.45 |
| Observed, last 36 months | 0.55 | 0.61 | 0.53 |
| Observed, last 60 months | 0.60 | 0.63 | 0.58 |
| Empirical Bayes (60 months) | 0.62 | 0.63 | 0.57 |
| ML forecast (best count model) | 0.66* | 0.61 | 0.57 |

\* validation period, used for early stopping and model selection.

Ranking by last year's crashes, the common practice, is the weakest option.
Five years of history, with or without EB, is consistently better. The ML
forecast matches EB but does not beat it out of sample, so the table is ranked
by EB, the method traffic engineers already recognize, and the ML forecast is
reported alongside it. The top 10 locations capture 12 to 15% of next-year
injury crashes against 18 to 19% with perfect foresight: year-to-year noise at
a single intersection is large.

VRU rankings are weaker (Spearman 0.33 to 0.39): injury crashes involving a
pedestrian, cyclist, or motorcyclist are rare and scattered.

### Concentration: is it 80/20?

Hypothesis: a small set of intersections, identifiable from past crashes,
accounts for a disproportionate and persistent share of future injury
crashes. `hotspots.py` tests it three ways (top 20% of locations):

| Test | All injury | Pedestrian, cyclist, motorcyclist |
|---|---|---|
| In-sample, citywide (naive): top 20% of 1,377 locations | 86% | 90% |
| Out-of-sample, panel: top 20% of 252 by crashes through 2022, share of 2023 onward | 44% | 47% |
| Same, with perfect foresight | 50% | 64% |
| Out-of-sample, citywide: those 50 locations (4.5% of all) vs every later injury crash | 36% | 39% |

The naive view clears 80/20 easily, but it is inflated twice: ranking and
scoring on the same years rewards one bad year, and hundreds of one-off
locations pad the denominator. Out of sample, among established
intersections, it is closer to 20/45, within a few points of perfect
foresight for all injury crashes. The defensible claim: about 4.5% of crash
locations, chosen from history alone, account for about a third of the next
three and a half years of injury crashes citywide.

Outputs: `concentration_through_<month>.csv`, `lorenz_through_<month>.csv`,
and `lorenz_through_<month>.png` (Lorenz curves, ranked by past crashes vs
perfect foresight).

### Output

`data/processed/hotspots/hotspots_through_<month>.csv`, one row per location,
ranked by `eb_inj_per_year`:

| Column | Meaning |
|---|---|
| `eb_inj_per_year` | EB expected injury-or-fatal crashes per year |
| `spf_inj_per_year` | Expected for a location of this type |
| `excess_inj_per_year` | EB minus SPF (potential for safety improvement) |
| `obs_inj_last12`, `obs_inj_last60` | Observed counts |
| `ml_inj_next12` | ML forecast, next 12 months |
| `vru_rank`, `*_vru_inj_*` | The same for VRU injury crashes |

### Hotspot limitations

- No traffic or pedestrian volume. The SPF knows location type only, so EB
  ranks expected crash frequency, not risk per vehicle or pedestrian. Volume
  (for example, city traffic counts) would sharpen the SPF and enable rates.
- The location set is fixed by crashes through 2022; a location that became
  dangerous later is absent until the threshold or train window changes.
- About 9% of Santa Monica Blvd records give no North/South direction and form
  their own keys.
- Mid-block crashes without a cross street are grouped by street.

---

## Dash app

The interactive front end lives in the `flask_apps` repo (`bh_traffic/`). It
reads only flat files exported from here:

```bash
make export_dash FLASK_APP_DATA=../flask_apps/bh_traffic/data
```

| Target | Does |
|---|---|
| `export_dash` | Writes the app's data files to `data/dash/` (and copies them to `FLASK_APP_DATA` if set) |
| `refresh_coords` | Re-downloads OpenStreetMap street geometry and rebuilds the coordinate cache |
| `audit_top` | Pre-release check of every top-10 location, to `data/audit/` |
| `brief` | One-page PDF brief for decision makers, to `data/brief/` |

`modeling/export_dash.py` writes the ranked hotspot table with coordinates,
monthly history and the 12-month forecast per location, the backtest,
concentration, and Lorenz files, count-model metrics, a SHAP sample,
severity-classifier test predictions and MLflow metrics, the leakage
audit (5-fold AUC with and without the reporting-rule columns), and crash
profiles.

### Crash profiles ("what stands out")

`hotspot_profile.csv` describes the injury crashes at each location over the
last 5 years (the EB window): BHPD collision type, primary collision factor,
and whether a pedestrian, cyclist, or motorcyclist was involved, each with
the citywide share for comparison. The primary collision factor is the
California Vehicle Code section on the report, grouped roughly as SWITRS
groups PCF violations (unsafe speed, left turn failing to yield, ran red
light, ran stop sign, and so on); BHPD's free-text cause fills in when the
code is missing (about 7% of injury crashes).

A location's "what stands out" is the type or cause most over-represented
there relative to the city, called out only with at least 4 crashes, at
least 20% of the location's injury crashes, and at least 1.5x the citywide
share. It is descriptive, with no significance test. This is what turns a
rank into something actionable: Robertson & Wilshire is a left-turn problem
(46% of injury crashes, 2.9x the city); Beverly Dr & Carmelita is stop-sign
running (31%, 7.2x).

### Before releasing: audit and brief

```bash
make export_dash
make audit_top    # data/audit/top_locations_audit.{csv,html}
make brief        # data/brief/bh_injury_hotspots_brief_<month>.pdf
```

`audit_top` lists every location in a top 10 under any ranking the app
offers, with the raw BHPD spellings merged into it, how far from the corner
the reports sit, its coordinates and their source, flags (no or approximate
coordinates, a pin far from the city, mid-block keys, most reports 250+ ft
from the corner), and two Google Maps links: the pin and an address search.
They should land on the same corner. Fix a wrong pin in
`data/external/location_coords_manual.csv` and re-export.

`brief` writes a one-page PDF from the same exported files: the headline
numbers, the top 10 with what stands out at each, the top 10 grouped by
pattern with the countermeasures an engineer would usually review first
(FHWA Proven Safety Countermeasures), the method, the limits, and the app
link. Options: `--outcome` (inj_count, vru_inj_count), `--rank-by` (eb,
excess, obs12, ml), `--top` (5 to 15), `--url`, `--contact`, `--no-logo`. The
layout shrinks slightly when needed so the brief is always one page.

The same module is copied into `flask_apps/bh_traffic/bh_brief.py`; the app's
"PDF brief" button builds it from the current map settings. Keep the two copies
identical.

Coordinates come from the OpenStreetMap drive network (`osmnx`): every node
where two named streets meet yields an intersection key, normalized with the
same `_norm_street` as preprocessing, so it joins `location_key` directly.
Mid-block keys take the mean of geocoded intersections on the same street and
are flagged approximate. The first run needs internet; results are cached in
`data/external/location_coords.csv`, and rows in
`data/external/location_coords_manual.csv` (`location_key,lat,lon`) override.

---

## Requirements

- Python 3.12
- See `requirements.txt`

```bash
make create_venv
source bh_venv/bin/activate
make requirements
```

Or with conda:

```bash
conda create -n conda_bh python=3.12
conda activate conda_bh
make requirements
```

---

## Quick start

```bash
make full_pipeline           # folders -> preprocessing -> classifier -> hotspots
make mlflow_ui               # inspect runs at http://localhost:5501
```

| Target | Does |
|---|---|
| `full_pipeline` | `create_folders`, `preproc_pipeline`, `train_eval_pipeline`, `hotspot_pipeline`, in order |
| `full_pipeline_stability` | `full_pipeline`, then `did_stability` (slow) |
| `clean_results` | Deletes `mlruns/`, `models/`, `data/processed/`; keeps `data/raw/` |
| `rerun_all` | `clean_results`, then `full_pipeline` |

Pass a differently named raw file with
`make full_pipeline RAW_DATA=$(pwd)/data/raw/<file>.csv`.

---

## Project layout

```
bh_traffic/
├── core/                        # shared utilities
│   ├── config.py                # paths, pipelines, model grids
│   ├── constants.py             # column names, target, leak_cols, MLflow names
│   ├── functions.py             # MLflow helpers, metrics, plots
│   └── model_registry.py        # MLflow-backed model loading
├── data/
│   ├── raw/                     # source CSV (not tracked)
│   ├── interim/
│   ├── processed/               # parquet artifacts + logs
│   │   ├── hotspots/            # ranked hotspot table + backtest
│   │   └── inference/
│   └── external/
├── modeling/
│   ├── train.py                 # classifier training + hyperparameter search
│   ├── evaluation.py            # classifier evaluation, calibration, metrics
│   ├── did_stability.py         # stability of the VRU AUC gap under ablation
│   ├── train_counts.py          # Poisson count models on the temporal split
│   ├── evaluation_counts.py     # count models vs naive baselines
│   ├── hotspots.py              # Empirical Bayes screening, backtest, ranked table
│   ├── export_dash.py           # flat files for the Dash app in flask_apps
│   ├── audit_top.py             # pre-release check of top-ranked locations
│   └── brief_pdf.py             # one-page PDF brief for decision makers
├── preprocessing/
│   ├── data_gen.py              # raw CSV -> df.parquet
│   ├── preprocessing.py         # cleaning, dedup, realignment, features, encoding
│   ├── feat_gen.py              # leakage removal -> X.parquet, y.parquet
│   └── panel_gen.py             # location x month panel -> X_counts, y_counts
├── notebooks/
├── models/
│   ├── results/<outcome>/       # training logs per model + pipeline
│   └── eval/<outcome>/          # evaluation logs
└── Makefile
```

---

## Features

| Group | Features |
|---|---|
| Time | year, month, weekday, weekend, hour, cyclical hour, night, AM/PM rush |
| Place | primary street (top 30), intersection vs mid-block, feet from cross street, direction |
| Cause | other improper driving (top 30), vehicle code section (top 25), less `leak_cols` |
| Geometry | type of collision (broadside, rear end, head-on, ...) |
| Party | party type, involved party, collision partner, pedestrian / bicycle / motorcycle flags, `vru` |

Categorical levels are learned during the training stage (top n per column,
`core/constants.py: cat_top_n`), with the rest collapsed to `OTHER` and blanks
to `MISSING`. Levels are computed on the full dataset before the split; this
uses no label information. They are stored in MLflow so inference reproduces
the same dummy columns.

---

## The Makefile

The Makefile is the orchestration layer. Every stage is a target, targets
compose into pipelines, and all stdout is teed to a log file so runs are
reproducible after the fact.

### Configuration variables

Set at the top of the file:

| Variable | Purpose |
|---|---|
| `PROJECT_NAME` | Project identifier |
| `PYTHON_VERSION` | Interpreter version (3.12) |
| `VENV_DIR` | Virtualenv directory (`bh_venv`) |
| `CONDA_ENV_NAME` | Conda environment name |
| `RAW_DATA` | Path to source CSV |
| `PROCESSED_DATA` | Path to `df.parquet` |
| `OUTCOMES` | Outcome variables to loop over (`injury_or_fatal`) |
| `PIPELINES` | Variants to run (`orig`); `smote`, `under`, and `orig_no_party` are defined in `config.py` |
| `SCORING` | Tuning metric (`average_precision`) |
| `PRETRAINED` | `0` = train from scratch, `1` = calibrate an existing model |

`RAW_DATA` and `PROCESSED_DATA` use `?=`, so they can be overridden at the
command line without editing the file:

```bash
make data_gen RAW_DATA=/path/to/other.csv
```

### Environment and setup

| Target | Does |
|---|---|
| `init_config` | Interactive rename of project + variables (cross-platform sed) |
| `check_vars` | Prints the variables that still need real values |
| `create_venv` | Builds the virtualenv |
| `activate_venv` | Prints the activation command |
| `clean_venv` | Removes the virtualenv |
| `requirements` | Upgrades pip, installs `requirements.txt` |
| `create_folders` | Scaffolds the full directory tree, per-outcome subdirs |
| `clean` | Deletes `.pyc` files and `__pycache__` |
| `clean_dir` | Removes `data/` entirely |
| `mlflow_ui` | MLflow UI on port 5501, backend `mlruns` |

### Preprocessing

| Target | Input | Output |
|---|---|---|
| `data_gen` | Raw CSV | `data/processed/df.parquet` |
| `data_prep_preprocessing_training` | `df.parquet` | `df_sans_zero.parquet` |
| `feat_gen_training` | `df_sans_zero.parquet` | `X.parquet`, `y.parquet` |
| `preproc_pipeline` | — | All three, in order |
| `clean_data` | — | Removes the processed parquet + csv |

### Training

One target per algorithm, each looping over `OUTCOMES` × `PIPELINES`:

| Target | Model |
|---|---|
| `train_logistic_regression` | `--model-type lr` |
| `train_random_forest` | `--model-type rf` |
| `train_xgboost` | `--model-type xgb` |
| `train_catboost` | `--model-type cat` |
| `train_all_models` | All four |

Training stratifies the train/validation/test split on `vru` (and `y`), so
every split carries the same share of pedestrian, bicycle, and motorcycle
collisions. Logs land in `models/results/<outcome>/<model>_<pipeline>.txt`.

### Evaluation

| Target | Model |
|---|---|
| `eval_logistic_regression` | lr |
| `eval_random_forest` | rf |
| `eval_xgboost` | xgb |
| `eval_catboost` | cat |
| `eval_all_models` | All four |

Logs land in `models/eval/<outcome>/<model>_eval_<pipeline>.txt`.

### Party ablation

```make
cat_no_party:
	$(MAKE) train_catboost PIPELINES=orig_no_party
	$(MAKE) eval_catboost PIPELINES=orig_no_party
```

A struck pedestrian is almost always injured, so party features carry much of
the signal and partly restate the outcome. The ablated arm retrains CatBoost
without them (party type, collision partner, pedestrian / bicycle / motorcycle
flags, and `vru`), leaving place, time, cause, and collision geometry. `vru`
stays in `X` for stratification and the subgroup audit but is excluded from
the transformer.

Some cause levels also identify pedestrian crashes (for example
`pcf_section_21950`, `other_imp_drive_fail_to_yield_to_pedestrian`; injury
rate 1.0). Unless they are added to `party_cols_extra` in
`core/constants.py`, the ablated model still receives pedestrian information
through the cause fields.

`make did_stability` then refits both arms across repeated stratified splits
and reports whether removing party features changes the VRU minus non-VRU ROC
AUC gap consistently, or only on the original partition.

### Hotspot count models

| Target | Does |
|---|---|
| `panel_gen` | `df_sans_zero.parquet` -> `panel.parquet`, `X_counts.parquet`, `y_counts.parquet`, `X_counts_forecast.parquet` |
| `train_count_models` | Loops `COUNT_OUTCOMES` x `COUNT_MODELS` (`pr`, `xgb`, `cat`) |
| `eval_count_models` | Each count model vs the history-rate and last-12-months baselines |
| `hotspots` | Backtest of rankings, then `data/processed/hotspots/hotspots_through_<month>.csv` |
| `hotspot_pipeline` | All four, in order (run after `preproc_pipeline`) |

Logs land in `models/results/<count outcome>/` and `models/eval/<count outcome>/`.

### Explainability, inference, composite pipelines

| Target | Does |
|---|---|
| `model_explainer` | Selects the best model by K-Fold Average Precision |
| `model_explanations_training` | SHAP values on training data, top 5 features |
| `model_explanations_inference` | SHAP values on inference data |
| `data_prep_preprocessing_inference` | Preprocessing in inference mode |
| `feat_gen_inference` | Feature generation in inference mode |
| `predict` | Best-model predictions to CSV |
| `preproc_pipeline` | data_gen → preprocessing → feat_gen |
| `train_eval_pipeline` | train_all → eval_all → cat_no_party |
| `preproc_train_eval` | preproc → train_all → eval_all → cat_no_party |
| `preproc_pipeline_inf` | inference preprocessing → feat_gen → predict |

### Help

`.DEFAULT_GOAL := help`, so a bare `make` prints available rules.

---

## Reproducing the analysis

```bash
make create_folders
make preproc_pipeline
make train_eval_pipeline     # severity classifier
make did_stability
make hotspot_pipeline        # hotspot screening
```

---

## Limitations

- Most non-injury collisions are absent by reporting practice; results
  describe reported collisions only.
- No traffic or pedestrian volume, so place effects mix exposure and risk.
- Injury severity is not recorded in the BHPD export; minor injuries and the
  19 fatalities share one class. Severity requires the CCRS victim files.
- Injury prevalence drops from about 0.67 to about 0.60 in 2025 and 2026,
  coinciding with the batch of shuffled fields; it may reflect a change in
  reporting practice.
- Coverage runs through June 2026 as of this export.

---

## Citation

If you use this code, please cite the source dataset:

> City of Beverly Hills. Historic Collision Records of Beverly Hills (New).
> Beverly Hills Police Department Traffic Bureau. City of Beverly Hills Open
> Data.

---

## License

Code released under MIT. Source data belongs to the City of Beverly Hills and
is subject to the portal's license terms.
