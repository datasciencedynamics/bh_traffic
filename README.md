<p align="left">
  <img src="https://raw.githubusercontent.com/datasciencedynamics/datasciencedynamics.github.io/refs/heads/main/data_science_dynamics_logo.svg" alt="Data Science Dynamics" width="250"/>
</p>

# Beverly Hills Collision Injury Prediction

Machine learning pipeline for predicting whether a traffic collision in Beverly
Hills results in injury, using the Beverly Hills Police Department (BHPD)
Traffic Bureau collision records published on the City of Beverly Hills Open
Data portal (records since January 1, 2015).

The project trains four classifiers (logistic regression, random forest,
XGBoost, CatBoost), evaluates them under a common protocol, and audits model
behavior for vulnerable road users (pedestrians, bicyclists, motorcyclists and
scooter riders) versus vehicle-only collisions, including an ablation that
removes every "who was involved" feature from the model.

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
| Records | 8,551 rows (ObjectId); 8,503 unique accident numbers |
| Coverage | January 2015 onward |
| Raw file | `data/raw/bh_collisions.csv` |
| Outcome | Any injury or fatality (`injury`) |

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
- **`Accident Type` tags.** About 30 inconsistent spellings carry hit-and-run,
  DUI, and CPD tags; these are kept as flags and the severity part is dropped.
- **Duplicates.** 48 rows share an accident number with another row. The
  index is `ObjectId`; `accident_number` is retained (string, excluded from
  `X`) for linking to the state CCRS / SWITRS records.

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
make create_folders          # scaffold data/, models/, notebooks/
make preproc_train_eval      # full pipeline: preprocess -> train -> evaluate
make mlflow_ui               # inspect runs at http://localhost:5501
```

---

## Project layout

```
bh_collisions/
├── core/                        # shared utilities
│   ├── config.py                # paths, pipelines, model grids
│   ├── constants.py             # column names, target, MLflow names
│   ├── functions.py             # MLflow helpers, metrics, plots
│   └── model_registry.py        # MLflow-backed model loading
├── data/
│   ├── raw/                     # source CSV (not tracked)
│   ├── interim/
│   ├── processed/               # parquet artifacts + logs
│   │   └── inference/
│   └── external/
├── modeling/
│   ├── train.py                 # model training + hyperparameter search
│   ├── evaluation.py            # held-out evaluation, calibration, metrics
│   └── did_stability.py         # stability of the VRU AUC gap under ablation
├── preprocessing/
│   ├── data_gen.py              # raw CSV -> df.parquet
│   ├── preprocessing.py         # cleaning, realignment, features, encoding
│   └── feat_gen.py              # feature construction -> X.parquet, y.parquet
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
| Cause | other improper driving (top 30), vehicle code section (top 25) |
| Report tags | hit-and-run, DUI, CPD |
| Geometry | type of collision (broadside, rear end, head-on, ...) |
| Party | party type, involved party, collision partner, pedestrian / bicycle / motorcycle flags, `vru` |

Categorical levels are learned on training data, collapsed to `OTHER` /
`MISSING` beyond the top n (`core/constants.py: cat_top_n`), and stored in
MLflow so inference reproduces the same dummy columns.

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
| `OUTCOMES` | Outcome variables to loop over (`injury`) |
| `PIPELINES` | Sampling variants (`orig`, `smote`, `under`, `orig_no_party`) |
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

`make did_stability` then refits both arms across repeated stratified splits
and reports whether removing party features changes the VRU minus non-VRU ROC
AUC gap consistently, or only on the original partition.

### Explainability, inference, composite pipelines

Same targets as the template: `model_explainer`,
`model_explanations_training`, `model_explanations_inference`,
`data_prep_preprocessing_inference`, `feat_gen_inference`, `predict`,
`preproc_pipeline`, `train_eval_pipeline`, `preproc_train_eval`,
`preproc_pipeline_inf`.

### Help

`.DEFAULT_GOAL := help`, so a bare `make` prints available rules.

---

## Reproducing the analysis

```bash
make create_folders
make preproc_pipeline
make train_eval_pipeline
make did_stability
```

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
