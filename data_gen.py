from pathlib import Path
import typer
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

# Import functions and constants
from core.functions import (
    mlflow_load_model,
    log_mlflow_metrics,
)

from core.constants import location_key_var

from core.config import (
    PROCESSED_DATA_DIR,
    count_model_definitions,
)

app = typer.Typer()

## Panel feature prefix for each outcome, used by the naive baselines
BASELINE_PREFIX = {"inj_count": "inj", "vru_inj_count": "vru_inj"}


def row_metrics(y_true, y_pred):
    """Location-month accuracy: deviance, MAE, and total calibration."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-6, None)
    return {
        "Poisson Deviance": mean_poisson_deviance(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "Calibration Ratio": y_pred.sum() / max(np.sum(y_true), 1e-9),
    }


def ranking_metrics(y_true, y_pred, locations, top=(10, 20)):
    """Location-level ranking over the whole period: how well summed
    predictions order locations by their actual crash totals."""
    actual = pd.Series(np.asarray(y_true), index=locations).groupby(level=0).sum()
    pred = pd.Series(np.asarray(y_pred), index=locations).groupby(level=0).sum()
    order = pred.sort_values(ascending=False).index
    out = {"Spearman": spearmanr(pred, actual.loc[pred.index]).correlation}
    for n in top:
        out[f"Top {n} Capture"] = actual.loc[order[:n]].sum() / actual.sum()
        out[f"Top {n} Oracle"] = (
            actual.sort_values(ascending=False).iloc[:n].sum() / actual.sum()
        )
    return out, pred, actual


################################################################################
# ---- STEP 1: Define command-line arguments with default values ----
################################################################################


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    model_type: str = "xgb",
    pipeline_type: str = "orig",
    outcome: str = "inj_count",
    features_path: Path = PROCESSED_DATA_DIR / "X_counts.parquet",
    labels_path: Path = PROCESSED_DATA_DIR / "y_counts.parquet",
    panel_path: Path = PROCESSED_DATA_DIR / "panel.parquet",
    # -----------------------------------------
):

    ################################################################################
    # STEP 2: Load Model Configuration
    ################################################################################

    estimator_name = count_model_definitions[model_type]["estimator_name"]
    experiment_name = f"{outcome}_model"
    run_name = f"{estimator_name}_{pipeline_type}_training"
    model_name = f"{estimator_name}_{outcome}"

    print(run_name)
    print(model_name)

    ################################################################################
    # STEP 3: Load Pre-Trained Model from MLflow
    ################################################################################

    model = mlflow_load_model(
        experiment_name=experiment_name,
        run_name=run_name,
        model_name=model_name,
    )

    ################################################################################
    # STEP 4: Load Processed Data (Features, Labels, Panel Keys)
    ################################################################################

    X = pd.read_parquet(features_path)
    y = pd.read_parquet(labels_path)[outcome].squeeze()
    panel = pd.read_parquet(panel_path, columns=[location_key_var, "split"]).loc[
        X.index
    ]

    ################################################################################
    # STEP 5: Predictions and Naive Baselines
    ################################################################################
    # Two baselines a traffic engineer could compute without a model, both from
    # the same 12-month-lagged history the model sees:
    #   history rate: average monthly count since 2015
    #   last 12 months: the most recent 12 months on record, per month
    ################################################################################

    prefix = BASELINE_PREFIX[outcome]
    preds = {
        estimator_name: pd.Series(model.predict(X), index=X.index),
        "history_rate": X[f"{prefix}_hist_rate"],
        "last_12_months": X[f"{prefix}_roll12"] / 12,
    }

    ################################################################################
    # STEP 6: Compute Metrics by Split
    ################################################################################

    rows = []
    test_totals = {}
    for split in ("train", "valid", "test"):
        mask = panel["split"] == split
        for name, p in preds.items():
            r = {"split": split, "model": name}
            r.update(row_metrics(y[mask], p[mask]))
            rank, pred_tot, actual_tot = ranking_metrics(
                y[mask], p[mask], panel.loc[mask, location_key_var]
            )
            r.update(rank)
            rows.append(r)
            if split == "test":
                test_totals[name] = pred_tot
        if split == "test":
            test_totals["actual"] = actual_tot

    results = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"\n{'=' * 80}\nCount Model Evaluation: {outcome}\n{'=' * 80}")
    print(results.round(4).to_string(index=False))

    ################################################################################
    # STEP 7: Location Calibration Plot (Test Period)
    ################################################################################

    fig, ax = plt.subplots(figsize=(6, 6))
    actual = test_totals["actual"]
    pred = test_totals[estimator_name].loc[actual.index]
    ax.scatter(pred, actual, s=14, alpha=0.6)
    lim = max(actual.max(), pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1, color="grey")
    ax.set_xlabel("Predicted crashes, test period")
    ax.set_ylabel("Observed crashes, test period")
    ax.set_title(f"{estimator_name}: {outcome} by location")
    fig.tight_layout()

    ################################################################################
    # STEP 8: Log Experiment Details to MLflow
    ################################################################################

    own = results[results["model"] == estimator_name]
    metrics = pd.Series(
        {
            f"{r['split']} {k}": r[k]
            for _, r in own.iterrows()
            for k in r.index
            if k not in ("split", "model")
        }
    )

    log_mlflow_metrics(
        experiment_name=experiment_name,
        run_name=run_name,
        metrics=metrics,
        images={f"{estimator_name}_{outcome}_test_location_calibration.png": fig},
    )

    ################################################################################
    # STEP 9: Completion Message
    ################################################################################

    logger.success("Count model evaluation complete.")
    # -----------------------------------------


if __name__ == "__main__":

    app()
