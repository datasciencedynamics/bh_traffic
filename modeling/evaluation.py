from pathlib import Path
import typer
from loguru import logger
import pandas as pd

# Import functions and constants
from core.functions import (
    mlflow_load_model,
    return_model_metrics,
    return_model_plots,
    log_mlflow_metrics,
    mlflow_log_parameters_model,
)

from core.constants import target_outcome

from core.config import (
    PROCESSED_DATA_DIR,
    model_definitions,
)

app = typer.Typer()

################################################################################
# ---- STEP 1: Define command-line arguments with default values ----
################################################################################


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    model_type: str = "lr",
    pipeline_type: str = "orig",
    outcome: str = "injury_or_fatal",
    features_path: Path = PROCESSED_DATA_DIR / "X.parquet",
    labels_path: Path = PROCESSED_DATA_DIR / "y.parquet",
    scoring: str = "average_precision",
    # -----------------------------------------
):

    ################################################################################
    # STEP 2: Load Model Configuration & Pipeline Settings
    ################################################################################

    estimator_name = model_definitions[model_type]["estimator_name"]

    # Must match train.py: the party-ablation variant registers under a distinct
    # model_name so it forms its own group in the registry.
    name_suffix = "_no_party" if pipeline_type == "orig_no_party" else ""

    print(f"{estimator_name}_{pipeline_type}_training")
    print(f"{estimator_name}_{outcome}{name_suffix}")

    ################################################################################
    # STEP 3: Load Pre-Trained Model from MLflow
    ################################################################################

    model = mlflow_load_model(
        experiment_name=f"{outcome}_model",
        run_name=f"{estimator_name}_{pipeline_type}_training",
        model_name=f"{estimator_name}_{outcome}{name_suffix}",
    )

    # Print model thresholdon
    print(f"Model Threshold: {model.threshold}")

    ################################################################################
    # STEP 4: Load Processed Data (Features & Labels)
    ################################################################################

    X = pd.read_parquet(features_path)
    y = pd.read_parquet(labels_path)
    y = y[target_outcome[0]].squeeze()  # coerce into a series

    ################################################################################
    # STEP 5: Split Data into Train, Validation, and Test Sets
    ################################################################################

    X_train, y_train = model.get_train_data(X, y)
    X_valid, y_valid = model.get_valid_data(X, y)
    X_test, y_test = model.get_test_data(X, y)

    ################################################################################
    # STEP 6: Log Updated Model
    ################################################################################

    mlflow_log_parameters_model(
        experiment_name=f"{outcome}_model",
        run_name=f"{estimator_name}_{pipeline_type}_training",
        model_name=f"{estimator_name}_{outcome}{name_suffix}",
        model=model,
    )

    ################################################################################
    # STEP 7: Compute and Evaluate Model Performance Metrics
    ################################################################################

    all_inputs = {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
        "valid": (X_valid, y_valid),
    }
    metrics = return_model_metrics(
        inputs=all_inputs,
        model=model,
        estimator_name=estimator_name,
    )

    print(metrics)

    ################################################################################
    # STEP 8: Generate and Save Model Evaluation Plots
    ################################################################################

    # Generate evaluation plots
    all_plots = return_model_plots(
        inputs=all_inputs,
        model=model,
        estimator_name=estimator_name,
        scoring=scoring,
    )

    ################################################################################
    # STEP 9: Log Experiment Details to MLflow
    ################################################################################

    log_mlflow_metrics(
        experiment_name=f"{outcome}_model",
        run_name=f"{estimator_name}_{pipeline_type}_training",
        metrics=metrics[estimator_name],
        images=all_plots,
    )

    ################################################################################
    # STEP 10: Completion Message
    ################################################################################

    logger.success("Modeling evaluation complete.")
    # -----------------------------------------


if __name__ == "__main__":

    app()
