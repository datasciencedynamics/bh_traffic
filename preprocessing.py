from pathlib import Path

import typer
from loguru import logger
import pandas as pd
from model_tuner import Model

################################################################################
# Step 1. Import Configurations and Constants
################################################################################

from core.config import (
    PROCESSED_DATA_DIR,
    count_model_definitions,
    count_pipelines,
    count_numerical_cols,
    rstate,
)

from core.functions import (
    clean_feature_selection_params,
    mlflow_log_parameters_model,
    adjust_preprocessing_pipeline,
    mlflow_load_model,
)

app = typer.Typer()

################################################################################
# Step 2. Define CLI Arguments with Default Values
################################################################################


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ---
    model_type: str = "xgb",
    pipeline_type: str = "orig",
    outcome: str = "inj_count",
    features_path: Path = PROCESSED_DATA_DIR / "X_counts.parquet",
    labels_path: Path = PROCESSED_DATA_DIR / "y_counts.parquet",
    panel_path: Path = PROCESSED_DATA_DIR / "panel.parquet",
    scoring: str = "neg_mean_poisson_deviance",
    pretrained: int = 0,
    # -----------------------------------------
):

    ################################################################################
    # Step 3. Load Feature and Label Datasets
    ################################################################################

    X = pd.read_parquet(features_path)  # read in X
    y = pd.read_parquet(labels_path)[outcome].squeeze()  # coerce into a series

    ################################################################################
    # Step 3a. Temporal Splits From the Panel
    ################################################################################
    # Rows are location-months. A random split would let a month in 2025 train
    # the model that is then scored on 2019, so train / valid / test follow
    # the calendar set in core/constants.py and are passed to model_tuner as
    # custom_splits.

    split = pd.read_parquet(panel_path, columns=["split"])["split"].loc[X.index]
    idx = {name: split.index[split == name] for name in ("train", "valid", "test")}

    custom_splits = {
        "X_train": X.loc[idx["train"]],
        "y_train": y.loc[idx["train"]],
        "X_valid": X.loc[idx["valid"]],
        "y_valid": y.loc[idx["valid"]],
        "X_test": X.loc[idx["test"]],
        "y_test": y.loc[idx["test"]],
    }

    for name, rows in idx.items():
        print(
            f"{name:>5}: {len(rows):>6,} location-months, {y.loc[rows].sum():>5.0f} crashes"
        )

    ################################################################################
    # Step 4. Retrieve Model and Pipeline Configurations
    ################################################################################

    clc = count_model_definitions[model_type]["clc"]
    estimator_name = count_model_definitions[model_type]["estimator_name"]
    pipeline_steps = count_pipelines[pipeline_type]["pipeline"]
    sampler = count_pipelines[pipeline_type]["sampler"]
    feature_selection = count_pipelines[pipeline_type]["feature_selection"]

    # Set the parameters
    tuned_parameters = count_model_definitions[model_type]["tuned_parameters"]
    randomized_grid = count_model_definitions[model_type]["randomized_grid"]
    n_iter = count_model_definitions[model_type]["n_iter"]
    early_stop = count_model_definitions[model_type]["early"]

    ################################################################################
    # Step 5. Clean up pipeline
    ################################################################################

    clean_feature_selection_params(pipeline_steps, tuned_parameters)

    # Skip imputer and scaler for 'xgb' and 'cat'; both handle NaN natively
    pipeline_steps = adjust_preprocessing_pipeline(
        model_type,
        pipeline_steps,
        count_numerical_cols,
        [],
        sampler=sampler,
    )

    ################################################################################
    # Step 6. Printing Outcome
    ################################################################################

    print()
    print(f"Outcome:")
    print("-" * 60)
    print()
    print("=" * 60)
    print(f"{outcome}")
    print("=" * 60)

    ################################################################################
    # Step 7. Define and Initialize the Model Pipeline
    ################################################################################

    logger.info(f"Training {estimator_name} for {outcome} ...")

    experiment_name = f"{outcome}_model"
    run_name = f"{estimator_name}_{pipeline_type}_training"
    model_name = f"{estimator_name}_{outcome}"

    if pretrained:

        print("Loading Pretrained Model...")
        model = mlflow_load_model(
            experiment_name=experiment_name,
            run_name=run_name,
            model_name=model_name,
        )

    else:
        model = Model(
            pipeline_steps=pipeline_steps,
            name=estimator_name,
            model_type="regression",
            estimator_name=estimator_name,
            calibrate=False,
            estimator=clc,
            kfold=False,
            grid=tuned_parameters,
            n_jobs=5,
            randomized_grid=randomized_grid,
            n_iter=n_iter,
            scoring=[scoring],
            random_state=rstate,
            boost_early=early_stop,
            imbalance_sampler=sampler,
            feature_selection=feature_selection,
        )

        ################################################################################
        # Step 8. Perform Hyperparameter Tuning on the Temporal Splits
        ################################################################################

        model.grid_search_param_tuning(X, y, custom_splits=custom_splits)

    ################################################################################
    # Step 9. Extract Training, Validation, and Test Splits
    ################################################################################

    X_train, y_train = model.get_train_data(X, y)
    X_valid, y_valid = model.get_valid_data(X, y)
    X_test, y_test = model.get_test_data(X, y)

    ################################################################################
    # Step 10. Train the Model
    ################################################################################

    # Boosting algorithms use the validation months for early stopping.

    if not pretrained:
        if model_type in {"xgb", "cat"}:
            model.fit(
                X_train,
                y_train,
                validation_data=(X_valid, y_valid),
                score=scoring,
            )
        else:
            model.fit(
                X_train,
                y_train,
                score=scoring,
            )

    ################################################################################
    # Step 11. See Results in Terminal and Store Model in MLFlow
    ################################################################################

    for label, (X_s, y_s) in {
        "Training": (X_train, y_train),
        "Validation": (X_valid, y_valid),
        "Test": (X_test, y_test),
    }.items():
        print(f"\n{'=' * 60}\n{label} Results\n{'=' * 60}")
        model.return_metrics(X=X_s, y=y_s)

    if pretrained:
        mlflow_log_parameters_model(
            experiment_name=experiment_name,
            run_name=run_name,
            model_name=model_name,
            model=model,
        )

    else:
        mlflow_log_parameters_model(
            model_type=model_type,
            n_iter=n_iter,
            kfold=False,
            outcome=outcome,
            experiment_name=experiment_name,
            run_name=run_name,
            model_name=model_name,
            model=model,
            hyperparam_dict=model.best_params_per_score[scoring],
        )

    logger.success("Count model training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
