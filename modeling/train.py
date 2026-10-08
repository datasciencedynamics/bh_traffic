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
    model_definitions,
    rstate,
    pipelines,
    numerical_cols,
    numerical_cols_no_party,
    categorical_cols,
)

from core.constants import target_outcome, vru_var

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
    model_type: str = "lr",
    pipeline_type: str = "orig",
    outcome: str = "injury_or_fatal",
    features_path: Path = PROCESSED_DATA_DIR / "X.parquet",
    labels_path: Path = PROCESSED_DATA_DIR / "y.parquet",
    scoring: str = "average_precision",
    pretrained: int = 0,
    # -----------------------------------------
):

    ################################################################################
    # Step 3. Load Feature and Label Datasets
    ################################################################################

    X = pd.read_parquet(features_path)  # read in X
    y = pd.read_parquet(labels_path)  # read in y
    y = y[target_outcome[0]].squeeze()  # coerce into a series

    ################################################################################
    # Step 4. Retrieve Model and Pipeline Configurations
    ################################################################################

    clc = model_definitions[model_type]["clc"]
    estimator_name = model_definitions[model_type]["estimator_name"]
    pipeline_steps = pipelines[pipeline_type]["pipeline"]
    sampler = pipelines[pipeline_type]["sampler"]
    feature_selection = pipelines[pipeline_type]["feature_selection"]

    # Set the parameters
    tuned_parameters = model_definitions[model_type]["tuned_parameters"]
    randomized_grid = model_definitions[model_type]["randomized_grid"]
    n_iter = model_definitions[model_type]["n_iter"]
    early_stop = model_definitions[model_type]["early"]

    print("Sampler", sampler)

    ################################################################################
    # Step 4a. Model name suffix for the party-ablation variant
    ################################################################################
    # The ablated model registers under a distinct model_name so it forms its own
    # group in the registry rather than competing with the primary model when
    # selecting the best run per algorithm.

    name_suffix = "_no_party" if pipeline_type == "orig_no_party" else ""

    ################################################################################
    # Step 5. Clean up pipeline
    # Step 5a. Clean up tuned_parameters by removing feature selection keys if
    # RFE isn't in the pipeline
    ################################################################################
    clean_feature_selection_params(pipeline_steps, tuned_parameters)

    # Step 5b. Adjust preproc. pipe. to skip imputer and scaler for 'rf', 'xgb', 'cat'

    num_cols = (
        numerical_cols_no_party
        if pipeline_type.endswith("_no_party")
        else numerical_cols
    )

    pipeline_steps = adjust_preprocessing_pipeline(
        model_type,
        pipeline_steps,
        num_cols,
        categorical_cols,
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

    if pretrained:

        print("Loading Pretrained Model...")
        model = mlflow_load_model(
            experiment_name=f"{outcome}_model",
            run_name=f"{estimator_name}_{pipeline_type}_training",
            model_name=f"{estimator_name}_{outcome}{name_suffix}",
        )

    else:
        model = Model(
            pipeline_steps=pipeline_steps,
            name=estimator_name,
            model_type="classification",
            estimator_name=estimator_name,
            calibrate=True,
            estimator=clc,
            kfold=False,
            grid=tuned_parameters,
            n_jobs=5,
            randomized_grid=randomized_grid,
            n_iter=n_iter,
            scoring=[scoring],
            random_state=rstate,
            stratify_cols=[vru_var],
            stratify_y=True,
            boost_early=early_stop,
            imbalance_sampler=sampler,
            feature_selection=feature_selection,
        )

        ################################################################################
        # Step 8. Perform Hyperparameter Tuning
        ################################################################################

        model.grid_search_param_tuning(X, y, f1_beta_tune=True, betas=[1])

        ################################################################################
        # Step 9. Extract Training, Validation, and Test Splits
        ################################################################################

    X_train, y_train = model.get_train_data(X, y)
    X_valid, y_valid = model.get_valid_data(X, y)
    X_test, y_test = model.get_test_data(X, y)

    ################################################################################
    # Step 10. Train the Model
    ################################################################################

    # Boosting algorithms like XGBoost and CatBoost benefit from validation data
    # during training to optimize early stopping and prevent overfitting.
    # Hence, we explicitly provide the validation dataset in the `fit` method
    # for these models. For other models, validation data is not required at this
    # stage.

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
    # Step 11. Calibrate the Model If Necessary
    ################################################################################

    # ## If we need to update isotonic method
    # model.calibration_method = "isotonic"

    if model.calibrate:
        model.calibrateModel(X, y, score=scoring, f1_beta_tune=True)

    ################################################################################
    # Step 12. See Results in Terminal and Store Model in MLFlow
    ################################################################################

    # see the results printed to the terminal for reference
    print(f"\n{'=' * 60}\nTraining Results\n{'=' * 60}")
    model.return_metrics(
        X=X_train,
        y=y_train,
        optimal_threshold=True,
        print_threshold=True,
        model_metrics=True,
    )
    print(f"\n{'=' * 60}\nValidation Results\n{'=' * 60}")
    model.return_metrics(
        X=X_valid,
        y=y_valid,
        optimal_threshold=True,
        print_threshold=True,
        model_metrics=True,
    )
    print(f"\n{'=' * 60}\nTest Results\n{'=' * 60}")
    model.return_metrics(
        X=X_test,
        y=y_test,
        optimal_threshold=True,
        print_threshold=True,
        model_metrics=True,
    )
    if pretrained:
        mlflow_log_parameters_model(
            experiment_name=f"{outcome}_model",
            run_name=f"{estimator_name}_{pipeline_type}_training",
            model_name=f"{estimator_name}_{outcome}{name_suffix}",
            model=model,
        )

    else:
        mlflow_log_parameters_model(
            model_type=model_type,
            n_iter=n_iter,
            kfold=False,
            outcome=outcome,
            experiment_name=f"{outcome}_model",
            run_name=f"{estimator_name}_{pipeline_type}_training",
            model_name=f"{estimator_name}_{outcome}{name_suffix}",
            model=model,
            hyperparam_dict=model.best_params_per_score[scoring],
        )

    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
