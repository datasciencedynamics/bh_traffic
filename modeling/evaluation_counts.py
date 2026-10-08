from pathlib import Path
import typer
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

# Import functions and constants
from core.functions import (
    mlflow_load_model,
    log_mlflow_metrics,
    lorenz,
    share_at,
)

from core.constants import location_key_var, concentration_share, plot_colors

from core.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    count_model_definitions,
)

app = typer.Typer()

## Panel feature prefix for each outcome, used by the naive baselines
BASELINE_PREFIX = {"inj_count": "inj", "vru_inj_count": "vru_inj"}

INK, INK_2 = plot_colors["ink"], plot_colors["ink_2"]
GRID, SURFACE = plot_colors["grid"], plot_colors["surface"]
SERIES = plot_colors["series"]


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


def style_axes(ax):
    """Recessive grid and axes, text in ink tokens."""
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def shap_values_for(model, X_rows, X_background):
    """SHAP values for the fitted estimator inside the model_tuner pipeline.

    Trees use TreeExplainer; the Poisson GLM uses LinearExplainer. Both
    explain the log of the expected count (the Poisson link), so values add
    on the log scale and exponentiate to multiplicative effects.
    """
    pre = model.estimator[:-1]
    est = model.estimator[-1]
    names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    Xt = pd.DataFrame(pre.transform(X_rows), columns=names, index=X_rows.index)

    if hasattr(est, "coef_"):
        bg = pd.DataFrame(pre.transform(X_background), columns=names)
        values = shap.LinearExplainer(est, bg).shap_values(Xt)
    elif hasattr(est, "get_booster"):
        ## XGBoost: native TreeSHAP (identical algorithm). shap's TreeExplainer
        ## cannot parse the base_score format of XGBoost >= 2.1. The last
        ## column of pred_contribs is the bias term.
        import xgboost as xgb

        contribs = est.get_booster().predict(
            xgb.DMatrix(Xt.to_numpy(), feature_names=None), pred_contribs=True
        )
        values = contribs[:, :-1]
    else:
        values = shap.TreeExplainer(est).shap_values(Xt)
    return np.asarray(values), Xt


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
    eval_dir: Path = MODELS_DIR / "eval",
    shap_top_n: int = 15,
    # -----------------------------------------
):

    ################################################################################
    # STEP 2: Load Model Configuration
    ################################################################################

    estimator_name = count_model_definitions[model_type]["estimator_name"]
    experiment_name = f"{outcome}_model"
    run_name = f"{estimator_name}_{pipeline_type}_training"
    model_name = f"{estimator_name}_{outcome}"
    out_dir = eval_dir / outcome
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{estimator_name}_{pipeline_type}"

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

    actual = test_totals["actual"]
    fig_cal, ax = plt.subplots(figsize=(6, 6))
    fig_cal.patch.set_facecolor(SURFACE)
    style_axes(ax)
    pred = test_totals[estimator_name].loc[actual.index]
    lim = max(actual.max(), pred.max()) * 1.05
    ax.plot(
        [0, lim],
        [0, lim],
        color=INK_2,
        linewidth=1,
        linestyle=":",
        label="Predicted = observed",
    )
    ax.scatter(
        pred,
        actual,
        s=40,
        color=SERIES[0],
        edgecolor=SURFACE,
        linewidth=1.5,
        alpha=0.85,
        label=estimator_name,
    )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted crashes, test period", color=INK_2)
    ax.set_ylabel("Observed crashes, test period", color=INK_2)
    ax.set_title(
        f"{estimator_name}: {outcome} by location", color=INK, fontsize=11, loc="left"
    )
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    fig_cal.tight_layout()

    ################################################################################
    # STEP 8: Lorenz Curves (Test Period)
    ################################################################################
    # Locations ranked by each method's summed test-period predictions, scored
    # on observed test-period crashes. Perfect foresight ranks by the observed
    # crashes themselves; it is the ceiling.
    ################################################################################

    q = concentration_share
    lorenz_curves = {
        estimator_name: lorenz(test_totals[estimator_name], actual),
        "history_rate": lorenz(test_totals["history_rate"], actual),
        "last_12_months": lorenz(test_totals["last_12_months"], actual),
        "perfect foresight": lorenz(actual, actual),
    }
    lorenz_summary = pd.Series(
        {name: share_at(c, q) for name, c in lorenz_curves.items()},
        name=f"top_{q:.0%}_share",
    )
    print(f"\nTest period: share of crashes at the top {q:.0%} of locations")
    print(lorenz_summary.round(3).to_string())

    fig_lor, ax = plt.subplots(figsize=(6.5, 5.5))
    fig_lor.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.plot(
        [0, 100],
        [0, 100],
        color=INK_2,
        linewidth=1,
        linestyle=":",
        label="No concentration",
    )
    styles = {
        estimator_name: (SERIES[0], "-"),
        "history_rate": (SERIES[2], "-."),
        "last_12_months": (SERIES[3], (0, (1, 1.5))),
        "perfect foresight": (SERIES[1], "--"),
    }
    for name, c in lorenz_curves.items():
        color, dash = styles[name]
        ax.plot(
            c["location_share"] * 100,
            c["crash_share"] * 100,
            color=color,
            linewidth=2,
            linestyle=dash,
            label=f"{name} ({lorenz_summary[name]:.0%} at top {q:.0%})",
        )
    ax.axvline(q * 100, color=GRID, linewidth=1, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel(f"Locations, ranked (% of {len(actual)})", color=INK_2)
    ax.set_ylabel("Share of test-period crashes (%)", color=INK_2)
    ax.set_title(
        f"{estimator_name}: {outcome}, Lorenz curves (test period)",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax.legend(frameon=False, loc="lower right", labelcolor=INK, fontsize=9)
    fig_lor.tight_layout()

    ################################################################################
    # STEP 9: SHAP Values (Test Period)
    ################################################################################
    # Explains the log expected count. Mean |SHAP| ranks features; the
    # beeswarm shows direction (high feature values in warm colors).
    ################################################################################

    test_rows = X[panel["split"] == "test"]
    train_rows = X[panel["split"] == "train"]
    background = train_rows.sample(min(500, len(train_rows)), random_state=222)
    sv, Xt = shap_values_for(model, test_rows, background)

    importance = pd.Series(
        np.abs(sv).mean(axis=0), index=Xt.columns, name="mean_abs_shap"
    ).sort_values(ascending=False)
    importance.round(5).to_csv(out_dir / f"{stem}_shap_importance.csv")
    print(f"\nTop {shap_top_n} features by mean |SHAP| (log expected count):")
    print(importance.head(shap_top_n).round(4).to_string())

    ## summary_plot draws into the current figure, so open a fresh one
    fig_shap = plt.figure()
    shap.summary_plot(sv, Xt, max_display=shap_top_n, show=False, plot_size=(8, 6))
    fig_shap.patch.set_facecolor(SURFACE)
    plt.title(
        f"{estimator_name}: {outcome}, SHAP on log expected count (test)",
        color=INK,
        fontsize=11,
        loc="left",
    )
    fig_shap.tight_layout()

    ################################################################################
    # STEP 10: Save Figures and Log Experiment Details to MLflow
    ################################################################################

    images = {
        f"{stem}_{outcome}_test_location_calibration.png": fig_cal,
        f"{stem}_{outcome}_test_lorenz.png": fig_lor,
        f"{stem}_{outcome}_test_shap_beeswarm.png": fig_shap,
    }
    for fname, fig in images.items():
        fig.savefig(out_dir / fname, dpi=150, facecolor=SURFACE, bbox_inches="tight")

    own = results[results["model"] == estimator_name]
    metrics = pd.Series(
        {
            f"{r['split']} {k}": r[k]
            for _, r in own.iterrows()
            for k in r.index
            if k not in ("split", "model")
        }
    )
    metrics[f"test Top {q * 100:.0f} Pct Lorenz Share"] = lorenz_summary[estimator_name]

    log_mlflow_metrics(
        experiment_name=experiment_name,
        run_name=run_name,
        metrics=metrics,
        images=images,
    )
    for fig in images.values():
        plt.close(fig)

    print(f"\nFigures saved to {out_dir}")

    ################################################################################
    # STEP 11: Completion Message
    ################################################################################

    logger.success("Count model evaluation complete.")
    # -----------------------------------------


if __name__ == "__main__":

    app()
