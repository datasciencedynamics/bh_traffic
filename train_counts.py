#!/usr/bin/env python
"""Stability of the group-wise AUC gap difference-in-differences.

Groups are vulnerable road users (VRU: pedestrian, bicyclist, motorcyclist or
scooter) versus vehicle-only collisions. The ablation removes every party
feature (who was involved), leaving place, time, cause, and collision geometry.

The bootstrap in the subgroup audit resamples one fixed held-out set, so it
answers whether the change in the VRU-minus-non-VRU ROC AUC gap survives
resampling those rows. It cannot say whether the finding is a property of
which rows landed in the test set to begin with. This script answers that
second question by refitting both models across repeated stratified splits
of the full dataset.

ROC AUC is threshold-independent, so no threshold selection is needed here.
Hyperparameters are held at their tuned values; re-tuning inside each split
would answer a third question and cost far more. Those hyperparameters were
selected on the original partition, so each split inherits a choice that saw
data now sitting in its own test fold. That is a mild optimism and should be
recorded as a limitation.

Run from the project root, or via `make did_stability`.

Usage
-----
    python modeling/did_stability.py \
        --features-path ./data/processed/X.parquet \
        --labels-path ./data/processed/y.parquet \
        --outcome injury_or_fatal
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.auto import tqdm

app = typer.Typer(add_completion=False, help=__doc__)

FitPredict = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame, bool], np.ndarray]


# --------------------------------------------------------------------------
# core
# --------------------------------------------------------------------------
def progress_stream():
    """Stream for the progress bar, and whether it needs closing.

    The Makefile runs this as `... 2>&1 | tee`, so stderr is a pipe and a bar
    written there would land in the log as carriage returns while never
    reaching the terminal. Writing to /dev/tty reaches the terminal directly
    and stays out of the piped log. Falls back to stderr where there is no
    controlling terminal, and the caller suppresses the bar in that case.
    """
    try:
        return open("/dev/tty", "w"), True
    except OSError:
        return sys.stderr, False


def auc_gap(y: np.ndarray, p: np.ndarray, grp: np.ndarray) -> float:
    """Group minus rest ROC AUC. NaN if either group lacks both classes."""
    rest = ~grp
    if len(np.unique(y[grp])) < 2 or len(np.unique(y[rest])) < 2:
        return np.nan
    return roc_auc_score(y[grp], p[grp]) - roc_auc_score(y[rest], p[rest])


def did_stability(
    X: pd.DataFrame,
    y: np.ndarray,
    fit_predict: FitPredict,
    group_col: str = "vru",
    group_value: int = 1,
    n_splits: int = 20,
    test_size: float = 0.2,
    seed: int = 222,
    progress: bool = True,
) -> pd.DataFrame:
    """One row per split: group AUCs, the two gaps, and their difference.

    Stratifies on group x outcome so every split carries the same number of
    VRU events. Plain outcome stratification lets that count drift, which
    would make the estimate look unstable for reasons unrelated to the finding.

    Each split refits both models, so the bar ticks once per two CatBoost fits.
    It writes to stderr and is suppressed when stderr is not a terminal, which
    keeps tee'd Makefile logs free of carriage returns.
    """
    X = X.reset_index(drop=True)
    y = np.asarray(y).astype(int)
    grp_all = (X[group_col] == group_value).to_numpy()

    strata = np.char.add(grp_all.astype(str), y.astype(str))
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=seed
    )

    splits = list(splitter.split(X, strata))
    rows = []

    stream, is_tty = progress_stream()
    bar = tqdm(
        splits,
        desc="splits",
        unit="split",
        file=stream,
        disable=not progress or not (is_tty or stream.isatty()),
        leave=False,
        dynamic_ncols=True,
    )

    for i, (tr, te) in enumerate(bar):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]
        grp_te = grp_all[te]

        p_pri = np.asarray(fit_predict(X_tr, y_tr, X_te, False))
        p_abl = np.asarray(fit_predict(X_tr, y_tr, X_te, True))

        g_pri = auc_gap(y_te, p_pri, grp_te)
        g_abl = auc_gap(y_te, p_abl, grp_te)

        rows.append(
            {
                "split": i,
                "n_test": len(te),
                "n_group_events": int(y_te[grp_te].sum()),
                "n_rest_events": int(y_te[~grp_te].sum()),
                "auc_pri": roc_auc_score(y_te, p_pri),
                "auc_abl": roc_auc_score(y_te, p_abl),
                "gap_pri": g_pri,
                "gap_abl": g_abl,
                "did": g_abl - g_pri,
            }
        )
        bar.set_postfix(did=f"{g_abl - g_pri:+.3f}")

    bar.close()
    if is_tty:
        stream.close()
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    """Distributional summary of the per-split difference-in-differences."""
    d = df["did"].dropna()
    if d.empty:
        raise ValueError("every split was dropped; check group event counts")
    return {
        "n_splits": len(d),
        "n_dropped": int(df["did"].isna().sum()),
        "did_median": float(d.median()),
        "did_mean": float(d.mean()),
        "did_q25": float(d.quantile(0.25)),
        "did_q75": float(d.quantile(0.75)),
        "did_min": float(d.min()),
        "did_max": float(d.max()),
        "frac_positive": float((d > 0).mean()),
    }


# --------------------------------------------------------------------------
# model wiring
# --------------------------------------------------------------------------
def find_catboost(obj, depth: int = 0):
    """Walk down to the actual CatBoostClassifier.

    model_tuner wraps the estimator, sometimes behind a sklearn Pipeline. The
    wrapper also has get_params(), so matching on that alone returns tuning
    params like cv that CatBoost will not accept.
    """
    from catboost import CatBoostClassifier

    if isinstance(obj, CatBoostClassifier):
        return obj
    if depth > 5 or obj is None:
        return None

    if hasattr(obj, "steps"):  # sklearn Pipeline
        for _, step in obj.steps:
            found = find_catboost(step, depth + 1)
            if found is not None:
                return found

    for attr in (
        "estimator",
        "estimator_",
        "best_estimator_",
        "model",
        "clf",
        "base_estimator",
        "calibrated_classifiers_",
    ):
        sub = getattr(obj, attr, None)
        if sub is None or sub is obj:
            continue
        if isinstance(sub, (list, tuple)):
            for item in sub:
                found = find_catboost(item, depth + 1)
                if found is not None:
                    return found
            continue
        found = find_catboost(sub, depth + 1)
        if found is not None:
            return found
    return None


def get_estimator(model):
    """Return the CatBoostClassifier inside a champion, or raise informatively."""
    est = find_catboost(model)
    if est is not None:
        return est
    public = ", ".join(a for a in dir(model) if not a.startswith("_"))
    raise AttributeError(
        f"no CatBoostClassifier found inside {type(model).__name__}: {public}"
    )


def clean_params(params: dict) -> tuple[dict, list[str]]:
    """Keep only keys CatBoostClassifier accepts. Returns (kept, dropped)."""
    import inspect

    from catboost import CatBoostClassifier

    allowed = set(inspect.signature(CatBoostClassifier.__init__).parameters) - {"self"}
    explicit = {"verbose", "logging_level", "silent", "allow_writing_files"}

    kept, dropped = {}, []
    for k, v in params.items():
        if k in explicit:
            continue
        (kept.__setitem__(k, v) if k in allowed else dropped.append(k))
    return kept, dropped


def make_fit_predict(
    pri_params: dict, abl_params: dict, ablate_cols: list[str]
) -> FitPredict:
    """Close over both hyperparameter sets and return the refit callable."""
    from catboost import CatBoostClassifier

    drop = set(ablate_cols)

    def fit_predict(X_tr, y_tr, X_te, ablate):
        params = abl_params if ablate else pri_params
        cols = [c for c in X_tr.columns if not (ablate and c in drop)]
        ## early stopping needs an eval set; refits use the tuned tree count
        params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        model = CatBoostClassifier(**params, verbose=0, allow_writing_files=False)
        model.fit(X_tr[cols], y_tr)
        return model.predict_proba(X_te[cols])[:, 1]

    return fit_predict


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def report(df: pd.DataFrame) -> dict:
    """Print the per-split table and the summary. Returns the summary."""
    typer.echo("\n" + df.round(3).to_string(index=False))

    s = summarize(df)
    typer.echo("\nDifference-in-differences across splits")
    typer.echo(f"  median          {s['did_median']:+.4f}")
    typer.echo(f"  mean            {s['did_mean']:+.4f}")
    typer.echo(f"  IQR             {s['did_q25']:+.4f} to {s['did_q75']:+.4f}")
    typer.echo(f"  range           {s['did_min']:+.4f} to {s['did_max']:+.4f}")
    typer.echo(f"  positive        {s['frac_positive']:.0%} of {s['n_splits']} splits")
    if s["n_dropped"]:
        typer.echo(f"  dropped         {s['n_dropped']} (a group lacked both classes)")

    if s["did_q25"] > 0 and s["frac_positive"] >= 0.75:
        typer.echo("\nIQR sits above zero; the widening holds across partitions.")
    elif s["did_q75"] < 0 and s["frac_positive"] <= 0.25:
        typer.echo("\nIQR sits below zero; the narrowing holds across partitions.")
    else:
        typer.echo(
            "\nIQR straddles zero; the bootstrap result may be specific to the "
            "original partition."
        )
    return s


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------
@app.command()
def run(
    features_path: Path = typer.Option(
        Path("./data/processed/X.parquet"), "--features-path", help="feature matrix"
    ),
    labels_path: Path = typer.Option(
        Path("./data/processed/y.parquet"), "--labels-path", help="labels"
    ),
    outcome: str = typer.Option("injury_or_fatal", "--outcome", help="outcome column"),
    group_col: str = typer.Option("vru", "--group-col", help="subgroup column"),
    group_value: int = typer.Option(1, "--group-value", help="value denoting group"),
    primary_key: str = typer.Option(
        "cat_injury_or_fatal",
        "--primary-key",
        help="registry key, model with party features",
    ),
    ablated_key: str = typer.Option(
        "cat_injury_or_fatal_no_party",
        "--ablated-key",
        help="registry key, model without party features",
    ),
    metric_name: str = typer.Option(
        "valid Average Precision", "--metric-name", help="champion selection metric"
    ),
    n_splits: int = typer.Option(20, "--n-splits", min=2, help="stratified splits"),
    test_size: float = typer.Option(0.2, "--test-size", min=0.05, max=0.5),
    seed: int = typer.Option(222, "--seed"),
    progress: bool = typer.Option(
        True, "--progress/--no-progress", help="show the per-split bar"
    ),
    out: Optional[Path] = typer.Option(None, "--out", help="write per-split CSV here"),
) -> None:
    """Refit both models across repeated splits and report DiD stability."""
    from core.config import party_cols
    from core.model_registry import load_best_per_algo

    X = pd.read_parquet(features_path)
    y = pd.read_parquet(labels_path)[outcome].squeeze()

    if group_col not in X.columns:
        raise typer.BadParameter(f"{group_col!r} not in X: {list(X.columns)}")

    champs = load_best_per_algo(metric=metric_name)
    for key in (primary_key, ablated_key):
        if key not in champs:
            raise typer.BadParameter(f"{key!r} not in registry: {list(champs)}")

    pri_params, pri_dropped = clean_params(
        get_estimator(champs[primary_key]).get_params()
    )
    abl_params, abl_dropped = clean_params(
        get_estimator(champs[ablated_key]).get_params()
    )

    typer.echo(f"catboost params: {len(pri_params)} primary, {len(abl_params)} ablated")
    dropped = sorted(set(pri_dropped) | set(abl_dropped))
    if dropped:
        typer.echo(f"  ignored (not CatBoost args): {', '.join(dropped)}")
    typer.echo(f"ablated party columns: {len(party_cols)}")

    typer.echo(
        f"dataset {len(X)} rows, {int(np.asarray(y).sum())} events | "
        f"{n_splits} splits, test_size={test_size}, seed={seed}"
    )

    df = did_stability(
        X,
        y,
        fit_predict=make_fit_predict(pri_params, abl_params, party_cols),
        group_col=group_col,
        group_value=group_value,
        n_splits=n_splits,
        test_size=test_size,
        seed=seed,
        progress=progress,
    )

    report(df)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        typer.echo(f"\nwrote {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
