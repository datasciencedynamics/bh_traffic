"""
model_registry.py
=================

MLflow-backed model loading for bh_collisions. No hard-coded experiment ids, run
ids, or paths.

Two things this handles that plain MLflow does not, for this project:

  * Several tracking roots live under mlruns/ (preprocessing, models, ...).
    All of them are indexed, not just one.
  * The recorded artifact_location points into a different project directory
    (nuforc_media/), because this mlruns tree was copied. Every MLflow artifact
    API will therefore resolve to a path that does not exist here. Artifacts
    are instead rebased onto the store root actually found on disk.

Naming
------
Artifact folders are named "<algo>_<TARGET>", and several runs share one folder
name (six runs all write cat_dramatic/). So the folder gives the ALGO and the
run name gives the VARIANT:

    algo    : cat, lr, cat_feats_and_text, cat_text_only
    variant : cat_orig, cat_smote, cat_rfe, ...   (the MLflow run name)

``variant`` is the addressable key. ``algo`` is what you group by.

Selection
---------
Default policy is "newest" (latest start_time wins a tie). Selecting by best
test metric is available but OFF by default, because picking the max-metric run
IS model selection on the evaluation set. Use it to inspect, pin by name to load.

Usage
-----
    from core.model_registry import available, load, load_all, rank

    available()                      # everything on disk, with metrics
    rank("cat", metric="roc_auc")    # see the six cat runs ordered
    model = load("cat_smote")        # pin the variant you actually reported
    models = load_all()              # one model per variant
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from model_tuner import loadObjects

PROJECT_ROOT = Path(os.environ.get("BH_ROOT", Path(__file__).resolve().parents[1]))

# Trailing token on artifact folders: "<algo>_<TARGET>".
TARGET = os.environ.get("BH_TARGET", "")

MODEL_FILE = "model.pkl"


@dataclass(frozen=True)
class ModelEntry:
    algo: str  # from the artifact folder: cat, lr, ...
    variant: str  # from the run name: cat_orig, cat_smote, ...
    path: Path  # local, verified to exist
    run_id: str
    experiment_id: str
    experiment_name: str
    store_root: Path
    start_time: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.experiment_name}/{self.variant}"


# ---------------------------------------------------------------------------
# filesystem walk
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:  # noqa: BLE001
        return {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except Exception:  # noqa: BLE001
        return ""


def _read_metrics(run_dir: Path) -> Dict[str, float]:
    """FileStore writes one file per metric: '<timestamp> <value> <step>'."""
    out: Dict[str, float] = {}
    mdir = run_dir / "metrics"
    if not mdir.is_dir():
        return out
    for f in mdir.iterdir():
        if not f.is_file():
            continue
        lines = [ln for ln in _read_text(f).splitlines() if ln.strip()]
        if not lines:
            continue
        parts = lines[-1].split()
        if len(parts) >= 2:
            try:
                out[f.name] = float(parts[1])
            except ValueError:
                pass
    return out


def _strip_target(name: str) -> str:
    if TARGET and name.endswith(f"_{TARGET}"):
        return name[: -len(f"_{TARGET}")]
    return name


def _find_models_in_run(run_dir: Path) -> List[tuple]:
    """[(algo, path)] for every model.pkl this run wrote, on the LOCAL disk."""
    artifacts = run_dir / "artifacts"
    if not artifacts.is_dir():
        return []

    found = []
    direct = artifacts / MODEL_FILE
    if direct.is_file():
        found.append((_strip_target(run_dir.name), direct))

    for sub in sorted(p for p in artifacts.iterdir() if p.is_dir()):
        pkl = sub / MODEL_FILE
        if pkl.is_file():
            found.append((_strip_target(sub.name), pkl))
    return found


@lru_cache(maxsize=1)
def _index() -> List[ModelEntry]:
    """Index every model.pkl in every MLflow store under the project root."""
    entries: List[ModelEntry] = []
    skip = {".git", "node_modules", ".ipynb_checkpoints"}

    for exp_meta in PROJECT_ROOT.rglob("meta.yaml"):
        if any(p in skip for p in exp_meta.parts):
            continue

        meta = _read_yaml(exp_meta)
        if "experiment_id" not in meta or "run_id" in meta:
            continue  # this is a run, not an experiment

        exp_dir = exp_meta.parent
        exp_id = str(meta["experiment_id"])
        exp_name = str(meta.get("name") or exp_id)

        for run_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            run_meta = _read_yaml(run_dir / "meta.yaml")
            if not run_meta:
                continue
            if str(run_meta.get("lifecycle_stage", "active")) == "deleted":
                continue

            models = _find_models_in_run(run_dir)
            if not models:
                continue

            run_id = str(
                run_meta.get("run_id") or run_meta.get("run_uuid") or run_dir.name
            )
            run_name = (
                _read_text(run_dir / "tags" / "mlflow.runName")
                or str(run_meta.get("run_name") or "")
                or run_id[:8]
            )
            metrics = _read_metrics(run_dir)

            for algo, path in models:
                entries.append(
                    ModelEntry(
                        algo=algo,
                        variant=_strip_target(run_name),
                        path=path,
                        run_id=run_id,
                        experiment_id=exp_id,
                        experiment_name=exp_name,
                        store_root=exp_dir.parent,
                        start_time=int(run_meta.get("start_time") or 0),
                        metrics=metrics,
                    )
                )

    if not entries:
        raise RuntimeError(
            f"No model.pkl found under any MLflow store in {PROJECT_ROOT}."
        )

    entries.sort(key=lambda e: e.start_time, reverse=True)
    return entries


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------


def _matches(name: str) -> List[ModelEntry]:
    """Entries whose variant, algo, run_id, or qualified key matches `name`."""
    return [e for e in _index() if name in (e.variant, e.algo, e.run_id, e.key)]


def rank(
    name: str,
    metric: str = "roc_auc",
    ascending: bool = False,
    experiment: Optional[str] = None,
) -> pd.DataFrame:
    """
    Order every run matching `name` by a metric. For LOOKING, not for loading.

    e.g. rank("cat") shows all six cat runs so you can see which is cat_smote.
    """
    hits = _matches(name)
    if experiment:
        hits = [e for e in hits if experiment in (e.experiment_id, e.experiment_name)]
    if not hits:
        raise LookupError(f"Nothing matches '{name}'. Try available().")

    df = pd.DataFrame(
        [
            {
                "variant": e.variant,
                "algo": e.algo,
                "experiment": e.experiment_name,
                "run_id": e.run_id,
                **e.metrics,
            }
            for e in hits
        ]
    )
    if metric in df.columns:
        df = df.sort_values(metric, ascending=ascending)
    return df.reset_index(drop=True)


def resolve(
    name: str,
    experiment: Optional[str] = None,
    policy: str = "newest",
    metric: str = "roc_auc",
) -> ModelEntry:
    """
    Resolve one entry.

    name    : a variant (cat_smote), an algo (cat), a run_id, or 'exp/variant'
    policy  : 'newest' (default) or 'best' (max `metric`; see the docstring
              warning about selecting on the evaluation set)
    """
    if "/" in name and not experiment:
        experiment, name = name.split("/", 1)

    hits = _matches(name)
    if experiment:
        hits = [e for e in hits if experiment in (e.experiment_id, e.experiment_name)]
    if not hits:
        raise LookupError(
            f"No model for '{name}'"
            + (f" in experiment '{experiment}'" if experiment else "")
            + ".\nVariants available:\n  "
            + "\n  ".join(sorted({e.variant for e in _index()}))
        )

    if len(hits) > 1:
        if policy == "best":
            scored = [e for e in hits if metric in e.metrics]
            if not scored:
                raise LookupError(
                    f"policy='best' but no run matching '{name}' logged "
                    f"'{metric}'. Metrics present: "
                    f"{sorted({m for e in hits for m in e.metrics})}"
                )
            hits = sorted(scored, key=lambda e: e.metrics[metric], reverse=True)
        else:
            hits = sorted(hits, key=lambda e: e.start_time, reverse=True)

        print(
            f"[model_registry] '{name}' matched {len(hits)} runs; "
            f"policy='{policy}' selected {hits[0].variant} "
            f"({hits[0].experiment_name}, {hits[0].run_id[:8]}). "
            "Pass a variant name to pin it."
        )

    return hits[0]


# ---------------------------------------------------------------------------
# metric-based selection
# ---------------------------------------------------------------------------

# Whatever you happened to call average precision when you logged it.
AP_ALIASES = (
    "average_precision",
    "average_precision_score",
    "avg_precision",
    "aucpr",
    "pr_auc",
    "ap",
)


def metric_names() -> List[str]:
    """Every metric key logged anywhere in the store."""
    return sorted({m for e in _index() for m in e.metrics})


def resolve_metric(metric: str) -> str:
    """
    Map a loose metric name onto the key actually logged.

    'average_precision' will find 'test_average_precision', 'val_ap', etc.
    Raises if the guess is ambiguous, rather than silently ranking on the
    wrong split.
    """
    names = metric_names()
    if metric in names:
        return metric

    candidates = AP_ALIASES if metric.lower() in AP_ALIASES else (metric,)
    for cand in candidates:
        low = cand.lower()
        exact = [n for n in names if n.lower() == low]
        if exact:
            return exact[0]
        partial = [n for n in names if low in n.lower()]
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            raise LookupError(
                f"'{metric}' is ambiguous. Matches: {partial}\n"
                "Pass the exact key so the ranking split is unambiguous."
            )

    raise LookupError(f"No metric like '{metric}'. Logged metrics: {names}")


def _best_entries(
    metric: str,
    experiment: Optional[str] = None,
    per_experiment: bool = True,
    ascending: bool = False,
) -> tuple:
    """(resolved_metric, {group_key: winning ModelEntry})."""
    key = resolve_metric(metric)

    entries = [e for e in _index() if key in e.metrics]
    if experiment:
        entries = [
            e for e in entries if experiment in (e.experiment_id, e.experiment_name)
        ]
    if not entries:
        raise LookupError(f"No runs logged '{key}'.")

    winners: Dict[tuple, ModelEntry] = {}
    for e in entries:
        group = (e.experiment_name, e.algo) if per_experiment else (e.algo,)
        cur = winners.get(group)
        if cur is None:
            winners[group] = e
            continue
        better = (
            e.metrics[key] < cur.metrics[key]
            if ascending
            else e.metrics[key] > cur.metrics[key]
        )
        if better:
            winners[group] = e

    return key, winners


def best_per_algo(
    metric: str = "average_precision",
    experiment: Optional[str] = None,
    per_experiment: bool = True,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    The winning run for each algo, ranked by `metric`. Inspect before loading.

    per_experiment=True keeps the full-text and summarized-text versions of
    cat_feats_and_text separate instead of letting them compete.
    """
    key, winners = _best_entries(metric, experiment, per_experiment, ascending)

    rows = [
        {
            "algo": e.algo,
            "winner": e.variant,
            "experiment": e.experiment_name,
            key: e.metrics[key],
            "run_id": e.run_id,
            "n_candidates": sum(
                1
                for o in _index()
                if o.algo == e.algo
                and key in o.metrics
                and (not per_experiment or o.experiment_name == e.experiment_name)
            ),
            **{m: v for m, v in e.metrics.items() if m != key},
        }
        for e in winners.values()
    ]
    return (
        pd.DataFrame(rows).sort_values(key, ascending=ascending).reset_index(drop=True)
    )


def load_best_per_algo(
    metric: str = "average_precision",
    experiment: Optional[str] = None,
    per_experiment: bool = True,
    ascending: bool = False,
    qualified: bool = False,
) -> Dict[str, object]:
    """
    Load the top run for each algo by `metric`.

    NOTE: if `metric` is computed on your test set, this is model selection on
    the test set and the winning score is optimistically biased. Fine for
    exploration. For anything you report, pin the variant by name.
    """
    _, winners = _best_entries(metric, experiment, per_experiment, ascending)
    out = {}
    for e in winners.values():
        out[e.key if qualified else e.algo] = loadObjects(str(e.path))
    return out


# ---------------------------------------------------------------------------
# validation-set selection (the honest path)
# ---------------------------------------------------------------------------


def _proba(model, X):
    """Positive-class scores from a model_tuner model, sklearn, or CatBoost."""
    for attr in ("predict_proba", "decision_function", "predict"):
        fn = getattr(model, attr, None)
        if fn is None:
            continue
        out = fn(X)
        if attr == "predict_proba":
            arr = out.values if hasattr(out, "values") else out
            return arr[:, 1] if getattr(arr, "ndim", 1) > 1 else arr
        return out
    raise TypeError(f"{type(model)} exposes no scoring method.")


def _data_for(entry: ModelEntry, data):
    """
    data may be:
      (X, y)                      one matrix for every model
      {algo_or_variant: (X, y)}   text models need a different X than tabular
      callable(entry) -> (X, y)
    """
    if callable(data):
        return data(entry)
    if isinstance(data, dict):
        for k in (entry.variant, entry.algo, entry.key):
            if k in data:
                return data[k]
        raise KeyError(
            f"No validation data for '{entry.variant}' (algo '{entry.algo}'). "
            f"Keys given: {sorted(data)}"
        )
    return data


def score_candidates(data, scorer=None, name: Optional[str] = None) -> pd.DataFrame:
    """
    Score EVERY indexed model against a held-out set. Default scorer is
    average precision.

        from sklearn.metrics import average_precision_score
        score_candidates((X_valid, y_valid))
        score_candidates({"cat": (X_val, y_val),
                          "cat_feats_and_text": (X_val_text, y_val)})
    """
    if scorer is None:
        from sklearn.metrics import average_precision_score as scorer  # noqa: N813

    entries = _index() if name is None else _matches(name)
    rows = []
    for e in entries:
        try:
            X, y = _data_for(e, data)
            model = loadObjects(str(e.path))
            rows.append(
                {
                    "variant": e.variant,
                    "algo": e.algo,
                    "experiment": e.experiment_name,
                    "score": float(scorer(y, _proba(model, X))),
                    "run_id": e.run_id,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "variant": e.variant,
                    "algo": e.algo,
                    "experiment": e.experiment_name,
                    "score": float("nan"),
                    "run_id": e.run_id,
                    "error": str(exc)[:120],
                }
            )

    return (
        pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    )


def select_on_validation(
    data,
    scorer=None,
    per_experiment: bool = True,
) -> pd.DataFrame:
    """
    Champion of each family, chosen on the data you pass. Pass VALIDATION data.
    Then report test metrics for these winners only.
    """
    df = score_candidates(data, scorer=scorer).dropna(subset=["score"])
    group = ["experiment", "algo"] if per_experiment else ["algo"]
    idx = df.groupby(group)["score"].idxmax()
    return df.loc[idx].sort_values("score", ascending=False).reset_index(drop=True)


def load_selected(
    selection: pd.DataFrame, qualified: bool = False
) -> Dict[str, object]:
    """Load the models named in a select_on_validation() frame."""
    out = {}
    for _, row in selection.iterrows():
        entry = next(e for e in _index() if e.run_id == row["run_id"])
        out[entry.key if qualified else entry.algo] = loadObjects(str(entry.path))
    return out


@lru_cache(maxsize=None)
def load(
    name: str,
    experiment: Optional[str] = None,
    policy: str = "newest",
    metric: str = "roc_auc",
):
    """Load one model. Cached."""
    return loadObjects(str(resolve(name, experiment, policy, metric).path))


load_model = load  # alias


def load_all(
    only: Optional[List[str]] = None, qualified: bool = False
) -> Dict[str, object]:
    """
    One model per VARIANT (not per algo), so the six cat runs stay distinct.

    qualified=True keys by '<experiment_name>/<variant>' so nothing collides
    across the full-text and summarized-text experiments.
    """
    entries = _index()
    if only:
        entries = [
            e for e in entries if e.variant in only or e.algo in only or e.key in only
        ]

    out: Dict[str, object] = {}
    for e in entries:  # newest-first
        key = e.key if qualified else e.variant
        if key not in out:
            out[key] = loadObjects(str(e.path))
    return out


def variants() -> List[str]:
    return sorted({e.variant for e in _index()})


def algos() -> List[str]:
    return sorted({e.algo for e in _index()})


def available() -> pd.DataFrame:
    """Every model found on disk, newest first, with its logged metrics."""
    return pd.DataFrame(
        [
            {
                "variant": e.variant,
                "algo": e.algo,
                "experiment": e.experiment_name,
                "experiment_id": e.experiment_id,
                "run_id": e.run_id,
                "path": str(e.path.relative_to(PROJECT_ROOT)),
                **e.metrics,
            }
            for e in _index()
        ]
    )


def diagnose() -> None:
    entries = _index()
    print(f"project root : {PROJECT_ROOT}")
    print(f"stores       : {sorted({str(e.store_root) for e in entries})}")
    print(f"models       : {len(entries)}")
    print(f"algos        : {algos()}")
    print(f"variants     : {variants()}")
    print()
    print(available().to_string(index=False))


if __name__ == "__main__":
    diagnose()