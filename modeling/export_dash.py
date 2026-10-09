#!/usr/bin/env python
"""Export flat files for the Beverly Hills collisions Dash app (flask_apps).

Everything the app shows comes from the files this script writes to
`data/dash/`; copy that folder's contents into `flask_apps/bh_traffic/data/`.
The app needs no access to this project, MLflow, or the models.

Files
-----
    meta.json                 counts, dates, splits, chosen models, headline numbers
    hotspots.csv              ranked locations with coordinates (EB, SPF, ML, observed)
    monthly.csv               observed and fitted monthly counts per location
    forecast_monthly.csv      ML forecast, next 12 months, per location
    citywide_monthly.csv      every crash in the city by month (not only panel locations)
    backtest.csv              ranking backtest (from `make hotspots`)
    concentration.csv         Pareto test summary (from `make hotspots`)
    lorenz.csv                out-of-sample Lorenz curves (from `make hotspots`)
    count_metrics.csv         count models vs baselines by split
    shap_importance.csv       mean |SHAP| per feature, best count model per outcome
    shap_sample.csv           SHAP values for a test-row sample, top features
    classifier_preds.csv      severity classifier test-set predictions, every run
    classifier_metrics.csv    severity classifier metrics from MLflow
    leakage_audit.csv         AUC with and without the reporting-rule columns
    reporting_rule.csv        injury share in tagged vs untagged reports
    hotspot_profile.csv       per location, injury crashes in the last 5 years:
                              collision type, primary cause, who was involved,
                              with citywide shares for comparison

Coordinates
-----------
Intersections are geocoded from the OpenStreetMap drive network (osmnx): every
OSM node where two named streets meet gives an intersection key, built with
the same street normalization as preprocessing.py, so it matches
`location_key` directly. Mid-block keys take the mean of the geocoded
intersections on that street and are flagged approximate. Results are cached
in data/external/location_coords.csv; rows in
data/external/location_coords_manual.csv (location_key, lat, lon) override.

Run from the project root, or via `make export_dash`.
"""

from __future__ import annotations

import glob
import json
import os
import re
import unicodedata
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.config import PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, model_definitions
from core.constants import (
    location_key_var,
    count_outcomes,
    target_outcome,
    vru_var,
    leak_cols,
    eb_window_months,
    train_end,
    valid_end,
    first_target_month,
    forecast_horizon,
    min_location_crashes,
)
from core.functions import mlflow_load_model
from preprocessing.preprocessing import _norm_street
from modeling.hotspots import best_count_model, empirical_bayes, window_counts
from modeling.evaluation_counts import (
    row_metrics,
    ranking_metrics,
    shap_values_for,
    BASELINE_PREFIX,
)

app = typer.Typer(add_completion=False, help=__doc__)

## Beverly Hills city center; the OSM graph covers this radius
BH_CENTER = (34.0736, -118.4004)
BH_RADIUS_M = 4500

## OSM spells out suffixes BHPD abbreviates; mapped before _norm_street
OSM_WORDS = {
    r"\bROAD\b": "RD",
    r"\bPLACE\b": "PL",
    r"\bLANE\b": "LN",
    r"\bTERRACE\b": "TER",
    r"\bCIRCLE\b": "CIR",
    r"\bCOURT\b": "CT",
    r"\bPARKWAY\b": "PKWY",
}

## OSM names the big Santa Monica Blvd through Beverly Hills (State Route 2)
## plain "Santa Monica Boulevard"; BHPD calls it N Santa Monica Blvd (the
## small parallel road is "South Santa Monica Boulevard" in both). Without
## this alias every N Santa Monica intersection fails to geocode.
OSM_ALIASES = {"SANTA MONICA BLVD": ["N SANTA MONICA BLVD"]}

CLASSIFIER_RUNS = {
    "lr": ("lr", "orig"),
    "rf": ("rf", "orig"),
    "xgb": ("xgb", "orig"),
    "cat": ("cat", "orig"),
    "cat_no_party": ("cat", "orig_no_party"),
}


# --------------------------------------------------------------------------
# geocoding
# --------------------------------------------------------------------------
def norm_osm_names(names: pd.Series) -> pd.Series:
    """OSM street names -> the street form used in location_key."""
    ascii_names = names.map(
        lambda s: unicodedata.normalize("NFKD", str(s))
        .encode("ascii", "ignore")
        .decode()
    )
    s = ascii_names.str.upper()
    for pattern, repl in OSM_WORDS.items():
        s = s.str.replace(pattern, repl, regex=True)
    return _norm_street(s)


def intersections_from_graph(G) -> pd.DataFrame:
    """One row per intersection key: mean node coordinates and spread.

    A node joins every pair of distinct normalized street names on its
    incident edges. Divided roads meet at several nodes, so coordinates are
    averaged and the spread (meters) is kept as a quality flag.
    """
    raw = []
    for u, v, d in G.edges(data=True):
        names = d.get("name")
        if not names:
            continue
        for n in [names] if isinstance(names, str) else list(names):
            raw.append((u, n))
            raw.append((v, n))
    if not raw:
        return pd.DataFrame(columns=[location_key_var, "lat", "lon", "spread_m"])

    df = pd.DataFrame(raw, columns=["node", "name"]).drop_duplicates()
    df["street"] = norm_osm_names(df["name"]).astype("string")
    df = df.dropna(subset=["street"])

    def with_aliases(streets):
        out = set(streets)
        for st in streets:
            out.update(OSM_ALIASES.get(st, []))
        return sorted(out)

    ## a street and its own alias are one road, not an intersection
    same_road = {frozenset((k, a)) for k, al in OSM_ALIASES.items() for a in al}

    node_streets = df.groupby("node")["street"].agg(lambda s: sorted(set(s)))
    pts = defaultdict(list)
    for node, streets in node_streets.items():
        streets = with_aliases(streets)
        if len(streets) < 2:
            continue
        y, x = G.nodes[node]["y"], G.nodes[node]["x"]
        for a, b in combinations(streets, 2):
            if frozenset((a, b)) in same_road:
                continue
            pts[f"{a} / {b}"].append((y, x))

    rows = []
    for key, p in pts.items():
        arr = np.asarray(p)
        lat, lon = arr.mean(axis=0)
        dy = (arr[:, 0] - lat) * 111_000
        dx = (arr[:, 1] - lon) * 111_000 * np.cos(np.radians(lat))
        rows.append(
            {
                location_key_var: key,
                "lat": lat,
                "lon": lon,
                "spread_m": float(np.sqrt(dx**2 + dy**2).max()),
            }
        )
    return pd.DataFrame(rows)


def geocode(keys: pd.Index, cache: Path, manual: Path, refresh: bool) -> pd.DataFrame:
    """Coordinates for every key, from cache or OSM, plus manual overrides."""
    if cache.exists() and not refresh:
        coords = pd.read_csv(cache)
        print(f"Coordinates from cache: {cache}")
    else:
        import osmnx as ox

        print(f"Downloading OSM drive network ({BH_RADIUS_M / 1000:.1f} km) ...")
        G = ox.graph_from_point(BH_CENTER, dist=BH_RADIUS_M, network_type="drive")
        inter = intersections_from_graph(G)
        coords = pd.DataFrame({location_key_var: keys}).merge(
            inter, on=location_key_var, how="left"
        )
        coords["approx"] = coords["spread_m"] > 250

        ## Mid-block keys: mean of geocoded intersections on the same street
        mid = coords[location_key_var].str.endswith("(MIDBLOCK)")
        found = coords.dropna(subset=["lat"])
        for i in coords.index[mid]:
            street = coords.at[i, location_key_var].replace(" (MIDBLOCK)", "")
            on_street = found[
                found[location_key_var].str.split(" / ").map(lambda p: street in p)
            ]
            if len(on_street):
                coords.loc[i, ["lat", "lon"]] = on_street[["lat", "lon"]].mean().values
                coords.at[i, "approx"] = True

        cache.parent.mkdir(parents=True, exist_ok=True)
        coords.to_csv(cache, index=False)
        print(f"Cached coordinates: {cache}")

    if manual.exists():
        m = pd.read_csv(manual)[[location_key_var, "lat", "lon"]]
        coords = coords.set_index(location_key_var)
        coords.loc[m[location_key_var], ["lat", "lon"]] = m.set_index(location_key_var)[
            ["lat", "lon"]
        ].values
        coords.loc[m[location_key_var], "approx"] = False
        coords = coords.reset_index()
        print(f"Applied {len(m)} manual coordinates from {manual}")

    coords = coords.set_index(location_key_var).reindex(keys).reset_index()
    missing = coords["lat"].isna()
    print(
        f"Geocoded {(~missing).sum()} of {len(coords)} locations "
        f"({coords['approx'].fillna(False).sum()} approximate)."
    )
    if missing.any():
        print("Not geocoded (add to location_coords_manual.csv to place them):")
        for k in coords.loc[missing, location_key_var]:
            print(f"  {k}")
    return coords


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def latest(pattern: str) -> Path:
    files = sorted(glob.glob(pattern))
    if not files:
        raise typer.BadParameter(
            f"nothing matches {pattern}; run `make hotspots` first"
        )
    return Path(files[-1])


def crash_level(crash_path: Path) -> pd.DataFrame:
    c = pd.read_parquet(crash_path).dropna(subset=["year", "month"])
    c["period"] = pd.PeriodIndex.from_fields(
        year=c["year"].astype(int), month=c["month"].astype(int), freq="M"
    )
    c["injury_or_fatal"] = c[target_outcome[0]].astype(int)
    c["vru_injury_or_fatal"] = (c[target_outcome[0]] * c[vru_var]).astype(int)
    return c


def report_tags(raw_path: Path, index: pd.Index) -> pd.DataFrame:
    """Hit-and-run / DUI / CPD tags from the raw Accident Type, by ObjectId."""
    raw = pd.read_parquet(raw_path).set_index("ObjectId")
    at = raw["Accident Type"].fillna("").astype(str).str.upper().str.replace(" ", "")
    tags = pd.DataFrame(
        {
            "hit_run": at.str.contains("HIT&RUN", regex=False),
            "dui": at.str.contains("DUI", regex=False),
            "cpd": at.str.contains("CPD", regex=False),
        }
    ).astype(int)
    return tags.reindex(index)


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# crash profiles: what kind of injury crashes happen at each location
# --------------------------------------------------------------------------
## BHPD "Type of accident" -> plain collision type
CRASH_TYPES = {
    "BROADSIDE": "Broadside (T-bone)",
    "REAR END": "Rear-end",
    "SIDE SWIPE": "Sideswipe",
    "HEAD-ON": "Head-on",
    "HEAD ON": "Head-on",
    "HIT OBJECT": "Hit object",
    "VEHICLE/PEDESTRIAN": "Vehicle hits pedestrian",
}

## Primary collision factor (California Vehicle Code section) -> plain cause,
## grouped roughly the way SWITRS groups PCF violations
PCF_GROUPS = {
    "Unsafe speed": ["22350", "22351", "22352"],
    "Left turn, failed to yield": ["21801"],
    "Failed to yield (other)": ["21800", "21802", "21803", "21804", "21806"],
    "Failed to yield to pedestrian": ["21950", "21951", "21952"],
    "Ran red light": ["21453", "21451", "21452", "21457"],
    "Ran stop sign": ["22450"],
    "Unsafe turn or lane change": [
        "22107", "21658", "22100", "22101", "22102", "22103", "22104", "22105",
        "21460", "21650", "21651", "21750", "21755",
    ],
    "Unsafe starting or backing": ["22106"],
    "Following too close": ["21703"],
    "Opening door into traffic": ["22517"],
    "DUI": ["23152", "23153"],
    "Pedestrian violation": ["21954", "21955", "21456", "21949", "21961"],
    "Cyclist violation": ["21200", "21201", "21202", "21208"],
}
PCF_LOOKUP = {code: g for g, codes in PCF_GROUPS.items() for code in codes}

## Fallback when the vehicle code is missing: BHPD's free-text cause.
## Order matters (first match wins).
PCF_TEXT = [
    ("YIELD TO PED", "Failed to yield to pedestrian"),
    ("PEDESTRIAN ROW", "Failed to yield to pedestrian"),
    ("PEDESTRIAN RIGHT", "Failed to yield to pedestrian"),
    ("PEDES", "Pedestrian violation"),
    ("LEFT TURN", "Left turn, failed to yield"),
    ("YIELD", "Failed to yield (other)"),
    ("SPEED", "Unsafe speed"),
    ("RED", "Ran red light"),
    ("STOP SIGN", "Ran stop sign"),
    ("TURN", "Unsafe turn or lane change"),
    ("LANE", "Unsafe turn or lane change"),
    ("BACKING", "Unsafe starting or backing"),
    ("FOLLOW", "Following too close"),
    ("DOOR", "Opening door into traffic"),
    ("DUI", "DUI"),
    ("INFLUENCE", "DUI"),
    ("BICYCLE", "Cyclist violation"),
]

WHO = {
    "ped_involved": "Pedestrian involved",
    "bike_involved": "Cyclist involved",
    "moto_involved": "Motorcyclist involved",
}

_VC = re.compile(r"(\d{5})")


def _cause(code: str, text: str) -> str:
    m = _VC.search(code or "")
    if m and m.group(1) in PCF_LOOKUP:
        return PCF_LOOKUP[m.group(1)]
    t = (text or "").upper()
    for key, group in PCF_TEXT:
        if key in t:
            return group
    return "Other or unknown"


def crash_profiles(
    raw_path: Path, crash_path: Path, panel_end: pd.Period, months: int
) -> pd.DataFrame:
    """Collision type, primary cause, and who was involved, for injury crashes
    in the last `months` months, per location and citywide (key "__CITY__").

    Long rows: location_key, dimension, category, n, n_location, share,
    city_share, ratio (share here / share citywide). Type and cause are one
    category per crash; "who" categories can overlap.
    """
    raw = pd.read_parquet(raw_path).set_index("ObjectId")
    c = crash_level(crash_path).set_index("object_id")
    c = c[(c["period"] > panel_end - months) & (c["period"] <= panel_end)]
    c = c[c["injury_or_fatal"] == 1].copy()
    r = raw.reindex(c.index)
    t = r["Type of accident"].fillna("").astype(str).str.strip().str.upper()
    c["type"] = t.map(CRASH_TYPES).fillna("Other")
    c["cause"] = [
        _cause(str(code), str(text))
        for code, text in zip(r["PCF VC"].fillna(""), r["Other Imp Drive"].fillna(""))
    ]

    n_loc = c.groupby(location_key_var).size()
    n_city = len(c)
    rows = []
    for dim in ("type", "cause"):
        city = c[dim].value_counts() / n_city
        g = c.groupby([location_key_var, dim]).size().rename("n").reset_index()
        g = g.rename(columns={dim: "category"})
        g["dimension"] = dim
        g["city_share"] = g["category"].map(city)
        rows.append(g)
        rows.append(
            pd.DataFrame(
                {
                    location_key_var: "__CITY__",
                    "category": city.index,
                    "n": (city * n_city).round().astype(int).values,
                    "dimension": dim,
                    "city_share": city.values,
                }
            )
        )
    for col, label in WHO.items():
        if col not in c.columns:
            continue
        city = float(c[col].mean())
        g = c.groupby(location_key_var)[col].sum().rename("n").reset_index()
        g = g[g["n"] > 0].copy()
        g["category"], g["dimension"], g["city_share"] = label, "who", city
        rows.append(g)
        rows.append(
            pd.DataFrame(
                {
                    location_key_var: ["__CITY__"],
                    "category": [label],
                    "n": [int(c[col].sum())],
                    "dimension": ["who"],
                    "city_share": [city],
                }
            )
        )

    out = pd.concat(rows, ignore_index=True)
    out["n"] = out["n"].astype(int)
    out["n_location"] = out[location_key_var].map(n_loc).fillna(n_city).astype(int)
    out["share"] = out["n"] / out["n_location"]
    out["ratio"] = out["share"] / out["city_share"]
    cols = [
        location_key_var, "dimension", "category", "n", "n_location", "share",
        "city_share", "ratio",
    ]
    return out[cols].sort_values(
        [location_key_var, "dimension", "n"], ascending=[True, True, False]
    )


@app.command()
def run(
    out_dir: Path = typer.Option(PROCESSED_DATA_DIR.parent / "dash"),
    hotspot_dir: Path = typer.Option(PROCESSED_DATA_DIR / "hotspots"),
    authors: str = typer.Option("Leonid Shpaner", "--authors"),
    refresh_coords: bool = typer.Option(False, "--refresh-coords/--cached-coords"),
    shap_rows: int = typer.Option(1500, "--shap-rows"),
    shap_features: int = typer.Option(12, "--shap-features"),
) -> None:
    """Write every file the Dash app reads to `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    P = PROCESSED_DATA_DIR

    panel = pd.read_parquet(P / "panel.parquet")
    panel["period"] = pd.PeriodIndex(panel["period"], freq="M")
    observed = panel[panel["split"] != "forecast"]
    future = panel[panel["split"] == "forecast"]
    X = pd.read_parquet(P / "X_counts.parquet")
    Y = pd.read_parquet(P / "y_counts.parquet")
    X_fc = pd.read_parquet(P / "X_counts_forecast.parquet")
    panel_end = observed["period"].max()
    stamp = str(panel_end)

    ############################################################################
    # Hotspot table + coordinates
    ############################################################################
    hot = pd.read_csv(latest(str(hotspot_dir / "hotspots_through_*.csv")))
    coords = geocode(
        pd.Index(hot[location_key_var]),
        EXTERNAL_DATA_DIR / "location_coords.csv",
        EXTERNAL_DATA_DIR / "location_coords_manual.csv",
        refresh_coords,
    )
    hot = hot.merge(coords[[location_key_var, "lat", "lon", "approx"]], how="left")
    totals = observed.groupby(location_key_var)[count_outcomes].sum()
    hot["inj_total"] = hot[location_key_var].map(totals["inj_count"])
    hot["vru_inj_total"] = hot[location_key_var].map(totals["vru_inj_count"])
    hot["midblock"] = hot[location_key_var].str.endswith("(MIDBLOCK)")
    hot.round(4).to_csv(out_dir / "hotspots.csv", index=False)

    ############################################################################
    # Best count model per outcome: fitted history, forecast, metrics, SHAP
    ############################################################################
    monthly = observed[[location_key_var, "period", "split", *count_outcomes]].copy()
    fc_rows, metric_rows, imp_rows, shap_rows_out, best = [], [], [], [], {}

    for outcome in count_outcomes:
        tag = BASELINE_PREFIX[outcome]
        name, model = best_count_model(outcome, X, Y[outcome], panel)
        if model is None:
            raise typer.BadParameter(
                f"no count model for {outcome}; run train_count_models"
            )
        best[outcome] = name
        print(f"{outcome}: best count model {name}")

        fitted = pd.Series(model.predict(X), index=X.index)
        monthly[f"fit_{tag}"] = fitted.reindex(monthly.index)

        fc = pd.Series(model.predict(X_fc), index=X_fc.index)
        fc_rows.append(
            pd.DataFrame(
                {
                    location_key_var: future.loc[fc.index, location_key_var],
                    "period": future.loc[fc.index, "period"].astype(str),
                    "outcome": outcome,
                    "pred": fc.values,
                }
            )
        )

        ## Metrics: best model and the two naive baselines, by split
        preds = {
            name: fitted,
            "history_rate": X[f"{tag}_hist_rate"],
            "last_12_months": X[f"{tag}_roll12"] / 12,
        }
        for split in ("train", "valid", "test"):
            mask = panel.loc[X.index, "split"] == split
            for m, p in preds.items():
                r = {"outcome": outcome, "split": split, "model": m}
                r.update(row_metrics(Y.loc[mask, outcome], p[mask]))
                rank, _, _ = ranking_metrics(
                    Y.loc[mask, outcome],
                    p[mask],
                    panel.loc[X.index[mask], location_key_var],
                )
                r.update(rank)
                metric_rows.append(r)

        ## SHAP on the test months
        test_idx = X.index[panel.loc[X.index, "split"] == "test"]
        train_idx = X.index[panel.loc[X.index, "split"] == "train"]
        bg = X.loc[train_idx].sample(min(500, len(train_idx)), random_state=222)
        sv, Xt = shap_values_for(model, X.loc[test_idx], bg)
        imp = pd.Series(np.abs(sv).mean(axis=0), index=Xt.columns).sort_values(
            ascending=False
        )
        direction = pd.Series(
            [
                np.corrcoef(Xt[c], sv[:, j])[0, 1] if Xt[c].std() > 0 else 0.0
                for j, c in enumerate(Xt.columns)
            ],
            index=Xt.columns,
        )
        imp_rows.append(
            pd.DataFrame(
                {
                    "outcome": outcome,
                    "model": name,
                    "feature": imp.index,
                    "mean_abs_shap": imp.values,
                    "direction": direction.loc[imp.index].fillna(0).values,
                }
            )
        )
        rng = np.random.default_rng(222)
        pick = rng.choice(len(Xt), size=min(shap_rows, len(Xt)), replace=False)
        for feat in imp.index[:shap_features]:
            j = Xt.columns.get_loc(feat)
            vals = Xt[feat].to_numpy()[pick]
            pct = pd.Series(Xt[feat]).rank(pct=True).to_numpy()[pick]
            shap_rows_out.append(
                pd.DataFrame(
                    {
                        "outcome": outcome,
                        "feature": feat,
                        "shap": sv[pick, j],
                        "value": vals,
                        "value_pct": pct,
                    }
                )
            )

    monthly["period"] = monthly["period"].astype(str)
    monthly.round(4).to_csv(out_dir / "monthly.csv", index=False)
    pd.concat(fc_rows).round(4).to_csv(out_dir / "forecast_monthly.csv", index=False)
    pd.DataFrame(metric_rows).round(5).to_csv(
        out_dir / "count_metrics.csv", index=False
    )
    pd.concat(imp_rows).round(5).to_csv(out_dir / "shap_importance.csv", index=False)
    pd.concat(shap_rows_out).round(4).to_csv(out_dir / "shap_sample.csv", index=False)

    ############################################################################
    # Citywide monthly (all crashes, every location)
    ############################################################################
    crashes = crash_level(P / "df_sans_zero.parquet")
    crashes = crashes[crashes["period"] <= panel_end]
    city = (
        crashes.groupby("period")
        .agg(
            crashes=("injury_or_fatal", "size"),
            injury_or_fatal=("injury_or_fatal", "sum"),
            vru_injury_or_fatal=("vru_injury_or_fatal", "sum"),
        )
        .reset_index()
    )
    city["period"] = city["period"].astype(str)
    city.to_csv(out_dir / "citywide_monthly.csv", index=False)

    ############################################################################
    # Crash profiles: what kind of injury crashes happen at each location
    ############################################################################
    prof = crash_profiles(
        P / "df.parquet", P / "df_sans_zero.parquet", panel_end, eb_window_months
    )
    prof.round(4).to_csv(out_dir / "hotspot_profile.csv", index=False)

    ############################################################################
    # Hotspot artifacts written by `make hotspots`
    ############################################################################
    for name in ("backtest", "concentration", "lorenz"):
        src = latest(str(hotspot_dir / f"{name}_through_*.csv"))
        pd.read_csv(src).to_csv(out_dir / f"{name}.csv", index=False)

    ############################################################################
    # Severity classifier: test predictions and MLflow metrics
    ############################################################################
    Xc = pd.read_parquet(P / "X.parquet")
    yc = pd.read_parquet(P / "y.parquet")[target_outcome[0]].squeeze()
    preds, cmetrics, test_index = {}, [], None
    import mlflow

    for key, (algo, pipe) in CLASSIFIER_RUNS.items():
        est = model_definitions[algo]["estimator_name"]
        suffix = "_no_party" if pipe == "orig_no_party" else ""
        try:
            model = mlflow_load_model(
                experiment_name=f"{target_outcome[0]}_model",
                run_name=f"{est}_{pipe}_training",
                model_name=f"{est}_{target_outcome[0]}{suffix}",
            )
        except Exception:
            model = None
        if model is None:
            print(f"classifier {key}: not found, skipped")
            continue
        Xt_, yt_ = model.get_test_data(Xc, yc)
        if test_index is None:
            test_index = Xt_.index
        elif not Xt_.index.equals(test_index):
            print(f"classifier {key}: different test split, skipped")
            continue
        preds[f"p_{key}"] = model.predict_proba(Xt_)[:, 1]

        run = mlflow.search_runs(
            experiment_names=[f"{target_outcome[0]}_model"],
            filter_string=f"tags.mlflow.runName = '{est}_{pipe}_training'",
        )
        if len(run):
            r = run.iloc[0]
            for col in [c for c in run.columns if c.startswith("metrics.")]:
                cmetrics.append(
                    {"run": key, "metric": col[len("metrics.") :], "value": r[col]}
                )

    if test_index is not None:
        cp = pd.DataFrame(
            {"y": yc.loc[test_index].values, "vru": Xc.loc[test_index, vru_var].values}
        )
        for k, v in preds.items():
            cp[k] = v
        cp.round(5).to_csv(out_dir / "classifier_preds.csv", index=False)
    pd.DataFrame(cmetrics).to_csv(out_dir / "classifier_metrics.csv", index=False)

    ############################################################################
    # Leakage audit: what the reporting-rule columns do to discrimination
    ############################################################################
    dsz = pd.read_parquet(P / "df_sans_zero.parquet").set_index("object_id")
    tags = report_tags(P / "df.parquet", Xc.index)
    statute = dsz.reindex(Xc.index)[[c for c in leak_cols if c in dsz.columns]]
    leaky = pd.concat([Xc, tags, statute], axis=1)
    leaky = leaky.loc[:, ~leaky.columns.duplicated()]
    lr = make_pipeline(
        SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=3000, C=0.1)
    )
    sets = {
        "Report tags only (3 columns)": tags,
        "All features, leakage included": leaky,
        "Leakage removed (model as trained)": Xc,
    }
    audit = []
    for label, Z in sets.items():
        auc = cross_val_score(lr, Z, yc, cv=5, scoring="roc_auc")
        audit.append(
            {
                "feature_set": label,
                "n_features": Z.shape[1],
                "auc": auc.mean(),
                "auc_sd": auc.std(),
            }
        )
        print(f"  {label}: AUC {auc.mean():.3f}")
    pd.DataFrame(audit).round(4).to_csv(out_dir / "leakage_audit.csv", index=False)

    tagged = tags.max(axis=1).astype(bool)
    rule = pd.DataFrame(
        {
            "reports": ["Tagged hit-and-run, DUI, or CPD", "Untagged"],
            "n": [int(tagged.sum()), int((~tagged).sum())],
            "injury_share": [yc[tagged].mean(), yc[~tagged].mean()],
        }
    )
    rule.round(4).to_csv(out_dir / "reporting_rule.csv", index=False)

    ############################################################################
    # Meta
    ############################################################################
    conc = pd.read_csv(out_dir / "concentration.csv")
    cw = conc[
        (conc["outcome"] == "inj_count") & (conc["test"] == "out-of-sample, citywide")
    ]
    alpha = {}
    for outcome in count_outcomes:
        obs = window_counts(observed, outcome, panel_end, eb_window_months).reindex(
            pd.Index(hot[location_key_var]), fill_value=0
        )
        alpha[outcome] = float(
            empirical_bayes(obs, eb_window_months / 12)["alpha"].iloc[0]
        )

    first = crashes["period"].min()
    meta = {
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "authors": authors,
        "n_crashes": int(len(crashes)),
        "n_injury": int(crashes["injury_or_fatal"].sum()),
        "n_vru_injury": int(crashes["vru_injury_or_fatal"].sum()),
        "n_fatal_note": "Fatal and injury crashes share one class (19 fatal).",
        "first_month": str(first),
        "last_month": stamp,
        "n_locations_any": int(crashes[location_key_var].nunique()),
        "n_panel_locations": int(hot.shape[0]),
        "min_location_crashes": min_location_crashes,
        "panel_share_injury": float(
            crashes.loc[
                crashes[location_key_var].isin(hot[location_key_var]), "injury_or_fatal"
            ].sum()
            / crashes["injury_or_fatal"].sum()
        ),
        "first_target_month": first_target_month,
        "train_end": train_end,
        "valid_end": valid_end,
        "forecast_horizon": forecast_horizon,
        "forecast_from": str(panel_end + 1),
        "forecast_to": str(panel_end + forecast_horizon),
        "eb_window_months": eb_window_months,
        "eb_alpha": alpha,
        "best_count_model": best,
        "top_share_citywide": (
            float(cw[[c for c in cw.columns if c.startswith("top_")][0]].iloc[0])
            if len(cw)
            else None
        ),
        "top_locations": int(cw["top_locations"].iloc[0]) if len(cw) else None,
        "top_locations_share_of_all": (
            float(cw["top_locations_share_of_all"].iloc[0]) if len(cw) else None
        ),
        "classifier_test_n": int(len(test_index)) if test_index is not None else 0,
        "data_note": "City of Beverly Hills Open Data, Historic Collision Records of "
        "Beverly Hills (BHPD Traffic Bureau). Street geometry: OpenStreetMap contributors.",
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote {len(os.listdir(out_dir))} files to {out_dir}")
    print("Copy them into flask_apps/bh_traffic/data/")


def main() -> None:
    app()


if __name__ == "__main__":
    main()