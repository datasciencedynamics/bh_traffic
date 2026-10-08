#!/usr/bin/env python
"""Hotspot screening: ranked locations by expected injury-or-fatal crashes.

Two estimates per location, both reported as crashes per year:

Empirical Bayes (EB), the Highway Safety Manual network-screening method.
    A safety performance function (SPF) gives the expected count for a
    location of its type (Poisson GLM on mid-block and corridor flags, fit to
    the last `eb_window_months` of counts). EB then blends that expectation
    with the location's own observed count, trusting the observation more
    when the location has more expected crashes:

        w  = 1 / (1 + alpha * SPF)
        EB = w * SPF + (1 - w) * observed

    alpha is the negative binomial overdispersion, estimated by moments. EB
    corrects regression to the mean: a quiet location with one bad year is
    pulled back toward its type, a persistently bad one is not. "Excess"
    (EB minus SPF) is the HSM potential for safety improvement.

ML forecast.
    The best trained count model (lowest validation Poisson deviance) summed
    over the next 12 months. Its features look back at least 12 months, so
    the forecast uses only data already on record.

Backtest.
    At each cutoff in `backtest_cutoffs`, rank locations using data through
    the cutoff only, then score against the crashes observed in the next 12
    months. Compared: observed counts over the last 12, 36, and 60 months, EB,
    and the ML forecast.

Concentration (Pareto) test.
    Hypothesis: a small set of intersections, identifiable from past crashes,
    accounts for a disproportionate and persistent share of future injury
    crashes. Measured three ways: in-sample citywide (the naive 80/20 view,
    inflated by noise and one-off locations), out-of-sample within the panel
    (rank through train_end, score the crashes after it), and out-of-sample
    citywide. Each out-of-sample share is shown against perfect foresight.

Run from the project root, or via `make hotspots`.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import PoissonRegressor

from core.constants import (
    location_key_var,
    count_outcomes,
    corridors,
    eb_window_months,
    backtest_cutoffs,
    top_n_hotspots,
    valid_end,
    train_end,
    forecast_horizon,
    concentration_share,
    plot_colors,
)
from core.config import PROCESSED_DATA_DIR, count_model_definitions
from core.functions import mlflow_load_model, lorenz, share_at, locations_for

app = typer.Typer(add_completion=False, help=__doc__)

LABEL = {"inj_count": "inj", "vru_inj_count": "vru_inj"}


# --------------------------------------------------------------------------
# site features and Empirical Bayes
# --------------------------------------------------------------------------
def site_features(locations: pd.Index) -> pd.DataFrame:
    """Static descriptors for the SPF: mid-block flag and corridor flags."""
    S = pd.DataFrame(index=locations)
    S["midblock"] = locations.str.contains("MIDBLOCK", regex=False).astype(int)
    for corridor in corridors:
        S[f"on_{corridor.lower().replace(' ', '_')}"] = locations.str.contains(
            corridor, regex=False
        ).astype(int)
    return S.loc[:, S.std() > 0]


def empirical_bayes(observed: pd.Series, years: float) -> pd.DataFrame:
    """EB estimate per location from counts observed over `years` years.

    Returns annual SPF, EB, and excess, plus the weight and alpha used.
    """
    S = site_features(observed.index)
    spf = PoissonRegressor(alpha=0.0, max_iter=3000).fit(S, observed / years)
    mu = spf.predict(S) * years  # expected count over the window

    ## Method-of-moments overdispersion: Var = mu + alpha * mu^2
    alpha = max(1e-3, float((((observed - mu) ** 2 - mu) / mu**2).mean()))
    w = 1.0 / (1.0 + alpha * mu)
    eb = w * mu + (1.0 - w) * observed

    return pd.DataFrame(
        {
            "spf_per_year": mu / years,
            "eb_per_year": eb / years,
            "excess_per_year": (eb - mu) / years,
            "eb_weight": w,
            "alpha": alpha,
        },
        index=observed.index,
    )


def window_counts(counts: pd.DataFrame, outcome: str, end, months: int) -> pd.Series:
    """Per-location total over the `months` months ending at `end`."""
    mask = (counts["period"] > end - months) & (counts["period"] <= end)
    return counts.loc[mask].groupby(location_key_var)[outcome].sum()


# --------------------------------------------------------------------------
# best ML count model, selected on validation deviance
# --------------------------------------------------------------------------
def best_count_model(outcome, X, y, panel):
    """Load every trained count model for `outcome`, keep the lowest
    validation Poisson deviance. Returns (name, model) or (None, None)."""
    from sklearn.metrics import mean_poisson_deviance

    valid = panel.loc[X.index, "split"] == "valid"
    scores, models = {}, {}
    for model_type, spec in count_model_definitions.items():
        name = spec["estimator_name"]
        try:
            model = mlflow_load_model(
                experiment_name=f"{outcome}_model",
                run_name=f"{name}_orig_training",
                model_name=f"{name}_{outcome}",
            )
        except Exception:
            continue
        if model is None:
            continue
        p = np.clip(model.predict(X[valid]), 1e-6, None)
        scores[name] = mean_poisson_deviance(y[valid], p)
        models[name] = model

    if not scores:
        return None, None
    print(f"  validation deviance: { {k: round(v, 4) for k, v in scores.items()} }")
    best = min(scores, key=scores.get)
    return best, models[best]


# --------------------------------------------------------------------------
# backtest
# --------------------------------------------------------------------------
def score_ranking(pred: pd.Series, actual: pd.Series, top: int) -> dict:
    order = pred.sort_values(ascending=False).index
    return {
        "spearman": spearmanr(pred, actual.loc[pred.index]).correlation,
        "top10_capture": actual.loc[order[:10]].sum() / actual.sum(),
        f"top{top}_capture": actual.loc[order[:top]].sum() / actual.sum(),
    }


def backtest(counts, outcome, locations, ml_pred, top):
    rows = []
    for cut in backtest_cutoffs:
        c = pd.Period(cut, "M")
        future = window_counts(counts, outcome, c + forecast_horizon, forecast_horizon)
        future = future.reindex(locations, fill_value=0)
        if future.sum() == 0:
            continue

        rankings = {}
        for m in (12, 36, 60):
            rankings[f"observed_last_{m}"] = (
                window_counts(counts, outcome, c, m).reindex(locations, fill_value=0)
                / m
                * 12
            )
        obs = window_counts(counts, outcome, c, eb_window_months).reindex(
            locations, fill_value=0
        )
        rankings["empirical_bayes"] = empirical_bayes(obs, eb_window_months / 12)[
            "eb_per_year"
        ]
        if ml_pred is not None:
            in_window = (counts["period"] > c) & (
                counts["period"] <= c + forecast_horizon
            )
            rankings["ml_forecast"] = (
                ml_pred[in_window]
                .groupby(counts.loc[in_window, location_key_var])
                .sum()
                .reindex(locations, fill_value=0)
            )

        oracle = future.sort_values(ascending=False)
        for name, pred in rankings.items():
            r = {"cutoff": cut, "ranking": name}
            r.update(score_ranking(pred, future, top))
            rows.append(r)
        rows.append(
            {
                "cutoff": cut,
                "ranking": "oracle (perfect foresight)",
                "spearman": 1.0,
                "top10_capture": oracle.iloc[:10].sum() / future.sum(),
                f"top{top}_capture": oracle.iloc[:top].sum() / future.sum(),
            }
        )
        if ml_pred is not None and c < pd.Period(valid_end, "M"):
            rows[-2]["ranking"] += " (validation period)"
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# concentration (Pareto) test
# --------------------------------------------------------------------------
SOURCE = {"inj_count": "injury_or_fatal", "vru_inj_count": "vru_injury_or_fatal"}
TITLE = {
    "inj_count": "All injury-or-fatal crashes",
    "vru_inj_count": "Pedestrian, cyclist, motorcyclist",
}

INK, INK_2 = plot_colors["ink"], plot_colors["ink_2"]
GRID, SURFACE = plot_colors["grid"], plot_colors["surface"]
BLUE, ORANGE = plot_colors["series"][:2]


def concentration(crashes, panel, out_dir, stamp, q=concentration_share):
    """In-sample and out-of-sample concentration, plus Lorenz curves."""
    split = pd.Period(train_end, "M")
    crashes = crashes.dropna(subset=[location_key_var, "period"])
    rows, curves = [], []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, outcome in zip(axes, count_outcomes):
        src = SOURCE[outcome]
        c = crashes[crashes[src] == 1]

        ## 1. In-sample, citywide: every location with any crash is the base
        all_locs = crashes[location_key_var].unique()
        totals = c.groupby(location_key_var).size().reindex(all_locs, fill_value=0)
        ins = lorenz(totals, totals)
        rows.append(
            {
                "outcome": outcome,
                "test": "in-sample, citywide (naive)",
                "locations": len(all_locs),
                f"top_{q:.0%}_share": share_at(ins, q),
                "locations_for_80pct": locations_for(ins),
                "oracle_top_share": share_at(ins, q),
            }
        )

        ## 2. Out-of-sample, panel: rank through train_end, score after it
        obs = panel[panel["split"] != "forecast"]
        past = obs[obs["period"] <= split].groupby(location_key_var)[outcome].sum()
        future = obs[obs["period"] > split].groupby(location_key_var)[outcome].sum()
        oos = lorenz(past, future)
        orc = lorenz(future, future)
        rows.append(
            {
                "outcome": outcome,
                "test": "out-of-sample, panel",
                "locations": len(past),
                f"top_{q:.0%}_share": share_at(oos, q),
                "locations_for_80pct": locations_for(oos),
                "oracle_top_share": share_at(orc, q),
            }
        )

        ## 3. Out-of-sample, citywide: the same top locations against every
        ## injury crash after train_end, wherever it happened
        k = int(round(q * len(past)))
        top = past.sort_values(ascending=False).index[:k]
        after = c[c["period"] > split]
        before_locs = crashes.loc[
            crashes["period"] <= split, location_key_var
        ].nunique()
        rows.append(
            {
                "outcome": outcome,
                "test": "out-of-sample, citywide",
                "locations": before_locs,
                f"top_{q:.0%}_share": after[location_key_var].isin(top).mean(),
                "locations_for_80pct": np.nan,
                "oracle_top_share": np.nan,
                "top_locations": k,
                "top_locations_share_of_all": k / before_locs,
            }
        )

        curves.append(
            pd.concat(
                [
                    oos.assign(outcome=outcome, curve="ranked_by_past"),
                    orc.assign(outcome=outcome, curve="perfect_foresight"),
                ]
            )
        )

        ## Lorenz plot: ranked by past vs perfect foresight
        ax.set_facecolor(SURFACE)
        ax.plot(
            [0, 100],
            [0, 100],
            color=INK_2,
            linewidth=1,
            linestyle=":",
            label="No concentration",
        )
        ax.plot(
            orc["location_share"] * 100,
            orc["crash_share"] * 100,
            color=ORANGE,
            linewidth=2,
            linestyle="--",
            label="Perfect foresight",
        )
        ax.plot(
            oos["location_share"] * 100,
            oos["crash_share"] * 100,
            color=BLUE,
            linewidth=2,
            label="Ranked by past crashes",
        )
        ax.axvline(q * 100, color=GRID, linewidth=1, zorder=0)

        s_oos, s_orc = share_at(oos, q), share_at(orc, q)
        ax.scatter(
            [q * 100, q * 100],
            [s_oos * 100, s_orc * 100],
            s=40,
            color=[BLUE, ORANGE],
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
        )
        ax.annotate(
            f"{s_oos:.0%}",
            (q * 100, s_oos * 100),
            xytext=(8, -14),
            textcoords="offset points",
            color=INK,
            fontsize=10,
        )
        ax.annotate(
            f"{s_orc:.0%}",
            (q * 100, s_orc * 100),
            xytext=(8, 4),
            textcoords="offset points",
            color=INK,
            fontsize=10,
        )

        ax.set_title(TITLE[outcome], color=INK, fontsize=11, loc="left")
        ax.set_xlabel(f"Locations, ranked (% of {len(past)})", color=INK_2)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK_2)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_ylabel(f"Share of injury crashes after {train_end} (%)", color=INK_2)
    axes[0].legend(frameon=False, loc="lower right", labelcolor=INK)
    fig.suptitle(
        f"Top {q:.0%} of intersections, ranked on crashes through {train_end}, "
        "and their share of later injury crashes",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()

    summary = pd.DataFrame(rows)
    summary.round(4).to_csv(out_dir / f"concentration_through_{stamp}.csv", index=False)
    pd.concat(curves).round(4).to_csv(
        out_dir / f"lorenz_through_{stamp}.csv", index=False
    )
    fig_path = out_dir / f"lorenz_through_{stamp}.png"
    fig.savefig(fig_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return summary, fig_path


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------
@app.command()
def run(
    panel_path: Path = typer.Option(PROCESSED_DATA_DIR / "panel.parquet"),
    features_path: Path = typer.Option(PROCESSED_DATA_DIR / "X_counts.parquet"),
    labels_path: Path = typer.Option(PROCESSED_DATA_DIR / "y_counts.parquet"),
    forecast_path: Path = typer.Option(
        PROCESSED_DATA_DIR / "X_counts_forecast.parquet"
    ),
    crash_path: Path = typer.Option(PROCESSED_DATA_DIR / "df_sans_zero.parquet"),
    out_dir: Path = typer.Option(PROCESSED_DATA_DIR / "hotspots"),
    top: int = typer.Option(top_n_hotspots, "--top", help="rows to print"),
) -> None:
    """Backtest the rankings, then write the current hotspot table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    panel = pd.read_parquet(panel_path)
    panel["period"] = pd.PeriodIndex(panel["period"], freq="M")
    observed = panel[panel["split"] != "forecast"].copy()
    future_rows = panel[panel["split"] == "forecast"]

    X = pd.read_parquet(features_path)
    Y = pd.read_parquet(labels_path)
    X_fc = pd.read_parquet(forecast_path)

    locations = pd.Index(sorted(panel[location_key_var].unique()))
    panel_end = observed["period"].max()
    years = eb_window_months / 12

    print(f"Locations: {len(locations)} | history through {panel_end}")
    print(f"EB window: {eb_window_months} months ending {panel_end}\n")

    table = pd.DataFrame(index=locations)
    table.index.name = location_key_var
    backtests = []

    for outcome in count_outcomes:
        tag = LABEL[outcome]
        print(f"{'=' * 80}\n{outcome}\n{'=' * 80}")

        ########################################################################
        # ML model, chosen on validation deviance
        ########################################################################
        name, model = best_count_model(outcome, X, Y[outcome], panel)
        ml_pred = None
        if model is not None:
            print(f"  best ML count model: {name}")
            ml_pred = pd.Series(model.predict(X), index=X.index).loc[observed.index]
            fc = pd.Series(model.predict(X_fc), index=X_fc.index)
            table[f"ml_{tag}_next12"] = fc.groupby(
                future_rows.loc[fc.index, location_key_var]
            ).sum()
        else:
            print("  no trained count model found; ML columns skipped")

        ########################################################################
        # Backtest
        ########################################################################
        bt = backtest(observed, outcome, locations, ml_pred, top)
        bt.insert(0, "outcome", outcome)
        backtests.append(bt)
        print("\nBacktest: rank at cutoff, score on the next 12 months")
        print(bt.drop(columns="outcome").round(3).to_string(index=False))

        ########################################################################
        # Current EB screening
        ########################################################################
        obs = window_counts(observed, outcome, panel_end, eb_window_months).reindex(
            locations, fill_value=0
        )
        eb = empirical_bayes(obs, years)
        print(f"\n  overdispersion alpha = {eb['alpha'].iloc[0]:.3f}")

        table[f"eb_{tag}_per_year"] = eb["eb_per_year"]
        table[f"spf_{tag}_per_year"] = eb["spf_per_year"]
        table[f"excess_{tag}_per_year"] = eb["excess_per_year"]
        table[f"obs_{tag}_last12"] = window_counts(observed, outcome, panel_end, 12)
        table[f"obs_{tag}_last60"] = obs

    ############################################################################
    # Ranked table: EB injury-or-fatal crashes per year
    ############################################################################
    table = table.fillna(0)
    table = table.sort_values("eb_inj_per_year", ascending=False)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table["vru_rank"] = (
        table["eb_vru_inj_per_year"].rank(ascending=False, method="min").astype(int)
    )

    stamp = str(panel_end)
    table_path = out_dir / f"hotspots_through_{stamp}.csv"
    bt_path = out_dir / f"backtest_through_{stamp}.csv"
    table.round(3).to_csv(table_path)
    pd.concat(backtests).round(4).to_csv(bt_path, index=False)

    show = [
        "rank",
        "eb_inj_per_year",
        "excess_inj_per_year",
        "obs_inj_last12",
        "ml_inj_next12",
        "vru_rank",
        "eb_vru_inj_per_year",
        "obs_vru_inj_last12",
    ]
    show = [c for c in show if c in table.columns]
    print(
        f"\n{'=' * 80}\nTop {top} locations by EB injury-or-fatal crashes per year\n{'=' * 80}"
    )
    print(table[show].head(top).round(2).to_string())

    ############################################################################
    # Concentration (Pareto) test
    ############################################################################
    crashes = pd.read_parquet(crash_path)
    crashes = crashes.dropna(subset=["year", "month"])
    crashes["period"] = pd.PeriodIndex.from_fields(
        year=crashes["year"].astype(int), month=crashes["month"].astype(int), freq="M"
    )
    crashes = crashes[crashes["period"] <= panel_end]
    from core.constants import target_outcome, vru_var

    crashes["injury_or_fatal"] = crashes[target_outcome[0]].astype(int)
    crashes["vru_injury_or_fatal"] = (
        crashes[target_outcome[0]] * crashes[vru_var]
    ).astype(int)

    conc, fig_path = concentration(crashes, panel, out_dir, stamp)
    print(f"\n{'=' * 80}\nConcentration: is it 80/20?\n{'=' * 80}")
    print(conc.round(3).to_string(index=False))
    print(f"\nwrote {table_path}")
    print(f"wrote {fig_path}")
    print(f"wrote {bt_path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
