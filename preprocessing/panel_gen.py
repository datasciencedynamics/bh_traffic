################################################################################
######################### Step 1: Import Requisite Libraries ###################
################################################################################

import os
import numpy as np
import pandas as pd
import typer

from core.functions import mlflow_dumpArtifact

from core.constants import (
    var_index,
    location_key_var,
    target_outcome,
    vru_var,
    count_outcomes,
    count_sources,
    panel_start,
    first_target_month,
    train_end,
    valid_end,
    forecast_horizon,
    min_location_crashes,
    corridors,
    exp_artifact_name,
    preproc_run_name,
    panel_artifact_name,
)

################################################################################
####################### Location x Month Panel for Hotspots #####################
################################################################################
# One row per location (intersection or mid-block segment) per month, with
# the number of injury-or-fatal crashes and of those involving a vulnerable
# road user (VRU). Every feature for month t uses crashes up to t - 12 only,
# so a forecast for any of the next 12 months is built from data already in
# hand, and the test period is scored at the same horizon it is used at.
################################################################################

################################################################################
################ Step 2: Define Typer Application ##############################
################################################################################

app = typer.Typer()


def _rolling_sum(series, window, lag):
    """Sum over months [t - lag - window + 1, t - lag]; NaN until full."""
    return series.shift(lag).rolling(window, min_periods=window).sum()


def _expanding_mean(series, lag):
    """Mean monthly count from panel start through t - lag."""
    return series.shift(lag).expanding(min_periods=1).mean()


################################################################################
################ Step 3: Define Main Function ##################################
################################################################################


@app.command()
def main(
    input_data_file: str = "./data/processed/df_sans_zero.parquet",
    data_path: str = "./data/processed",
):
    """
    Builds the location x month panel, lagged features, temporal split labels,
    and forecast rows for the next `forecast_horizon` months.

    Args:
        input_data_file (str): Path to df_sans_zero.parquet from preprocessing.
        data_path (str): Directory for panel outputs.
    """

    ############################################################################
    ################ Step 4: Load Crash-Level Data #############################
    ############################################################################

    df = pd.read_parquet(input_data_file)

    if df.index.name != var_index:
        try:
            df.set_index(var_index, inplace=True)
            print(f"Index set to '{var_index}'.")
        except KeyError:
            print(f"Warning: '{var_index}' not found - using default index.")

    n_raw = len(df)
    df = df.dropna(subset=[location_key_var, "year", "month"])
    print(f"Crashes with location and date: {len(df):,} of {n_raw:,}")

    df["period"] = pd.PeriodIndex.from_fields(
        year=df["year"].astype(int), month=df["month"].astype(int), freq="M"
    )

    ## Crash-level indicators summed into monthly counts
    outcome = target_outcome[0]
    df["injury_or_fatal"] = df[outcome].astype(int)
    df["vru_injury_or_fatal"] = (df[outcome] * df[vru_var]).astype(int)
    df["all_count"] = 1

    ############################################################################
    ################ Step 5: Panel Calendar ####################################
    ############################################################################
    # The panel ends at the latest month on record, unless that month looks
    # partial (fewer than half the crashes of the trailing 12-month median),
    # in which case it ends one month earlier.
    ############################################################################

    monthly = df.groupby("period").size()
    panel_end = monthly.index.max()
    trailing = monthly.loc[: panel_end - 1].tail(12).median()
    if monthly[panel_end] < 0.5 * trailing:
        print(
            f"Last month {panel_end} has {monthly[panel_end]} crashes vs a "
            f"trailing median of {trailing:.0f}; treating it as partial."
        )
        panel_end = panel_end - 1
    history = pd.period_range(panel_start, panel_end, freq="M")
    future = pd.period_range(panel_end + 1, panel_end + forecast_horizon, freq="M")
    months = history.append(future)

    print(f"History: {history[0]} to {history[-1]} ({len(history)} months)")
    print(f"Forecast: {future[0]} to {future[-1]}")

    ############################################################################
    ################ Step 6: Select Locations ##################################
    ############################################################################
    # Keep locations with at least `min_location_crashes` crashes through
    # train_end. Selecting on later data would let the test period choose
    # which locations get scored.
    ############################################################################

    pre = df[df["period"] <= pd.Period(train_end, "M")]
    loc_counts = pre[location_key_var].value_counts()
    locations = sorted(loc_counts[loc_counts >= min_location_crashes].index)

    covered = df[location_key_var].isin(locations)
    print(
        f"Locations: {len(locations)} with >= {min_location_crashes} crashes "
        f"through {train_end}; they hold {covered.mean():.1%} of crashes and "
        f"{df.loc[covered, 'injury_or_fatal'].sum() / df['injury_or_fatal'].sum():.1%} "
        "of injury-or-fatal crashes."
    )

    ############################################################################
    ################ Step 7: Monthly Counts ####################################
    ############################################################################

    count_cols = ["injury_or_fatal", "vru_injury_or_fatal", "all_count"]
    counts = (
        df[covered]
        .groupby([location_key_var, "period"])[count_cols]
        .sum()
        .reindex(
            pd.MultiIndex.from_product(
                [locations, months], names=[location_key_var, "period"]
            ),
            fill_value=0,
        )
        .reset_index()
        .sort_values([location_key_var, "period"])
    )

    ## Future months carry no observed counts
    is_future = counts["period"] > panel_end
    counts.loc[is_future, count_cols] = np.nan

    ############################################################################
    ################ Step 8: Lagged Features (all >= forecast_horizon) #########
    ############################################################################

    H = forecast_horizon
    grp = counts.groupby(location_key_var, sort=False)
    feats = pd.DataFrame(index=counts.index)

    short = {
        "injury_or_fatal": "inj",
        "vru_injury_or_fatal": "vru_inj",
        "all_count": "all",
    }
    for col, name in short.items():
        feats[f"{name}_same_month_ly"] = grp[col].shift(H)
        feats[f"{name}_roll12"] = grp[col].transform(_rolling_sum, window=12, lag=H)
        feats[f"{name}_roll24"] = grp[col].transform(_rolling_sum, window=24, lag=H)
        feats[f"{name}_hist_rate"] = grp[col].transform(_expanding_mean, lag=H)

    ## Citywide trend, from all crashes (not only panel locations)
    city = (
        df.groupby("period")[["injury_or_fatal", "all_count"]]
        .sum()
        .reindex(months, fill_value=0)
    )
    city.loc[city.index > panel_end] = np.nan
    city_roll = city.shift(H).rolling(12, min_periods=12).sum()
    feats["city_inj_roll12"] = counts["period"].map(city_roll["injury_or_fatal"])
    feats["city_all_roll12"] = counts["period"].map(city_roll["all_count"])

    ## Calendar
    month_num = counts["period"].dt.month
    feats["month_sin"] = np.sin(2 * np.pi * month_num / 12)
    feats["month_cos"] = np.cos(2 * np.pi * month_num / 12)
    feats["covid"] = (
        (counts["period"] >= pd.Period("2020-04", "M"))
        & (counts["period"] <= pd.Period("2021-06", "M"))
    ).astype(int)

    ## Static location descriptors
    key = counts[location_key_var]
    feats["midblock"] = key.str.contains("MIDBLOCK", regex=False).astype(int)
    for corridor in corridors:
        name = corridor.lower().replace(" ", "_")
        feats[f"on_{name}"] = key.str.contains(corridor, regex=False).astype(int)

    ############################################################################
    ################ Step 9: Temporal Split Labels #############################
    ############################################################################

    period = counts["period"]
    split = pd.Series("drop", index=counts.index)
    split[
        (period >= pd.Period(first_target_month, "M"))
        & (period <= pd.Period(train_end, "M"))
    ] = "train"
    split[
        (period > pd.Period(train_end, "M")) & (period <= pd.Period(valid_end, "M"))
    ] = "valid"
    split[(period > pd.Period(valid_end, "M")) & (period <= panel_end)] = "test"
    split[is_future] = "forecast"

    panel = pd.concat(
        [counts[[location_key_var, "period"]], split.rename("split"), feats], axis=1
    )
    for out in count_outcomes:
        panel[out] = counts[count_sources[out]]

    panel = panel[panel["split"] != "drop"].reset_index(drop=True)
    panel.index.name = "panel_id"
    panel["period"] = panel["period"].astype(str)

    print("\nRows by split:")
    print(panel["split"].value_counts().reindex(["train", "valid", "test", "forecast"]))

    for out in count_outcomes:
        observed = panel[panel["split"] != "forecast"]
        print(
            f"{out}: mean {observed[out].mean():.3f} per location-month, "
            f"{(observed[out] > 0).mean():.1%} of location-months nonzero"
        )

    ############################################################################
    ################ Step 10: Save Panel, X, y, and Column List ################
    ############################################################################

    X_counts_columns_list = feats.columns.to_list()

    mlflow_dumpArtifact(
        experiment_name=exp_artifact_name,
        run_name=preproc_run_name,
        obj_name=panel_artifact_name,
        obj=X_counts_columns_list,
    )
    mlflow_dumpArtifact(
        experiment_name=exp_artifact_name,
        run_name=preproc_run_name,
        obj_name="panel_locations",
        obj=locations,
    )

    observed = panel["split"] != "forecast"
    panel.to_parquet(os.path.join(data_path, "panel.parquet"))
    panel.loc[observed, X_counts_columns_list].to_parquet(
        os.path.join(data_path, "X_counts.parquet")
    )
    panel.loc[observed, count_outcomes].to_parquet(
        os.path.join(data_path, "y_counts.parquet")
    )
    panel.loc[~observed, X_counts_columns_list].to_parquet(
        os.path.join(data_path, "X_counts_forecast.parquet")
    )

    print(f"\nFeatures: {len(X_counts_columns_list)}")
    print(f"Saved panel ({panel.shape[0]:,} rows) to {data_path}")


################################################################################

if __name__ == "__main__":
    app()
