################################################################################
######################### Import Requisite Libraries ###########################
import os
import re
import sys
import typer
import numpy as np
import pandas as pd

# import pickling scripts
from model_tuner.pickleObjects import dumpObjects

# sys.path.insert(0, "core")

################################################################################

from core.constants import (
    var_index,
    accident_number,
    rename_map,
    date_var,
    time_var,
    injured_var,
    killed_var,
    accident_type_var,
    cat_top_n,
    vru_var,
    target_outcome,
    preproc_run_name,
    exp_artifact_name,
)

# import all user-defined functions and constants
from core.functions import (
    mlflow_dumpArtifact,
    mlflow_loadArtifact,
    safe_to_numeric,
)

app = typer.Typer()


def _norm_text(series):
    """Upper-case, trim, and collapse internal whitespace."""
    return (
        series.astype("string")
        .str.upper()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def _slug(level):
    """Column-safe suffix for a one-hot level."""
    return re.sub(r"[^0-9a-z]+", "_", str(level).lower()).strip("_") or "blank"

@app.command()
def main(
    input_data_file: str = "./data/processed/df.parquet",
    output_data_file: str = "./data/processed/df_sans_zero.parquet",
    stage: str = "training",
    data_path: str = "./data/processed",
):
    """
    Main script execution replacing sys.argv with typer.

    Args:
        input_data_file (str): Path to the input parquet file.
        output_data_file (str): Path to save the processed parquet file.
        stage (str): Processing stage (e.g., 'training' or 'inference').
    """
    ############################################################################
    # Step 1. Read the input data file
    ############################################################################

    df = pd.read_parquet(input_data_file)

    ###########################################################################
    # Step 2. Rename Columns for Consistency
    ###########################################################################
    # The BHPD export uses title-case names with spaces ("Accident Number").
    # Rename to snake_case before anything else so the index lookup below and
    # every downstream reference use one naming scheme.
    renamed = {k: v for k, v in rename_map.items() if k in df.columns}
    df.rename(columns=renamed, inplace=True)

    for old, new in renamed.items():
        print(f"Renamed '{old}' to '{new}'.")

    # Set index if not already set
    if df.index.name != var_index:
        try:
            df.set_index(var_index, inplace=True)
            print(f"\nIndex set to '{var_index}'.")
        except KeyError:
            print(
                f"Warning: '{var_index}' not found in columns - "
                "proceeding with default integer index."
            )
    else:
        print(f"Index '{var_index}' already set - skipping.")

    if stage == "training":

        ########################################################################
        # Step 2a. Drop Duplicate Accident Reports
        ########################################################################
        # 48 report numbers appear twice. Twins share the same outcome, so a
        # random split could place one copy in train and the other in test.
        # Keep the first occurrence.
        ########################################################################
        n_before = len(df)
        df = df[~df[accident_number].duplicated(keep="first")]
        print(f"\nDropped {n_before - len(df)} duplicate accident reports.")

        df_object = df.select_dtypes(["object", "string"])
        print()
        print(
            "The following columns have strings and may need to be removed from "
            "modeling and/or otherwise transformed with `categorical_transformer` "
            f"\nas handled accordingly in the `config.py` file. This list is stored "
            f"as an artifact in MLflow for future reference if necessary for "
            f"retrieval at a later time. \n \n"
            f"There are {df_object.shape[1]} string columns:\n \n"
            f"{df_object.columns.to_list()}. \n "
        )
        ########################################################################
        # Step 3. String Columns Handling
        ########################################################################
        # String columns are identified and should be removed before modeling
        # because machine learning models typically require numerical inputs.
        # Keeping string columns in the dataset may lead to errors or
        # unintended behavior unless explicitly encoded.
        #
        # To ensure consistency between training and inference,
        # we save the list of string columns and track it using MLflow.
        ########################################################################

        # Extract column names to a list
        string_cols_list = df_object.columns.to_list()

        ########################################################################
        # Step 4. Save and Log String Column List
        ########################################################################
        # Save the list of string columns for consistency across training and
        # inference and log them in MLflow for reproducibility.
        # This list of string columns is dumped (stored) only to inform of what
        # the string columns are; no further action is taken; we do not need to
        # load this list into production, since it is only there for us to
        # see what the columns are.
        ########################################################################

        # Dump the string_cols_list into a pickle file for future reference
        dumpObjects(
            string_cols_list,
            os.path.join(data_path, "string_cols_list.pkl"),
        )

        # Log the string column list as an artifact in MLflow
        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="string_cols_list",
            obj=string_cols_list,
        )

    ############################################################################
    ###################### Re-engineering Selected Features ####################
    ############################################################################

    ########################################################################
    # Step 5. Ensure Numeric Data
    ########################################################################
    # Convert any possible numeric values that may have been incorrectly
    # classified as non-numeric. This avoids accidental labeling errors.
    ########################################################################

    # Convert possible numeric columns to actual numeric types
    df = df.apply(lambda x: safe_to_numeric(x))

    ########################################################################
    # Step 6. Target Construction (training only)
    ########################################################################
    # injury = 1 if anyone was injured or killed. The injured/killed counts and
    # the free-text Accident Type disagree on ~100 rows, so a collision is
    # positive if EITHER source says injury. Every column that encodes the
    # outcome (counts and accident type) is dropped after the flags below are
    # derived, so nothing leaks into X.
    ########################################################################

    acc_type = _norm_text(df[accident_type_var]).fillna("").str.replace(" ", "")

    if stage == "training":
        type_injury = (
            acc_type.str.contains("INJURY") & ~acc_type.str.contains("NONINJURY")
        ) | acc_type.str.contains("FATAL")
        count_injury = (df[injured_var].fillna(0) > 0) | (df[killed_var].fillna(0) > 0)
        df[target_outcome[0]] = (type_injury | count_injury).astype(int)
        print(
            f"\nTarget '{target_outcome[0]}' prevalence: "
            f"{df[target_outcome[0]].mean():.3f}"
        )

    ########################################################################
    # Step 7. Feature Engineering
    ########################################################################
    # - Accident Type also carries hit-and-run, DUI, and CPD tags in ~30
    #   inconsistent spellings; keep those tags as flags, drop the severity.
    # - Date and time -> year, month, weekday, hour, cyclical hour, night,
    #   rush-hour flags.
    # - Location "A / B" is an intersection; "8670 WILSHIRE BLVD" is mid-block.
    #   The primary street (house number stripped) becomes a categorical.
    # - PCF VC -> vehicle code section (e.g., "22350 VC-I" -> "22350").
    # - Party fields -> pedestrian / bicycle / motorcycle flags and the
    #   vulnerable road user (VRU) flag used for stratification and audit.
    ########################################################################

    df["hit_run"] = acc_type.str.contains("HIT&RUN").astype(int)
    df["dui"] = acc_type.str.contains("DUI").astype(int)
    df["cpd"] = acc_type.str.contains("CPD").astype(int)

    ## Rows from late 2025 onward arrive with Day of Week / Time / Date
    ## shuffled (two different permutations). Detect by a weekday name sitting
    ## in Time, then reassign each of the three values by its pattern: a "/"
    ## marks the date, a weekday name the day, anything else the time.
    weekdays = {
        "MONDAY",
        "TUESDAY",
        "WEDNESDAY",
        "THURSDAY",
        "FRIDAY",
        "SATURDAY",
        "SUNDAY",
    }
    rotated = _norm_text(df[time_var]).isin(weekdays).fillna(False).to_numpy()
    if rotated.any():
        trio = df.loc[rotated, ["day_of_week", time_var, date_var]].astype("string")

        def _pick(row, kind):
            for v in row:
                if pd.isna(v):
                    continue
                v_up = v.strip().upper()
                is_date, is_dow = "/" in v_up, v_up in weekdays
                if (
                    (kind == "date" and is_date)
                    or (kind == "dow" and is_dow)
                    or (kind == "time" and not is_date and not is_dow)
                ):
                    return v
            return pd.NA

        for col, kind in (
            (date_var, "date"),
            (time_var, "time"),
            ("day_of_week", "dow"),
        ):
            df.loc[rotated, col] = trio.apply(_pick, axis=1, kind=kind)
        print(
            f"\nRealigned {int(rotated.sum())} rows with shuffled "
            "day_of_week / time / date fields."
        )

    ## The export mixes formats by era: dates as "8/27/2015 0:00" and
    ## "6/12/2025"; times as "16:08:00", "3:15:00 PM", and military "1458".
    dt = pd.to_datetime(df[date_var], format="mixed", errors="coerce")
    raw_time = df[time_var].astype("string").str.strip()
    military = raw_time.str.fullmatch(r"\d{3,4}").fillna(False)
    raw_time = raw_time.mask(military, raw_time.str.zfill(4))
    tm = pd.to_datetime(raw_time, format="%H:%M:%S", errors="coerce")
    for fmt in ("%I:%M:%S %p", "%H%M", "%H:%M"):
        tm = tm.fillna(pd.to_datetime(raw_time, format=fmt, errors="coerce"))
    hour = tm.dt.hour + tm.dt.minute / 60

    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["weekday"] = dt.dt.dayofweek
    df["weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["hour"] = hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["night"] = ((hour >= 21) | (hour < 6)).astype(float).where(hour.notna())
    df["rush_am"] = (
        hour.between(7, 10, inclusive="left").astype(float).where(hour.notna())
    )
    df["rush_pm"] = (
        hour.between(16, 19, inclusive="left").astype(float).where(hour.notna())
    )

    location = _norm_text(df["location"]).fillna("")
    df["at_intersection"] = location.str.contains("/").astype(int)
    df["street"] = (
        location.str.split("/")
        .str[0]
        .str.strip()
        .str.replace(r"^\d+\s+", "", regex=True)
        .replace("", pd.NA)
    )
    df["feet_from"] = df["feet_from"].fillna(0)

    df["pcf_section"] = _norm_text(df["pcf_vc"]).str.extract(r"^(\d+)")[0]

    for col in [
        "other_imp_drive",
        "type_of_accident",
        "type",
        "involved_party",
        "collision_with",
        "direction_from",
    ]:
        df[col] = _norm_text(df[col])
    df["type_of_accident"] = df["type_of_accident"].replace("HEAD ON", "HEAD-ON")

    party_text = (
        df["type"].fillna("")
        + "|"
        + df["involved_party"].fillna("")
        + "|"
        + df["collision_with"].fillna("")
    )
    df["ped_involved"] = party_text.str.contains("PED").astype(int)
    df["bike_involved"] = party_text.str.contains("BICYC").astype(int)
    df["moto_involved"] = party_text.str.contains("MOTORCYCLE|SCOOTER").astype(int)
    df[vru_var] = df[["ped_involved", "bike_involved", "moto_involved"]].max(axis=1)

    ########################################################################
    # Step 8. One-Hot Encode Categorical Fields
    ########################################################################
    # Training: keep the top-n levels per column (constants.cat_top_n); the
    # rest collapse to OTHER, missing to MISSING. The level lists are stored
    # in MLflow so inference produces the identical dummy columns.
    ########################################################################

    if stage == "training":
        cat_levels = {
            col: df[col].value_counts().head(n).index.tolist()
            for col, n in cat_top_n.items()
        }
        dumpObjects(cat_levels, os.path.join(data_path, "cat_levels.pkl"))
        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="cat_levels",
            obj=cat_levels,
        )

    if stage == "inference":
        cat_levels = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="cat_levels",
        )

    dummies = []
    for col, levels in cat_levels.items():
        binned = df[col].where(df[col].isin(levels), "OTHER")
        binned = binned.where(df[col].notna(), "MISSING")
        for level in list(levels) + ["OTHER", "MISSING"]:
            dummies.append(
                (binned == level).astype(int).rename(f"{col}_{_slug(level)}")
            )
    df = pd.concat([df] + dummies, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    ## Drop outcome-encoding columns and raw fields now represented as features.
    ## accident_number (string) is retained for linking to CCRS / SWITRS and
    ## is removed from X by feat_gen.py, which keeps numeric columns only.
    drop_cols = [
        injured_var,
        killed_var,
        accident_type_var,
        date_var,
        time_var,
        "hit_run",
        "dui",
        "cpd",
        "location",
        "from_street",
        "day_of_week",
        "pcf_vc",
    ] + list(cat_levels)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print(f"\nShape after feature engineering: {df.shape}")

    ################################################################################
    # Step 9. Zero Variance Columns
    ################################################################################

    # Select only numeric columns s/t .var() can be applied since you can only
    # call this function on numeric columns; otherwise, if you include a mix
    # (object and numeric), it will throw the following FutureWarning:
    # Dropping of nuisance columns in DataFrame reductions
    # (with 'numeric_only=None') is deprecated; in a future version this will
    # raise TypeError.  Select only valid columns before calling the reduction.

    ################################################################################

    if stage == "training":
        # Extract numeric columns to compute variance and identify
        # zero-variance features
        numeric_cols = df.select_dtypes(include=["number"]).columns
        var_indf = df[numeric_cols].var()

        # identify zero variance columns
        zero_var = var_indf[var_indf == 0]
        # capture zero-variance cols in list
        zero_varlist_list = list(zero_var.index)

        print("*" * 80)
        print(f"Zero-variance columns: {zero_varlist_list}")
        print("*" * 80)

        ########################################################################
        # Step 10. Save and Log Zero Variance Columns List
        ########################################################################
        # Save the list of string columns for consistency across training and
        # inference and log them in MLflow for reproducibility.
        ########################################################################

        dumpObjects(
            zero_varlist_list,
            os.path.join(data_path, "zero_varlist_list.pkl"),
        )

        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="zero_varlist_list",
            obj=zero_varlist_list,
        )

    if stage == "inference":

        ########################################################################
        # Load Previously Saved Zero Variance Columns List
        ########################################################################

        # load zero_var_list
        zero_varlist_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="zero_varlist_list",
        )

    ########################################################################
    # Step 11. Remove zero variance cols from main df, and assign to new var
    # df_sans_zero
    ########################################################################
    df_sans_zero = df.drop(columns=[c for c in zero_varlist_list if c in df.columns])

    print(f"Sans Zero Var Shape: {df_sans_zero.shape}")

    print()
    print(f"Original shape: {df.shape[1]} columns.")
    print(f"Reduced by {df.shape[1]-df_sans_zero.shape[1]} zero variance columns.")
    print(f"Now there are {df_sans_zero.shape[1]} columns.")
    print()

    ############################################################################
    # Step 12. Save Processed Data
    ############################################################################

    # Save out the dataframe to parquet file
    print(df_sans_zero.shape)
    df_sans_zero.reset_index().to_parquet(output_data_file)


if __name__ == "__main__":
    app()
