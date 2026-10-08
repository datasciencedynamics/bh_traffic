################################################################################
############################# Path Variables ###################################
################################################################################

import os

model_output = "model_output"  # model output path

################################################################################
############################# Mlflow Variables #################################
################################################################################

mlflow_artifacts_data = "./mlruns/preprocessing"
mlflow_models_data = "./mlruns/models"
mlflow_models_copy = "./mlruns/models_copy"

artifact_data = "artifacts/"  # path to store mlflow artifacts
profile_data = "profile_data"  # path to store pandas profiles in
data_path = "data/processed/"


# One Hot Encoded Vars to Be Omitted
cat_vars = []


################################################################################
########################## Variable/DataFrame Constants ########################
################################################################################

var_index = "object_id"  # id index (Accident Number is not unique)
accident_number = "accident_number"  # BHPD report number, kept for joins
date_var = "date"
time_var = "time"
injured_var = "injured"
killed_var = "killed"
accident_type_var = "accident_type"
main_df = "df.parquet"  # main dataframe file name

## Raw BHPD column name -> snake_case
rename_map = {
    "Location": "location",
    "Feet From": "feet_from",
    "Direction From": "direction_from",
    "From Street": "from_street",
    "Accident Number": "accident_number",
    "Day of Week": "day_of_week",
    "Injured": "injured",
    "Killed": "killed",
    "Other Imp Drive": "other_imp_drive",
    "PCF VC": "pcf_vc",
    "Accident Type": "accident_type",
    "Type of accident": "type_of_accident",
    "Type": "type",
    "Involved Party": "involved_party",
    "Collision With": "collision_with",
    "ObjectId": "object_id",
    "Time": "time",
    "Date": "date",
}

## Categorical columns one-hot encoded in preprocessing.py, with the number of
## most frequent levels kept per column (the rest collapse to "OTHER")
cat_top_n = {
    "street": 30,
    "other_imp_drive": 30,
    "pcf_section": 25,
    "type_of_accident": 12,
    "type": 10,
    "involved_party": 10,
    "collision_with": 15,
    "direction_from": 5,
}

## Party columns: who was involved. Near-deterministic for injury (a struck
## pedestrian is almost always injured), so the ablation variant removes them.
party_prefixes = ["type_", "involved_party_", "collision_with_"]
party_cols_extra = ["ped_involved", "bike_involved", "moto_involved"]

## Vulnerable road user flag: stratification and subgroup audit column
vru_var = "vru"

## DataBricks
databricks_username = "/" + "/".join(os.getcwd().split("/")[2:-1]) + "/"


################################################################################

# The below artificat name is used for preprocessing alone
exp_artifact_name = "preprocessing"
preproc_run_name = "preprocessing"
artifact_run_id = "preprocessing"
artifact_name = "preprocessing"


################################################################################
############################## SHAP Constants ##################################

shap_artifact_name = "explainer"
shap_run_name = "explainer"
shap_artifacts_data = "./mlruns/explainer"


################################################################################
############################### Target Outcome #################################

target_outcome = ["injury_or_fatal"]

leak_cols = [
    "hit_run",
    "dui",
    "cpd",
    "pcf_section_23152",
    "other_imp_drive_dui",
    "other_imp_drive_driving_under_influence_of_alcohol_42104",
]


################################################################################
########################## Hotspot Count Model Constants #######################
################################################################################

## Intersection key built in preprocessing.py (string, excluded from X)
location_key_var = "location_key"

## Street suffix variants normalized before building location_key
street_suffixes = {
    r"\bBL\b": "BLVD",
    r"\bBOULEVARD\b": "BLVD",
    r"\bAV\b": "AVE",
    r"\bAVENUE\b": "AVE",
    r"\bDRIVE\b": "DR",
    r"\bSTREET\b": "ST",
    r"\bCYN\b": "CANYON",
}

## Panel outcomes: monthly counts per location
count_outcomes = ["inj_count", "vru_inj_count"]
count_sources = {
    "inj_count": "injury_or_fatal",  # any injury or death
    "vru_inj_count": "vru_injury_or_fatal",  # injury or death involving a VRU
}

## Panel calendar. Features look back at least `forecast_horizon` months, so
## every feature for a month up to 12 months ahead is already observed.
panel_start = "2015-01"  # first month of history
first_target_month = "2017-01"  # first month with full lagged features
train_end = "2022-12"  # train: first_target_month .. train_end
valid_end = "2023-12"  # valid: train_end + 1 .. valid_end; test: after
forecast_horizon = 12  # months

## Locations kept in the panel: at least this many crashes (any severity)
## through train_end. Selection uses no test-period data.
min_location_crashes = 5

## Major corridors flagged as static location features
corridors = [
    "WILSHIRE",
    "SANTA MONICA",
    "OLYMPIC",
    "SUNSET",
    "LA CIENEGA",
    "ROBERTSON",
    "BEVERLY DR",
    "CANON",
    "RODEO",
    "BURTON",
    "DOHENY",
    "CRESCENT",
]

## Empirical Bayes network screening
eb_window_months = 60  # observed history combined with the SPF
backtest_cutoffs = ["2022-12", "2023-12", "2024-12"]  # rank, then score next 12
top_n_hotspots = 25

## Count-model MLflow artifact names
panel_artifact_name = "X_counts_columns_list"
