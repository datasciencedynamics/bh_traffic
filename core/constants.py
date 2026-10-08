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

target_outcome = ["injury"]
