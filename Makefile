# Makefile
# ------------------------------------------------------------------------------
# GLOBALS
# ------------------------------------------------------------------------------
PROJECT_NAME = bh_collisions
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python
VENV_DIR = bh_venv
CONDA_ENV_NAME = conda_bh
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROJECT_DIRECTORY := $(abspath $(MAKEFILE_DIR))

RAW_DATA       ?= $(PROJECT_DIRECTORY)/data/raw/bh_collisions.csv
PROCESSED_DATA ?= $(PROJECT_DIRECTORY)/data/processed/df.parquet
DATA_GEN_SCRIPT = $(PROJECT_DIRECTORY)/preprocessing/data_gen.py
CSV_BACKUP     ?= --no-csv-backup


############################## Training Globals ################################

# Define variables for looping
OUTCOMES = injury_or_fatal
PIPELINES = orig 
SCORING = average_precision
PRETRAINED ?= 0  # 0 if you want to train the models, 1 if calibrate pretrained

########################### Hotspot Count Globals ##############################

# Monthly crash counts per location (preprocessing/panel_gen.py)
COUNT_OUTCOMES = inj_count vru_inj_count
COUNT_MODELS = pr xgb cat
COUNT_SCORING = neg_mean_poisson_deviance

############################# Production Globals ###############################

# Model outcome variable used in production 
EXPLAN_OUTCOME = injury_or_fatal # explainer outcome variable
PROD_OUTCOME = injury_or_fatal # production outcome variable


# ------------------------------------------------------------------------------
# COMMANDS
# ------------------------------------------------------------------------------
.PHONY: init_config
init_config:
	@CURRENT_DIR=$$(sed -n 's/^PROJECT_DIRECTORY = //p' Makefile); \
	\
	read -p "Enter project name: " project_name; \
	read -p "Enter Python version (e.g., 3.10.12): " python_version; \
	read -p "Enter Python interpreter (default: python): " python_interpreter; \
	read -p "Enter virtual environment directory name: " venv_dir; \
	read -p "Enter conda environment name: " conda_env; \
	python_interpreter=$${python_interpreter:-python}; \
	\
	if [ -d "$$CURRENT_DIR" ] && [ "$$CURRENT_DIR" != "$$project_name" ]; then \
		mv "$$CURRENT_DIR" "$$project_name"; \
	fi; \
	\
	# Cross-platform sed command (works on both macOS and Linux) \
	if [ "$$(uname)" = "Darwin" ]; then \
		sed -i '' \
			-e "s/^PROJECT_NAME = .*/PROJECT_NAME = $${project_name}/" \
			-e "s/^PYTHON_VERSION = .*/PYTHON_VERSION = $${python_version}/" \
			-e "s/^PYTHON_INTERPRETER = .*/PYTHON_INTERPRETER = $${python_interpreter}/" \
			-e "s/^VENV_DIR = .*/VENV_DIR = $${venv_dir}/" \
			-e "s/^CONDA_ENV_NAME = .*/CONDA_ENV_NAME = $${conda_env}/" \
			-e "s|^PROJECT_DIRECTORY = .*|PROJECT_DIRECTORY = $${project_name}|" \
			Makefile; \
	else \
		sed -i \
			-e "s/^PROJECT_NAME = .*/PROJECT_NAME = $${project_name}/" \
			-e "s/^PYTHON_VERSION = .*/PYTHON_VERSION = $${python_version}/" \
			-e "s/^PYTHON_INTERPRETER = .*/PYTHON_INTERPRETER = $${python_interpreter}/" \
			-e "s/^VENV_DIR = .*/VENV_DIR = $${venv_dir}/" \
			-e "s/^CONDA_ENV_NAME = .*/CONDA_ENV_NAME = $${conda_env}/" \
			-e "s|^PROJECT_DIRECTORY = .*|PROJECT_DIRECTORY = $${project_name}|" \
			Makefile; \
	fi; \
	\
	# Replace project name in Python files and other text files only \
	if [ "$$(uname)" = "Darwin" ]; then \
		find "./$$project_name" -type f \( -name "*.py" -o -name "*.txt" -o -name "*.md" -o -name "*.yaml" -o -name "*.json" \) -exec sed -i '' "s/$$CURRENT_DIR/$$project_name/g" {} \;; \
	else \
		find "./$$project_name" -type f \( -name "*.py" -o -name "*.txt" -o -name "*.md" -o -name "*.yaml" -o -name "*.json" \) -exec sed -i "s/$$CURRENT_DIR/$$project_name/g" {} \;; \
	fi; \
	\
	echo "Configuration updated successfully. Folder '$$CURRENT_DIR' -> '$$project_name'."

.PHONY: check_vars
check_vars:
	@echo "Dummy configuration detected."
	@echo ""
	@echo "Please update the following variables in your Makefile before proceeding:"
	@echo " - PROJECT_NAME"
	@echo " - PYTHON_VERSION"
	@echo " - VENV_DIR"
	@echo " - CONDA_ENV_NAME"
	@echo " - OUTCOMES"
	@echo " - PIPELINES"
	@echo " - SCORING"
	@echo " - EXPLAN_OUTCOME"
	@echo " - PROD_OUTCOME"
	@echo ""
	@echo "Once you've replaced the dummy values, you can run your full pipeline commands safely."

## Set up python interpreter environment
create_conda_env:
	@echo "Run 'conda create -n $(CONDA_ENV_NAME) python=$(PYTHON_VERSION)' to create conda environment"

# Target to create a virtual environment
create_venv:
	# Create the virtual environment using the specified Python version
	$(PYTHON_INTERPRETER) -m venv $(VENV_DIR)
	@echo "Virtual environment created with $(PYTHON_INTERPRETER)$(PYTHON_VERSION)"

# Target to activate the virtual environment (Unix-based systems)
activate_venv:
	@echo "Run 'conda deactivate' to deactivate the $(CONDA_ENV_NAME) conda environment"
	@echo "Run 'source $(VENV_DIR)/bin/activate' to activate the virtual environment"

# Target to clean the virtual environment
clean_venv:
	rm -rf $(VENV_DIR)
	@echo "Virtual environment removed"

## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


setup_dir_venv: create_folders create_conda_env create_venv activate_venv 

.PHONY: mlflow_ui
mlflow_ui:
	mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5501

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

################################################################################
####################### Preprocessing (+) Dataprep Pipeline ####################
################################################################################
# clean directories
clean_dir:
	@echo "Cleaning directory..."
	rm -rf data/

################################################################################
################################ Folder Creation  ##############################
################################################################################

.PHONY: create_folders
create_folders:
# Create data subdirectories
	mkdir -p data/external data/interim data/processed data/raw data/processed/inference
	mkdir -p models/results models/eval
	mkdir -p modeling preprocessing
	mkdir -p notebooks
	mkdir -p core

	touch data/interim/.gitkeep
	touch data/processed/.gitkeep
	touch data/processed/inference/.gitkeep
	touch models/results/.gitkeep
	touch models/eval/.gitkeep

	touch modeling/__init__.py
	touch preprocessing/__init__.py
	touch core/__init__.py

# Create models subdirectories for each outcome
	@for outcome in $(OUTCOMES) $(COUNT_OUTCOMES); do \
		mkdir -p models/results/$$outcome; \
		mkdir -p models/eval/$$outcome; \
	done
	mkdir -p data/processed/hotspots

.PHONY: data_gen
data_gen: $(PROCESSED_DATA)

$(PROCESSED_DATA): $(RAW_DATA) $(DATA_GEN_SCRIPT)
	$(PYTHON_INTERPRETER) $(DATA_GEN_SCRIPT) \
		--input-data-file $(RAW_DATA) \
		--output-data-file $(PROCESSED_DATA) \
		$(CSV_BACKUP)

.PHONY: clean_data
clean_data:
	rm -f $(PROCESSED_DATA) $(basename $(PROCESSED_DATA)).csv

data_prep_preprocessing_training:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/preprocessing.py \
		--input-data-file ./data/processed/df.parquet \
		--output-data-file ./data/processed/df_sans_zero.parquet \
		--stage training \
		--data-path ./data/processed \
	2>&1 | tee data/processed/preprocessing.txt

.PHONY: feat_gen_training
feat_gen_training:
	@mkdir -p data/processed
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/feat_gen.py \
		--input-data-file ./data/processed/df_sans_zero.parquet \
		--stage training \
		--data-path ./data/processed \
	2>&1 | tee data/processed/feat_gen.txt


preproc_pipeline: data_gen data_prep_preprocessing_training feat_gen_training

################################################################################
################################# Training #####################################
######################### Imb Learn Models, Ablation ##########################
################################################################################

train_logistic_regression:
	@echo "Pretrained is set to: $(PRETRAINED)"
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			mkdir -p models/results/$$outcome; \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/train.py \
				--model-type lr \
				--pipeline-type "$$pipeline" \
				--features-path ./data/processed/X.parquet \
				--labels-path ./data/processed/y.parquet \
				--outcome "$$outcome" \
				--pretrained "$(PRETRAINED)" \
				--scoring "$(SCORING)" \
				2>&1 | tee models/results/$$outcome/lr_$$pipeline$$( [ "$(PRETRAINED)" -eq 1 ] && echo "_prefit" ).txt; \
		done; \
	done

train_random_forest:
	@echo "Pretrained is set to: $(PRETRAINED)"
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			mkdir -p models/results/$$outcome; \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/train.py \
				--model-type rf \
				--pipeline-type "$$pipeline" \
				--features-path ./data/processed/X.parquet \
				--labels-path ./data/processed/y.parquet \
				--outcome "$$outcome" \
				--pretrained "$(PRETRAINED)" \
				--scoring "$(SCORING)" \
				2>&1 | tee models/results/$$outcome/rf_$$pipeline$$( [ "$(PRETRAINED)" -eq 1 ] && echo "_prefit" ).txt; \
		done; \
	done

train_xgboost:
	@echo "Pretrained is set to: $(PRETRAINED)"
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			mkdir -p models/results/$$outcome; \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/train.py \
				--model-type xgb \
				--pipeline-type "$$pipeline" \
				--features-path ./data/processed/X.parquet \
				--labels-path ./data/processed/y.parquet \
				--outcome "$$outcome" \
				--pretrained "$(PRETRAINED)" \
				--scoring "$(SCORING)" \
				2>&1 | tee models/results/$$outcome/xgb_$$pipeline$$( [ "$(PRETRAINED)" -eq 1 ] && echo "_prefit" ).txt; \
		done; \
	done

train_catboost:
	@echo "Pretrained is set to: $(PRETRAINED)"
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			mkdir -p models/results/$$outcome; \
			"$(PYTHON_INTERPRETER)" $(PROJECT_DIRECTORY)/modeling/train.py \
				--model-type cat \
				--pipeline-type "$$pipeline" \
				--features-path ./data/processed/X.parquet \
				--labels-path ./data/processed/y.parquet \
				--outcome "$$outcome" \
				--pretrained "$(PRETRAINED)" \
				--scoring "$(SCORING)" \
				2>&1 | tee models/results/$$outcome/cat_$$pipeline$$( [ "$(PRETRAINED)" -eq 1 ] && echo "_prefit" ).txt; \
		done; \
	done



train_all_models: train_logistic_regression train_random_forest train_xgboost train_catboost

################################################################################
############################## Model Evaluation ################################
################################################################################

eval_logistic_regression:
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/evaluation.py \
			--model-type lr \
			--pipeline-type $$pipeline \
			--features-path ./data/processed/X.parquet \
			--labels-path ./data/processed/y.parquet \
			--outcome $$outcome \
			--scoring $(SCORING) 2>&1 | tee models/eval/$$outcome/lr_eval_$$pipeline.txt; \
		done; \
	done

eval_random_forest:
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/evaluation.py \
			--model-type rf \
			--pipeline-type $$pipeline \
			--features-path ./data/processed/X.parquet \
			--labels-path ./data/processed/y.parquet \
			--outcome $$outcome \
			--scoring $(SCORING) 2>&1 | tee models/eval/$$outcome/rf_eval_$$pipeline.txt; \
		done; \
	done

eval_xgboost:
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/evaluation.py \
			--model-type xgb \
			--pipeline-type $$pipeline \
			--features-path ./data/processed/X.parquet \
			--labels-path ./data/processed/y.parquet \
			--outcome $$outcome \
			--scoring $(SCORING) 2>&1 | tee models/eval/$$outcome/xgb_eval_$$pipeline.txt; \
		done; \
	done

eval_catboost:
	@for outcome in $(OUTCOMES); do \
		for pipeline in $(PIPELINES); do \
			$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/evaluation.py \
			--model-type cat \
			--pipeline-type $$pipeline \
			--features-path ./data/processed/X.parquet \
			--labels-path ./data/processed/y.parquet \
			--outcome $$outcome \
			--scoring $(SCORING) 2>&1 | tee models/eval/$$outcome/cat_eval_$$pipeline.txt; \
		done; \
	done

## Party-ablated CatBoost: retrain and evaluate without who-was-involved features
.PHONY: cat_no_party
cat_no_party:
	$(MAKE) train_catboost PIPELINES=orig_no_party
	$(MAKE) eval_catboost PIPELINES=orig_no_party

eval_all_models: eval_logistic_regression eval_random_forest eval_xgboost eval_catboost 

train_eval_pipeline: train_all_models eval_all_models cat_no_party

################################################################################
########## Preprocessing, Feature Generation, Training and Evaluation ##########
################################################################################

# This pipeline is to run consecutively the full preprocessing, training, and 
# evaluation pipeline in one command

preproc_train_eval: preproc_pipeline train_all_models eval_all_models cat_no_party


################################################################################
######################## Hotspot Count Models (Panel) ##########################
################################################################################
# Location x month panel of injury-or-fatal crash counts, Poisson count models
# on a temporal split, and Empirical Bayes hotspot screening with a backtest.

## Build the location x month panel from df_sans_zero.parquet
.PHONY: panel_gen
panel_gen:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/panel_gen.py \
		--input-data-file ./data/processed/df_sans_zero.parquet \
		--data-path ./data/processed \
	2>&1 | tee data/processed/panel_gen.txt

## Train every count model for every count outcome
.PHONY: train_count_models
train_count_models:
	@for outcome in $(COUNT_OUTCOMES); do \
		for model in $(COUNT_MODELS); do \
			mkdir -p models/results/$$outcome; \
			$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/train_counts.py \
				--model-type "$$model" \
				--outcome "$$outcome" \
				--features-path ./data/processed/X_counts.parquet \
				--labels-path ./data/processed/y_counts.parquet \
				--panel-path ./data/processed/panel.parquet \
				--scoring "$(COUNT_SCORING)" \
				--pretrained "$(PRETRAINED)" \
				2>&1 | tee models/results/$$outcome/$${model}_orig$$( [ "$(PRETRAINED)" -eq 1 ] && echo "_prefit" ).txt; \
		done; \
	done

## Evaluate every count model against naive baselines
.PHONY: eval_count_models
eval_count_models:
	@for outcome in $(COUNT_OUTCOMES); do \
		for model in $(COUNT_MODELS); do \
			mkdir -p models/eval/$$outcome; \
			$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/evaluation_counts.py \
				--model-type "$$model" \
				--outcome "$$outcome" \
				--features-path ./data/processed/X_counts.parquet \
				--labels-path ./data/processed/y_counts.parquet \
				--panel-path ./data/processed/panel.parquet \
				2>&1 | tee models/eval/$$outcome/$${model}_eval_orig.txt; \
		done; \
	done

## Backtest rankings, concentration test, and the ranked hotspot table
.PHONY: hotspots
hotspots:
	@mkdir -p data/processed/hotspots
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/hotspots.py \
		--panel-path ./data/processed/panel.parquet \
		--features-path ./data/processed/X_counts.parquet \
		--labels-path ./data/processed/y_counts.parquet \
		--forecast-path ./data/processed/X_counts_forecast.parquet \
		--crash-path ./data/processed/df_sans_zero.parquet \
		--out-dir ./data/processed/hotspots \
	2>&1 | tee data/processed/hotspots/hotspots.txt

## Panel, count models, evaluation, and hotspot table in one command
.PHONY: hotspot_pipeline
hotspot_pipeline: panel_gen train_count_models eval_count_models hotspots

################################################################################
############################ Dash App Export ###################################
################################################################################
# Flat files for the Dash app in flask_apps/bh_traffic. Run after
# hotspot_pipeline and train_eval_pipeline, then copy data/dash/* into
# flask_apps/bh_traffic/data/. Coordinates come from OpenStreetMap on the
# first run (internet needed) and are cached in data/external/.

DASH_DIR ?= $(PROJECT_DIRECTORY)/data/dash
FLASK_APP_DATA ?=

## Write the Dash app's data files to data/dash (geocodes on first run)
.PHONY: export_dash
export_dash:
	@mkdir -p $(DASH_DIR) data/external
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/export_dash.py \
		--out-dir $(DASH_DIR) \
	2>&1 | tee $(DASH_DIR)/export_dash.txt
	@if [ -n "$(FLASK_APP_DATA)" ]; then \
		mkdir -p "$(FLASK_APP_DATA)"; \
		cp $(DASH_DIR)/*.csv $(DASH_DIR)/meta.json "$(FLASK_APP_DATA)"/; \
		echo "Copied data files to $(FLASK_APP_DATA)"; \
	fi

## Re-download OSM street geometry and rebuild the coordinate cache
.PHONY: refresh_coords
refresh_coords:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/export_dash.py \
		--out-dir $(DASH_DIR) --refresh-coords

## One-page PDF brief for city decision makers (data/brief); run after export_dash
.PHONY: brief
brief:
	@mkdir -p $(PROJECT_DIRECTORY)/data/brief
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/brief_pdf.py \
		--dash-dir $(DASH_DIR) --out-dir $(PROJECT_DIRECTORY)/data/brief

## Pre-release check of every top-10 location (pins, spellings, flags); run after export_dash
.PHONY: audit_top
audit_top:
	@mkdir -p $(PROJECT_DIRECTORY)/data/audit
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/audit_top.py \
		--dash-dir $(DASH_DIR) --out-dir $(PROJECT_DIRECTORY)/data/audit

################################################################################
############################## Master Pipeline #################################
################################################################################
# Everything from raw CSV to hotspot table, in order. Recursive $(MAKE) calls
# keep the stages sequential even under `make -j`. did_stability (slow,
# optional) and mlflow_ui (a server that never exits) are left out.

## Full run: folders, preprocessing, severity classifier, hotspot screening
.PHONY: full_pipeline
full_pipeline:
	$(MAKE) create_folders
	$(MAKE) preproc_pipeline
	$(MAKE) train_eval_pipeline
	$(MAKE) hotspot_pipeline

## Full run plus the DiD stability analysis (slow)
.PHONY: full_pipeline_stability
full_pipeline_stability: full_pipeline
	$(MAKE) did_stability

## Delete all results (mlruns, models, data/processed); keeps data/raw
.PHONY: clean_results
clean_results:
	rm -rf mlruns models data/processed

## Clean rerun from raw CSV: clean_results, then full_pipeline
.PHONY: rerun_all
rerun_all: clean_results
	$(MAKE) full_pipeline

################################################################################
#################### Best Model Explainer and Explanations #####################
################################################################################

.PHONY: model_explainer
model_explainer:
	@for outcome in $(EXPLAN_OUTCOME); do \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/explainer.py \
			--input-data-file ./data/processed/X.parquet \
			--outcome $$outcome \
			--metric-name "K-Fold Average Precision" \
			--mode max; \
	done

.PHONY: model_explanations_training
model_explanations_training:
	@for outcome in $(EXPLAN_OUTCOME); do \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/explanations_training.py \
			--features-path ./data/processed/X.parquet \
			--labels-path ./data/processed/y_$$outcome.parquet \
			--outcome $$outcome \
			--metric-name "K-Fold AUC ROC" \
			--mode max \
			--top-n 5 \
			--hold-out "kfold" \
			--shap-val-flag 1 \
			--explanations-path ./data/processed/shap_predictions_$$outcome.csv; \
	done

model_explaining_training: model_explainer model_explanations_training

.PHONY: model_explanations_inference
model_explanations_inference:
	@for outcome in $(EXPLAN_OUTCOME); do \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/explanations_inference.py \
			--features-path ./data/processed/inference/X.parquet \
			--outcome $$outcome \
			--metric-name "K-Fold AUC ROC" \
			--mode max \
			--top-n 5 \
			--shap-val-flag 1 \
			--explanations-path ./data/processed/inference/shap_predictions_$$outcome.csv; \
	done

################################################################################
################################################################################

DID_SPLITS ?= 20
DID_METRIC ?= valid Average Precision

## Stability of the party-ablation VRU AUC gap across repeated splits
.PHONY: did_stability
did_stability:
	@for outcome in $(OUTCOMES); do \
		mkdir -p models/eval/$$outcome; \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/did_stability.py \
			--features-path ./data/processed/X.parquet \
			--labels-path ./data/processed/y.parquet \
			--outcome $$outcome \
			--metric-name "$(DID_METRIC)" \
			--n-splits $(DID_SPLITS) \
			--out models/eval/$$outcome/did_stability.csv \
			2>&1 | tee models/eval/$$outcome/did_stability.txt; \
	done

################################################################################
################################# Production ###################################
############################### Model Predict ##################################
################################################################################

.PHONY: data_prep_preprocessing_inference
data_prep_preprocessing_inference:
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/preprocessing.py \
	--input-data-file ./data/processed/inference/df_inference.parquet \
	--output-data-file ./data/processed/inference/df_inference_process.parquet \
	--stage inference \
	--data-path ./data/processed

.PHONY: feat_gen_inference
feat_gen_inference: 
	$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/preprocessing/feat_gen.py \
	--input-data-file ./data/processed/inference/df_inference_process.parquet \
	--stage inference \
	--data-path ./data/processed/inference

.PHONY: predict
predict:
	@for outcome in $(PROD_OUTCOME); do \
		$(PYTHON_INTERPRETER) $(PROJECT_DIRECTORY)/modeling/predict.py \
			--input-data-file data/processed/inference/X.parquet \
			--predictions-path ./data/processed/inference/predictions_$$outcome.csv \
			--outcome $$outcome \
			--metric-name "K-Fold Average Precision" \
			--mode max; \
	done


.PHONY: preproc_pipeline_inf
preproc_pipeline_inf: data_prep_preprocessing_inference feat_gen_inference predict
#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)