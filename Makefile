SHELL := /bin/sh
.DELETE_ON_ERROR:

ifneq (,$(wildcard .env))
include .env
export
endif

PYTHON ?= python3
ROOT := $(abspath .)
SCRIPTS_DIR := $(ROOT)/scripts
OUTPUT_DIR := $(ROOT)/output

.PHONY: install test data audit county-values county-value-coverage benchmark-data benchmark-estimators selection-data select-logistic select-mixture-logistic evaluate-spline-purchaser-interactions diagnose-logistic-calibration diagnose-mixture-calibration diagnose-threshold-subgroups plausibility-checks evaluate-gradient-boosting evaluate-rf-mixture estimate-mixture-shares train train-logistic validate classify figures all clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/

# --- Pipeline stages -----------------------------------------------------
# Each target wraps one scripts/*.py CLI (see MIGRATION_PLAN.md, "Execution
# order"). This is the pipeline contract; targets are filled in as each
# stage is ported, so not all scripts exist yet.

data:
	$(PYTHON) $(SCRIPTS_DIR)/prepare_training_data.py

audit:
	$(PYTHON) $(SCRIPTS_DIR)/audit_sample.py

county-values:
	$(PYTHON) $(SCRIPTS_DIR)/build_county_values.py

county-value-coverage:
	$(PYTHON) $(SCRIPTS_DIR)/audit_county_value_coverage.py

benchmark-data:
	$(PYTHON) $(SCRIPTS_DIR)/prepare_benchmark_data.py

benchmark-estimators: benchmark-data
	$(PYTHON) $(SCRIPTS_DIR)/benchmark_estimators.py

selection-data:
	$(PYTHON) $(SCRIPTS_DIR)/prepare_selection_data.py

select-logistic: selection-data
	$(PYTHON) $(SCRIPTS_DIR)/select_logistic.py

select-mixture-logistic: selection-data
	$(PYTHON) $(SCRIPTS_DIR)/select_mixture_logistic.py

evaluate-spline-purchaser-interactions: selection-data
	$(PYTHON) $(SCRIPTS_DIR)/evaluate_spline_purchaser_interactions.py

diagnose-logistic-calibration: selection-data
	$(PYTHON) $(SCRIPTS_DIR)/diagnose_logistic_calibration.py

diagnose-mixture-calibration: selection-data
	$(PYTHON) $(SCRIPTS_DIR)/diagnose_mixture_calibration.py

diagnose-threshold-subgroups: selection-data
	$(PYTHON) $(SCRIPTS_DIR)/diagnose_threshold_subgroups.py

plausibility-checks: selection-data
	$(PYTHON) $(SCRIPTS_DIR)/run_plausibility_checks.py

evaluate-gradient-boosting: selection-data
	$(PYTHON) $(SCRIPTS_DIR)/evaluate_gradient_boosting.py

evaluate-rf-mixture: selection-data
	$(PYTHON) $(SCRIPTS_DIR)/evaluate_random_forest_mixture.py

estimate-mixture-shares: selection-data
	$(PYTHON) $(SCRIPTS_DIR)/estimate_mixture_shares.py

train: data
	$(PYTHON) $(SCRIPTS_DIR)/train_classifier.py

train-logistic: data
	$(PYTHON) $(SCRIPTS_DIR)/train_logistic_classifier.py

validate: train
	$(PYTHON) $(SCRIPTS_DIR)/validate_classifier.py

classify: train
	$(PYTHON) $(SCRIPTS_DIR)/classify_all_years.py

figures: classify
	$(PYTHON) $(SCRIPTS_DIR)/build_diagnostic_cells.py
	$(PYTHON) $(SCRIPTS_DIR)/make_figures.py

all: validate figures

clean:
	rm -rf $(OUTPUT_DIR)/model/* $(OUTPUT_DIR)/figures/* $(OUTPUT_DIR)/tables/*
