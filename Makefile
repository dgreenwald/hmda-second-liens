SHELL := /bin/sh
.DELETE_ON_ERROR:

ifneq (,$(wildcard .env))
include .env
export
endif

PYTHON ?= python3
ROOT := $(abspath .)
SCRIPTS_DIR := $(ROOT)/scripts
OUTPUT_DIR := $(abspath $(if $(HMDA_SECONDS_OUTPUT_DIR),$(HMDA_SECONDS_OUTPUT_DIR),output))

.PHONY: install test generate-hmda-parquet-jobs audit county-values county-value-coverage selection-data select-logistic generate-logistic-selection-coarse aggregate-logistic-selection-coarse generate-logistic-selection-refinement aggregate-logistic-selection-refinement finalize-logistic-selection select-mixture-logistic generate-density-ratio-pilot generate-first-order-logistic-grid evaluate-spline-purchaser-interactions diagnose-logistic-calibration diagnose-mixture-calibration diagnose-threshold-subgroups plausibility-checks evaluate-gradient-boosting evaluate-rf-mixture estimate-mixture-shares clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/

# --- Pipeline stages -----------------------------------------------------
# Each target wraps one scripts/*.py CLI (see documentation/MIGRATION_PLAN.md, "Execution
# order"). This is the pipeline contract; targets are filled in as each
# stage is ported, so not all scripts exist yet.

generate-hmda-parquet-jobs:
	$(PYTHON) $(SCRIPTS_DIR)/generate_hmda_parquet_slurm.py

audit:
	$(PYTHON) $(SCRIPTS_DIR)/audit_sample.py

county-values:
	$(PYTHON) $(SCRIPTS_DIR)/build_county_values.py

county-value-coverage:
	$(PYTHON) $(SCRIPTS_DIR)/audit_county_value_coverage.py

selection-data:
	$(PYTHON) $(SCRIPTS_DIR)/prepare_selection_data.py

select-logistic: selection-data
	$(PYTHON) $(SCRIPTS_DIR)/select_logistic.py

generate-logistic-selection-coarse:
	$(PYTHON) $(SCRIPTS_DIR)/generate_logistic_selection_slurm.py --stage coarse

aggregate-logistic-selection-coarse:
	$(PYTHON) $(SCRIPTS_DIR)/aggregate_logistic_selection_shards.py --manifest $(OUTPUT_DIR)/slurm/logistic_selection/coarse/logistic_selection_jobs.json

generate-logistic-selection-refinement:
	$(PYTHON) $(SCRIPTS_DIR)/generate_logistic_selection_slurm.py --stage refinement --coarse-summary $(OUTPUT_DIR)/tables/logistic_selection_core_coarse_summary.csv

aggregate-logistic-selection-refinement:
	$(PYTHON) $(SCRIPTS_DIR)/aggregate_logistic_selection_shards.py --manifest $(OUTPUT_DIR)/slurm/logistic_selection/refinement/logistic_selection_jobs.json --coarse-cells $(OUTPUT_DIR)/tables/logistic_selection_core_coarse_cells.csv

finalize-logistic-selection:
	$(PYTHON) $(SCRIPTS_DIR)/finalize_logistic_selection.py

select-mixture-logistic: selection-data
	$(PYTHON) $(SCRIPTS_DIR)/select_mixture_logistic.py

generate-density-ratio-pilot:
	$(PYTHON) $(SCRIPTS_DIR)/generate_density_ratio_slurm.py

generate-first-order-logistic-grid:
	$(PYTHON) $(SCRIPTS_DIR)/generate_first_order_logistic_slurm.py

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

clean:
	rm -rf $(OUTPUT_DIR)/model/* $(OUTPUT_DIR)/figures/* $(OUTPUT_DIR)/tables/*
