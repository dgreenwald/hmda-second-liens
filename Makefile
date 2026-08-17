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
FINALIZE_LOGISTIC_FLAGS ?=
FINALIZE_BOOSTING_FLAGS ?=
FINALIZE_HMDA_ONLY_BOOSTING_FLAGS ?=

.PHONY: install test generate-hmda-parquet-jobs download-zillow audit county-values county-value-coverage selection-data select-logistic generate-logistic-selection-coarse aggregate-logistic-selection-coarse generate-logistic-selection-refinement aggregate-logistic-selection-refinement generate-hmda-only-logistic-coarse aggregate-hmda-only-logistic-coarse generate-hmda-only-logistic-refinement aggregate-hmda-only-logistic-refinement generate-finalize-logistic-slurm submit-finalize-logistic finalize-logistic-selection generate-finalize-hmda-only-logistic-slurm select-mixture-logistic generate-density-ratio-pilot generate-first-order-logistic-grid generate-boosting-screen aggregate-boosting-screen generate-boosting-survivors aggregate-boosting-survivors generate-boosting-refinement aggregate-boosting-refinement finalize-boosting-selection generate-finalize-boosting-slurm submit-finalize-boosting finalize-boosting generate-hmda-only-boosting-screen aggregate-hmda-only-boosting-screen generate-hmda-only-boosting-survivors aggregate-hmda-only-boosting-survivors generate-hmda-only-boosting-refinement aggregate-hmda-only-boosting-refinement finalize-hmda-only-boosting-selection generate-finalize-hmda-only-boosting-slurm submit-finalize-hmda-only-boosting finalize-hmda-only-boosting evaluate-spline-purchaser-interactions diagnose-logistic-calibration diagnose-mixture-calibration diagnose-threshold-subgroups plausibility-checks evaluate-gradient-boosting evaluate-rf-mixture estimate-mixture-shares clean

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

download-zillow:
	$(PYTHON) $(SCRIPTS_DIR)/download_zillow.py

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

generate-hmda-only-logistic-coarse:
	$(PYTHON) $(SCRIPTS_DIR)/generate_logistic_selection_slurm.py --stage coarse --feature-set hmda_only

aggregate-hmda-only-logistic-coarse:
	$(PYTHON) $(SCRIPTS_DIR)/aggregate_logistic_selection_shards.py --manifest $(OUTPUT_DIR)/slurm/hmda_only_logistic_selection/coarse/logistic_selection_jobs.json --model-output $(OUTPUT_DIR)/model/logistic_hmda_only_selected.pkl

generate-hmda-only-logistic-refinement:
	$(PYTHON) $(SCRIPTS_DIR)/generate_logistic_selection_slurm.py --stage refinement --feature-set hmda_only --coarse-summary $(OUTPUT_DIR)/tables/logistic_selection_hmda_only_coarse_summary.csv

aggregate-hmda-only-logistic-refinement:
	$(PYTHON) $(SCRIPTS_DIR)/aggregate_logistic_selection_shards.py --manifest $(OUTPUT_DIR)/slurm/hmda_only_logistic_selection/refinement/logistic_selection_jobs.json --coarse-cells $(OUTPUT_DIR)/tables/logistic_selection_hmda_only_coarse_cells.csv --model-output $(OUTPUT_DIR)/model/logistic_hmda_only_selected.pkl

generate-finalize-logistic-slurm:
	$(PYTHON) $(SCRIPTS_DIR)/generate_finalize_logistic_slurm.py

submit-finalize-logistic:
	$(PYTHON) $(SCRIPTS_DIR)/generate_finalize_logistic_slurm.py --submit

finalize-logistic-selection:
	$(PYTHON) $(SCRIPTS_DIR)/finalize_logistic_selection.py

generate-finalize-hmda-only-logistic-slurm:
	$(PYTHON) $(SCRIPTS_DIR)/generate_finalize_logistic_slurm.py --feature-set hmda_only $(FINALIZE_LOGISTIC_FLAGS)

select-mixture-logistic: selection-data
	$(PYTHON) $(SCRIPTS_DIR)/select_mixture_logistic.py

generate-density-ratio-pilot:
	$(PYTHON) $(SCRIPTS_DIR)/generate_density_ratio_slurm.py

generate-first-order-logistic-grid:
	$(PYTHON) $(SCRIPTS_DIR)/generate_first_order_logistic_slurm.py

generate-boosting-screen:
	$(PYTHON) $(SCRIPTS_DIR)/generate_boosting_selection_slurm.py --stage boosting_screen

aggregate-boosting-screen:
	$(PYTHON) $(SCRIPTS_DIR)/aggregate_density_ratio_shards.py --manifest $(OUTPUT_DIR)/slurm/boosting/screen/density_ratio_jobs.json --output-dir $(OUTPUT_DIR)/tables/boosting_screen

generate-boosting-survivors:
	$(PYTHON) $(SCRIPTS_DIR)/generate_boosting_selection_slurm.py --stage boosting_survivors --prior-summary $(OUTPUT_DIR)/tables/boosting_screen/density_ratio_summary.csv

aggregate-boosting-survivors:
	$(PYTHON) $(SCRIPTS_DIR)/aggregate_density_ratio_shards.py --manifest $(OUTPUT_DIR)/slurm/boosting/survivors/density_ratio_jobs.json --output-dir $(OUTPUT_DIR)/tables/boosting_survivors

generate-boosting-refinement:
	$(PYTHON) $(SCRIPTS_DIR)/generate_boosting_selection_slurm.py --stage boosting_refinement --prior-summary $(OUTPUT_DIR)/tables/boosting_survivors/density_ratio_summary.csv

aggregate-boosting-refinement:
	$(PYTHON) $(SCRIPTS_DIR)/aggregate_density_ratio_shards.py --manifest $(OUTPUT_DIR)/slurm/boosting/refinement/density_ratio_jobs.json --output-dir $(OUTPUT_DIR)/tables/boosting_refinement

finalize-boosting-selection:
	$(PYTHON) $(SCRIPTS_DIR)/finalize_boosting_selection.py --survivor-dir $(OUTPUT_DIR)/tables/boosting_survivors --refinement-dir $(OUTPUT_DIR)/tables/boosting_refinement

generate-finalize-boosting-slurm:
	$(PYTHON) $(SCRIPTS_DIR)/generate_finalize_boosting_slurm.py $(FINALIZE_BOOSTING_FLAGS)

submit-finalize-boosting:
	$(PYTHON) $(SCRIPTS_DIR)/generate_finalize_boosting_slurm.py --submit

finalize-boosting:
	$(PYTHON) $(SCRIPTS_DIR)/finalize_boosting.py

generate-hmda-only-boosting-screen:
	$(PYTHON) $(SCRIPTS_DIR)/generate_hmda_only_boosting_slurm.py --stage hmda_only_boosting_screen

aggregate-hmda-only-boosting-screen:
	$(PYTHON) $(SCRIPTS_DIR)/aggregate_density_ratio_shards.py --manifest $(OUTPUT_DIR)/slurm/hmda_only_boosting/screen/density_ratio_jobs.json --output-dir $(OUTPUT_DIR)/tables/hmda_only_boosting_screen

generate-hmda-only-boosting-survivors:
	$(PYTHON) $(SCRIPTS_DIR)/generate_hmda_only_boosting_slurm.py --stage hmda_only_boosting_survivors --prior-summary $(OUTPUT_DIR)/tables/hmda_only_boosting_screen/density_ratio_summary.csv

aggregate-hmda-only-boosting-survivors:
	$(PYTHON) $(SCRIPTS_DIR)/aggregate_density_ratio_shards.py --manifest $(OUTPUT_DIR)/slurm/hmda_only_boosting/survivors/density_ratio_jobs.json --output-dir $(OUTPUT_DIR)/tables/hmda_only_boosting_survivors

generate-hmda-only-boosting-refinement:
	$(PYTHON) $(SCRIPTS_DIR)/generate_hmda_only_boosting_slurm.py --stage hmda_only_boosting_refinement --prior-summary $(OUTPUT_DIR)/tables/hmda_only_boosting_survivors/density_ratio_summary.csv

aggregate-hmda-only-boosting-refinement:
	$(PYTHON) $(SCRIPTS_DIR)/aggregate_density_ratio_shards.py --manifest $(OUTPUT_DIR)/slurm/hmda_only_boosting/refinement/density_ratio_jobs.json --output-dir $(OUTPUT_DIR)/tables/hmda_only_boosting_refinement

finalize-hmda-only-boosting-selection:
	$(PYTHON) $(SCRIPTS_DIR)/finalize_hmda_only_boosting_selection.py --survivor-dir $(OUTPUT_DIR)/tables/hmda_only_boosting_survivors --refinement-dir $(OUTPUT_DIR)/tables/hmda_only_boosting_refinement

generate-finalize-hmda-only-boosting-slurm:
	$(PYTHON) $(SCRIPTS_DIR)/generate_finalize_hmda_only_boosting_slurm.py $(FINALIZE_HMDA_ONLY_BOOSTING_FLAGS)

submit-finalize-hmda-only-boosting:
	$(PYTHON) $(SCRIPTS_DIR)/generate_finalize_hmda_only_boosting_slurm.py --submit

finalize-hmda-only-boosting:
	$(PYTHON) $(SCRIPTS_DIR)/finalize_hmda_only_boosting.py

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
