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

.PHONY: install test data audit county-values county-value-coverage train train-logistic validate classify figures all clean

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
