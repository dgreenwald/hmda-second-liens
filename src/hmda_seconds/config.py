"""Paths, feature lists, and hyperparameters for the HMDA second-lien classifier.

Centralizes what was scattered as module-level globals across
``classify_seconds.py`` in the original research script, and adds the
train/validate/apply year partition used for out-of-time validation (see
MIGRATION_PLAN.md, "Methodological improvements", item 1).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("HMDA_SECONDS_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PUBLIC_DIR = DATA_DIR / "public"

OUTPUT_DIR = Path(os.environ.get("HMDA_SECONDS_OUTPUT_DIR", REPO_ROOT / "output"))
MODEL_DIR = OUTPUT_DIR / "model"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# Directory holding one cleaned HMDA extract per year, named hmda<year>.parquet,
# in the same format produced by py_tools.datasets.hmda.load_hmda. See
# data/README.md for how these are built from raw LAR files; override locally
# to point at an already-materialized cache (e.g. an existing
# /data/hmda/save/ directory) via the HMDA_SECONDS_YEARLY_DIR env var.
HMDA_YEARLY_DIR = Path(
    os.environ.get("HMDA_SECONDS_YEARLY_DIR", RAW_DIR / "hmda" / "save")
)

FHFA_COUNTY_HPI_FILE = PUBLIC_DIR / "fhfa_county_hpi.parquet"

MODEL_FILE = MODEL_DIR / "rf_fit.pkl"

# Lien status is reliably reported in HMDA starting in 2004. Train on the
# same 2004-2007 window as the original script; unlike the original script,
# separately hold out every other labeled year (2008-2016) for out-of-time
# validation rather than only spot-checking a single year.
TRAIN_YEARS = range(2004, 2008)
VALIDATE_YEARS = range(2008, 2017)
APPLY_YEARS = range(1990, 2017)

assert set(TRAIN_YEARS).isdisjoint(VALIDATE_YEARS)
assert set(TRAIN_YEARS) | set(VALIDATE_YEARS) <= set(APPLY_YEARS)

LABEL_VAR = "lien_status"
CONTINUOUS_VARS = ["log_lti", "log_ltv"]
CATEGORY_VARS = ["purchaser_type", "loan_type", "has_edit_status", "loan_below_10k"]

RF_KWARGS = {
    "n_estimators": 50,
    "max_depth": 10,
    "random_state": 17,
}

TRAIN_SIZE = 0.25
TEST_SIZE = 0.75
