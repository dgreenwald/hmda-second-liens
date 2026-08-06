"""Paths, feature lists, and hyperparameters for the HMDA second-lien classifier.

Centralizes what was scattered as module-level globals across
``classify_seconds.py`` in the original research script, and adds the
train/validate/apply year partition used for out-of-time validation (see
MIGRATION_PLAN.md, "Methodological improvements", item 1).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

# Load unambiguously from the repo root rather than searching upward from
# the current working directory, so behavior doesn't depend on where a
# script is invoked from or on import order elsewhere pulling in a dotenv
# search as a side effect (as py_tools.datasets loaders do).
load_dotenv(REPO_ROOT / ".env", override=False)

DATA_DIR = Path(os.environ.get("HMDA_SECONDS_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PUBLIC_DIR = DATA_DIR / "public"

OUTPUT_DIR = Path(os.environ.get("HMDA_SECONDS_OUTPUT_DIR", REPO_ROOT / "output"))
MODEL_DIR = OUTPUT_DIR / "model"
MIXTURE_FOLD_MODEL_DIR = MODEL_DIR / "mixture_folds"
BOOSTING_FOLD_MODEL_DIR = MODEL_DIR / "boosting_folds"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# Large, regenerable, non-public intermediate files (the concatenated
# training extract, per-year classified outputs) live outside the repo
# entirely rather than under output/, following the same convention the
# original project used for /data/hmda/. Defaults to a repo-relative
# fallback so the pipeline still runs for someone without that local
# convention; override via HMDA_SECONDS_EXTERNAL_DIR (see .env).
EXTERNAL_DATA_DIR = Path(os.environ.get("HMDA_SECONDS_EXTERNAL_DIR", DATA_DIR))
INTERMEDIATE_DIR = EXTERNAL_DATA_DIR / "intermediate"

# Directory holding one cleaned HMDA extract per year, named hmda<year>.parquet,
# in the same format produced by py_tools.datasets.hmda.load_hmda. See
# data/README.md for how these are built from raw LAR files; override locally
# to point at an already-materialized cache (e.g. an existing
# /data/hmda/save/ directory) via the HMDA_SECONDS_YEARLY_DIR env var.
HMDA_YEARLY_DIR = Path(
    os.environ.get("HMDA_SECONDS_YEARLY_DIR", RAW_DIR / "hmda" / "save")
)

# Vendored public input: FHFA's HPI_AT_BDL_county.xlsx lives directly under
# this directory; py_tools.datasets.fhfa.load('county', data_dir=...) reads
# it from there and caches a derived parquet alongside it.
FHFA_DATA_DIR = PUBLIC_DIR

# Zillow Research county ZHVI used to put the FHFA appreciation indexes on a
# dollar scale. The source is the all-homes, middle-tier, smoothed and
# seasonally adjusted monthly series downloaded by py_tools.datasets.zillow.
# Pin the source vintage so a later Zillow history revision cannot silently
# change the fitted model.
ZILLOW_VINTAGE = os.environ.get("HMDA_SECONDS_ZILLOW_VINTAGE", "202608")
ZILLOW_ANCHOR_YEAR = 2017
ZILLOW_MIN_OVERLAP_YEARS = 1

MODEL_FILE = MODEL_DIR / "rf_fit.pkl"
LOGISTIC_MODEL_FILE = MODEL_DIR / "logistic_fit.pkl"
BENCHMARK_RF_MODEL_FILE = MODEL_DIR / "benchmark_rf_full.pkl"
BENCHMARK_LOGISTIC_MODEL_FILE = MODEL_DIR / "benchmark_logistic_full.pkl"

# Lien status is reliably reported in HMDA starting in 2004. Train on the
# same 2004-2007 window as the original script; unlike the original script,
# separately hold out every other labeled year (2008-2016) for out-of-time
# validation rather than only spot-checking a single year.
TRAIN_YEARS = range(2004, 2008)
VALIDATE_YEARS = range(2008, 2017)
APPLY_YEARS = range(1990, 2017)

assert set(TRAIN_YEARS).isdisjoint(VALIDATE_YEARS)
assert set(TRAIN_YEARS) | set(VALIDATE_YEARS) <= set(APPLY_YEARS)

TRAIN_PARQUET = (
    INTERMEDIATE_DIR / f"hmda_train_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet"
)
BENCHMARK_TRAIN_PARQUET = INTERMEDIATE_DIR / (
    f"hmda_train_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}_county_value.parquet"
)
SELECTION_DATA_DIR = INTERMEDIATE_DIR / "logistic_selection"
SELECTED_LOGISTIC_MODEL_FILE = MODEL_DIR / "logistic_selected.pkl"
SELECTED_BOOSTING_MODEL_FILE = MODEL_DIR / "boosting_challenger.pkl"
CLASSIFY_PARQUET = (
    INTERMEDIATE_DIR / f"hmda_classified_{min(APPLY_YEARS)}_{max(APPLY_YEARS)}.parquet"
)
# Small, aggregated (bin counts only, no loan-level data) -- unlike the
# other pipeline artifacts, this is a plausible candidate to vendor
# publicly alongside the letter for full figure reproducibility.
HISTOGRAM_CELLS_PARQUET = TABLE_DIR / "hmda_lti_histogram_cells.parquet"

LABEL_VAR = "lien_status"
CONTINUOUS_VARS = ["log_lti", "log_county_value_to_loan"]
# has_edit_status and loan_below_10k were dropped from the feature list after
# a real-data feature ablation (MIGRATION_PLAN.md, "Results (real data,
# items 1-6 all run)") showed removing either changes held-out error by
# <0.06pp -- both are within noise of a no-op, and loan_below_10k in
# particular is redundant with log_lti (which the RF can already split on
# directly) despite looking highly predictive in a marginal cross-tab.
# clean.clean_frame still computes both columns since they're cheap
# diagnostics; they're just no longer fed to the model.
CATEGORY_VARS = ["purchaser_type", "loan_type"]

# Canonical levels for each categorical feature. py_tools.get_labels_features
# builds dummy columns via pd.get_dummies(df[var]), which only creates
# columns for values actually present in whatever slice it's given -- unsafe
# once classify.py/validate.py encode one year (or one held-out split) at a
# time, since a category value absent from a particular slice would silently
# produce a differently-shaped or misaligned feature matrix instead of an
# error. clean.py pins every CATEGORY_VARS column to a pandas Categorical
# with these fixed categories so pd.get_dummies always emits the same
# columns in the same order, with an all-zero column for any value absent
# from a given slice, regardless of what's actually present in it.
CATEGORY_LEVELS = {
    "purchaser_type": list(range(10)),  # matches py_tools.datasets.hmda's own category range
    "loan_type": [1, 2, 3, 4],  # matches clean.clean_frame's loan_type.between(1, 4) filter
}

# Not used as model features, but retained (unlike the original script, which
# dropped them) so a released predicted-lien-status crosswalk can be joined
# back to a same-vintage raw LAR file by anyone who has one.
ID_VARS = ["resp_id", "seq_num"]

RF_KWARGS = {
    "n_estimators": 50,
    "max_depth": 10,
    "random_state": 17,
    # Parallelizes fitting/prediction across all available cores. Pure speed:
    # each tree's fit is seeded independently from random_state, so results
    # are unaffected by n_jobs (confirmed the hyperparameter grid comparing
    # n_estimators=50 vs 200 was run single-threaded before this change --
    # this only changes wall-clock time, not the fitted model).
    "n_jobs": -1,
}

# Starting specification for the formal logistic estimator. Regularization
# strength and the decision threshold remain objects for the next model-
# selection pass; keeping them explicit prevents sklearn default changes from
# silently changing the results.
LOGISTIC_KWARGS = {
    "C": 1.0,
    "l1_ratio": 0.0,
    "max_iter": 1000,
    "solver": "lbfgs",
}

LOGISTIC_SELECTION_COARSE_C = (1e-4, 1e-2, 1.0, 100.0)
LOGISTIC_SELECTION_SOLVER = "newton-cholesky"
LOGISTIC_SELECTION_MAX_ITER = 200
LOGISTIC_SELECTION_TOL = 1e-8
LOGISTIC_SELECTION_JOBS = int(os.environ.get("HMDA_SECONDS_SELECTION_JOBS", "1"))
LOGISTIC_SELECTION_THREADS_PER_JOB = max(
    1, (os.cpu_count() or 1) // LOGISTIC_SELECTION_JOBS
)

# Step 9 gradient-boosting challenger. The structure screen varies only the
# two principal tree-complexity controls; the winning structure is subsequently
# refined over iteration count and leaf L2 regularization.
BOOSTING_STRUCTURE_LEAF_NODES = (7, 15, 31)
BOOSTING_STRUCTURE_LEARNING_RATES = (0.05, 0.1)
BOOSTING_BASE_MAX_ITER = 200
BOOSTING_BASE_L2 = 1.0
BOOSTING_MIN_SAMPLES_LEAF = 1_000
BOOSTING_REFINEMENT_MAX_ITER = (100, 400)
BOOSTING_REFINEMENT_L2 = (0.0, 10.0)
BOOSTING_SCREEN_SURVIVORS = 2
BOOSTING_RANDOM_STATE = 17

TRAIN_SIZE = 0.25
TEST_SIZE = 0.75

PREDICTED_LABEL_VAR = f"{LABEL_VAR}_predicted"
PROB_SECOND_LIEN_VAR = "prob_second_lien"
FIRST_LIEN_CLASS = 1
SECOND_LIEN_CLASS = 2
