# Migration Plan: HMDA Second-Lien Classifier -> Standalone Letter

## Context

`~/research/frm/hmda_seconds` contains an exploratory script (`classify_seconds.py`, plus
some downstream scratch scripts) that trains a Random Forest to identify likely second liens
in HMDA loan records for years 1990-2016, since HMDA only reliably records `lien_status`
starting in 2004. This classifier is currently just a data-cleaning appendix (`sec:hmda-data`
in `~/research/frm/tex/draft/frm_revision_new_appendix.tex`) inside a much larger house-price
paper. It's a good candidate to spin off on its own: the underlying problem (imputing lien
status pre-2004) is a genuine, reusable methods contribution for any researcher using
pre-2004 HMDA data, the inputs are entirely public (unlike the licensed Fannie Mae data used
elsewhere in that project), and the existing code is a rough, hardcoded, path-dependent
research script rather than something citable or reusable by others.

Scope decisions (confirmed with user):
- **Classifier methodology only.** No downstream implied-PTI / bunching application
  (`plot_high_pti.py`, `plot_implied_pti.py`, `plot_hmda_distribution.py`, `vti_growth.py`,
  `distributional_regression.r` are all out of scope — they belong to the main paper, not
  this spin-off).
- **Fully public/reproducible**, but reusing `dgreenwald-py-tools` (pip: `dgreenwald-py-tools`,
  already published by the user) rather than reimplementing its utilities. The repo depends on
  it as a normal package dependency (`dgreenwald-py-tools[ml,datasets]`) instead of vendoring
  copies of `RandomForestWrapper`, `tic`/`toc`, or the dataset loaders.
- **No R.** Python-only.

This considerably narrows what needs to be ported: with the PTI application out of scope,
`misc_data.py`'s IRS/Census/FRED (`q_star`) machinery is not needed at all — the classifier
only ever needs the FHFA county HPI (for the `log_ltv` feature) and the HMDA LAR files
themselves. Only `classify_seconds.py`'s core logic (`clean_data`, RF train/evaluate/classify,
LTI-by-lien-status diagnostic histograms) needs to be ported.

Target repo: `/home/dan/research/hmda-second-liens` (this repo), remote
`git@github.com:dgreenwald/hmda-second-liens.git`.

## What already exists that we should reuse, not rebuild

- `py_tools.econometrics.machine_learning` (`RandomForestWrapper`, `complete_estimation`,
  `get_labels_features`, `plot_importance_random_forest`) — already a clean, documented,
  pip-installable wrapper around `sklearn.RandomForestClassifier`. Use as-is via the
  `dgreenwald-py-tools[ml]` extra.
- `py_tools.datasets.hmda` (`load_hmda`/`store`) — already parses the raw fixed-width HMDA LAR
  files for all years 1990-2016 into an HDF5 cache, including the exact column layouts that
  changed in 2004. Use as-is via `dgreenwald-py-tools[datasets]`.
- `py_tools.datasets.fhfa.load('county')` — loads the FHFA county HPI (`HPI_AT_BDL_county.xlsx`),
  which is small, public, and can be vendored directly in this repo's `data/` directory so the
  pipeline doesn't require a manual download for that piece.
- `py_tools.datasets.config.base_dir()` — resolves a data root from `PY_TOOLS_DATA_DIR` (env
  var or a discovered `.env` file), which lets this repo point all `py_tools` loaders at a
  repo-relative `data/raw/` directory via a checked-in `.env.example`, with no `py_tools` code
  changes needed.
- `~/research/frm/replication_package_proposal/code/diagnostics/{hmda_lien_diagnostics.py,
  build_hmda_lti_cells.py}` — an already-existing, already-polished reproduction of exactly the
  LTI-by-lien-status diagnostic figures this letter needs (argparse CLI, validated binned-cell
  intermediate format, deterministic output filenames). Use this as the direct template for the
  figure-generation script rather than porting the ad hoc plotting from `classify_seconds.py`
  and `plot_hmda_distribution.py`.
- `~/research/frm/tex/draft/frm_revision_new_appendix.tex` (`sec:hmda-data`) — the existing
  prose description of the method (features, hyperparameters, train/test split, 2006
  validation) is the direct source for this letter's methods section; reuse its language and
  figures as the starting point rather than redrafting from scratch.

## Target repo layout

```
hmda-second-liens/
  README.md                  # what this is, how to reproduce, how to use the released crosswalk
  LICENSE                    # exists
  pyproject.toml             # deps: dgreenwald-py-tools[ml,datasets], + repo-local package
  .env.example                # PY_TOOLS_DATA_DIR=./data/raw
  Makefile                    # data -> train -> validate -> classify -> figures -> paper
  src/hmda_seconds/
    __init__.py
    config.py                 # paths, feature lists, rf_kwargs (was scattered globals in classify_seconds.py)
    clean.py                  # load_and_clean/clean_data, ported from classify_seconds.py + the FHFA merge from misc_data.py
    train.py                  # fit RF on 2004-2007 via py_tools complete_estimation, save model
    validate.py               # NEW: out-of-time validation, baselines, confusion matrix, OOB score (see below)
    classify.py                # apply fitted RF to all years 1990-2016, write predicted-lien-status parquet
    diagnostics.py             # build binned LTI-by-year/lien-status cells (mirrors build_hmda_lti_cells.py)
  scripts/                     # thin argparse CLIs, one per Makefile target (mirrors replication_package_proposal style)
    prepare_training_data.py
    train_classifier.py
    validate_classifier.py
    classify_all_years.py
    build_diagnostic_cells.py
    make_figures.py            # wraps hmda_lien_diagnostics.py-style plotting
  data/
    raw/                       # .gitignored; user-populated HMDA LAR downloads
    public/                    # vendored small public inputs: FHFA county HPI parquet
    README.md                  # exact source URLs + instructions for the raw HMDA LAR files (too large to vendor)
  output/
    model/                     # fitted RF pickle (regenerable; consider whether to commit or regenerate-only)
    figures/
    tables/
  paper/
    letter.tex                 # short paper skeleton (see outline below)
    refs.bib
  tests/
    test_clean.py               # synthetic-data tests for clean_data's filters/feature construction
    test_validate.py            # tests for the out-of-time validation and baseline-comparison logic
```

This is a lighter-weight version of the `replication_package_proposal` conventions (argparse
scripts, `Makefile`-driven stages, `tests/`) without that package's heavier machinery
(`data-manifest.csv`, asset-manifest auditing, licensed-data gating) — not needed here since
every input is public and the scope is one self-contained result, not a whole paper's asset
inventory.

## Porting map (old -> new)

- `classify_seconds.py: clean_data` -> `src/hmda_seconds/clean.py`. Keep the sample
  restrictions (`action_taken==1`, `loan_purp==1`, `occupancy==1`, valid state/county, positive
  income/loan amount, `loan_type` in 1-4) and feature construction (`log_lti`, `log_ltv` via
  FHFA county HPI merge, `has_edit_status`, `loan_below_10k`) as-is; replace the hardcoded
  `/data/hmda/`, `df_fhfa` global-scope construction with an explicit function that takes the
  FHFA frame as an argument.
- `classify_seconds.py: run_list` stages `create_data`/`estimate`/`classify`/`combine`/`compare`
  -> becomes the `Makefile` targets / `scripts/*.py` CLIs, each doing one stage and writing one
  well-defined output, instead of one script with commented-in/out globals.
- `classify_seconds.py: rf_kwargs`, `continuous_vars`, `category_vars`, `label_var` ->
  `src/hmda_seconds/config.py` as named constants, imported everywhere instead of re-declared.
- `misc_data.py` -> **not ported**. Only the FHFA county-HPI load+balanced-panel logic
  (currently inlined at the top of `classify_seconds.py`) is needed and moves into `clean.py`;
  the FRED/IRS/Census pieces were only for the excluded PTI application.
- `plot_high_pti.py`, `plot_implied_pti.py`, `plot_hmda_distribution.py`, `vti_growth.py`,
  `distributional_regression.r` -> **not ported** (out of scope).
- `code/diagnostics/hmda_lien_diagnostics.py` + `build_hmda_lti_cells.py` (from
  `replication_package_proposal`) -> adapted into `src/hmda_seconds/diagnostics.py` +
  `scripts/make_figures.py`, generalized to run over the full 1990-2016 range rather than the
  fixed subset of years currently hardcoded there.

## Resolved: the lien_status binary-vs-4-class question

`classify_seconds.py`'s `clean_data` only requires `lien_status` to be non-null when training
— it doesn't explicitly restrict to `lien_status in {1, 2}`, even though HMDA's code list also
defines `3` (not secured by a lien) and `4` (not applicable). Checked directly against the real
cached data (`/data/hmda/save/hmda{2004..2007}.parquet`): within the actual sample restriction
(`action_taken==1`, `loan_purp==1`, `occupancy==1`, `loan_type` in 1-4), `lien_status` takes
only values `1` and `2` in every training year — codes `3`/`4` never appear. The purchase-loan
restriction alone rules them out in practice. No extra filter needed; `clean.py` ports the
original condition as-is.

## Local environment decisions (this machine)

- Real inputs are already available locally: `/data/hmda/save/hmda<year>.parquet` (1990-2018,
  matching the `py_tools.datasets.hmda.load_hmda` schema exactly) and
  `~/Dropbox/data/fhfa/HPI_AT_BDL_county.xlsx`. `config.HMDA_YEARLY_DIR` and
  `config.FHFA_DATA_DIR` are wired to these via `.env` (gitignored) so the pipeline runs for
  real on this machine while staying repo-relative/overridable by default for anyone else.
- `data/hmda_seconds_new/hmda_first_lien_purch_clean_<year>.parquet`-style files sitting
  alongside the yearly cache are leftovers from an unrelated older script
  (`hmda_first_purch.py`) — not part of this pipeline, ignore them.
- Large regenerable intermediate artifacts (concatenated training extract, per-year classified
  outputs) are written to `/data/hmda_seconds/intermediate/` (external to the repo, via
  `HMDA_SECONDS_EXTERNAL_DIR`), following the same convention the original project used for
  `/data/hmda/`, rather than living under the repo's `output/`.
- `resp_id`/`seq_num` are retained through `clean.py` (the original script dropped them) so a
  released predicted-lien-status crosswalk can be joined back to a same-vintage raw LAR file.
- **Bug found and fixed in step 4**: `config.py` originally never loaded `.env` itself. Step 2
  appeared to work only because `clean.py` happens to `import py_tools.datasets.fhfa` before
  `from . import config`, and that import has a side effect (computing `fhfa`'s module-level
  `default_dir` default argument calls `py_tools.datasets.config.base_dir()`, which triggers
  py_tools's *own* dotenv search) that incidentally populated `os.environ` first. Any script
  importing `hmda_seconds.config` directly (e.g. `scripts/train_classifier.py`) hit this in the
  wrong order and silently fell back to repo-relative defaults instead of `/data/hmda_seconds`.
  Fixed by having `config.py` call `load_dotenv(REPO_ROOT / ".env", override=False)` itself,
  anchored to the repo root rather than a cwd-relative search, and adding `python-dotenv` as a
  direct dependency instead of relying on it arriving transitively via `py_tools[datasets]`.
- **Latent bug found and fixed in step 5**: `py_tools.get_labels_features` builds dummy columns
  via `pd.get_dummies(df[var])`, which only creates columns for values *actually present* in
  whatever slice it's given. `classify.py` encodes one year at a time, so a category value
  absent from a given year would silently produce a differently-shaped/misaligned feature
  matrix instead of an error. Checked empirically against the real data: every year 1990-2016
  has all 4 `loan_type` values and all 10 `purchaser_type` values present, so this was not
  actually corrupting results, but it's not guaranteed by the code and `validate.py`'s baselines
  hit the identical pattern. Fixed at the source: `clean.clean_frame` now pins every
  `CATEGORY_VARS` column to a `pandas.Categorical` with fixed, explicit categories
  (`config.CATEGORY_LEVELS`), so `pd.get_dummies` always emits the same columns in the same
  order regardless of what's present in a given slice.
  - This fix landed *after* the step-2/step-4 real-data runs (`hmda_train_2004_2007.parquet`,
    `rf_fit.pkl`, `hmda_classified_1990_2016.parquet` were all built pre-fix). Confirmed the fix
    is a no-op for this specific dataset (same category coverage and sort order either way), so
    those cached artifacts were not regenerated; a future clean `make all` rebuild will use the
    fixed code path throughout. Flagged here rather than silently left inconsistent.
- **Bug found and fixed while running validate.py for real**: `validate.oob_score` originally
  fit on the *full* 19.35M-row training frame instead of the same `train_size=0.25` split
  `train.fit()` actually uses, making it a much larger (and slower -- roughly 4x the data,
  observed running 20+ minutes before being killed) fit than the deployed model, for a number
  that didn't even describe that model. Fixed to split with the same `train_size`/`test_size`/
  `random_state` as the headline model before fitting with `oob_score=True`, so it's both fast
  and a real cross-check against the deployed model's held-out error rate.

## Methodological improvements worth making (this is most of the letter's actual contribution)

The original code is a quick validity check (train on a random 25/75 split of pooled
2004-2007 data, eyeball the 2006 predicted-vs-actual LTI histograms). A few of these are cheap
and substantially strengthen the letter:

1. **Out-of-time validation across the full labeled range (highest priority).** HMDA actually
   has real `lien_status` for *every* year 2004-2016, not just 2004-2007 — the original code
   never uses 2008-2016 for validation, only for blind prediction. Train on 2004-2007 (as
   before) but evaluate out-of-time accuracy/precision/recall/confusion matrix separately for
   each held-out year 2008-2016. This directly tests generalization across changing mortgage-
   market composition (post-crash origination looks very different from 2004-2007), which is
   exactly the setting the classifier is used in for 1990-2003, and turns a single anecdotal
   "2006 looks right" check into a real validation section with a table/figure spanning 13
   years.
2. **Continuity check at the 2004 boundary.** Plot the predicted second-lien share by year for
   1990-2016 against the *actual* second-lien share for 2004-2016; confirm there's no
   discontinuity where the method switches from "actual" to "predicted" at 2004. Cheap, and
   directly supports the appendix's current qualitative claim that predictions are "highly
   consistent" with the labeled years.
3. **Baseline comparisons.** Report accuracy for one or two much simpler baselines (e.g.
   logistic regression on the same features, or a naive threshold rule on loan amount) next to
   the Random Forest, so the letter can justify the modeling choice rather than asserting it.
4. **Richer evaluation metrics.** Swap the current "error rate by class" for a confusion matrix,
   precision/recall/F1 by class, and ROC-AUC using `predict_proba`; report out-of-bag score
   (`oob_score=True`) as a cheap additional check alongside the held-out split.
5. **Feature importance / ablation.** `plot_importance_random_forest` already exists and isn't
   currently called with `plot=True` output kept — surface it as a table/figure, and consider a
   simple leave-one-feature-out ablation to show which features (LTI, LTV proxy, purchaser
   type, loan type, edit-status flag, sub-$10k flag) actually drive classification.
6. **Light hyperparameter justification.** `n_estimators=50, max_depth=10` look hand-picked;
   either show results are robust to reasonable alternatives (e.g. depth 8/10/15,
   n_estimators 50/200) or run a small grid search and report the chosen point, rather than
   asserting the values.
7. **The release itself is a contribution.** Package the fitted model and/or the full
   1990-2016 predicted-lien-status crosswalk (year, HMDA record identifiers available,
   predicted lien status, predicted probability) as the letter's citable public good for other
   HMDA researchers — this is arguably the main reason this is worth spinning off at all.

Items 1-2 are the ones I'd treat as required for the letter to feel like a real methods
contribution rather than a repackaged appendix; 3-6 are good value-per-effort additions;
item 7 is the deliverable that makes it citable.

### Acted on: feature list and RF settings (post-review)

Per the ablation/hyperparameter results below, `config.CATEGORY_VARS` now drops
`has_edit_status` and `loan_below_10k` (down to `["purchaser_type", "loan_type"]`;
`clean.clean_frame` still computes both columns, just doesn't feed them to the model), and
`config.RF_KWARGS` adds `n_jobs=-1` (pure speed, no effect on the fitted model since each
tree's fit is independently seeded from `random_state`). Hyperparameters
(`n_estimators=50, max_depth=10`) were left unchanged -- already validated as the best point in
the tested grid. Re-ran training for real: 1m34s wall time (was 7m57s single-threaded), 2.60%
overall held-out error (was 2.59%), second-lien error improved slightly to 4.57% (was 4.80%) --
confirms the dropped features weren't pulling their weight. `class_weight='balanced'` was
considered and explicitly rejected; see the base-rate-shift finding below.

### Results (real data, items 1-6 all run)

- **Feature ablation** (500k-row subsample, baseline err_rate 2.61%): dropping `log_lti` or
  `log_ltv` alone roughly doubles the error rate (4.76%, 3.65%); `purchaser_type` matters
  modestly (2.98% without it); `loan_type`, `has_edit_status`, `loan_below_10k` are each within
  ~0.06pp of the full-feature baseline. **Recommend dropping `has_edit_status` and
  `loan_below_10k`** — confirmed by direct held-out-error evidence, not just importance scores.
  (`loan_below_10k` is marginally *very* predictive of lien status on its own -- 84.5% of
  sub-$10k loans are second liens vs. 17.9% otherwise -- but redundant given `log_lti` is
  already a continuous feature the RF can split on directly.)
- **Hyperparameter grid** (same subsample): the original `n_estimators=50, max_depth=10` is
  already the best point tested (2.60% err_rate); `max_depth=15` is slightly *worse* (2.66%),
  `n_estimators=200` is statistically indistinguishable (2.60%). **No change recommended** —
  the original ad hoc choice holds up under scrutiny.
- **OOB score**: 0.9740 (2.60% OOB error), matching the 2.59% held-out-split error from step 4
  almost exactly — a good independent cross-check that the model isn't overfitting.
- **Out-of-time validation, 2008-2016 (the headline result)**: accuracy 97.9%-98.9% and ROC-AUC
  0.988-0.995 throughout — the model's ranking ability holds up well outside the training
  window. But **second-lien precision degrades sharply out-of-time (53%-66%, vs. recall staying
  strong at 85%-93%)**, while precision is presumably much better in-era (2004-2007, implicit in
  the 4.8% same-era second-lien error rate from step 4). The `continuity_check` explains why:
  predicted second-lien share is consistently *higher* than actual, by a small margin in
  2004-2007 (~1pp) but by a much larger *relative* margin post-2007 as the true second-lien
  share collapses from the piggyback-loan boom peak (~23% in 2006) to ~2% by the mid-2010s. The
  same absolute over-prediction becomes a much bigger precision problem once the base rate the
  model is up against is that low. This is exactly the kind of finding that justifies doing
  out-of-time validation instead of a single same-era spot check, and should be reported
  honestly in the letter rather than smoothed over.
  - **Revises the class_weight recommendation floated during step 5's live review**:
    `class_weight='balanced'` improves second-lien recall at the cost of precision on
    2004-2007-only held-out data, which looked attractive given the application cares more
    about not missing second liens. But the model *already* over-predicts second liens
    out-of-time — making it more trigger-happy toward the second-lien class via balanced
    weighting would compound that existing bias, not fix it. **Do not recommend
    `class_weight='balanced'`** without also addressing the base-rate shift (e.g., recalibrating
    the decision threshold per era, or reporting the precision caveat directly).
- **Baselines, out-of-time**: a plain logistic regression on the identical features tracks the
  RF closely (accuracy and ROC-AUC within ~0.001 of each other every year) — most of the RF's
  apparent power out-of-time comes from the features (`log_lti`, `log_ltv`), not from the
  Random Forest's flexibility per se. The naive single-threshold-on-`log_lti` baseline is
  clearly worse (accuracy ~93.5%-95.4%, second-lien precision only 22%-36%), showing the other
  features do add real value over the simplest possible rule.

## Paper skeleton (`paper/letter.tex`)

1. **Introduction** — HMDA is the near-universal source for U.S. mortgage originations, but
   `lien_status` (needed to exclude second liens/HELOCs/piggybacks, which have very different
   LTI/LTV characteristics — Figure showing raw bimodal LTI distribution) is unreliable before
   2004. Contribution: a validated ML imputation + a released crosswalk.
2. **Data** — HMDA sample restrictions (owner-occupied, single-family, purchase, loan types
   1-4), FHFA county HPI for the LTV proxy.
3. **Method** — features, Random Forest, training window, hyperparameters (post
   items 3/6 above).
4. **Validation** — out-of-time accuracy 2008-2016, confusion matrix, continuity check,
   baseline comparison, feature importance (items 1-5 above).
5. **Released crosswalk / how to use it** — describe the public artifact.
6. **Conclusion.**

The prose isn't drafted yet — this is the structural outline to build the results toward; each
section maps to one or two of the figures/tables produced by `scripts/make_figures.py` and
`scripts/validate_classifier.py`.

## Execution order

1. **Done.** Scaffold repo (`pyproject.toml`, `.env.example`, `Makefile`, package skeleton,
   `data/README.md` with exact HMDA LAR source URLs).
2. **Done.** Port `clean.py` (data prep + FHFA merge), get `prepare_training_data.py` producing
   the 2004-2007 training parquet from real local HMDA data. Folded step 3 in here too: checked
   the lien_status-in-{1,2} question against real data directly (see "Resolved" section above)
   rather than treating it as a separate pass.
3. ~~Resolve the lien_status-in-{1,2} question against real data~~ — done as part of step 2.
4. Port `train.py`/`classify.py` using `py_tools.econometrics.machine_learning` as-is. Adds
   `prob_second_lien` (via the underlying sklearn classifier's `predict_proba`, not exposed by
   `RandomForestWrapper` itself) alongside the hard predicted label, since both step 5's richer
   metrics and step 7's release need it. `classify_all_years.py` writes one combined
   `hmda_classified_1990_2016.parquet` rather than per-year files plus a separate combine stage.
5. Build `validate.py` (new out-of-time validation, baselines, confusion matrix, continuity
   check) — this is the new, most important code in the repo.
6. Adapt `diagnostics.py`/`make_figures.py` from the `replication_package_proposal` templates,
   generalized to 1990-2016.
7. Write `tests/` against synthetic data (a handful of fabricated rows covering the filter
   edge cases, not real HMDA data) so the pipeline logic is covered without needing raw data in
   CI.
8. `README.md` + package/export the crosswalk artifact.
9. Draft `paper/letter.tex` from the outline once the validation results exist.

## Verification

- `pytest tests/` passes on synthetic fixtures without requiring real HMDA data.
- End-to-end on real data (once the user points `.env` at a populated `data/raw/`):
  `make data && make train && make validate && make classify && make figures` reproduces the
  2004-2007 fit, the 2008-2016 out-of-time validation table, the full 1990-2016 predicted
  crosswalk, and the diagnostic figures (raw LTI histograms + actual-vs-predicted 2006
  comparison), matching the qualitative results already described in
  `frm_revision_new_appendix.tex`'s `sec:hmda-data`.
