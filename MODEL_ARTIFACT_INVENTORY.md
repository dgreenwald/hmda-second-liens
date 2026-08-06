# Current Model Artifact Inventory

This inventory freezes the model-artifact boundary before the density-ratio refactor. It
describes artifacts written by the current code, not the intended final schema. Existing
files listed here must remain readable during the migration or receive an explicit,
tested conversion path.

## Density-ratio artifacts in active use

| Artifact | Python payload | Producer and loader | Current deterministic naming | Refactor requirement |
|---|---|---|---|---|
| Selected raw logistic | `model_selection.SelectedLogisticModel` | `model_selection.save_selected_model` / `load_selected_model` | `output/model/logistic_selected.pkl` | Preserve as the incumbent-selection input and raw-logistic diagnostic model. Add an adapter rather than modifying the pickled class in place. |
| Complete logistic ratio fold | `mixture.DensityRatioModels` | `mixture.save_density_ratio_models` / `load_density_ratio_models` | `output/model/mixture_folds/all_ratio_variants__<spec>__c_<C>__train_<first>_<last>.pkl` | Preserve pooled, year-fixed-effect, and known-source-prior variants and their diagnostics. |
| Known-source-prior logistic fold | `mixture.KnownSourcePriorModel` | `mixture.save_known_source_prior_model` / `load_known_source_prior_model` | `output/model/mixture_folds/known_source_prior__<spec>__c_<C>__train_<first>_<last>.pkl` | This is the natural logistic implementation of the future fitted-model protocol. Preserve `log_ratio(frame)`, feature transformation, source years, specification, penalty, and fit diagnostics. |
| Mixture-logistic selection candidate | `mixture.KnownSourcePriorModel` | Same as above | `output/model/mixture_logistic_selection/known_source_prior__<spec>__c_<C>__train_<first>_<last>.pkl` | Keep every candidate/fold artifact. Do not resume or rewrite the partial search until the refactor reaches the common runner. |
| Selected mixture-native logistic | `mixture.KnownSourcePriorModel` | Same as above | `output/model/logistic_mixture_selected.pkl` | Reserved for the eventual mixture-native winner; do not overwrite the incumbent during refactoring. |
| Gradient-boosting fold | `gradient_boosting.BoostingDensityRatioModel` | `gradient_boosting.save_boosting_model` / `load_boosting_model` | `output/model/boosting_folds/<parameter_id>__train_<first>_<last>.pkl` | Preserve classifier, parameters, primitive feature order, source years, and `log_ratio(frame)`. |
| Gradient-boosting final challenger | `gradient_boosting.BoostingDensityRatioModel` | Same as above | `output/model/boosting_challenger.pkl` | Preserve as the saved 2004--2007 challenger refit. |
| Random Forest fold | `random_forest_mixture.RandomForestDensityRatioModel` | `random_forest_mixture.save_forest_model` / `load_forest_model` | `output/model/rf_mixture_folds/rf_50_depth_10__train_<first>_<last>.pkl` | Preserve classifier, explicit one-hot feature schema, source years, and `log_ratio(frame)`. |
| Random Forest final challenger | `random_forest_mixture.RandomForestDensityRatioModel` | Same as above | `output/model/rf_mixture_challenger.pkl` | Preserve as the saved 2004--2007 challenger refit. |

All current loaders intentionally accept trusted local pickle files and enforce the expected
top-level Python type. Pickle embeds module and class paths, so moving these classes outright
would break old files. During migration, leave compatibility definitions at their current
module paths or provide tested compatibility shims.

## Earlier classifier artifacts retained by the pipeline

These artifacts predate the density-ratio workflow but remain configured and may still be
used by older Makefile targets or comparisons.

| Configured path | Current purpose |
|---|---|
| `output/model/rf_fit.pkl` | Original primary Random Forest produced by `train.py` and consumed by the legacy classification/validation path. |
| `output/model/logistic_fit.pkl` | Formal non-mixture logistic estimator. |
| `output/model/benchmark_rf_full.pkl` | Full-sample Random Forest benchmark. |
| `output/model/benchmark_logistic_full.pkl` | Full-sample logistic benchmark. |

The refactor should not silently reinterpret these as density-ratio models. They remain
outside the new fitted-density-ratio protocol unless wrapped by an explicit adapter with a
well-defined source-prior correction.

## Current gaps to address after behavioral parity

- Artifact files do not have a common metadata sidecar or schema version.
- Save helpers write directly to their final paths rather than using atomic replacement.
- Fit diagnostics are not represented consistently across families.
- Model IDs are implicit in filenames and family-specific parameter objects.
- Software versions, training counts, weighting conventions, and feature-schema versions are
  not recorded uniformly.
- Checkpoint CSVs can be shared mutable outputs; they are unsuitable for parallel cluster
  writers.

These are inputs to refactoring steps 5 and 7. Step 1 records them but does not change artifact
formats or model behavior.
