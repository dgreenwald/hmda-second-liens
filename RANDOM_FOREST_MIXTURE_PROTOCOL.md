# Random-Forest Mixture Robustness Protocol

This exercise revisits the project's original Random Forest after introducing the annual
mixture-share adjustment. It is a fixed-specification robustness comparison, not a reopened
hyperparameter search.

## Estimator

Use the established full-sample Random Forest settings: 50 trees, maximum depth 10,
`random_state=17`, and all available cores. Supply the two untransformed continuous variables
and full one-hot encodings of the canonical purchaser- and loan-type levels. Do not include
splines, explicit interactions, year, geography, HPI growth, edit status, or small-loan status.

Within each four-year source window, weight first and second liens to equal total mass in every
source year. Do not apply an additional `class_weight`. Interpret the resulting balanced-prior
forest odds as the density ratio, clipping probabilities to `[1e-12, 1 - 1e-12]` before taking
log odds.

For every target year, estimate the second-lien mixture share from the target covariates and
construct adjusted probabilities using the fitted target share. Evaluate all 45 frozen reverse
cells and the separate 2008--2016 forward regime. Use the same Brier, log-loss, calibration,
and annual-share diagnostics as the logistic and boosting mixture estimators.

## Decision rule

Compare the fixed forest with both selected logistic and gradient boosting. Do not tune forest
depth, leaf size, feature subsampling, or tree count unless this fixed specification is
competitive enough that limited tuning could plausibly affect the substantive conclusion.

Save every source-window fit under `output/model/rf_mixture_folds/` and the final 2004--2007 fit
as `output/model/rf_mixture_challenger.pkl`. Persist only aggregate validation diagnostics.
