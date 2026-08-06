# Mixture-Native Logistic Reselection Protocol

This protocol is fixed before inspecting mixture-native logistic selection results. It asks
whether the earlier logistic specification remains optimal when the actual release estimator
uses equal source priors and a separately estimated target-year mixture share.

## Candidates and objective

Use a first-order, coordinate-wise neighborhood around the incumbent
`spline_lti__purchaser_type` specification with ridge `C=0.1`. Change exactly one declared
component at a time:

- continuous form: `linear__purchaser_type` or `spline_both__purchaser_type`, each at `C=0.1`;
- interaction structure: `spline_lti__none`, `spline_lti__loan_type`,
  `spline_lti__both`, or `spline_lti__purchaser_type_spline_lti`, each at `C=0.1`; and
- regularization: the incumbent specification at `C=0.01` or `C=1.0`.

Include the incumbent itself as the benchmark. Continue to exclude geography, year, HPI
growth, edit status, small-loan status, lender, MSA, and tract features.

For every source window, give first and second liens equal total weight within each source year.
Use the complete balanced-prior logistic log odds as the density-ratio score. Estimate a
separate mixture share from every target year's covariates and score the resulting adjusted
probabilities. The sole selection statistic is raw adjusted Brier, averaged within backward
horizon and then equally across horizons 1--9.

## First-order stopping rule

Evaluate all eight one-coordinate alternatives and the incumbent over the complete 45-cell
reverse design. Any strictly lower equal-horizon adjusted Brier counts as a local improvement.
If no alternative improves on the incumbent, stop the search and retain the incumbent. The
absence of a coordinate-wise improvement is a computational stopping rule, not proof that no
joint change could improve the objective.

If one or more alternatives improve, refine only the improving coordinate or coordinates. A
combined-change search is permitted only when at least two distinct coordinates improve on
their own, and its candidates must be declared before their results are inspected. Forward
results remain diagnostics and never enter selection.

The cluster manifest groups configurations that share a transformed design matrix. It contains
63 `(specification, source window)` jobs and 81 fitted models: nine incumbent jobs carrying
`C in {0.01, 0.1, 1}` and 54 single-configuration feature-neighbor jobs.

## Artifacts and diagnostics

Save every fitted specification/penalty/source-window model under the cluster output root's
`models/logistic/` tree. Save any final 2004--2007 refit as
`output/model/logistic_mixture_selected.pkl`. Do not overwrite the existing selected logistic
artifact until the final estimator decision.

Compare the winner with the existing mixture logistic, gradient boosting, and Random Forest in
all reverse cells. Apply the final refit to 2008--2016 and report the same probability and
calibration diagnostics. Persist aggregated outputs only.
