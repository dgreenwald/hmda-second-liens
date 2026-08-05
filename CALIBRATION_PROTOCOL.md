# Logistic Calibration Protocol

## Estimator

Calibration diagnostics use the fixed selected core estimator:
`spline_lti__purchaser_type` with ridge penalty `C=0.1`. Step 6 does not retune the model,
alter its features, adjust its intercept, or fit a post-hoc calibrator.

## Primary backward evaluation

For each of the 45 cells in the frozen reverse-temporal design, refit the selected
specification on its four later training years and predict the relevant earlier validation
year. Report:

- raw Brier score;
- log loss;
- observed and mean-predicted second-lien shares;
- mean predicted probability minus observed prevalence;
- calibration intercept and slope from a logistic regression of the outcome on the logit of
  the predicted probability; and
- the number of validation loans.

Average cell metrics within backward horizon and then average the nine horizons equally,
matching the model-selection objective. Preserve the complete cell table so heterogeneity by
training window and validation year remains visible.

## Reliability diagrams

Within each validation cell, divide predicted probabilities into ten approximate equal-count
bins. Retain each bin's observation count, probability range, mean predicted probability, and
observed second-lien share. Pool bin counts within backward horizon for the primary reliability
diagrams. Quantile bins are used because second liens are uncommon and fixed-width bins would
leave much of the probability range sparsely populated.

## Forward-regime robustness

Separately apply the final model fitted on 2004--2007 to each year from 2008 through 2016.
Report the same metrics and year-specific reliability diagrams. These results measure behavior
across the post-crisis regime but are not treated as evidence that supersedes the backward
validation exercise.

## Decision rule

Step 6 is diagnostic. Do not apply Platt scaling, isotonic regression, year-specific prevalence
matching, an intercept shift, or any other recalibration automatically. A recalibration proposal
would require a separate temporally defensible protocol and validation years not used to fit
the calibrator.
