# Threshold and Subgroup Diagnostic Protocol

## Frozen estimators

Step 7 compares the raw logistic probabilities with the known-source-prior
mixture probabilities. Both use the frozen `spline_lti__purchaser_type`
specification and ridge `C=0.1`. Every source-window model is persisted under
`output/model/mixture_folds/`; Step 7 does not retune features, regularization,
class weights, intercepts, or thresholds.

## Operating threshold

The canonical hard classification uses the previously declared threshold 0.5,
with probabilities exactly equal to 0.5 classified as second liens. Equal false-
positive and false-negative costs remain the declared operating loss. No
year-specific or subgroup-specific threshold is selected.

Report accuracy, second-lien precision, recall and F1, observed prevalence,
mean probability, hard-classification share, Brier score, and average precision
for every reverse and forward validation cell. Average reverse cells within
horizon and then average horizons equally.

## Precision-recall curves

Construct approximate precision-recall curves on a fixed grid of 243 thresholds:
zero, one, 0.5, and 241 probabilities corresponding to an equally spaced logit
grid from -12 to 12. Pool confusion counts within reverse horizon and plot the
raw and mixture estimators together. For the forward check, report one panel per
validation year. These curves are diagnostics and are not used to select a new
operating threshold.

## Subgroups

At threshold 0.5, report metrics and observation counts by:

- all four loan-type codes;
- all ten purchaser-type codes;
- Census region; and
- target-year deciles of `log_lti` and `log_county_value_to_loan`.

Target-year deciles are label-free and guarantee useful support while retaining
an ordinal interpretation across cells. Retain the original categorical codes
rather than combining economically different groups merely to increase sample
size. Flag cells with small counts, especially purchaser type 4, and do not draw
substantive conclusions from them.

Persist only aggregated cell, curve-count, and subgroup outputs. Do not save
loan-level diagnostic probabilities.
