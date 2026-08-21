# Step 9: Gradient-Boosting Challenger Protocol

This protocol is fixed before inspecting any gradient-boosting validation result. The
challenger asks whether flexible nonlinearities materially improve the already selected
known-source-prior mixture estimator.

## Estimator and features

Use scikit-learn's `HistGradientBoostingClassifier` with binary log loss. Supply only the four
primitive predictors used by the logistic model: `log_lti`, `log_county_value_to_loan`,
`purchaser_type`, and `loan_type`. Mark the latter two as native categorical predictors. Do not
supply logistic splines or hand-built interactions; tree splits are responsible for learning
nonlinearities and interactions. Continue to exclude year, HPI growth, edit status, small-loan
status, geography, lender, MSA, and tract variables.

Within each four-year source window, weight first and second liens to equal total mass in every
source year. The fitted balanced-prior log odds estimate `log(f_1(x) / f_0(x))`. Clip fitted
probabilities to `[1e-12, 1 - 1e-12]` before taking log odds. For every target year, estimate the
second-lien mixture share from the unlabeled target covariates and use that share to construct
adjusted probabilities. Unadjusted source-prior probabilities are not the selection object.

Disable the estimator's random internal early-stopping split. Iteration count is an explicit
candidate, and all selection uses the existing reverse temporal folds. Set `random_state=17`,
`min_samples_leaf=1000`, and retain the library's 255-bin default.

## Staged grid

The structure screen uses the latest source fold, 2013--2016, because that one fitted model is
evaluated at every backward horizon from one through nine. Cross:

- `max_leaf_nodes` in `{7, 15, 31}`; and
- `learning_rate` in `{0.05, 0.1}`,

holding `max_iter=200` and `l2_regularization=1`. Carry the two candidates with the lowest
equal-horizon mixture-adjusted Brier score into the complete 45-cell reverse design.

Select the better surviving structure over all 45 cells, then add four one-dimensional
refinements around it:

- `max_iter` in `{100, 400}`, holding L2 at 1; and
- `l2_regularization` in `{0, 10}`, holding iterations at 200.

The two surviving structures and all refinements are eligible for the final decision. This is
a deliberately bounded challenger search, not an attempt to exhaust the boosting parameter
space.

## Decision rule and artifacts

The sole selection statistic is raw Brier score of the mixture-adjusted target probabilities,
averaged within backward horizon and then equally across horizons 1--9. Report cell-level
differences from the frozen mixture-adjusted logistic estimator. A lower aggregate Brier alone
does not automatically replace logistic: the gain must also be temporally broad and large
enough to justify reduced interpretability and a more flexible density-ratio estimate.

Save every fitted fold/candidate object under `output/model/boosting_folds/`. Refit the selected
challenger on 2004--2007 and save it as `output/model/boosting_challenger.pkl`, regardless of
whether it replaces logistic. Persist only aggregate validation outputs, never loan-level
probabilities.

## Cluster reproduction

The original `make evaluate-gradient-boosting` entry point remains the sequential local
compatibility workflow. The same frozen screen, survivor, refinement, and 2004--2007 refit can
be reproduced with immutable Slurm shards and a batch final fit. See
`BOOSTING_SELECTION_CLUSTER.md` for the complete commands and retry rules. Cluster
finalization verifies the documented 7-leaf, learning-rate 0.05, 200-iteration, L2-1 winner
before writing the established `boosting_challenger_*` compatibility tables.
