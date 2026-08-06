# Logistic model-selection protocol

This protocol was frozen before running the logistic specification search. It governs Steps 4
and 5 of `recommendations_revised.md`.

## Objective

The sole model-selection statistic is raw out-of-sample Brier score,

\[
N^{-1}\sum_i(p_i-y_i)^2,
\]

where `y = 1` denotes a second lien. There is no class weighting, prevalence adjustment,
oracle recalibration, Brier-skill transformation, or threshold tuning. Other classification,
ranking, and calibration statistics may be reported but cannot determine the selected model.

## Reverse temporal folds

Every estimation window contains four consecutive later years. A fitted model is evaluated
on every available earlier labeled year:

| Training years | Validation years | Backward horizons |
|---|---|---|
| 2005--2008 | 2004 | 1 |
| 2006--2009 | 2004--2005 | 2--1 |
| ... | ... | ... |
| 2013--2016 | 2004--2012 | 9--1 |

This produces nine model fits per candidate/hyperparameter value and 45 out-of-sample cells.
For a cell, the horizon is the first training year minus the validation year. Scores are
averaged first across available cells within a horizon and then equally across horizons 1--9.
The number of contributing cells is always reported because support declines with horizon.

The final selected model is refit on 2004--2007 and applied backward to 1990--2003. Horizons
10--14 cannot be validated with the available labeled data. Existing forward 2008--2016
results remain a robustness check and do not tune the specification.

## Core features and candidates

Every core candidate includes:

- `log_lti`;
- `log_county_value_to_loan`;
- reference-coded purchaser-type indicators; and
- reference-coded loan-type indicators.

The candidate grid crosses three continuous functional forms with four interaction structures:

| Continuous form | None | Loan type | Purchaser type | Both |
|---|---:|---:|---:|---:|
| Linear main effects | yes | yes | yes | yes |
| Restricted cubic spline for `log_lti` | yes | yes | yes | yes |
| Restricted cubic splines for both continuous variables | yes | yes | yes | yes |

Continuous-by-indicator interactions are linear slope deviations around the shared main-effect
functions. There is no continuous-by-continuous tensor interaction. Four restricted-cubic-
spline knots are fixed at the 5th, 35th, 65th, and 95th percentiles of each fold's training
sample. Knot locations, centering, and scaling are fitted only on the training window and
applied unchanged to its earlier validation years. Spline tails are linear.

After completing this frozen grid, a focused challenger allows every fold-fitted `log_lti`
restricted-cubic-spline basis term to interact with purchaser-type indicators. It retains the
linear `log_county_value_to_loan` by purchaser interaction. The challenger uses the identical
reverse folds, raw-Brier objective, and staged ridge search; it does not retroactively alter
the original 12-candidate grid or automatically replace its winner.

The core grid excludes HPI growth, `has_edit_status`, `loan_below_10k`, year effects, lender
identity, and MSA or tract identifiers. HPI growth is excluded because it could proxy for the
boom-era financing and securitization regime rather than a relationship that transports to
the 1990s.

## Regularization grid

The primary penalty is L2 ridge with `class_weight=None`. For each of the 12 specifications,
the coarse grid is

\[
C\in\{10^{-4},10^{-2},1,100\}.
\]

After scoring every coarse value, two predeclared adjacent decades are added around that
specification's best coarse value. A boundary winner extends the grid outward by one decade.
All specifications survive to refinement. The selected object is the complete
`(feature specification, C)` pair with the lowest equally horizon-weighted raw Brier score.

The Newton--Cholesky solver, convergence tolerance, iteration limit, reference categories,
spline knot rule, and 0.5 hard-classification threshold are fixed implementation choices
rather than tuned hyperparameters. Newton--Cholesky is appropriate here because observations
number in the millions while the candidate designs contain fewer than 100 columns.
Nonconverged fits are rerun with a larger iteration allowance, not compared as if early
stopping were a model choice.

## Geographic challengers

After selecting the non-geographic core, two guarded challengers add either broad Census-
region indicators or state indicators and retune `C` under the same folds. They are not
automatically eligible to replace the core merely because aggregate Brier improves. Their
incremental Brier must be reported by year and horizon, along with category support and effect
stability across windows. The non-geographic model remains primary unless a geographic gain is
large, temporally broad, and substantively defensible for 1990--2003. No geographic
interactions are considered.
