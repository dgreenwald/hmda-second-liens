# Mixture-Native Logistic Reselection Protocol

This protocol is fixed before inspecting mixture-native logistic selection results. It asks
whether the earlier logistic specification remains optimal when the actual release estimator
uses equal source priors and a separately estimated target-year mixture share.

## Candidates and objective

Evaluate the original 12 core specifications formed by crossing three continuous functional
forms with four interaction structures, plus the previously tested full LTI-spline-by-
purchaser challenger. Continue to exclude geography, year, HPI growth, edit status, small-loan
status, lender, MSA, and tract features.

For every source window, give first and second liens equal total weight within each source year.
Use the complete balanced-prior logistic log odds as the density-ratio score. Estimate a
separate mixture share from every target year's covariates and score the resulting adjusted
probabilities. The sole selection statistic is raw adjusted Brier, averaged within backward
horizon and then equally across horizons 1--9.

## Staged grid

Screen all 13 specifications at the existing coarse ridge grid
`C in {1e-4, 1e-2, 1, 100}` on the 2013--2016 source window, which exposes every backward
horizon from one through nine. Within each specification, retain its best coarse penalty.

Carry four specifications into the complete 45-cell design. The current selected
`spline_lti__purchaser_type` specification is guaranteed a place; fill the other places with
the best screen specifications not already included. Evaluate all four coarse penalties for
every survivor in all 45 cells.

For each survivor, identify its best coarse penalty using the complete reverse design and add
the two adjacent decades around it, following the original logistic refinement rule. Evaluate
those refinements in all 45 cells. Select the specification/penalty pair with the lowest
equal-horizon adjusted Brier among all fully evaluated coarse and refined candidates.

The latest-window screen controls computation but does not provide the final score. Report the
screened-out specifications and the guaranteed survival of the existing primary explicitly.

## Artifacts and diagnostics

Save every fitted specification/penalty/source-window model under
`output/model/mixture_logistic_selection/`. Save the final 2004--2007 refit as
`output/model/logistic_mixture_selected.pkl`. Do not overwrite the existing selected logistic
artifact until the final estimator decision.

Compare the winner with the existing mixture logistic, gradient boosting, and Random Forest in
all reverse cells. Apply the final refit to 2008--2016 and report the same probability and
calibration diagnostics. Persist aggregated outputs only.
