# Step 8: Historical Plausibility Findings

## Scope

Step 8 applies the frozen final models; it does not refit or select a model. The source model is
the persisted 2004--2007 selected logistic specification (`spline_lti__purchaser_type`, ridge
`C=0.1`). The mixture series uses the corresponding equal-source-prior density ratio and fits
only the target-year mixture share. All outputs are annual aggregates. Loan-level historical
probabilities are not retained.

The public release figure shows four series for 1990--2016:

- the mixture mean probability, equal up to numerical tolerance to the fitted annual count
  share;
- the share of mixture-adjusted probabilities at or above the frozen 0.5 threshold;
- the raw logistic mean probability as a benchmark; and
- actual second-lien shares for labeled years 2004--2016.

## Internal evidence

The mixture mean is 9.14% in 2003 and 13.23% in 2004, a 4.09 percentage-point increase across
the lien-reporting boundary. Its 2004 estimate is close to the observed 13.54% share, missing by
-0.31 point. The raw logistic mean also increases materially, from 10.62% to 14.14% (3.52
points), and its 2004 error is +0.60 point. Thus, the discontinuity is not generated solely by
the annual mixture intercept; the observed feature distribution or sample composition also
changes at the boundary.

Across labeled years 2004--2016, the mean absolute annual share error is 0.30 percentage point
for the mixture mean and 1.22 points for the raw mean. The mixture series reproduces the
housing-boom pattern in labeled data: 13.23% in 2004, 19.86% in 2005, 22.32% in 2006, and
12.91% in 2007. In the unlabeled period, it rises from 2.68--4.88% in 1990--1996 to roughly 9%
in 2000--2003.

These findings support plausibility, not identification. The close 2004 fit and low
labeled-period errors show that the target-share likelihood works when the density-ratio
transport approximation is reasonably successful. They do not establish the unobserved
pre-2004 shares. Likewise, smoothness would not prove validity, and the observed boundary jump
is a warning that should remain visible in the paper rather than be adjusted away.

## External-source review

No reviewed public source supplies the exact validation target: the annual origination-count
share of subordinate liens among owner-occupied home-purchase loans in the HMDA analysis
sample.

- The FHFA National Mortgage Database core sample is a representative sample of closed-end
  **first-lien** residential mortgages. Its public products therefore do not provide the missing
  historical subordinate-lien origination share.
- The Federal Reserve Financial Accounts include junior-lien home-equity loans and HELOCs, but
  measure dollar flows or outstanding balances across purposes and property/occupancy types.
- The New York Fed Consumer Credit Panel reports credit-account and balance series, including
  HELOC balances, from 1999 onward. It does not isolate simultaneous home-purchase junior liens.
- The American Housing Survey can describe second mortgages and home-equity borrowing in the
  occupied housing stock, but it is a biennial housing-unit survey rather than an annual
  origination sample.

Accordingly, these series can be used only to ask whether broad historical movements are wildly
inconsistent with other evidence. They must not be rescaled or presented as numerical
validation targets without a defensible reconciliation of units, loan purpose, occupancy,
coverage, and timing. The machine-readable definition mapping is
`data/public/step8_external_source_comparison.csv` and is copied into `output/tables/` by the
Step 8 target.

## Generated outputs

- `output/tables/step8_annual_plausibility.csv`
- `output/tables/step8_boundary_continuity.csv`
- `output/tables/step8_external_source_comparison.csv`
- `output/figures/step8_predicted_actual_shares_1990_2016.pdf`
