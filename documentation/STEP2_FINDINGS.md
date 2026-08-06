# Step 2 Findings: Sample Construction and HPI Feature Audit

## Scope

This audit covers every HMDA year in the application window, 1990--2016. It
streams the raw yearly parquet files and reproduces the cumulative restrictions
in `src/hmda_seconds/clean.py`. No sample definition or model feature has been
changed as part of the audit.

The audit exactly reproduces the existing 19,351,821-observation 2004--2007
training extract.

## Sample attrition

Across all 27 years, the cumulative counts are:

| Stage | Observations | Retention from prior stage |
|---|---:|---:|
| Raw HMDA records | 542,378,694 | -- |
| Originated owner-occupied purchase loans | 95,451,637 | 17.60% |
| Nonmissing lien status where reported | 95,451,637 | 100.00% |
| Valid state and county | 92,429,861 | 96.83% |
| Positive, nonmissing income and loan amount | 90,232,646 | 97.62% |
| Loan type in 1--4 | 90,232,611 | approximately 100.00% |
| Matched to the balanced FHFA panel | 85,818,337 | 95.11% |
| Final model sample | 85,818,337 | 100.00% |

The low raw-to-final retention rate mainly reflects the project's intended
origination, purpose, and occupancy restrictions. The two material exclusions
within the target loan population are missing/invalid geography or income and
the balanced FHFA merge.

Detailed yearly results are written to:

- `output/tables/sample_attrition_by_year.csv`
- `output/tables/sample_attrition_by_lien_status.csv`

## Balanced versus year-specific FHFA coverage

The current balanced panel requires a county to have an HPI observation in
every year from 1990 through 2016. It contains 1,431 counties. Year-specific
FHFA coverage rises from 1,454 counties in 1990 to 2,723 counties in 2016.

Because excluded counties are generally small, the difference in loan coverage
is smaller than the difference in county counts:

- The balanced panel matches 85.82 million of 90.23 million otherwise eligible
  loans, or 95.11%.
- Year-specific matching would cover approximately 89.46 million loans, or
  99.14%.
- The balanced requirement therefore removes 3.64 million loans beyond those
  that lack an HPI in their own year.
- The incremental balanced-panel loss grows from zero in 1990 to approximately
  4%--6% of eligible loans in most years after 1998.

The exclusion is correlated with lien status. In 2004--2007, the balanced merge
retains 93.75%--95.13% of eligible first liens but 97.38%--98.00% of eligible
second liens. It raises the observed second-lien share by 0.35--0.56 percentage
points in those training years. The direction remains the same in every labeled
year.

### Recommendation

Use year-specific FHFA coverage as the provisional main sample and retain the
balanced panel as a robustness sample. This better matches the release target
of all eligible HMDA originations and sharply reduces geography-driven sample
selection. Report yearly coverage and the common-panel robustness series so
changes in county composition remain visible.

This recommendation should be implemented jointly with the HPI feature change
below. If the final feature requires lagged HPI growth, define coverage using
the observations needed for those lags rather than requiring the complete
1990--2016 panel.

Detailed coverage results are in
`output/tables/fhfa_coverage_by_year.csv`.

## HPI level comparability

The vendored FHFA workbook contains three county index columns:

1. A native HPI normalized to 100 when each county is first recorded.
2. An HPI with a common 1990 base.
3. An HPI with a common 2000 base.

The current loader and cleaning pipeline select the first, native HPI. Its
cross-county scale therefore depends on when each county's series began, not on
the relative level of local house prices.

The empirical consequences are large even within the 1,431 balanced counties:

- In 1990, the common-base index is exactly 100 in every county, while the
  native index ranges from 51.37 to 555.79.
- The county-specific ratio of native HPI to common-1990-base HPI ranges from
  0.514 to 5.558, with a median of 1.511.
- The native/common-base cross-county correlation is only 0.056 in 2000, 0.386
  in 2004, and 0.447 in 2016.

FHFA describes its HPI as a repeat-sales measure of price *changes*. Its
official FAQ states that index numbers alone do not have significance and are
used to calculate appreciation rates. The county workbook similarly describes
the series as cumulative appreciation indexes and explains their alternative
normalizations. See the [FHFA HPI FAQ](https://www.fhfa.gov/faqs/hpi) and
[FHFA Working Paper 16-01](https://www.fhfa.gov/working-paper-16-01-local-house-price-dynamics-new-indices-and-stylized-facts).

Detailed comparisons are in
`output/tables/fhfa_hpi_level_comparability.csv`.

### Implication for `log_ltv`

The current feature

\[
\texttt{log\_ltv}=\log(\text{native county HPI}/\text{loan amount})
\]

is not a defensible LTV proxy. County HPI is not a property-price level, and the
native index adds an arbitrary, county-specific normalization to the feature.
Its demonstrated predictive value may partly reflect a persistent accidental
geography code rather than housing equity or collateral value.

Simply substituting the 1990-base index would remove the arbitrary
normalization, but it still would not turn an appreciation index into a local
house-price level. The feature should therefore be revised rather than silently
rebased and retained under the `log_ltv` name.

### Recommendation

For the logistic specification work, compare at least these alternatives under
the predeclared temporal protocol:

1. Drop `log_ltv`.
2. Replace it with separate `log_loan_amount` and county HPI appreciation
   features based on a common normalization.
3. Test trailing one- and three-year log HPI growth using only information
   available by the origination year.
4. If a true local house-price-level proxy is desired, identify a separate
   public source with a comparable dollar-valued measure; do not label an HPI
   index ratio as LTV.

Until those comparisons are complete, existing RF and logistic results should
be described as results for the historical feature specification, not as
validation of an economically interpretable LTV measure.

## Step 2 decisions recommended for review

- Preserve the current pipeline outputs for reproducibility until model
  comparisons are rerun.
- Make year-specific FHFA matching the provisional main-sample candidate.
- Retain the balanced sample as a robustness check.
- Retire `log_ltv` in its current native-HPI form.
- Evaluate decomposed loan-amount and HPI-appreciation features during logistic
  optimization.
- Continue reporting sample coverage by year and lien status.

## Implemented county-value revision

The native-HPI feature has now been replaced in the production code by
`log_county_value_to_loan`. County Zillow ZHVI is annualized from complete calendar years,
and each county's FHFA history is scaled by the geometric mean Zillow/FHFA ratio over all
available overlap years. Alternative OLS-through-origin, median-log-ratio, and 2017-anchor
scales are retained for robustness. The merge is year-specific and requires at least one
overlap year; low-support county estimates remain identified in the diagnostic table.

This revised panel retains 99.754% of eligible 2004--2007 loans. It retains 99.718% of first
liens and 99.924% of second liens, increasing the training-sample second-lien share by only
0.030 percentage points. Full definitions and diagnostics are in
`COUNTY_VALUE_SCALING.md`.
