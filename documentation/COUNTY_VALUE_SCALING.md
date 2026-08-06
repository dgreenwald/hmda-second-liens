# County-value scaling decision

## Production definition

The native FHFA county HPI is an appreciation index whose level is not comparable across
counties. The production feature therefore scales each county's FHFA history to the dollar
level of Zillow county ZHVI.

Monthly Zillow all-homes, middle-tier, smoothed and seasonally adjusted ZHVI is averaged
within a calendar year when all 12 months are present. For county \(c\), the primary scale
uses every year in which annual Zillow ZHVI and FHFA HPI overlap:

\[
\log a_c = \operatorname{mean}_t\{\log Z_{ct} - \log H_{ct}\},
\qquad
\widehat V_{ct}=a_c H_{ct}.
\]

The loan-level feature is

\[
\texttt{log\_county\_value\_to\_loan}
=\log\{\widehat V_{ct}/(1000\times\texttt{loan\_amt})\},
\]

because HMDA reports `loan_amt` in thousands of dollars. This is a county-value-to-loan
proxy, not a property-specific LTV, and is named accordingly.

The source vintage is pinned at `202608`. The production panel requires at least one overlap
year and then uses every positive, year-specific FHFA observation from 1990--2016. It does
not require a county to have a balanced FHFA history.

## Why use the geometric mean ratio?

The estimator is the least-squares constant in log space. It gives each available overlap
year equal weight, is insensitive to the arbitrary native normalization of the FHFA series,
and does not let high-dollar later observations dominate the fit mechanically. It is also
transparent to reproduce with a county `groupby` mean of the log ratio.

The minimum of one overlap year preserves 2,690 scaled counties. Of these, 249 have exactly
one overlap year and 29 have two; these estimates are less well diagnosed and remain flagged
by `n_overlap_years` in the county scale table. Requiring three years would retain 2,412
counties and is a planned support robustness check, not the production restriction.

## Robustness estimators

The county diagnostic table also records:

1. OLS in levels through the origin, \(Z_{ct}=a_cH_{ct}+e_{ct}\).
2. The median annual log ratio.
3. The single-year 2017 ratio, where available.

Across the 2,690 primary counties, OLS and the geometric estimator are extremely close: the
median log OLS/geometric difference is 0.0009 and its 99th percentile is 0.0221. The median
log-ratio estimator has a median difference of zero and a 99th percentile absolute-direction
comparison of 0.0426. The 2017 anchor is less stable: its median log difference is 0.0319 and
its 90th percentile is 0.1179. This supports using the multi-year geometric scale as primary
and the alternatives as robustness checks.

The within-county geometric-fit log RMSE has a median of 0.0409 and a 90th percentile of
0.0917. These residuals reflect differences between the two series' concepts and dynamics;
the scaled value should not be presented as a direct estimate of an individual property's
market value.

## HMDA coverage

The exact streamed audit shows that the production panel matches 20,282,918 of 20,332,941
eligible 2004--2007 loans, or 99.754%. Annual match rates in that training window range from
99.704% to 99.778%. From 2000--2016, every annual loan match rate is at least 99.469%.

Training-period match rates are 99.718% for first liens and 99.924% for second liens. The
differential raises the observed second-lien share from 17.672% before the county-value merge
to 17.702% after it, a small increase of 0.030 percentage points.

Coverage is weaker by raw county-code count in early years but remains high by loan count.
In 1990, for example, 1,391 of 3,379 reported county codes match (41.17%), while 1,664,934 of
1,726,679 eligible loans match (96.42%). This indicates that the missing early counties are
mostly very low-volume, but the yearly loan and county rates should both continue to be
reported.

## Reproducible outputs

- `output/tables/zillow_fhfa_county_scales.csv`: scales, overlap support, and fit diagnostics.
- `output/tables/zillow_fhfa_annual_overlap.csv`: annual observations used to estimate scales.
- `output/tables/zillow_fhfa_support_summary.csv`: county counts at one-, three-, and five-year
  support cutoffs.
- `output/tables/county_value_coverage_by_year.csv`: exact HMDA loan and county coverage.
- `output/tables/county_value_coverage_by_lien_status.csv`: exact match rates separately for
  observed first and second liens.

Run `make county-values` and `make county-value-coverage` to regenerate these tables. Existing
training extracts and fitted models should be preserved until the revised feature
specifications are compared under the temporal model-selection protocol.
