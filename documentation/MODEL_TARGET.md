# Model Target and Release Outputs

## Target population

The model assigns first- versus second-lien status to each HMDA origination in
the cleaned analysis sample defined by `src/hmda_seconds/clean.py`. The released
annual shares are conditional on this eligible model sample, not on all raw HMDA
records.

The aggregate estimand is an **origination-count share**: every eligible loan
receives weight one. Loan amount is not used as an aggregation weight.

## Canonical release outputs

The release will contain all three of the following outputs.

### 1. Loan-level second-lien probability

For each eligible origination (i), release

\[
p_i = \Pr(y_i = 2 \mid x_i),
\]

where lien status 1 denotes a first lien and lien status 2 denotes a second
lien. Probabilities should be stored at sufficient precision that downstream
users can reproduce the canonical hard classification.

The probability is the primitive model output. It should not be interpreted as
a calibrated frequency in 1990--2003 unless the calibration analysis supports
that interpretation.

### 2. Canonical hard classification

The canonical classification uses a threshold of 0.5:

\[
\widehat{y}_i =
\begin{cases}
2 & \text{if } p_i \geq 0.5,\\
1 & \text{if } p_i < 0.5.
\end{cases}
\]

An observation with probability exactly equal to 0.5 is classified as a second
lien. The 0.5 threshold corresponds to the transparent default of equal false-
positive and false-negative classification costs. Alternative thresholds may
be reported as sensitivity analyses but are not separate canonical releases.

### 3. Annual origination-count shares

The primary aggregate output is the probability-implied expected share of
eligible originations that are second liens:

\[
\widehat{s}^{\mathrm{prob}}_t
= \frac{1}{N_t}\sum_{i \in t} p_i.
\]

The aggregate table will also report the share based on canonical hard labels:

\[
\widehat{s}^{\mathrm{hard}}_t
= \frac{1}{N_t}\sum_{i \in t} \mathbf{1}\{p_i \geq 0.5\}.
\]

The probability-implied share is the primary aggregate estimand. The hard-label
share is a classification diagnostic and convenience output. It discards
within-loan uncertainty and need not equal the mean probability.

For 2004 and later, when lien status is observed reliably, the aggregate table
will additionally report the actual second-lien count share on the identical
eligible sample.

## Required release fields

### Loan-level crosswalk

- `year`
- Available HMDA join identifiers, currently `resp_id` and `seq_num`
- `prob_second_lien`
- `lien_status_predicted_050`
- Model or release version

The crosswalk must identify the HMDA input vintage needed for a reliable join.

### Annual aggregate table

- `year`
- `n_model_sample`
- `mean_prob_second_lien`
- `share_predicted_second_lien_050`
- `actual_second_lien_share`, when observed
- Coverage or attrition fields needed to relate the model sample to the raw
  HMDA population
- Model or release version

## Interpretation and limitations

- All annual shares are unweighted loan-count shares among eligible
  originations. They are not loan-dollar shares.
- Summing probabilities gives the model-implied expected number of second liens;
  it does not eliminate calibration or temporal-transport error.
- The hard classification depends on the declared 0.5 threshold. Users may
  construct alternative classifications from the released probabilities, but
  those are not the canonical series.
- Smooth predictions around the 2004 reporting boundary are a plausibility
  check, not proof that the pre-2004 predictions are correct.
- Direct pre-2004 classification accuracy and calibration cannot be measured
  from HMDA because reliable lien-status labels are unavailable.

## Step 1 decision summary

- Release probabilities: **yes**.
- Release canonical hard labels: **yes**.
- Release annual aggregate shares: **yes**.
- Primary aggregate: **unweighted origination-count mean of predicted
  probabilities**.
- Secondary aggregate: **unweighted share classified as second liens at the
  0.5 threshold**.
- Loan-dollar weighting: **not part of the canonical release**.
