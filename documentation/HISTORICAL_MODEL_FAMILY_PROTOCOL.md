# Step 10 historical model-family protocol

## Purpose and frozen scope

This diagnostic extends the existing Step 8 historical plausibility pass. It does not fit,
select, or tune another model. Apply the four frozen 2004--2007 equal-source-prior finalists
from the common model-family comparison to every year from 1990 through 2016:

1. unrestricted ridge logistic;
2. unrestricted histogram gradient boosting;
3. HMDA-only ridge logistic; and
4. HMDA-only histogram gradient boosting.

All four models use the identical eligible sample defined by the unrestricted historical
cleaner. The HMDA-only variants omit the county-value predictor at prediction time but do not
change the target population. Estimate a separate mixture share from each model's density ratio
using only target-year characteristics. Never use target labels in estimation. Labels from
2004 onward are used only for reported plausibility metrics.

After cleaning, require finite values for all four primitive features so every finalist scores
the same observations. Report the clean-sample count, common model-sample count, excluded-row
count, and feature-specific non-finite counts in the immutable year shard. This final common
eligibility restriction is not model-family-specific and must not be implemented as imputation.

## Annual outputs

For each `(year, predictor_set, model_family)` cell report:

- model and configuration identity;
- clean, excluded, and common-model observation counts, plus the observed share when available;
- fitted mixture share and mean adjusted probability;
- hard second-lien share at the frozen 0.5 threshold;
- optimizer/EM convergence, boundary status, and optimizer-minus-EM difference;
- signed and absolute share error when labels are available; and
- fixed quantiles of the log density ratio and adjusted probability.

Persist one immutable aggregate JSON shard per target year. Loan-level probabilities, log
ratios, and target characteristics remain transient and must not be written.

## Support diagnostics

Use the unrestricted 2004--2007 logistic transform as the common source-support reference.
For each target year and continuous primitive predictor, standardize values using the frozen
source mean and standard deviation and report the mean, standard deviation, 1st, 5th, 50th,
95th, and 99th percentiles plus the fractions outside two and three source standard deviations.
For `log_lti`, also report fractions below the first and above the last frozen spline knot,
which correspond to the source 5th and 95th percentiles. These are diagnostics of covariate
transport, not sample restrictions.

For each categorical predictor and canonical level, report the annual count and share. For
each fitted model, report log-ratio and adjusted-probability quantiles plus the shares of
adjusted probabilities below 0.01 and above 0.99. The latter describe output saturation; they
are not clipping or exclusion rules.

The aggregation stage compares every pre-2004 support statistic with the 2004--2007 annual
envelope and reports whether it falls below, within, or above that labeled-source envelope.
It also reports annual boosting-minus-logistic differences within each predictor set. These
diagnostics describe empirical overlap and estimator behavior but cannot identify pre-2004
accuracy.

## Cluster and artifact contract

Use one cluster task per target year so each cleaned loan file is loaded once and scored by all
four models. A canonical manifest under `output/slurm/historical_model_family/` identifies the
comparison manifest, years, inputs, and output root. Completed matching year shards are reused;
conflicting shards are never overwritten. Submit a dependent aggregation job only after every
year task succeeds.

Aggregation requires exactly 27 year shards, four model cells per year, identical sample counts
within year, successful non-boundary mixture estimates, and mean-probability/share agreement.
It writes only annual, paired, continuous-support, categorical-support, and support-envelope
tables plus aggregate figures. The fitted model artifacts remain those produced by the common
model-family comparison and are not copied or refitted.

Generate and inspect the canonical array and dependent aggregation script with:

```bash
make historical-model-family-cluster
```

Submit both stages, with aggregation dependent on successful completion of every annual task,
only when explicitly requested:

```bash
make historical-model-family-cluster-submit
```

To rerun only validation and aggregation from completed immutable year shards:

```bash
make aggregate-historical-model-family
```

## Decision rule

Do not select a model merely because its unlabeled historical series is smoother. Interpret the
historical comparison together with the frozen reverse and forward labeled results. Large or
systematic family divergence, output saturation, or support excursions are warnings about
transportability. No result from this stage may trigger threshold selection, feature changes,
or hyperparameter tuning. The final Step 10 decision must report the unresolved inability to
observe pre-2004 lien status directly.
