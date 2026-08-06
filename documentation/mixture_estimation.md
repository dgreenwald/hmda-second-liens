# Mixture-Proportion Estimation Plan

## Objective

Estimate the annual second-lien **count share** using the distribution of loan characteristics
in each target year, rather than carrying the labeled-era prevalence embedded in a classifier
unchanged into another period.

This is a statistical mixture-proportion or quantification problem, not yet a structural
economic model. The selected logistic classifier remains the primary loan-level model. The
mixture estimator is initially a challenger for aggregate shares and a possible source of
prior-adjusted loan-level probabilities.

Step 6 motivates this exercise. In reverse validation, the selected model's equal-horizon mean
probability is 6.16% while the observed second-lien share is 13.63%. In the separate forward
check, the 2004--2007 model overpredicts every 2008--2016 share. The sign reversal suggests
that transporting the training-period prevalence is a central source of error.

## Estimand and notation

Let

- `S` denote a second lien;
- `F` denote a first lien;
- `X` denote the model characteristics; and
- \(\pi_t=P_t(S)\) denote the second-lien count share in target year \(t\).

The target-year feature distribution is

\[
f_t(x)=(1-\pi_t)f_F(x)+\pi_t f_S(x).
\]

The initial model treats the component distributions \(f_F\) and \(f_S\) as transportable from
a labeled source window and allows only \(\pi_t\) to change. This is the label-shift or
prior-probability-shift assumption:

\[
f_t(X\mid Y)=f_s(X\mid Y),
\qquad P_t(Y)\ne P_s(Y).
\]

This assumption differs from stability of \(P(Y\mid X)\). Neither assumption is automatically
weaker. The mixture approach is useful only to the extent that within-lien-class feature
distributions transport across time.

## Primary estimator: classifier density-ratio likelihood

The first implementation should reuse the selected logistic feature specification rather than
impose a new parametric joint distribution on all features. Fit source-year nuisance
intercepts alongside a common transported feature function:

\[
\operatorname{logit}P_s(S\mid X=x,T=t)=\alpha_t+g(x).
\]

The year intercepts absorb labeled-source prevalence changes while estimating the shared
feature coefficients. The reference-year intercept is unpenalized; scale the remaining year
indicators so their effective ridge penalty is negligible while retaining the selected
`C=0.1` penalty for `g(x)`. Do not extrapolate a source-year dummy into an unlabeled year.
Normalize `g(x)` so the implied density ratio averages one among source first liens, and use
the target mixture estimate to supply the customized target-year intercept.

For a labeled source window, let

\[
q_s(x)=P_s(S\mid X=x)
\]

be its fitted second-lien probability and let \(\pi_s\) be the source-window count share. Bayes'
rule implies the component-density ratio

\[
r(x)=\frac{f_S(x)}{f_F(x)}
=\frac{q_s(x)}{1-q_s(x)}\frac{1-\pi_s}{\pi_s}.
\]

For an unlabeled target sample \(x_1,\ldots,x_{N_t}\), terms involving \(f_F(x_i)\) do not
depend on \(\pi_t\). Estimate the target share by maximizing

\[
\ell_t(\pi)=
\sum_{i=1}^{N_t}\log\left[(1-\pi)+\pi r(x_i)\right],
\qquad 0\leq\pi\leq1.
\]

This is a one-dimensional concave likelihood when the density ratio is fixed. Estimate it with
a bounded optimizer and verify the first-order condition. Report estimates at or near a
boundary rather than silently truncating them.

### Equivalent EM implementation

For a trial value \(\pi\), calculate target-adjusted posterior probabilities

\[
w_i(\pi)=
\frac{\pi r(x_i)}{(1-\pi)+\pi r(x_i)}.
\]

Then update

\[
\pi\leftarrow\frac{1}{N_t}\sum_i w_i(\pi)
\]

until convergence. At the fixed point, the mean adjusted probability equals the estimated
target count share. The bounded likelihood optimizer and EM algorithm should agree to a tight
numerical tolerance and can serve as mutual implementation checks.

### Numerical and model checks

- Calculate likelihoods in log space.
- Clip probabilities only to prevent infinities, and report the fraction clipped.
- Use the unweighted source count share, matching the declared estimand.
- Check density-ratio normalization on labeled source observations, including whether
  \(E[r(X)\mid F]\) is close to one.
- Record optimizer convergence, score at the optimum, curvature, boundary distance, and EM
  iterations.
- Report each source-year fitted intercept, observed prior log-odds, and their difference. Under
  a common density ratio, the fitted intercept minus the observed prior log-odds should agree
  across source years and with the ratio normalization.
- Do not use target labels in estimation or tuning.

As an ablation, also fit the same feature function without source-year intercepts and normalize
its implied density ratio identically. This comparison does not create a competing historical
method; it measures whether absorbing source-year prevalence materially changes the transported
feature coefficients and target-share accuracy.

## Primary validation design

Use the frozen reverse-temporal triangle already used for logistic model selection:

- fit the selected logistic specification on each four-year later training window;
- treat every available earlier validation year as unlabeled;
- estimate its second-lien share using only its characteristics; and
- reveal its labels only after estimation to evaluate the share estimate.

This produces the same 45 validation cells and backward horizons 1--9. It directly matches the
1990--2003 application, unlike a validation exercise that estimates components on 2004--2007
and reports partially in-sample performance on 2004--2016.

### Comparators

For every cell, report:

1. Actual second-lien count share.
2. Raw mean probability from the source classifier.
3. Share of raw probabilities above 0.5.
4. Density-ratio mixture estimate \(\hat\pi_t\).
5. Share of mixture-adjusted probabilities above 0.5.

The aggregate-share comparison must not discard the existing classifier. The purpose is to
determine whether target-distribution adjustment improves the output that failed Step 6.

### Primary evaluation metrics

The primary metric is absolute count-share error in percentage points:

\[
\left|\hat\pi_t-\pi_t\right|.
\]

Also report signed error, squared error, and the fraction of cells in which each estimator has
the smallest absolute error. Average within backward horizon and then equally across horizons,
matching the existing temporal weighting convention.

For adjusted loan-level probabilities, report Brier score, log loss, calibration intercept and
slope, and hard-classification metrics as secondary diagnostics. Aggregate-share accuracy, not
loan-level Brier, determines whether the mixture adjustment succeeds at its primary task.

## Assumption and specification diagnostics

With millions of observations, conventional tests will reject economically negligible
differences. Emphasize effect sizes and graphical stability rather than p-values.

### Within-class stability

For each labeled year and lien class, report:

- means, standard deviations, and selected quantiles of both continuous variables;
- class-specific histograms or empirical distribution functions;
- purchaser- and loan-type shares;
- Wasserstein or other interpretable distribution distances across years; and
- the accuracy with which a classifier can distinguish source years within a fixed lien class.

A source-year classifier with strong discrimination is evidence against invariant component
distributions, even when lien status itself remains easy to classify.

### Feature-subset overidentification

Estimate \(\pi_t\) separately using:

- `log_lti` alone;
- `log_county_value_to_loan` alone;
- both continuous variables;
- purchaser type alone;
- loan type alone; and
- the selected combined specification.

If pure label shift is a reasonable approximation, these estimates should be broadly
consistent. Large disagreement indicates that at least some class-conditional feature
distributions changed and that a single mixture weight cannot reconcile the source and target.

### Source-window sensitivity

For each labeled target year, compare estimates obtained from multiple admissible later source
windows. Report dispersion across source windows alongside sampling uncertainty. This is more
informative for backward transport than an individual-loan bootstrap alone.

### Mixture goodness of fit

Compare the observed target feature distribution with the distribution implied by
\((1-\hat\pi_t)f_F+\hat\pi_t f_S\). For the density-ratio implementation, perform these checks
on low-dimensional feature margins and selected joint cells. A high likelihood does not by
itself demonstrate that the fixed-component model reproduces economically important aspects
of the target distribution.

## Direct generative robustness models

The classifier density-ratio method is primary because it reuses the selected flexible feature
model. Direct component-density estimates provide transparent robustness checks and reveal
which feature distributions identify the share.

### Univariate `log_lti`

Estimate class-specific `log_lti` densities in each source fold and then maximize

\[
\sum_i\log\left[(1-\pi)f_F(x_i)+\pi f_S(x_i)\right].
\]

Evaluate, in order:

1. Gaussian components as a deliberately simple benchmark.
2. Student-\(t\) components for heavier tails.
3. Reproducible histogram or spline densities for flexible shape.

The Gaussian model is not presumed correct. With millions of observations, small tail
misspecification can produce a precise but biased mixture weight. Prefer histograms or spline
densities to raw kernel density estimation for speed, reproducibility, and transparent
bandwidth or knot choices.

### Bivariate extension

Add `log_county_value_to_loan` only if the univariate estimators are reasonably stable and the
bivariate model improves reverse-fold share error consistently. Candidate specifications are a
bivariate Student-\(t\), a small Gaussian mixture within each lien class, or a two-dimensional
histogram/spline density with regularization.

Do not immediately build a joint density over every purchaser and loan-type cell. Sparse cells,
changing market structure, and categorical dependence could add more transport risk than
identifying information. Add categorical conditioning only after demonstrating incremental,
stable share accuracy.

## Uncertainty

Conditional profile-likelihood intervals for \(\pi_t\) should be reported but will be extremely
narrow because target samples contain millions of loans. They do not capture uncertainty about
component transport.

The primary uncertainty and sensitivity analysis should therefore include:

- source-year or source-window resampling;
- alternative admissible feature subsets;
- alternative direct-density specifications; and
- explicit sensitivity to bounded changes in component distributions.

Bayesian estimation is deferred. A beta prior on \(\pi_t\) alone contributes little at these
sample sizes, while a misspecified component model can yield a misleadingly concentrated
posterior. A hierarchical Bayesian model becomes useful only if the project adopts explicit,
defensible restrictions on how component distributions evolve through time.

Partial-identification bounds are also deferred. They would require explicit bounds on
class-conditional distribution drift; the general contaminated-data literature does not by
itself supply economically credible bounds for HMDA.

## Implementation structure

Add a separate pipeline stage without changing the selected logistic artifact:

- `src/hmda_seconds/mixture.py`: density-ratio construction, bounded MLE, EM, direct component
  densities, diagnostics, and result aggregation.
- `scripts/estimate_mixture_shares.py`: thin CLI.
- `src/hmda_seconds/mixture_calibration.py` and
  `scripts/diagnose_mixture_calibration.py`: Step 6 probability diagnostics for the frozen
  known-source-prior estimator.
- `tests/test_mixture.py`: synthetic mixtures with known shares, boundary cases, likelihood/EM
  agreement, and checkpoint behavior.
- `make estimate-mixture-shares`: reproducible entry point.

Write only aggregated share, metric, component-summary, and density-bin outputs under
`output/tables/`, plus diagnostic figures under `output/figures/`. Do not persist or release
loan-level target characteristics as part of this stage.

Every fitted model must be persisted when it is estimated. Reverse and forward fold artifacts
belong under `output/model/mixture_folds/` with deterministic names recording the estimator,
specification, regularization value, and training window. Each artifact must contain the fitted
feature transformer and all prediction parameters, so subsequent diagnostics load the exact
fit rather than estimating it again. Persisting fitted parameters does not expose loan-level
HMDA observations and is distinct from the prohibition on releasing microdata.

Suggested outputs include:

- `mixture_reverse_cell_shares.csv`;
- `mixture_reverse_horizon_summary.csv`;
- `mixture_component_stability.csv`;
- `mixture_feature_subset_comparison.csv`;
- `mixture_source_window_sensitivity.csv`; and
- estimated-versus-actual share and component-stability figures.

## Staged implementation agenda

### Stage 1: Density-ratio proof of concept

1. Implement the bounded likelihood and EM estimators.
2. Verify them on synthetic mixtures with known component densities and shares.
3. Recover density ratios from the selected logistic model in every reverse fold.
4. Produce the 45-cell share comparison against raw probability and hard-classification
   aggregation.

**Decision gate:** proceed only if the mixture estimate materially lowers equally
horizon-weighted absolute share error and improves a broad majority of horizons rather than a
small set of years.

### Stage 2: Test the identifying assumption

1. Measure annual class-conditional stability in 2004--2016.
2. Compare estimates across feature subsets and source windows.
3. Evaluate low-dimensional mixture goodness of fit.

**Decision gate:** do not interpret \(\hat\pi_t\) as a reliable historical share if different
features imply materially inconsistent shares or if no fixed-component mixture adequately fits
the labeled target distributions.

### Stage 3: Direct generative benchmarks

1. Fit Gaussian, Student-\(t\), and flexible univariate `log_lti` components within each source
   fold.
2. Compare their reverse-fold share errors with the density-ratio estimator.
3. Add a bivariate continuous model only if justified by stable improvement.

**Decision gate:** prefer the simplest component model with broad temporal performance and
acceptable goodness of fit. Do not select a density solely because it maximizes an enormous
in-sample likelihood.

### Stage 4: Historical application

Only after Stages 1--3 pass their decision gates:

1. Refit the chosen component or density-ratio model on the declared labeled source period.
2. Estimate annual shares for 1990--2003.
3. Produce adjusted posterior loan probabilities if supported.
4. Report specification and source-window sensitivity with the annual estimates.
5. Retain the original classifier outputs as a benchmark rather than overwriting them.

## Literature orientation

- Saerens, Latinne, and Decaestecker (2002), ["Adjusting the Outputs of a Classifier to New A
  Priori Probabilities: A Simple Procedure"](https://doi.org/10.1162/089976602753284446),
  develops the EM prior-adjustment approach that most directly matches the primary estimator.
- González et al. (2017), ["A Review on Quantification
  Learning"](https://doi.org/10.1145/3117807), surveys aggregate class-prevalence estimation and
  distinguishes classify-and-count, adjusted-count, and likelihood-based methods.
- Blanchard, Lee, and Scott (2010), ["Semi-Supervised Novelty
  Detection"](https://www.jmlr.org/papers/v11/blanchard10a.html), provides related theory on
  learning from labeled component and unlabeled mixture samples, though it is not the most
  direct formulation of this two-class prior-shift problem.
- Horowitz and Manski (1995), ["Identification and Robustness with Contaminated and Corrupted
  Data"](https://econpapers.repec.org/article/ecmemetrp/v_3a63_3ay_3a1995_3ai_3a2_3ap_3a281-302.htm),
  motivates bounds under weakened contamination assumptions but does not directly provide the
  HMDA component-drift restrictions needed here.
