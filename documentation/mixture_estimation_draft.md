# Structural Estimation of Lien Status via Finite Mixture Models

## The Idea

The current approach trains a discriminative classifier (Random Forest or logistic regression) on labeled 2004–2007 HMDA data to predict lien status from loan characteristics, then applies the fitted classifier to impute lien status for 1990–2003. This works well in-sample but, as the out-of-time validation revealed, the model systematically over-predicts second liens when the base rate shifts — second-lien precision degrades from ~95% in-era to 53–66% in 2008–2016, exactly because the post-crash second-lien share collapsed from ~23% to ~2% while the model's decision boundary stayed fixed.

The structural alternative inverts the question: instead of $P(\text{lien status} \mid X)$, estimate $f(X \mid \text{lien status})$ — the distribution of characteristics *within* each lien class. The observed (unlabeled) data is then a mixture:

$$f_{\text{observed}}(X) = \pi \cdot f_1(X) + (1 - \pi) \cdot f_2(X)$$

where $\pi$ is the first-lien share, $f_1$ is the first-lien characteristic distribution, and $f_2$ is the second-lien distribution. If $f_1$ and $f_2$ can be estimated from the labeled era and are stable enough across time, then the unlabeled data's *own shape* identifies $\pi$ — the mixing weight — without needing a classifier at all. The base-rate problem that plagues the discriminative classifier disappears, because the mixture model *estimates* the base rate from the data rather than inheriting it from the training sample.

## Assessment: This Is a Good Idea

This is a well-motivated approach that directly addresses the most concerning finding from the current validation exercise. Several things make it attractive here:

### Why It Fits This Problem Well

1. **The bimodal LTI distribution is the whole story.** The diagnostic histograms already show that `log_lti` alone almost perfectly separates the two classes — second liens cluster around $\ln(\text{LTI}) \approx 0$ (small loans relative to income) while first liens cluster around $\ln(\text{LTI}) \approx 1.2$–$1.5$. The feature ablation confirmed that `log_lti` drives almost all the classifier's power (dropping it roughly doubles the error rate). A mixture model on `log_lti` alone would be nearly sufficient, and adding `log_county_value_to_loan` would sharpen it further.

2. **The base-rate shift is the main failure mode.** The discriminative classifier's precision problem is entirely a base-rate artifact — it learned a decision boundary appropriate for a ~20% second-lien share, and that boundary produces too many false positives when the true share is 2%. A mixture model estimates the share directly from the data, so it automatically adapts to changing base rates.

3. **The component distributions are extremely well-separated.** Mixture models are notoriously fragile when components overlap heavily, but here the two classes have almost non-overlapping LTI distributions. This is about the most favorable setting possible for mixture identification.

4. **The labeled data provides strong anchoring.** Pure unsupervised mixture models suffer from label switching and initialisation sensitivity. Here we have 13 years (2004–2016) of fully labeled data to pin down the component distributions, and 14 years (1990–2003) where only the mixing weight is unknown. This is far more constrained than a typical mixture problem.

### Honest Caveats

1. **Parametric assumptions matter.** The discriminative approach is essentially nonparametric — the RF can fit any decision boundary. A mixture model requires specifying functional forms for $f_1$ and $f_2$. If the true component distributions are not well-approximated by the chosen family (e.g., Gaussian, Student-t, skew-normal), the estimated mixing weights will be biased. The diagnostic histograms suggest both components are unimodal and roughly log-concave, but they're not perfectly Gaussian — there's right-skew in the first-lien component and fat tails in the second-lien component.

2. **Stability of component distributions across time.** The approach assumes $f_1$ and $f_2$ are either constant or evolve in a parametrizable way. If the *shape* of the within-class distribution changes between 1990 and 2004 (not just the mixing weight), the estimated shares will be wrong. The discriminative classifier has the same implicit assumption — it also assumes the $P(\text{second lien} \mid X)$ relationship learned in 2004–2007 extrapolates backward — but the mixture model makes the assumption more explicit and testable.

3. **Loss of individual-level predictions.** The simplest mixture model only estimates the aggregate share $\pi_t$ per year (or per year $\times$ geography cell). It does not directly classify individual loans. Posterior assignment probabilities $P(\text{second lien} \mid X_i, \hat{\pi}, \hat{f}_1, \hat{f}_2)$ can be computed via Bayes' rule, but these are a derived quantity, not the primary estimand.

4. **Dimensionality.** A univariate mixture on `log_lti` alone is clean and identifiable. Extending to the full feature vector (`log_lti`, `log_county_value_to_loan`, `purchaser_type`, `loan_type`) requires multivariate component distributions, which means more parameters and potentially weaker identification. In practice, the near-sufficiency of `log_lti` for separating the classes suggests that a low-dimensional approach (1–2 continuous features) is the right starting point.

## Implementation Approaches

### Approach 1: Parametric MLE (Recommended Starting Point)

Model each year $t$'s observed `log_lti` distribution as:

$$f_t(x) = \pi_t \cdot f_1(x \mid \theta_1) + (1 - \pi_t) \cdot f_2(x \mid \theta_2)$$

where $f_1, f_2$ are parametric densities (Gaussian to start, then generalize) with parameters $\theta_1, \theta_2$ estimated from the labeled data.

#### Two-stage estimation (simplest)

**Stage 1** — Estimate component parameters from labeled data:

```python
# Using 2004–2007 where lien_status is observed
df_first = df_train[df_train["lien_status"] == 1]["log_lti"]
df_second = df_train[df_train["lien_status"] == 2]["log_lti"]

mu_1, sigma_1 = df_first.mean(), df_first.std()
mu_2, sigma_2 = df_second.mean(), df_second.std()
```

**Stage 2** — For each unlabeled year $t$, maximize:

$$\ell(\pi_t) = \sum_{i=1}^{N_t} \log\bigl[\pi_t \cdot \phi(x_i; \mu_1, \sigma_1) + (1 - \pi_t) \cdot \phi(x_i; \mu_2, \sigma_2)\bigr]$$

over $\pi_t \in [0, 1]$, where $\phi$ is the Gaussian density. This is a one-dimensional optimization per year — `scipy.optimize.minimize_scalar` suffices.

**Posterior classification** of individual loans follows from Bayes' rule:

$$P(\text{second lien} \mid x_i) = \frac{(1 - \hat{\pi}_t) \cdot \phi(x_i; \hat{\mu}_2, \hat{\sigma}_2)}{\hat{\pi}_t \cdot \phi(x_i; \hat{\mu}_1, \hat{\sigma}_1) + (1 - \hat{\pi}_t) \cdot \phi(x_i; \hat{\mu}_2, \hat{\sigma}_2)}$$

#### Advantages
- Transparent, closed-form gradients, trivially fast.
- Component parameters are *pinned* by labeled data, not estimated jointly — avoids label switching entirely.
- Easy to validate: run Stage 2 on 2004–2016 (where true $\pi_t$ is observed) and compare estimated vs. actual shares.

#### Disadvantages
- Gaussian may be a poor fit. Generalize to mixtures of skew-normal, Student-$t$, or kernel density estimates if needed.
- Treats $\theta_1, \theta_2$ as fixed across time, which is testable but may not hold.

### Approach 2: Joint MLE with Time-Varying Parameters

Allow component parameters to drift over time, pooling the labeled and unlabeled years in one likelihood:

$$\ell = \sum_{t \in \text{labeled}} \left[ \sum_{i: y_i = 1} \log f_1(x_i; \theta_{1t}) + \sum_{i: y_i = 2} \log f_2(x_i; \theta_{2t}) \right] + \sum_{t \in \text{unlabeled}} \sum_i \log\bigl[\pi_t \cdot f_1(x_i; \theta_{1t}) + (1 - \pi_t) \cdot f_2(x_i; \theta_{2t})\bigr]$$

with a smoothness penalty or random-walk prior on $\theta_{kt}$ across years to regularise the unlabeled-era parameters.

This is more ambitious but lets the data tell you whether the component shapes are stable. If the smoothness penalty is tight, this collapses to Approach 1. If it's loose, it accommodates secular trends in lending patterns.

### Approach 3: Bayesian Estimation

Place priors on the component parameters and mixing weights:

$$\pi_t \sim \text{Beta}(\alpha, \beta), \quad \mu_k \sim \mathcal{N}(\bar{\mu}_k, \sigma^2_{\mu}), \quad \sigma_k \sim \text{InvGamma}(a, b)$$

with informative priors centered on the labeled-era estimates. Sample via MCMC (e.g., PyMC, Stan, or NumPyro). The main advantage over MLE is that you get full posterior uncertainty on the mixing weights, which propagates into uncertainty on the estimated second-lien share per year — something the discriminative classifier cannot provide at all.

#### Practical notes
- The labeled-era data is so large (millions of loans per year) that the posterior on $\theta_1, \theta_2$ will be extremely tight. The interesting posterior uncertainty will be on $\pi_t$ for unlabeled years, especially early years with smaller HMDA samples.
- Hierarchical structure (random effects on $\theta_{kt}$ across years) is a natural fit.
- Computational cost is modest: the likelihood is a simple mixture of known densities, and only the mixing weight varies per year.

### Approach 4: Semiparametric / Kernel Density Approach

Instead of parametric $f_k$, estimate $\hat{f}_1, \hat{f}_2$ as kernel density estimates (KDEs) from the labeled data, then solve for $\pi_t$ by minimising the distance between the observed KDE and $\pi_t \hat{f}_1 + (1-\pi_t) \hat{f}_2$. This avoids parametric misspecification but requires choosing bandwidths and is less statistically efficient.

## Recommended Implementation Path

### Step 1: Proof of Concept — Univariate Gaussian Mixture on `log_lti`

Use Approach 1 (two-stage MLE) with Gaussian components:

1. Estimate $(\mu_1, \sigma_1)$ and $(\mu_2, \sigma_2)$ from labeled 2004–2007 data.
2. For each year 2004–2016 (where truth is known), estimate $\hat{\pi}_t$ by MLE and compare to actual.
3. Plot estimated vs. actual second-lien share. This is the key validation figure.
4. Compute posterior individual-level probabilities and compare to the RF's `prob_second_lien`.

This is a weekend's work and immediately tells you whether the approach has legs.

### Step 2: Diagnose Parametric Fit

- QQ-plots of `log_lti` within each class against the fitted Gaussian.
- Check whether $(\mu_k, \sigma_k)$ are stable across years 2004–2016. If yes, the two-stage approach is justified. If they drift, you need Approach 2 or 3.
- Try skew-normal or Student-$t$ components if Gaussian is visibly poor.

### Step 3: Extend to Bivariate (`log_lti`, `log_county_value_to_loan`)

The two continuous features are the workhorses. A bivariate normal mixture is only 5 parameters per component (2 means, 2 variances, 1 correlation) plus one mixing weight. Still extremely tractable.

### Step 4 (Optional): Bayesian Version for Uncertainty Quantification

If the paper wants to report credible intervals on the pre-2004 second-lien share (which would be a real contribution — no existing imputation provides uncertainty), implement the Bayesian version with PyMC or NumPyro.

### Step 5: Integrate with Existing Pipeline

This approach is *complementary* to the discriminative classifier, not a replacement. The natural structure is:

- `src/hmda_seconds/mixture.py` — component distribution estimation and mixing-weight MLE.
- `scripts/estimate_mixture.py` — CLI wrapper.
- Validation: run both the RF/logistic classifier and the mixture model on 2004–2016 and compare.
- The mixture model's estimated $\pi_t$ provides an aggregate cross-check on the classifier's implied shares, even if individual-loan predictions still come from the classifier.

## Relationship to Existing Code

The mixture approach would sit alongside, not replace, the existing pipeline:

| Aspect | Current (Discriminative) | Proposed (Mixture) |
|--------|--------------------------|---------------------|
| Estimand | $P(\text{second lien} \mid X)$ | $f(X \mid \text{lien status})$, then $\pi_t$ |
| Base-rate sensitivity | Inherits training-era base rate | Estimates base rate from data |
| Individual predictions | Direct | Via Bayes' rule (derived) |
| Parametric assumptions | None (RF) or linear (logistic) | Requires specifying $f_1, f_2$ |
| Uncertainty quantification | None | Natural (Bayesian version) |
| Aggregate share validation | Indirect (continuity check) | Direct estimand |
| Feature handling | Arbitrary dimension | Best in low dimension |
| Existing code reuse | `clean.py` (data prep), `config.py` | Same data prep; new estimation module |

## Connection to the Literature

This is a well-studied problem in statistics:

- **Finite mixture models with known components** — when $f_1, f_2$ are estimated from labeled data and only $\pi$ is unknown, this is the classical "mixture proportion estimation" problem. See Blanchard, Lee, and Scott (2010, "Semi-supervised novelty detection," *JMLR*) for theory.
- **Prevalence estimation / quantification learning** — the ML literature calls the problem of estimating class proportions in an unlabeled sample "quantification learning" or "prevalence estimation." The two-stage MLE approach above is equivalent to the "adjusted classify and count" method. See González et al. (2017, "A Review on Quantification Learning," *ACM Computing Surveys*).
- **Partial identification / ecological inference** — if you don't want to assume $f_k$ is exactly the same pre- and post-2004, the mixture decomposition still bounds the mixing weight under weaker assumptions (e.g., the components' supports don't fully overlap). See Horowitz and Manski (1995) for the partial identification flavor.

The connection to quantification learning is particularly apt: the existing pipeline is doing "classify and count" (classify every loan, then count the predicted second liens), which is known to be biased when the test-set class proportions differ from training — exactly the problem observed. The mixture MLE is the textbook fix.

## Concrete Code Sketch

Here is a minimal implementation that could serve as the starting point for `src/hmda_seconds/mixture.py`:

```python
"""Mixture-model estimation of second-lien shares.

Estimates the first- and second-lien component distributions of log_lti
from labeled data, then estimates the mixing weight (first-lien share)
for each unlabeled year by maximum likelihood.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats
from dataclasses import dataclass

from . import config


@dataclass
class GaussianComponent:
    """Parameters of a univariate Gaussian component."""
    mu: float
    sigma: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return stats.norm.logpdf(x, loc=self.mu, scale=self.sigma)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return stats.norm.pdf(x, loc=self.mu, scale=self.sigma)


@dataclass
class MixtureComponents:
    """Estimated component distributions from labeled data."""
    first_lien: GaussianComponent
    second_lien: GaussianComponent

    @classmethod
    def from_labeled_data(
        cls, df: pd.DataFrame, feature: str = "log_lti"
    ) -> "MixtureComponents":
        first = df.loc[df[config.LABEL_VAR] == config.FIRST_LIEN_CLASS, feature]
        second = df.loc[df[config.LABEL_VAR] == config.SECOND_LIEN_CLASS, feature]
        return cls(
            first_lien=GaussianComponent(mu=first.mean(), sigma=first.std()),
            second_lien=GaussianComponent(mu=second.mean(), sigma=second.std()),
        )


def mixture_loglik(
    pi: float, x: np.ndarray, comp: MixtureComponents
) -> float:
    """Log-likelihood of the mixture f(x) = pi * f1(x) + (1-pi) * f2(x)."""
    log_f1 = comp.first_lien.logpdf(x)
    log_f2 = comp.second_lien.logpdf(x)
    # Use logsumexp for numerical stability
    log_mix = np.logaddexp(
        np.log(pi) + log_f1,
        np.log(1.0 - pi) + log_f2,
    )
    return float(np.sum(log_mix))


def estimate_mixing_weight(
    x: np.ndarray, comp: MixtureComponents
) -> tuple[float, float]:
    """MLE of first-lien share pi for an unlabeled sample.

    Returns (pi_hat, loglik_at_optimum).
    """
    result = optimize.minimize_scalar(
        lambda pi: -mixture_loglik(pi, x, comp),
        bounds=(1e-6, 1.0 - 1e-6),
        method="bounded",
    )
    return result.x, -result.fun


def posterior_second_lien_prob(
    x: np.ndarray, pi: float, comp: MixtureComponents
) -> np.ndarray:
    """Posterior P(second lien | x_i) via Bayes' rule."""
    log_f1 = comp.first_lien.logpdf(x)
    log_f2 = comp.second_lien.logpdf(x)
    log_num = np.log(1.0 - pi) + log_f2
    log_den = np.logaddexp(np.log(pi) + log_f1, log_num)
    return np.exp(log_num - log_den)


def estimate_shares_by_year(
    df: pd.DataFrame,
    comp: MixtureComponents,
    years=None,
    feature: str = "log_lti",
) -> pd.DataFrame:
    """Estimate first-lien share pi_t for each year by MLE."""
    if years is None:
        years = config.APPLY_YEARS

    rows = []
    for year in years:
        sub = df.loc[df["year"] == year, feature].dropna()
        if sub.empty:
            continue
        pi_hat, loglik = estimate_mixing_weight(sub.to_numpy(), comp)
        actual = None
        if config.LABEL_VAR in df.columns:
            labeled = df.loc[
                (df["year"] == year) & df[config.LABEL_VAR].notna()
            ]
            if not labeled.empty:
                actual = (
                    labeled[config.LABEL_VAR] == config.FIRST_LIEN_CLASS
                ).mean()
        rows.append({
            "year": year,
            "pi_hat_first_lien": pi_hat,
            "actual_first_lien_share": actual,
            "second_lien_share_hat": 1.0 - pi_hat,
            "loglik": loglik,
            "n": len(sub),
        })

    return pd.DataFrame(rows).set_index("year")
```

## Summary

The mixture-model idea is sound, well-motivated by the specific failure mode of the existing approach (base-rate sensitivity), and feasible to implement. The key insight — that the unlabeled data's own distributional shape identifies the mixing weight — is correct and sidesteps the main weakness of discriminative classification for this problem. The labeled data provides unusually strong anchoring for the component parameters, making this a much easier mixture problem than the typical unsupervised case.

The recommended path is to implement the simplest version first (univariate Gaussian on `log_lti`, two-stage MLE), validate it against the 13 years of labeled data, and then extend as needed. This can coexist with the existing classifier pipeline — the mixture model provides aggregate share estimates and the discriminative model provides individual-level predictions, and the two serve as cross-checks on each other.
