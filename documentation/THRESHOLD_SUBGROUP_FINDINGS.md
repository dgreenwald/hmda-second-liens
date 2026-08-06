# Step 7 Threshold and Subgroup Findings

## Canonical threshold

With reverse-validation metrics averaged within horizon and then equally across
horizons, performance at threshold 0.5 is:

| Metric | Raw logistic | Known-prior mixture |
| --- | ---: | ---: |
| Accuracy | 0.9026 | 0.9545 |
| Second-lien precision | 0.8891 | 0.8718 |
| Second-lien recall | 0.3737 | 0.7242 |
| Second-lien F1 | 0.4896 | 0.7884 |
| Hard-classified second-lien share | 0.0480 | 0.1138 |
| Observed second-lien share | 0.1363 | 0.1363 |
| Average precision | 0.8787 | 0.8702 |

The mixture adjustment sacrifices 1.7 percentage points of precision but nearly
doubles recall. Its F1 remains between 0.750 and 0.807 at every reverse horizon;
the raw model's F1 falls from 0.754 at horizon one to roughly 0.36-0.39 at
horizons six through nine. Mean average precision is slightly lower for the
mixture score, showing that the large hard-classification improvement comes
primarily from its target-specific probability level rather than improved
ranking.

The hard-classified mixture share has the same remaining long-horizon
underprediction documented for the mean probability. No evidence supports
replacing the canonical 0.5 threshold with a threshold selected from these
validation labels.

## Forward robustness

Across 2008-2016, with years equally weighted:

| Metric | Raw logistic | Known-prior mixture |
| --- | ---: | ---: |
| Accuracy | 0.9826 | 0.9898 |
| Second-lien precision | 0.5299 | 0.8385 |
| Second-lien recall | 0.9128 | 0.6013 |
| Second-lien F1 | 0.6691 | 0.6989 |
| Hard-classified second-lien share | 0.0334 | 0.0141 |
| Observed second-lien share | 0.0196 | 0.0196 |
| Average precision | 0.8085 | 0.8200 |

The mixture adjustment corrects the raw model's post-crisis false-positive
problem. Its F1 is lower in 2008 and 2009 but higher in every year from 2010
through 2016. This tradeoff is consistent with the adjustment responding to the
post-crisis decline in prevalence.

## Subgroups

The mixture estimator improves reverse F1 in 35 of the 38 reported subgroups.
It improves Brier score in 21 groups; most Brier deteriorations occur in groups
whose second-lien prevalence is below 0.5 percent and are small in absolute
terms.

Regional performance is broad rather than concentrated:

| Region | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Midwest | 0.842 | 0.792 | 0.815 |
| Northeast | 0.853 | 0.697 | 0.763 |
| South | 0.853 | 0.772 | 0.809 |
| West | 0.929 | 0.655 | 0.762 |

Loan type 1, which contains most second liens, has precision 0.872, recall 0.723,
and F1 0.788. Loan types 2-4 have second-lien prevalence below 0.5 percent;
their lower hard-classification metrics should be interpreted together with
their very low Brier errors. Purchaser types 2 and 3 likewise have prevalence
below 0.1 percent. Purchaser type 4 has as few as 187 observations in a cell and
is too sparse for a stable conclusion.

The continuous-variable deciles locate the main shape limitation. The first
`log_lti` target decile has second-lien prevalence 0.728 and F1 0.882. In the
second and third deciles, prevalence falls to 0.502 and 0.123 while recall falls
to 0.417 and 0.118. Higher LTI deciles contain almost no second liens, so their
near-zero hard-classification recall has little effect on aggregate error. For
county value-to-loan, the mixture generates its largest improvements in deciles
8-10, where second liens are concentrated.

## Decision

Retain 0.5 as the canonical hard-classification threshold. The known-prior
mixture adjustment materially improves the precision-recall operating point and
performs credibly across regions and economically important subgroups. Advance
to Step 8 plausibility checks without additional threshold or subgroup tuning.
