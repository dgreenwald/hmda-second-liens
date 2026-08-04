# Full-sample estimator benchmark

## Design

The benchmark fits logistic regression and the Random Forest on exactly the same 20,282,918
HMDA loans from 2004--2007. Both use `log_lti`, `log_county_value_to_loan`, purchaser-type
indicators, and loan-type indicators. The Random Forest uses the production configuration of
50 trees, maximum depth 10, and random seed 17. Logistic regression uses the formal baseline
configuration in `config.py`. No temporal validation observation is used for estimation.

Both models are evaluated on the same 24,017,830 cleaned loans from 2008--2016. Hard
classifications use a 0.5 second-lien probability threshold. Metrics are reported separately
for every validation year and for the pooled sample.

The historical Random Forest trained on a random 25% of the old training extract remains at
`output/model/rf_fit.pkl`, with its earlier results under `output/tables/out_of_time_metrics.csv`.
It was not overwritten. Because it also uses the retired native-HPI feature and the old sample
construction, it is retained as a historical specification rather than mixed into the fair
headline comparison.

## Pooled results

| Metric | Full-sample RF | Full-sample logit | Logit minus RF |
|---|---:|---:|---:|
| Accuracy | 0.983763 | 0.985723 | +0.001960 |
| Second-lien precision | 0.551734 | 0.588880 | +0.037146 |
| Second-lien recall | 0.899954 | 0.891466 | -0.008488 |
| Second-lien F1 | 0.684079 | 0.709248 | +0.025169 |
| ROC-AUC | 0.991166 | 0.990270 | -0.000896 |
| Average precision | 0.762845 | 0.823678 | +0.060833 |
| Log loss | 0.039718 | 0.045478 | +0.005760 |
| Brier score | 0.011991 | 0.011244 | -0.000747 |
| Calibration intercept | -1.674187 | -2.149617 | -0.475430 |
| Calibration slope | 1.054476 | 0.662717 | -0.391759 |

Logit has higher accuracy, second-lien precision and F1, average precision, and a lower Brier
score. RF has slightly higher recall and ROC-AUC, lower log loss, and a calibration slope much
closer to one. Both models substantially overpredict the pooled second-lien share: the
observed share is 1.953%, compared with mean probabilities of 3.507% for RF and 3.467% for
logit. The negative calibration intercepts therefore matter even though discrimination is
excellent.

The yearly pattern is consistent. Logit has higher accuracy, second-lien F1, average
precision, and lower Brier score in all nine validation years. RF has higher ROC-AUC and lower
log loss in all nine years. Logit's yearly accuracy advantage ranges from 0.125 to 0.271
percentage points; its second-lien F1 advantage ranges from 0.007 to 0.035.

## Paired comparison

On the pooled validation sample, RF alone is correct for 27,445 observations and logit alone
is correct for 74,514. The net accuracy advantage is 47,069 classifications, or 0.196
percentage points. McNemar's p-value is numerically zero at this sample size. The effect size,
not the p-value, is the informative quantity.

## Computational comparison

Logit fits in 28.0 seconds and predicts the full validation sample in 8.3 seconds. RF fits in
200.2 seconds and predicts in 13.8 seconds. The serialized logit is 833 bytes; the RF is
4,248,734 bytes. Timings are machine-specific, but the relative simplicity advantage is
unambiguous.

## Decision

The full-sample RF does not deliver a broad or operationally decisive improvement over
logistic regression. Logistic regression is therefore the provisional primary estimator,
based on its consistently better hard-classification performance, second-lien F1, average
precision, Brier score, speed, and compactness. The RF remains a substantive robustness model
because its log loss and calibration slope are better, not merely a ceremonial validation
check.

This decision is provisional. The next stages should predeclare the temporal model-selection
protocol and then optimize the logistic specification. The marked base-rate error in both
models should be handled in the later calibration stage rather than silently corrected here.

## Reproducible outputs

- `output/tables/benchmark_model_summary.csv`
- `output/tables/benchmark_metrics_by_year.csv`
- `output/tables/benchmark_metrics_pooled.csv`
- `output/tables/benchmark_mcnemar_by_year.csv`
- `output/tables/benchmark_mcnemar_pooled.csv`
- `output/model/benchmark_rf_full.pkl`
- `output/model/benchmark_logistic_full.pkl`

Run `make benchmark-data` to construct the preserved common training extract and
`make benchmark-estimators` to refit and evaluate both estimators.
