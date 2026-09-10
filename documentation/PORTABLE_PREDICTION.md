# Portable logistic prediction

Prediction requires Python 3.11+ and NumPy. The base package includes both benchmark
models with saved annual intercepts for 1990–2016. Alternatively, load an explicit JSON
or copy the standalone export's `predict.py` and `model.json` into the receiving project.
Neither route requires scikit-learn, pandas,
SciPy, or the dataset loaders. A pandas DataFrame is accepted if the receiving project
already uses pandas.

## Install the prediction package

From a local checkout, `pip install .` installs only the package and NumPy. Alternatively,
install a wheel built in the development environment:

```bash
# In the development checkout:
python -m build --wheel
# In the receiving project's environment:
pip install /path/to/hmda_second_liens-0.2.0-py3-none-any.whl
```

To install from Git without publishing to PyPI, replace `COMMIT_SHA` with the full
revision containing the desired code:

```bash
pip install "hmda-second-liens @ git+https://github.com/dgreenwald/hmda-second-liens.git@COMMIT_SHA"
```

Use the package import in the receiving project:

```python
from hmda_seconds import load_benchmark

model = load_benchmark()  # feature_set="core"
probabilities = model.predict_proba_second_lien(inputs, year=2001)
```

Package version 0.2.0 identifies the bundled benchmark snapshot. Pin the package version
or Git revision in the consuming project. Loading a benchmark reads packaged resources
without network access, environment configuration, or training imports. Each load returns
a fresh model. Unknown feature-set names and years outside 1990–2016 raise errors.
For custom artifacts, `hmda_seconds.portable_predict.load_model(path)` remains available.
The wheel includes research source modules, but their dependencies are optional.

For research and export, use `pip install ".[train]"` from the checkout. For editable
development, use `pip install -e ".[train,dev]"` or `make install`. The `train` extra
includes data loading, fitting, plotting, and export dependencies; `dev` contains only
test, lint, and build tools. Fresh research environments must request `train` explicitly.

## HMDA-only model

The same NumPy-only predictor supports the selected restricted specification,
`hmda_only__spline_lti__none`, with ridge `C=1`, trained on 2004–2007. It uses a
`log_lti` spline and purchaser- and loan-type indicators, with no interactions.
It requires only `log_lti`, `purchaser_type`, and `loan_type`; county values are
neither required nor used. Load the restricted JSON to select this model:

```python
from hmda_seconds import load_benchmark

model = load_benchmark(feature_set="hmda_only")
print(model.required_columns)
# ('log_lti', 'purchaser_type', 'loan_type')
probabilities = model.predict_proba_second_lien({
    "log_lti": [0.4, 0.8],
    "purchaser_type": [0, 1],
    "loan_type": [1, 2],
}, year=2001)
```

Export the restricted **known-source-prior** fit (not the raw selected pickle):

```bash
python scripts/export_portable_logistic.py \
  --feature-set hmda_only \
  --model /path/to/saved/hmda_only_known_source_prior_fit.pkl
```

`--model` must point to the saved 2004–2007 logistic density-ratio artifact used by
the historical comparison, with its metadata sidecar. Its exact path is recorded
in the forward comparison shard. Without `--model`, the exporter looks in the
configured mixture-fold directory for
`known_source_prior__hmda_only__spline_lti__none__c_1__train_2004_2007.pkl`.

For `--feature-set hmda_only`, annual estimates default to
`output/tables/historical_model_family_annual.csv` and output defaults to
`output/model/portable_logistic_hmda_only/`, separate from the core export.
Override these with `--annual-file` and `--output-dir`; `--years` works for both models.
The exporter also accepts the original annual-table schema with
`mixture_model_id`, `mixture_model_sha256`, and `mixture_optimizer_converged`.
The equivalent Make invocation is
`make export-portable-logistic EXPORT_PORTABLE_LOGISTIC_FLAGS="--feature-set hmda_only --model /path/to/fit.pkl"`.

The exporter selects only the matching predictor-set/logistic rows from the
historical table and verifies the model identity and artifact digest. Restricted
predictions use the restricted model's estimated annual shares and intercepts.
They never borrow intercepts from the unrestricted model.

New historical comparison runs record `model_sha256` in each annual row. Older
immutable historical shards without matching digests are rejected on reuse and
remain untouched. Generate historical estimates into a **new output root** using
the saved comparison fits, then aggregate them before export. For example,
`make historical-model-family-cluster HISTORICAL_MODEL_FAMILY_FLAGS="--output-root output/historical_model_family_with_digests"`
prepares that workflow without submitting it. Follow the printed submission
instructions from the historical comparison workflow. Do not manually attach a
digest to an old annual table: its generating artifact cannot be verified that way.

The restricted fit was selected on the common sample that retained county-value
availability restrictions. Omitting the county predictor at inference does not
change the population used to estimate its saved annual intercepts; see the
[HMDA-only selection protocol](HMDA_ONLY_MODEL_SELECTION_PROTOCOL.md).
Previously exported core JSON files remain compatible with the updated predictor.

## Refresh the packaged snapshot

In a training/development checkout, explicitly import the current results snapshot:

```bash
make bundle-logistic-models RESULTS_ROOT=/path/to/current/results
```

This resolves the two 2004–2007 forward logistic artifacts beneath the supplied root,
validates their metadata and payload hashes, and checks all 27 annual historical shards
against `tables/historical_model_family_annual.csv`. It never searches `sync_backups`,
refits a model, estimates new shares, or edits the source results.

Both models are compared to the original fits on synthetic category combinations,
spline knots, and tails for every year at probability tolerance `1e-12`. The generated
JSONs are staged under the configured `output/model/benchmark_release/` and then copied
to the version-controlled package resources. To keep staging local when your environment
points output at the source results, pass
`HMDA_SECONDS_OUTPUT_DIR=/absolute/path/to/checkout/output` to Make.

The initial release uses the latest main results snapshot reviewed for this release.
Its historical annual outputs identify the models by ID but predate required artifact
digests in annual rows. The snapshot importer records that association honestly, along
with SHA-256 hashes and relative filenames for all contributing model, metadata,
comparison, annual-table, and historical-shard files. It does not claim those older
annual rows cryptographically bind the fitted artifacts. This explicitly selected
release-import path does not weaken the ordinary exporter's digest requirements.

Review and commit both generated resources together, and release refreshed snapshots
under a new package version. `make test-lightweight-install` verifies the wheel built
from a source distribution, including both bundled models, in a fresh NumPy-only
environment. No result tables, pickles, or loan-level observations are packaged.

The predictor uses one fixed coefficient vector from the final known-source-prior
ridge logistic model, with saved target-year intercepts:

```text
score = transformed features × fixed coefficients + intercept[year]
probability = 1 / (1 + exp(−score))
intercept[year] = density-ratio offset + log(annual share / (1 − annual share))
```

These are the mixture model's coefficients, not the separately fitted raw logistic
coefficients. Annual shares come from the full historical application sample. Applying
the predictor to a subset or in batches retains those intercepts and does not estimate
new shares. The mean probability in a subset need not equal the full-year mixture share.

## Export in this repository

In a training installation, restore the saved final known-source-prior model and its `.metadata.json` sidecar, plus
the raw fit needed by the historical application. Generate annual estimates with
`make plausibility-checks`, then run:

```bash
make export-portable-logistic
```

The default output is `output/model/portable_logistic/`, respecting
`HMDA_SECONDS_OUTPUT_DIR`. The default model uses the frozen
`spline_lti__purchaser_type` specification, `C=0.1`, trained on 2004–2007; the default
year coverage is 1990–2016. To select paths or a subset of years:

```bash
python scripts/export_portable_logistic.py \
  --model output/model/mixture_folds/known_source_prior__spline_lti__purchaser_type__c_0p1__train_2004_2007.pkl \
  --annual-file output/tables/step8_annual_plausibility.csv \
  --output-dir output/model/portable_logistic \
  --years 2000 2001 2002 2003
```

The exporter never fits models or estimates shares. It requires annual rows with
matching model identity and payload digest, successful optimization, and valid shares.
Older historical CSVs without provenance must be regenerated through
`make plausibility-checks`; that stage reuses the saved fits. Boundary estimates remain
exportable and their diagnostics are retained in JSON. Unsupported specifications and
missing years are errors.

## Predict in the receiving project

```python
from predict import load_model

model = load_model("model.json")
inputs = {
    "log_lti": [0.4, 0.8],
    "log_county_value_to_loan": [1.0, 1.2],
    "purchaser_type": [0, 1],
    "loan_type": [1, 2],
}
probabilities = model.predict_proba_second_lien(inputs, year=2001)
labels = model.predict(inputs, year=2001)
# For mixed-year batches: year=[2000, 2001]
```

`inputs` may also be a DataFrame. Columns must be finite, equally sized
one-dimensional arrays. Outputs are NumPy arrays in input order. Labels are `2` for
probabilities at or above 0.5, and `1` otherwise. An unknown year raises an error;
there is no interpolation or fallback intercept.

Prepare inputs using the original definitions:

| Column | Definition |
| --- | --- |
| `log_lti` | Natural log of loan amount divided by applicant income, in the same units. The original HMDA columns are both in thousands of dollars. |
| `log_county_value_to_loan` | Natural log of county value in dollars divided by loan amount in dollars (`1000 × loan_amt` for the original HMDA data). |
| `purchaser_type` | Integer HMDA codes 0–9; reference category 0. |
| `loan_type` | Integer HMDA codes 1–4; reference category 1. |

County values must use the same county/year FHFA HPI series scaled to Zillow dollar
values as the fitted model's inputs. See `clean.build_county_value_panel` and
`county_values.build_county_value_panel` for the existing construction; a raw HPI index
is not a dollar value. Preserve the source data vintage and scaling convention when
preparing inputs elsewhere. Sample construction and county-value preparation are
outside the portable module. The intended population remains the project's
owner-occupied home-purchase sample, including subsets of that population.

The JSON stores the fitted spline knots, centering, scaling, category ordering, annual
shares and intercepts, and source provenance. Do not refit transformations on prediction
data. No loan-level observations are included in the export; prediction returns arrays
in memory and does not write them to disk.
