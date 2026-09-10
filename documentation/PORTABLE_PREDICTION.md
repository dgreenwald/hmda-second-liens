# Portable logistic prediction

Prediction requires Python 3.11+ and NumPy. Install the base package and supply a
separately exported `model.json`, or copy the standalone export's `predict.py` and
`model.json` into the receiving project. Neither route requires scikit-learn, pandas,
SciPy, or the dataset loaders. A pandas DataFrame is accepted if the receiving project
already uses pandas.

## Install the prediction package

From a local checkout, `pip install .` installs only the package and NumPy. Alternatively,
install a wheel built in the development environment:

```bash
# In the development checkout:
python -m build --wheel
# In the receiving project's environment:
pip install /path/to/hmda_second_liens-0.1.0-py3-none-any.whl
```

To install from Git without publishing to PyPI, replace `COMMIT_SHA` with the full
revision containing the desired code:

```bash
pip install "hmda-second-liens @ git+https://github.com/dgreenwald/hmda-second-liens.git@COMMIT_SHA"
```

Use the package import in the receiving project:

```python
from hmda_seconds.portable_predict import load_model

model = load_model("model.json")
probabilities = model.predict_proba_second_lien(inputs, year=2001)
```

The JSON remains separately versioned: keep its provenance and annual coverage with
the consuming project. Installation does not bundle, download, or select a fitted model.
The wheel includes research source modules, but their dependencies are optional.

For research and export, use `pip install ".[train]"` from the checkout. For editable
development, use `pip install -e ".[train,dev]"` or `make install`. The `train` extra
includes data loading, fitting, plotting, and export dependencies; `dev` contains only
test, lint, and build tools. Fresh research environments must request `train` explicitly.

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
