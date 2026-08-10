# Data

Three inputs feed the classifier: raw HMDA loan-level records, the FHFA county house-price
index, and Zillow Research county ZHVI. FHFA supplies the long appreciation history; Zillow
puts each county series on a comparable dollar level. Nothing under `data/raw/` is committed
to this repo; `data/public/` holds small vendored public inputs.

## HMDA loan-level records

### Parallel raw-to-parquet conversion

Generate a Slurm array with one task for every supported year/source pair:

```bash
python scripts/generate_hmda_parquet_slurm.py \
    --account torch_pr_609_general \
    --max-concurrent 8
```

This writes a JSON manifest and Slurm script under the repository's `output/slurm/hmda/`
directory, regardless of the shell's current directory. By default it does not submit.
After inspection, submit with `sbatch output/slurm/hmda/hmda_parquet_jobs.slurm`. Repeated
`--year` and `--source` options select a subset (for example, `--year 2004 --source nara`).
Existing valid parquet files are skipped unless `--overwrite` is passed.
Log paths are uniquely keyed by the Slurm array and task IDs.
The generated script uses absolute manifest and log paths, so its `.out` and `.err` files are
written beside the manifest and Slurm script regardless of the directory used for `sbatch`.
By default, tasks use the same Python environment and `py_tools` data configuration as a simple
batch script. Pass `--activate PATH` or `--data-dir PATH` only when the batch environment needs
an explicit override.
Use `--source all` to request every available source explicitly; the generator expands it into
separate year/source array tasks. Omitting `--source` has the same behavior.

Compare the converted CFPB and FFIEC three-year releases for 2017 with aggregate-only output:

```bash
python scripts/compare_hmda_sources.py \
    --output output/tables/hmda_2017_source_comparison.json
```

The report includes row and column counts, file sizes, common and source-specific variables,
type differences, field completeness, geographic cardinality, and categorical coverage. It
does not retain loan-level values.

Add `--submit` to submit the generated script immediately with `sbatch`. Without this flag,
generation remains inspection-only. For example, a single conversion can be generated and
submitted with `--year 2003 --source nara --submit`.

The cleaning and estimation pipeline reads these source-specific converted files through
`py_tools.datasets.hmda.load(year=..., source="auto")`. Set `HMDA_SECONDS_HMDA_DATA_DIR` to
the root containing the `raw/` and `parquet/` directories when it differs from
`$PY_TOOLS_DATA_DIR/hmda`. The automatic source policy uses NARA through 2014, CFPB for
2015--2016, and the declared FFIEC releases thereafter.

Before model selection, `make selection-data` writes narrow cleaned 2004--2016 extracts under
`$HMDA_SECONDS_EXTERNAL_DIR/intermediate/logistic_selection`. These are derived inputs—not a
second download—and contain only the filtered sample and estimator columns needed by the
parallel grid.

**Sourcing raw LAR files for years 1990-2016 is not yet solved end-to-end and needs a follow-up
pass before this repo can be reproduced from scratch by someone without an existing cache.**
What's confirmed so far:

- 2007-2017: [CFPB's historic HMDA data page](https://www.consumerfinance.gov/data-research/hmda/historic-data/)
  serves per-year national LAR files at a confirmed URL pattern
  (`https://files.consumerfinance.gov/hmda-historic-loan-data/hmda_<year>_nationwide_..._labels.zip`).
  **Caution:** the convenient pre-packaged file is already filtered to first-lien
  owner-occupied records — i.e. second liens pre-stripped. That's fine for the 2008-2016
  *validation* years (no training signal needed from second liens there beyond checking
  predicted vs. actual shares), but **unsuitable for the 2004-2007 training window**, which
  needs the unfiltered file with `lien_status` intact for both classes.
- 2011+: `ffiec.gov/hmdarawdata/LAR/National/<year>HMDALAR - National.zip` is the URL pattern
  used historically in this project (`~/research/frm/empirical/loan_level/hmda/download.py`),
  though direct automated fetches to ffiec.gov were blocked (403) as of this writing — worth
  retrying manually in a browser.
- Pre-2007 (needed for the 2004-2007 training window and the full 1990-2003 application
  range): FFIEC's own historic raw-data page states LAR/TS raw data are available for
  1990-2008 via NTIS (`ntis.gov`), a paid/contact-based distributor, not a direct download.
  A promising unverified lead: [National Archives, "HMDA Data Files, 1981-2014"](https://catalog.archives.gov/search-within/2456161?levelOfDescription=fileUnit&limit=20&sort=naId%3Aasc)
  — this is a JS-rendered search UI that couldn't be scraped automatically; needs manual
  browsing to confirm which years/files it actually holds and whether they're directly
  downloadable. openICPSR project 151921 ("Historical Home Mortgage Disclosure Act (HMDA)
  Data") is a second unverified lead.

Do not commit anything derived from raw HMDA microdata beyond the aggregated/binned outputs
this letter is designed to release publicly.

## FHFA county house price index

`data/public/HPI_AT_BDL_county.xlsx` — vendored directly (public FHFA "All-Transactions House
Price Index" at the county level, ~3.4MB). Loaded via
`py_tools.datasets.fhfa.load('county', data_dir=...)` pointed at `data/public/`; the derived
parquet cache that call writes alongside the source file is regenerable and gitignored.

## Zillow county ZHVI

The pipeline uses Zillow's all-homes, middle-tier, smoothed and seasonally adjusted monthly
county ZHVI. The source vintage is pinned by `HMDA_SECONDS_ZILLOW_VINTAGE` and defaults to
`202608`. `py_tools.datasets.zillow` stores the raw CSV and its derived parquet under the
`PY_TOOLS_DATA_DIR` Zillow directory; loading never downloads implicitly.

Fetch the pinned input once with:

```bash
make download-zillow
```

The target uses `HMDA_SECONDS_ZILLOW_VINTAGE` and the `py_tools` data root. Run
`python scripts/download_zillow.py --help` for explicit data-directory and overwrite options.

Run `make county-values` to regenerate the public scaling diagnostics, and
`make county-value-coverage` to audit the scaled panel against the local HMDA files. The
estimation and robustness choices are documented in `COUNTY_VALUE_SCALING.md`.
