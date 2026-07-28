# Data

Two inputs feed the classifier: the raw HMDA loan-level records, and the FHFA county house
price index (used to build the `log_ltv` proxy feature). Nothing under `data/raw/` is
committed to this repo; `data/public/` holds small vendored public inputs.

## HMDA loan-level records

The pipeline expects one cleaned parquet per year at
`${HMDA_SECONDS_YEARLY_DIR:-data/raw/hmda/save}/hmda<year>.parquet`, in the schema produced by
`py_tools.datasets.hmda.load_hmda` (see that module for the exact fixed-width layouts, which
change in 2004). `HMDA_SECONDS_YEARLY_DIR` (see `.env.example`) can point this at an
already-materialized local cache instead.

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

Until this is resolved, local development uses an existing pre-built cache pointed to via
`HMDA_SECONDS_YEARLY_DIR` (see `.env.example`). Do not commit anything derived from raw HMDA
microdata beyond the aggregated/binned outputs this letter is designed to release publicly.

## FHFA county house price index

`data/public/HPI_AT_BDL_county.xlsx` — vendored directly (public FHFA "All-Transactions House
Price Index" at the county level, ~3.4MB). Loaded via
`py_tools.datasets.fhfa.load('county', data_dir=...)` pointed at `data/public/`; the derived
parquet cache that call writes alongside the source file is regenerable and gitignored.
