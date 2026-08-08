"""Aggregate comparisons of source-specific HMDA parquet files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

FIELD_ALIASES = {
    "year": ("activity_year", "as_of_year", "asof_date"),
    "loan_type": ("loan_type",),
    "loan_purpose": ("loan_purpose", "loan_purp"),
    "occupancy": ("occupancy_type", "owner_occupancy", "occupancy"),
    "loan_amount": ("loan_amount", "loan_amount_000s", "loan_amt"),
    "action_taken": ("action_taken",),
    "msa": ("derived_msa_md", "msa_md", "msamd", "prop_msa"),
    "state": ("state_code",),
    "county": ("county_code",),
    "census_tract": ("census_tract", "census_tract_number"),
    "income": ("income", "applicant_income_000s", "app_income"),
    "purchaser_type": ("purchaser_type",),
    "lien_status": ("lien_status",),
}
CATEGORICAL_FIELDS = {
    "year",
    "loan_type",
    "loan_purpose",
    "occupancy",
    "action_taken",
    "state",
    "purchaser_type",
    "lien_status",
}
DISTINCT_FIELDS = CATEGORICAL_FIELDS | {"msa", "county", "census_tract"}
MISSING_TOKENS = {None, "", "NA", "N/A", "NULL", "null"}


def compare_sources(
    year: int,
    sources: tuple[str, ...],
    *,
    data_dir: str | Path,
    batch_size: int = 250_000,
) -> dict:
    """Return schema, size, completeness, and coverage aggregates."""
    reports = {
        source: _source_report(year, source, data_dir, batch_size) for source in sources
    }
    column_sets = {source: set(report["columns"]) for source, report in reports.items()}
    common = sorted(set.intersection(*column_sets.values()))
    return {
        "year": int(year),
        "sources": reports,
        "schema_comparison": {
            "common_columns": common,
            "source_specific_columns": {
                source: sorted(columns - set(common))
                for source, columns in column_sets.items()
            },
            "common_dtype_differences": {
                column: {
                    source: reports[source]["column_types"][column]
                    for source in sources
                }
                for column in common
                if len(
                    {reports[source]["column_types"][column] for source in sources}
                )
                > 1
            },
        },
    }


def write_report(report: dict, output: str | Path | None = None) -> str:
    """Serialize an aggregate comparison as stable JSON."""
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return text


def _source_report(year: int, source: str, data_dir: str | Path, batch_size: int) -> dict:
    path = Path(data_dir) / "parquet" / source / str(year) / "lar.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"HMDA parquet file not found: {path}")
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    columns = schema.names
    selected = {
        field: next((name for name in aliases if name in columns), None)
        for field, aliases in FIELD_ALIASES.items()
    }
    scan_columns = sorted({name for name in selected.values() if name is not None})
    non_null = Counter()
    distinct = {field: set() for field in DISTINCT_FIELDS}
    frequencies = {field: Counter() for field in CATEGORICAL_FIELDS}

    for batch in parquet.iter_batches(batch_size=batch_size, columns=scan_columns):
        for field, column in selected.items():
            if column is None:
                continue
            array = batch.column(batch.schema.get_field_index(column))
            if field in DISTINCT_FIELDS:
                present = [
                    value
                    for value in array.to_pylist()
                    if value not in MISSING_TOKENS
                ]
                non_null[field] += len(present)
                distinct[field].update(present)
                if field in CATEGORICAL_FIELDS:
                    frequencies[field].update(str(value) for value in present)
            else:
                non_null[field] += len(array) - array.null_count

    rows = parquet.metadata.num_rows
    coverage = {}
    for field, column in selected.items():
        if column is None:
            coverage[field] = {"column": None}
            continue
        item = {
            "column": column,
            "non_null": non_null[field],
            "non_null_share": non_null[field] / rows if rows else None,
        }
        if field in DISTINCT_FIELDS:
            item["distinct_non_null"] = len(distinct[field])
        if field in CATEGORICAL_FIELDS:
            item["counts"] = dict(sorted(frequencies[field].items()))
        coverage[field] = item

    return {
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "rows": rows,
        "row_groups": parquet.metadata.num_row_groups,
        "column_count": len(columns),
        "columns": columns,
        "column_types": {field.name: str(field.type) for field in schema},
        "coverage": coverage,
    }
