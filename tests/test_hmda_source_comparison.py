import json

import pyarrow as pa
import pyarrow.parquet as pq

from hmda_seconds.hmda_source_comparison import compare_sources, write_report


def test_compare_sources_reports_schema_sample_size_and_coverage(tmp_path):
    cfpb = tmp_path / "parquet/cfpb/2017/lar.parquet"
    three_year = tmp_path / "parquet/ffiec_three_year/2017/lar.parquet"
    cfpb.parent.mkdir(parents=True)
    three_year.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "loan_purp": [1, 1, 2],
                "state_code": ["01", "01", "NA"],
                "cfpb_only": [1, 2, 3],
            }
        ),
        cfpb,
    )
    pq.write_table(
        pa.table(
            {
                "loan_purpose": [1, 2],
                "state_code": ["01", "02"],
                "three_year_only": [1, 2],
            }
        ),
        three_year,
    )

    report = compare_sources(
        2017, ("cfpb", "ffiec_three_year"), data_dir=tmp_path, batch_size=2
    )

    assert report["sources"]["cfpb"]["rows"] == 3
    assert report["sources"]["ffiec_three_year"]["rows"] == 2
    assert report["schema_comparison"]["common_columns"] == ["state_code"]
    assert report["sources"]["cfpb"]["coverage"]["state"] == {
        "column": "state_code",
        "non_null": 2,
        "non_null_share": 2 / 3,
        "distinct_non_null": 1,
        "counts": {"01": 2},
    }
    assert json.loads(write_report(report))["year"] == 2017
