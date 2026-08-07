#!/usr/bin/env python3
"""Run one source-specific annual HMDA parquet conversion."""

from __future__ import annotations

import argparse

from hmda_seconds.hmda_conversion import conversion_job_name, run_conversion_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--job-index", required=True, type=int)
    parser.add_argument("--data-dir")
    parser.add_argument("--print-job-name", action="store_true")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_job_name:
        print(conversion_job_name(args.manifest, args.job_index))
        return
    if args.data_dir is None:
        raise SystemExit("--data-dir is required unless --print-job-name is used")
    output = run_conversion_job(
        args.manifest,
        args.job_index,
        data_dir=args.data_dir,
        chunksize=args.chunksize,
        compression=args.compression,
        overwrite=args.overwrite,
    )
    print(output)


if __name__ == "__main__":
    main()
