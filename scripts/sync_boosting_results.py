#!/usr/bin/env python3
"""Sync validated logistic and boosting selection results from the Torch DTN."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmda_seconds.boosting_result_sync import request_from_environment, sync_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user")
    parser.add_argument("--host")
    parser.add_argument("--remote-repo")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = request_from_environment(
        user=args.user,
        host=args.host,
        remote_repo=args.remote_repo,
        output_dir=args.output_dir,
    )
    backup = sync_results(
        request,
        apply=args.apply,
        staging_dir=args.staging_dir,
    )
    if backup is not None:
        print(f"Selection results synchronized; any prior outputs are under {backup}")


if __name__ == "__main__":
    main()
