"""Generate both packaged benchmarks from an explicit current results snapshot."""

import argparse
from pathlib import Path

from hmda_seconds.benchmark_release import bundle_logistic_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    for path in bundle_logistic_models(args.results_root):
        print(path)


if __name__ == "__main__":
    main()
