"""Load packaged benchmark snapshots without research dependencies."""

import json
from importlib.resources import files

from .portable_predict import PortableLogisticModel


def load_benchmark(feature_set: str = "core") -> PortableLogisticModel:
    """Load a fresh benchmark with saved annual intercepts for 1990–2016."""
    if feature_set not in ("core", "hmda_only"):
        raise ValueError("feature_set must be core or hmda_only")
    resource = files("hmda_seconds").joinpath("benchmarks", f"{feature_set}.json")
    return PortableLogisticModel(json.loads(resource.read_text(encoding="utf-8")))
