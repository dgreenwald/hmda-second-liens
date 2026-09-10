"""Export the frozen known-source-prior model and historical annual intercepts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, portable_predict
from .density_ratio import artifacts
from .density_ratio.families.logistic import (
    known_source_prior_model_path,
    load_known_source_prior_model,
)
from .logistic_features import FeatureSpecification


def default_model_path() -> Path:
    return known_source_prior_model_path(
        config.TRAIN_YEARS,
        FeatureSpecification("spline_lti", "purchaser_type"),
        0.1,
    )


def _atomic_write(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(content)
            file.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def export_portable_logistic(
    model_file: str | Path | None = None,
    annual_file: str | Path = config.TABLE_DIR / "step8_annual_plausibility.csv",
    output_dir: str | Path = config.MODEL_DIR / "portable_logistic",
    years=tuple(config.APPLY_YEARS),
) -> Path:
    """Export existing fits and aggregates only; never fit or estimate shares."""
    model_file = default_model_path() if model_file is None else Path(model_file)
    annual_file = Path(annual_file)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Missing known-source-prior fit: {model_file}. Restore the saved final "
            "mixture fit before exporting; export never retrains."
        )
    if not annual_file.exists():
        raise FileNotFoundError(
            f"Missing annual estimates: {annual_file}. Run make plausibility-checks "
            "with the saved final fits before exporting."
        )
    model = load_known_source_prior_model(model_file)
    metadata = artifacts.load_metadata(model_file)
    if (
        model.specification.name != portable_predict.SPECIFICATION
        or model.regularization_c != 0.1
        or tuple(model.train_years) != tuple(config.TRAIN_YEARS)
    ):
        raise ValueError("Export requires the frozen C=0.1 final 2004–2007 core model")
    if model.transformer.specification != model.specification:
        raise ValueError("Model and transformer specifications differ")
    annual_bytes = annual_file.read_bytes()
    # Read the same bytes whose digest is recorded in the portable provenance.
    annual = pd.read_csv(BytesIO(annual_bytes))
    required = {
        "year",
        "mixture_share",
        "mixture_optimizer_converged",
        "mixture_at_boundary",
        "mixture_model_id",
        "mixture_model_sha256",
    }
    if not required <= set(annual):
        raise ValueError(
            "Annual estimates lack required diagnostics/model provenance; "
            "rerun make plausibility-checks using the saved fits."
        )
    year_values = pd.to_numeric(annual["year"], errors="raise").to_numpy(float)
    if not np.isfinite(year_values).all() or np.any(
        year_values != np.floor(year_values)
    ):
        raise ValueError("Annual years must be finite integers")
    if annual["year"].duplicated().any():
        raise ValueError("Annual estimates must have unique years")
    requested = np.asarray(tuple(years), dtype=float)
    if (
        requested.ndim != 1
        or not len(requested)
        or not np.isfinite(requested).all()
        or np.any(requested != np.floor(requested))
        or len(set(requested)) != len(requested)
    ):
        raise ValueError("Requested years must be unique finite integers")
    annual = annual.set_index("year")
    missing = set(requested) - set(annual.index)
    if missing:
        raise ValueError(f"Missing annual estimates for {sorted(missing)}")
    selected = annual.loc[requested.astype(int)]
    if not (
        selected["mixture_model_id"].eq(model.model_id).all()
        and selected["mixture_model_sha256"].eq(metadata.payload_sha256).all()
    ):
        raise ValueError("Annual estimates do not match the saved mixture model")
    if not selected["mixture_optimizer_converged"].eq(True).all():
        raise ValueError("Annual mixture optimization did not converge")
    if not selected["mixture_at_boundary"].isin([True, False]).all():
        raise ValueError("Annual boundary diagnostics must be boolean")
    transformer = model.transformer
    offset = float(model.ratio.log_ratio_offset)
    annual_payload = {}
    for year, row in selected.iterrows():
        share = float(row["mixture_share"])
        if not np.isfinite(share) or not 0 < share < 1:
            raise ValueError(
                "Annual shares must be finite and strictly between 0 and 1"
            )
        annual_payload[str(int(year))] = {
            "mixture_share": share,
            "intercept": offset + float(np.log(share) - np.log1p(-share)),
            "optimizer_converged": True,
            "at_boundary": bool(row["mixture_at_boundary"]),
        }
    payload = {
        "format_version": portable_predict.FORMAT_VERSION,
        "specification": model.specification.name,
        "category_levels": config.CATEGORY_LEVELS,
        "feature_names": list(transformer.feature_names_),
        "coefficients": model.ratio.feature_coefficients.tolist(),
        "density_ratio_offset": offset,
        "knots": transformer.knots_["log_lti"].tolist(),
        "raw_location": [
            transformer.raw_location_[v] for v in portable_predict.CONTINUOUS
        ],
        "raw_scale": [transformer.raw_scale_[v] for v in portable_predict.CONTINUOUS],
        "basis_location": transformer.basis_location_["log_lti"].tolist(),
        "basis_scale": transformer.basis_scale_["log_lti"].tolist(),
        "annual": annual_payload,
        "provenance": {
            "model_id": model.model_id,
            "source_artifact_sha256": metadata.payload_sha256,
            "annual_table_sha256": hashlib.sha256(annual_bytes).hexdigest(),
            "train_years": [int(year) for year in model.train_years],
            "regularization_c": model.regularization_c,
            "software_versions": metadata.software_versions,
            "weighting": metadata.weighting,
            "source_prior": metadata.source_prior,
        },
    }
    portable_predict.PortableLogisticModel(payload)
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        output_dir / "predict.py", Path(portable_predict.__file__).read_bytes()
    )
    model_path = output_dir / "model.json"
    _atomic_write(model_path, serialized)
    return model_path
