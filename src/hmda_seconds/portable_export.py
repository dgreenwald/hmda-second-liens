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


def default_model_path(feature_set: str = "core") -> Path:
    if feature_set not in ("core", "hmda_only"):
        raise ValueError("feature_set must be core or hmda_only")
    return known_source_prior_model_path(
        config.TRAIN_YEARS,
        FeatureSpecification(
            "spline_lti",
            "none" if feature_set == "hmda_only" else "purchaser_type",
            feature_set=feature_set,
        ),
        1.0 if feature_set == "hmda_only" else 0.1,
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
    annual_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    years=tuple(config.APPLY_YEARS),
    *,
    feature_set: str = "core",
) -> Path:
    """Export existing fits and aggregates only; never fit or estimate shares."""
    default_path = default_model_path(feature_set)
    model_file = default_path if model_file is None else Path(model_file)
    restricted = feature_set == "hmda_only"
    if annual_file is None:
        annual_file = config.TABLE_DIR / (
            "historical_model_family_annual.csv"
            if restricted
            else "step8_annual_plausibility.csv"
        )
    if output_dir is None:
        output_dir = config.MODEL_DIR / (
            "portable_logistic_hmda_only" if restricted else "portable_logistic"
        )
    annual_file = Path(annual_file)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Missing known-source-prior fit: {model_file}. Restore the saved final "
            "mixture fit before exporting; export never retrains."
        )
    if not annual_file.exists():
        raise FileNotFoundError(
            f"Missing annual estimates: {annual_file}. Generate historical annual "
            "estimates with the saved matching final fits before exporting."
        )
    model = load_known_source_prior_model(model_file)
    metadata = artifacts.load_metadata(model_file)
    if (
        model.specification.name
        != (
            portable_predict.HMDA_ONLY_SPECIFICATION
            if restricted
            else portable_predict.SPECIFICATION
        )
        or model.regularization_c != (1.0 if restricted else 0.1)
        or tuple(model.train_years) != tuple(config.TRAIN_YEARS)
    ):
        raise ValueError(
            f"Export requires the frozen final 2004–2007 {feature_set} model"
        )
    if model.transformer.specification != model.specification:
        raise ValueError("Model and transformer specifications differ")
    annual_bytes = annual_file.read_bytes()
    # Read the same bytes whose digest is recorded in the portable provenance.
    annual = pd.read_csv(BytesIO(annual_bytes))
    if {"predictor_set", "model_family"} <= set(annual):
        annual = annual.loc[
            annual["predictor_set"].eq("hmda_only" if restricted else "unrestricted")
            & annual["model_family"].eq("logistic")
        ].rename(
            columns={
                "model_id": "mixture_model_id",
                "model_sha256": "mixture_model_sha256",
                "optimizer_converged": "mixture_optimizer_converged",
            }
        )
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
            "regenerate historical estimates with model digests using the saved fits."
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
    payload = portable_payload(model, metadata, selected)
    payload["provenance"]["annual_table_sha256"] = hashlib.sha256(
        annual_bytes
    ).hexdigest()
    serialized = serialize_payload(payload)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        output_dir / "predict.py", Path(portable_predict.__file__).read_bytes()
    )
    model_path = output_dir / "model.json"
    _atomic_write(model_path, serialized)
    return model_path


def portable_payload(model, metadata, selected: pd.DataFrame) -> dict:
    """Encode a fitted model and already provenance-validated annual rows."""
    transformer = model.transformer
    continuous = portable_predict.continuous_variables(model.specification.name)
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
        "raw_location": [transformer.raw_location_[v] for v in continuous],
        "raw_scale": [transformer.raw_scale_[v] for v in continuous],
        "basis_location": transformer.basis_location_["log_lti"].tolist(),
        "basis_scale": transformer.basis_scale_["log_lti"].tolist(),
        "annual": annual_payload,
        "provenance": {
            "model_id": model.model_id,
            "source_artifact_sha256": metadata.payload_sha256,
            "train_years": [int(year) for year in model.train_years],
            "regularization_c": model.regularization_c,
            "software_versions": metadata.software_versions,
            "weighting": metadata.weighting,
            "source_prior": metadata.source_prior,
        },
    }
    portable_predict.PortableLogisticModel(payload)
    return payload


def serialize_payload(payload: dict) -> bytes:
    """Validate and deterministically serialize a portable model."""
    portable_predict.PortableLogisticModel(payload)
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
