"""Atomic model artifacts with versioned, hash-validated metadata sidecars."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pickle
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd

from .protocols import ModelArtifactMetadata, ModelConfiguration

METADATA_SUFFIX = ".metadata.json"
T = TypeVar("T")


def metadata_path(artifact_path: str | Path) -> Path:
    """Return the sidecar path without changing the payload suffix."""
    path = Path(artifact_path)
    return path.with_name(path.name + METADATA_SUFFIX)


def training_counts(
    training: pd.DataFrame,
    *,
    label_var: str,
    first_lien_class: int,
    second_lien_class: int,
) -> tuple[int, int, int]:
    """Return validated binary training counts."""
    labels = training[label_var].to_numpy()
    unknown = set(pd.unique(labels)) - {first_lien_class, second_lien_class}
    if unknown:
        raise ValueError(f"Unknown training labels: {sorted(unknown)}")
    n_first = int(np.sum(labels == first_lien_class))
    n_second = int(np.sum(labels == second_lien_class))
    return len(training), n_first, n_second


def build_metadata(
    *,
    model_id: str,
    configuration: ModelConfiguration,
    train_years: tuple[int, ...],
    counts: tuple[int, int, int],
    feature_names: tuple[str, ...],
    weighting: str,
    source_prior: str,
    artifact_path: str | Path,
) -> ModelArtifactMetadata:
    """Build common metadata before the payload digest is known."""
    n_training, n_first, n_second = counts
    return ModelArtifactMetadata(
        model_id=model_id,
        configuration=configuration,
        train_years=train_years,
        n_training=n_training,
        n_first_lien=n_first,
        n_second_lien=n_second,
        feature_names=feature_names,
        weighting=weighting,
        source_prior=source_prior,
        artifact_path=str(Path(artifact_path)),
        software_versions=_software_versions(),
    )


def save_pickle_artifact(
    payload: object,
    artifact_path: str | Path,
    metadata: ModelArtifactMetadata,
) -> ModelArtifactMetadata:
    """Atomically write a pickle and its hash-bound JSON metadata sidecar."""
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_temp = _temporary_path(path)
    sidecar = metadata_path(path)
    metadata_temp = _temporary_path(sidecar)
    try:
        with payload_temp.open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        finalized = replace(
            metadata,
            artifact_path=str(path),
            payload_sha256=_sha256(payload_temp),
        )
        metadata_temp.write_text(
            json.dumps(finalized.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        os.replace(metadata_temp, sidecar)
        # Replace the payload last. An interruption between the two replaces
        # leaves either a missing payload or a detectable sidecar/hash mismatch,
        # never an unverified new payload that looks like a legacy artifact.
        os.replace(payload_temp, path)
        return finalized
    finally:
        payload_temp.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)


def save_fitted_model(
    payload: object,
    artifact_path: str | Path,
    *,
    model_id: str,
    configuration: ModelConfiguration,
    train_years: tuple[int, ...],
    counts: tuple[int, int, int],
    feature_names: tuple[str, ...],
    weighting: str,
    source_prior: str,
) -> ModelArtifactMetadata:
    """Build metadata and atomically persist one fitted family model."""
    metadata = build_metadata(
        model_id=model_id,
        configuration=configuration,
        train_years=train_years,
        counts=counts,
        feature_names=feature_names,
        weighting=weighting,
        source_prior=source_prior,
        artifact_path=artifact_path,
    )
    return save_pickle_artifact(payload, artifact_path, metadata)


def load_pickle_artifact(
    artifact_path: str | Path,
    expected_type: type[T],
    *,
    allow_legacy: bool = False,
) -> tuple[T, ModelArtifactMetadata | None]:
    """Load a trusted pickle after validating any available sidecar."""
    path = Path(artifact_path)
    metadata = load_metadata(path, allow_legacy=allow_legacy)
    if metadata is not None:
        digest = _sha256(path)
        if digest != metadata.payload_sha256:
            raise ValueError(f"Artifact hash does not match {metadata_path(path)}")
    with path.open("rb") as file:
        payload = pickle.load(file)
    if not isinstance(payload, expected_type):
        raise TypeError(
            f"Expected {expected_type.__name__}, got {type(payload).__name__}"
        )
    return payload, metadata


def load_metadata(
    artifact_path: str | Path, *, allow_legacy: bool = False
) -> ModelArtifactMetadata | None:
    """Load a sidecar, optionally accepting a metadata-free legacy artifact."""
    sidecar = metadata_path(artifact_path)
    if not sidecar.exists():
        if allow_legacy:
            return None
        raise FileNotFoundError(f"Missing artifact metadata: {sidecar}")
    values = json.loads(sidecar.read_text())
    if not isinstance(values, dict):
        raise TypeError("Artifact metadata must be a JSON object")
    metadata = ModelArtifactMetadata.from_dict(values)
    if metadata.payload_sha256 is None:
        raise ValueError("Artifact metadata is missing payload_sha256")
    return metadata


def write_sidecar_for_existing_artifact(
    artifact_path: str | Path, metadata: ModelArtifactMetadata
) -> ModelArtifactMetadata:
    """Atomically add hash-bound metadata to an externally written artifact."""
    path = Path(artifact_path)
    digest = _sha256(path)
    finalized = replace(
        metadata,
        artifact_path=str(path),
        payload_sha256=digest,
    )
    values = (json.dumps(finalized.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(metadata_path(path), values)
    return finalized


def validate_existing_artifact(
    artifact_path: str | Path, *, allow_legacy: bool = False
) -> ModelArtifactMetadata | None:
    """Validate an external-format artifact against its optional sidecar."""
    path = Path(artifact_path)
    metadata = load_metadata(path, allow_legacy=allow_legacy)
    if metadata is not None:
        digest = _sha256(path)
        if digest != metadata.payload_sha256:
            raise ValueError(f"Artifact hash does not match {metadata_path(path)}")
    return metadata


def validate_metadata_identity(
    metadata: ModelArtifactMetadata | None,
    *,
    model_id: str,
    train_years: tuple[int, ...],
) -> None:
    """Ensure sidecar identity agrees with the deserialized fitted object."""
    if metadata is None:
        return
    if metadata.model_id != model_id:
        raise ValueError(
            f"Artifact model_id {metadata.model_id!r} does not match {model_id!r}"
        )
    if metadata.train_years != train_years:
        raise ValueError("Artifact metadata training years do not match the model")


def _software_versions() -> tuple[tuple[str, str], ...]:
    packages = ("hmda-second-liens", "numpy", "pandas", "scikit-learn", "scipy")
    versions = []
    for package in packages:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        versions.append((package, version))
    return tuple(sorted(versions))


def _atomic_write(path: Path, values: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        temporary.write_bytes(values)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _temporary_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
