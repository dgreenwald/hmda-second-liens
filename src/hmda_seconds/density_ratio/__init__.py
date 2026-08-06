"""Shared contracts for density-ratio fitting and evaluation."""

from .folds import forward_fold, reverse_folds, temporal_fold
from .protocols import (
    DensityRatioFamily,
    EvaluationResult,
    FittedDensityRatioModel,
    JobSpecification,
    ModelArtifactMetadata,
    ModelConfiguration,
    TemporalFold,
)

__all__ = [
    "DensityRatioFamily",
    "EvaluationResult",
    "FittedDensityRatioModel",
    "JobSpecification",
    "ModelArtifactMetadata",
    "ModelConfiguration",
    "TemporalFold",
    "forward_fold",
    "reverse_folds",
    "temporal_fold",
]
