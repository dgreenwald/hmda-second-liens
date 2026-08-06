"""Shared contracts for density-ratio fitting and evaluation."""

from .adapters import (
    ExistingFittedModelAdapter,
    adapt_boosting_model,
    adapt_known_source_prior_model,
    adapt_random_forest_model,
)
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
    "ExistingFittedModelAdapter",
    "FittedDensityRatioModel",
    "JobSpecification",
    "ModelArtifactMetadata",
    "ModelConfiguration",
    "TemporalFold",
    "adapt_boosting_model",
    "adapt_known_source_prior_model",
    "adapt_random_forest_model",
]
