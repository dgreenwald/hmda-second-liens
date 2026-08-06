import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from hmda_seconds.density_ratio import (
    EvaluationResult,
    FittedDensityRatioModel,
    JobSpecification,
    ModelArtifactMetadata,
    ModelConfiguration,
    TemporalFold,
)


@dataclass
class FakeFittedModel:
    model_id: str = "fake__train_2005_2008"
    train_years: tuple[int, ...] = (2005, 2006, 2007, 2008)

    def log_ratio(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["score"].to_numpy(dtype=float)


def test_fitted_model_protocol_uses_structural_typing():
    fitted = FakeFittedModel()

    assert isinstance(fitted, FittedDensityRatioModel)
    assert fitted.log_ratio(pd.DataFrame({"score": [1, 2]})) == pytest.approx(
        [1.0, 2.0]
    )


def test_boundary_records_have_stable_json_representations():
    configuration = ModelConfiguration.from_mapping(
        "logistic",
        "linear__none",
        {"C": 0.1, "solver": "newton-cholesky"},
        random_seed=17,
    )
    fold = TemporalFold(
        fold_id="reverse__train_2005_2008",
        train_years=(2005, 2006, 2007, 2008),
        target_years=(2004,),
        direction="reverse",
        horizons=(1,),
    )
    metadata = ModelArtifactMetadata(
        model_id="logistic__linear_none__train_2005_2008",
        configuration=configuration,
        train_years=fold.train_years,
        n_training=100,
        n_first_lien=80,
        n_second_lien=20,
        feature_names=("log_lti", "log_county_value_to_loan"),
        weighting="equal_class_mass_within_source_year",
        source_prior="one_half",
        artifact_path="output/model/example.pkl",
        software_versions=(("numpy", "2.0"), ("scikit-learn", "1.6")),
    )
    result = EvaluationResult(
        model_id=metadata.model_id,
        fold_id=fold.fold_id,
        target_year=2004,
        horizon=1,
        n_observations=50,
        actual_second_share=0.2,
        mixture_share=0.21,
        mean_probability=0.21,
        hard_share_050=0.1,
        brier_score=0.12,
        log_loss=0.4,
        calibration_mean_error=0.01,
        calibration_intercept=-0.1,
        calibration_slope=0.9,
        optimizer_converged=True,
        mixture_at_boundary=False,
    )
    job = JobSpecification(
        stage="coarse",
        family=configuration.family,
        specification=configuration.specification,
        train_years=fold.train_years,
        configurations=(configuration,),
        input_paths=(("selection_data", "data/selection"),),
        output_root="output/search",
    )

    payload = {
        "configuration": configuration.to_dict(),
        "fold": fold.to_dict(),
        "metadata": metadata.to_dict(),
        "result": result.to_dict(),
        "job": job.to_dict(),
    }
    encoded = json.dumps(payload, sort_keys=True)

    assert '"schema_version": 1' in encoded
    assert configuration.hyperparameters == (
        ("C", 0.1),
        ("solver", "newton-cholesky"),
    )
    assert job.to_dict()["input_paths"] == {"selection_data": "data/selection"}


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (
            lambda: ModelConfiguration(
                "logistic", "linear", (("solver", "x"), ("C", 1.0))
            ),
            "unique, sorted",
        ),
        (
            lambda: TemporalFold(
                "overlap", (2004, 2005), (2005,), "reverse", (1,)
            ),
            "must not overlap",
        ),
        (
            lambda: EvaluationResult(
                "model",
                "fold",
                2004,
                1,
                10,
                0.2,
                1.1,
                0.2,
                0.1,
                0.1,
                0.3,
                0.0,
                0.0,
                1.0,
                True,
                False,
            ),
            "mixture_share",
        ),
    ],
)
def test_boundary_records_reject_invalid_state(record, message):
    with pytest.raises(ValueError, match=message):
        record()
