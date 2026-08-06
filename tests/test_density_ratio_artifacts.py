import json
import pickle

import pandas as pd
import pytest

from hmda_seconds.density_ratio import artifacts
from hmda_seconds.density_ratio.protocols import ModelConfiguration


def _metadata(path):
    return artifacts.build_metadata(
        model_id="logistic__linear__train_2005_2006",
        configuration=ModelConfiguration.from_mapping(
            "logistic", "linear", {"C": 0.1}, random_seed=17
        ),
        train_years=(2005, 2006),
        counts=(10, 8, 2),
        feature_names=("x",),
        weighting="equal_class_mass_within_source_year",
        source_prior="one_half",
        artifact_path=path,
    )


def test_atomic_pickle_round_trip_writes_hash_bound_metadata(tmp_path):
    path = tmp_path / "model.pkl"
    finalized = artifacts.save_pickle_artifact(
        {"coefficient": 1.5}, path, _metadata(path)
    )

    restored, metadata = artifacts.load_pickle_artifact(path, dict)

    assert restored == {"coefficient": 1.5}
    assert metadata == finalized
    assert metadata.payload_sha256 is not None
    assert artifacts.metadata_path(path).exists()
    values = json.loads(artifacts.metadata_path(path).read_text())
    assert values["schema_version"] == 1
    assert values["model_id"] == finalized.model_id


def test_hash_validation_rejects_modified_payload(tmp_path):
    path = tmp_path / "model.pkl"
    artifacts.save_pickle_artifact({"value": 1}, path, _metadata(path))
    path.write_bytes(pickle.dumps({"value": 2}))

    with pytest.raises(ValueError, match="hash does not match"):
        artifacts.load_pickle_artifact(path, dict)


def test_metadata_free_pickle_is_rejected_by_default(tmp_path):
    path = tmp_path / "legacy.pkl"
    path.write_bytes(pickle.dumps([1, 2, 3]))

    with pytest.raises(FileNotFoundError, match="Missing artifact metadata"):
        artifacts.load_pickle_artifact(path, list)
    restored, metadata = artifacts.load_pickle_artifact(
        path, list, allow_legacy=True
    )
    assert restored == [1, 2, 3]
    assert metadata is None


def test_training_counts_reject_unknown_lien_classes():
    frame = pd.DataFrame({"lien_status": [1, 2, 3]})

    with pytest.raises(ValueError, match="Unknown training labels"):
        artifacts.training_counts(
            frame,
            label_var="lien_status",
            first_lien_class=1,
            second_lien_class=2,
        )
