from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from hmda_seconds.density_ratio import artifacts, folds
from hmda_seconds.density_ratio.protocols import (
    EvaluationResult,
    JobSpecification,
    ModelConfiguration,
)
from hmda_seconds.density_ratio.shards import (
    PlannedJob,
    ResultShard,
    ShardModel,
    aggregate_shards,
    read_shard,
    write_shard,
)


def configuration(c_value=0.1):
    return ModelConfiguration.from_mapping(
        "logistic", "linear__none", {"C": c_value}
    )


def plan(train_years=(2005, 2006), target_years=(2004,), c_value=0.1, root="out"):
    candidate = configuration(c_value)
    job = JobSpecification(
        stage="coarse",
        family="logistic",
        specification="linear__none",
        train_years=train_years,
        configurations=(candidate,),
        input_paths=(),
        output_root=root,
    )
    fold = folds.temporal_fold(train_years, target_years, direction="reverse")
    return PlannedJob(job, fold)


def result(model_id, fold, year, brier):
    return EvaluationResult(
        model_id=model_id,
        fold_id=fold.fold_id,
        target_year=year,
        horizon=fold.horizon_for(year),
        n_observations=100,
        actual_second_share=0.2,
        mixture_share=0.18,
        mean_probability=0.18,
        hard_share_050=0.1,
        brier_score=brier,
        log_loss=0.4,
        calibration_mean_error=-0.02,
        calibration_intercept=0.0,
        calibration_slope=1.0,
        optimizer_converged=True,
        mixture_at_boundary=False,
    )


def shard(planned, briers):
    model_id = (
        f"logistic__linear__none__c_0p1__train_"
        f"{planned.fold.train_years[0]}_{planned.fold.train_years[-1]}"
    )
    return ResultShard(
        job=planned.job,
        fold=planned.fold,
        models=(ShardModel(model_id, planned.job.configurations[0], "model.pkl"),),
        results=tuple(
            result(model_id, planned.fold, year, brier)
            for year, brier in zip(planned.fold.target_years, briers, strict=True)
        ),
    )


def test_shard_round_trip_is_idempotent_and_conflicts_are_rejected(tmp_path):
    planned = plan(root=str(tmp_path))
    original = shard(planned, [0.2])
    path = tmp_path / "result.json"

    assert write_shard(original, path) == original
    assert write_shard(original, path) == original
    assert read_shard(path) == original
    with pytest.raises(FileExistsError, match="Conflicting"):
        write_shard(shard(planned, [0.3]), path)


def test_shard_rejects_missing_and_duplicate_cells():
    planned = plan(target_years=(2003, 2004))
    complete = shard(planned, [0.1, 0.2])
    with pytest.raises(ValueError, match="missing, duplicate, or unexpected"):
        ResultShard(
            job=complete.job,
            fold=complete.fold,
            models=complete.models,
            results=complete.results[:1],
        )
    with pytest.raises(ValueError, match="missing, duplicate, or unexpected"):
        ResultShard(
            job=complete.job,
            fold=complete.fold,
            models=complete.models,
            results=(complete.results[0], complete.results[0]),
        )


def test_aggregation_is_order_independent_and_weights_horizons_equally(tmp_path):
    first = plan((2005, 2006), (2004,), root=str(tmp_path))
    second = plan((2006, 2007), (2004, 2005), root=str(tmp_path))
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_shard(shard(first, [0.1]), first_path)
    write_shard(shard(second, [0.8, 0.3]), second_path)

    forward = aggregate_shards([first, second], [first_path, second_path])
    reverse = aggregate_shards([first, second], [second_path, first_path])

    pd.testing.assert_frame_equal(forward.cells, reverse.cells)
    pd.testing.assert_frame_equal(forward.horizons, reverse.horizons)
    pd.testing.assert_frame_equal(forward.summary, reverse.summary)
    # Horizon 1 averages 0.1 and 0.3; horizon 2 contributes 0.8.
    assert forward.summary.iloc[0]["selection_brier"] == pytest.approx(0.5)
    assert forward.summary.iloc[0]["n_cells"] == 3


def test_aggregator_detects_missing_duplicate_and_incompatible_shards(tmp_path):
    planned = plan(root=str(tmp_path))
    path = tmp_path / "one.json"
    write_shard(shard(planned, [0.2]), path)
    with pytest.raises(ValueError, match="missing shards"):
        aggregate_shards([planned], [])
    with pytest.raises(ValueError, match="duplicate shard"):
        aggregate_shards([planned], [path, path])

    incompatible = plan(c_value=1.0, root=str(tmp_path))
    with pytest.raises(ValueError, match="incompatible"):
        aggregate_shards([incompatible], [path])


def test_malformed_shard_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": 1}')
    with pytest.raises((KeyError, TypeError)):
        read_shard(path)


@dataclass
class FakeModel:
    model_id: str
    train_years: tuple[int, ...]
    fail: bool = False

    def log_ratio(self, frame):
        if self.fail:
            raise RuntimeError("interrupted evaluation")
        return frame["score"].to_numpy()


class FakeFamily:
    family_name = "logistic"

    def __init__(self, artifact_dir, fail=False):
        self.artifact_dir = Path(artifact_dir)
        self.fail = fail
        self.fit_calls = 0

    def fit_many(self, training, configurations, *, train_years):
        self.fit_calls += 1
        candidate = configurations[0]
        model = FakeModel(
            f"fake__train_{train_years[0]}_{train_years[-1]}",
            train_years,
            self.fail,
        )
        path = self.artifact_dir / f"{model.model_id}.pkl"
        counts = artifacts.training_counts(
            training,
            label_var="lien_status",
            first_lien_class=1,
            second_lien_class=2,
        )
        metadata = artifacts.build_metadata(
            model_id=model.model_id,
            configuration=candidate,
            train_years=train_years,
            counts=counts,
            feature_names=("score",),
            weighting="test_equal_prior",
            source_prior="one_half",
            artifact_path=path,
        )
        artifacts.save_pickle_artifact(model, path, metadata)
        return {model.model_id: model}


def yearly_data():
    return {
        year: pd.DataFrame(
            {
                "year": [year] * 6,
                "lien_status": [1, 2, 1, 2, 1, 2],
                "score": [-2, -1, -0.2, 0.2, 1, 2],
            }
        )
        for year in (2004, 2005, 2006)
    }


def test_interrupted_runner_publishes_no_partial_shard_and_retry_is_idempotent(
    tmp_path,
):
    from hmda_seconds.density_ratio.runner import run_job

    planned = plan(root=str(tmp_path))
    path = tmp_path / "job.json"
    broken = FakeFamily(tmp_path / "models", fail=True)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_job(
            planned,
            yearly_data(),
            broken,
            artifact_root=broken.artifact_dir,
            output_path=path,
        )
    assert not path.exists()
    assert next(broken.artifact_dir.glob("*.pkl")).exists()

    working = FakeFamily(tmp_path / "models")
    completed = run_job(
        planned,
        yearly_data(),
        working,
        artifact_root=working.artifact_dir,
        output_path=path,
    )
    assert path.exists() and len(completed.results) == 1
    run_job(
        planned,
        yearly_data(),
        working,
        artifact_root=working.artifact_dir,
        output_path=path,
    )
    assert working.fit_calls == 1


def test_local_runner_uses_single_job_worker_for_each_plan(tmp_path):
    from hmda_seconds.density_ratio.runner import run_local

    first = plan(root=str(tmp_path))
    second = plan(
        train_years=(2006, 2007),
        target_years=(2004, 2005),
        root=str(tmp_path),
    )
    data = yearly_data()
    data[2007] = data[2006].assign(year=2007)
    family = FakeFamily(tmp_path / "models")
    paths = run_local(
        [first, second],
        data,
        {"logistic": family},
        artifact_roots={"logistic": family.artifact_dir},
    )

    assert family.fit_calls == 2
    assert len(paths) == 2 and all(path.exists() for path in paths)
    combined = aggregate_shards([first, second], reversed(paths))
    assert len(combined.cells) == 3
