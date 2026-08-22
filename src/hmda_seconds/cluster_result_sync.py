"""Transfer and validate model-selection results from the Torch DTN."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pandas as pd

from . import config, model_selection
from . import model_selection_cluster as logistic_cluster
from .density_ratio import artifacts
from .density_ratio.families.gradient_boosting import (
    BOOSTING_FEATURES,
    HMDA_ONLY_BOOSTING_FEATURES,
    HMDA_ONLY_SPECIFICATION,
    SPECIFICATION,
    BoostingParameters,
)
from .density_ratio.shards import (
    aggregate_shards,
    read_manifest,
    read_shard,
    shard_path,
)

DEFAULT_DTN_HOST = "dtn.torch.hpc.nyu.edu"
EXPECTED_SELECTION_BRIER = 0.026648
EXPECTED_WINNER = BoostingParameters(
    max_leaf_nodes=7,
    learning_rate=0.05,
    max_iter=200,
    l2_regularization=1.0,
    min_samples_leaf=1_000,
)

TRANSFER_PATHS = (
    "output/raw_logistic_selection/",
    "output/hmda_only_raw_logistic_selection/",
    "output/boosting_selection/",
    "output/hmda_only_boosting_selection/",
    "output/slurm/logistic_selection/",
    "output/slurm/hmda_only_logistic_selection/",
    "output/slurm/boosting/",
    "output/slurm/hmda_only_boosting/",
    "output/tables/logistic_selection_core_coarse_cells.csv",
    "output/tables/logistic_selection_core_coarse_horizons.csv",
    "output/tables/logistic_selection_core_coarse_summary.csv",
    "output/tables/logistic_selection_core_refinement_cells.csv",
    "output/tables/logistic_selection_core_refinement_horizons.csv",
    "output/tables/logistic_selection_core_refinement_summary.csv",
    "output/tables/logistic_selection_core_cells.csv",
    "output/tables/logistic_selection_core_horizons.csv",
    "output/tables/logistic_selection_core_summary.csv",
    "output/tables/logistic_selection_decision.csv",
    "output/tables/logistic_selection_hmda_only_coarse_cells.csv",
    "output/tables/logistic_selection_hmda_only_coarse_horizons.csv",
    "output/tables/logistic_selection_hmda_only_coarse_summary.csv",
    "output/tables/logistic_selection_hmda_only_refinement_cells.csv",
    "output/tables/logistic_selection_hmda_only_refinement_horizons.csv",
    "output/tables/logistic_selection_hmda_only_refinement_summary.csv",
    "output/tables/logistic_selection_hmda_only_cells.csv",
    "output/tables/logistic_selection_hmda_only_horizons.csv",
    "output/tables/logistic_selection_hmda_only_summary.csv",
    "output/tables/logistic_selection_hmda_only_decision.csv",
    "output/tables/boosting_screen/",
    "output/tables/boosting_survivors/",
    "output/tables/boosting_refinement/",
    "output/tables/boosting_challenger_cells.csv",
    "output/tables/boosting_challenger_horizons.csv",
    "output/tables/boosting_challenger_summary.csv",
    "output/tables/boosting_challenger_decision.csv",
    "output/tables/hmda_only_boosting_screen/",
    "output/tables/hmda_only_boosting_survivors/",
    "output/tables/hmda_only_boosting_refinement/",
    "output/tables/hmda_only_boosting_cells.csv",
    "output/tables/hmda_only_boosting_horizons.csv",
    "output/tables/hmda_only_boosting_summary.csv",
    "output/tables/hmda_only_boosting_decision.csv",
    "output/model/logistic_selected.pkl",
    "output/model/logistic_selected.pkl.metadata.json",
    "output/model/logistic_hmda_only_selected.pkl",
    "output/model/logistic_hmda_only_selected.pkl.metadata.json",
    "output/model/boosting_challenger.pkl",
    "output/model/boosting_challenger.pkl.metadata.json",
    "output/model/boosting_hmda_only_challenger.pkl",
    "output/model/boosting_hmda_only_challenger.pkl.metadata.json",
)

PROMOTION_PATHS = (
    "raw_logistic_selection",
    "hmda_only_raw_logistic_selection",
    "boosting_selection",
    "hmda_only_boosting_selection",
    "slurm/logistic_selection",
    "slurm/hmda_only_logistic_selection",
    "slurm/boosting",
    "slurm/hmda_only_boosting",
    *tuple(
        path.removeprefix("output/")
        for path in TRANSFER_PATHS
        if path.startswith("output/tables/logistic_")
    ),
    "tables/boosting_screen",
    "tables/boosting_survivors",
    "tables/boosting_refinement",
    "tables/boosting_challenger_cells.csv",
    "tables/boosting_challenger_horizons.csv",
    "tables/boosting_challenger_summary.csv",
    "tables/boosting_challenger_decision.csv",
    "tables/hmda_only_boosting_screen",
    "tables/hmda_only_boosting_survivors",
    "tables/hmda_only_boosting_refinement",
    "tables/hmda_only_boosting_cells.csv",
    "tables/hmda_only_boosting_horizons.csv",
    "tables/hmda_only_boosting_summary.csv",
    "tables/hmda_only_boosting_decision.csv",
    "model/logistic_selected.pkl",
    "model/logistic_selected.pkl.metadata.json",
    "model/logistic_hmda_only_selected.pkl",
    "model/logistic_hmda_only_selected.pkl.metadata.json",
    "model/boosting_challenger.pkl",
    "model/boosting_challenger.pkl.metadata.json",
    "model/boosting_hmda_only_challenger.pkl",
    "model/boosting_hmda_only_challenger.pkl.metadata.json",
)

STAGES = {
    "screen": ("boosting_screen", 6, 9),
    "survivors": ("boosting_survivors", 2, 45),
    "refinement": ("boosting_refinement", 4, 45),
}

HMDA_ONLY_STAGES = {
    "screen": ("hmda_only_boosting_screen", 6, 9),
    "survivors": ("hmda_only_boosting_survivors", 2, 45),
    "refinement": ("hmda_only_boosting_refinement", 4, 45),
}


@dataclass(frozen=True)
class SyncRequest:
    """Resolved connection and destination settings for one transfer."""

    user: str
    remote_repo: PurePosixPath
    output_dir: Path = config.OUTPUT_DIR
    host: str = DEFAULT_DTN_HOST

    def __post_init__(self) -> None:
        if not self.user.strip():
            raise ValueError("Cluster user must be nonempty")
        if not self.host.strip():
            raise ValueError("Cluster host must be nonempty")
        if not self.remote_repo.is_absolute():
            raise ValueError("Remote repository path must be absolute")


def transfer_command(request: SyncRequest, files_from: Path, staging: Path) -> list[str]:
    """Build the sole authenticated transfer command."""
    source = f"{request.user}@{request.host}:{request.remote_repo.as_posix().rstrip('/')}/"
    return [
        "rsync",
        "--archive",
        "--recursive",
        "--checksum",
        "--partial",
        "--relative",
        "--protect-args",
        f"--files-from={files_from}",
        "-e",
        "ssh",
        source,
        f"{staging}/",
    ]


def describe_sync(request: SyncRequest) -> str:
    """Return a network-free transfer plan."""
    paths = "\n".join(f"  - {path}" for path in TRANSFER_PATHS)
    return (
        f"Torch DTN: {request.user}@{request.host}\n"
        f"Remote repository: {request.remote_repo}\n"
        f"Local output: {request.output_dir}\n"
        "Apply mode uses one authenticated rsync invocation.\n"
        f"Transfer paths:\n{paths}"
    )


def sync_results(
    request: SyncRequest,
    *,
    apply: bool = False,
    staging_dir: str | Path | None = None,
) -> Path | None:
    """Transfer once, validate locally, then back up and promote results."""
    if not apply:
        if staging_dir is not None:
            raise ValueError("--staging-dir is only valid with --apply")
        print(describe_sync(request))
        return None

    request.output_dir.mkdir(parents=True, exist_ok=True)
    if staging_dir is None:
        staging = Path(
            tempfile.mkdtemp(prefix=".cluster-sync-staging-", dir=request.output_dir)
        )
    else:
        staging = Path(staging_dir).resolve()
        output_dir = request.output_dir.resolve()
        if (
            staging.parent != output_dir
            or not staging.name.startswith(".cluster-sync-staging-")
            or not staging.is_dir()
        ):
            raise ValueError(
                "Resume staging must be an existing sync staging directory "
                "directly under the configured output directory"
            )
    try:
        files_from = staging / "transfer-files.txt"
        files_from.write_text("\n".join(TRANSFER_PATHS) + "\n")
        subprocess.run(
            transfer_command(request, files_from, staging),
            check=True,
        )
        staged_output = staging / "output"
        validate_staged_results(staged_output)
        backup = promote_results(staged_output, request.output_dir)
        shutil.rmtree(staging)
        return backup
    except Exception:
        print(f"Sync failed; staged files retained at {staging}")
        raise
    finally:
        if staging.exists() and not any(staging.iterdir()):
            staging.rmdir()


def validate_staged_results(staged_output: Path) -> None:
    """Validate all transferred shards, aggregate tables, and model artifacts."""
    missing = [path for path in PROMOTION_PATHS if not (staged_output / path).exists()]
    if missing:
        raise FileNotFoundError(f"Transferred boosting results are missing: {missing}")

    for root in (
        "raw_logistic_selection",
        "hmda_only_raw_logistic_selection",
        "boosting_selection",
        "hmda_only_boosting_selection",
    ):
        _validate_artifact_tree(staged_output / root / "models")

    _validate_boosting_variant(
        staged_output,
        slurm_root="boosting",
        result_root="boosting_selection",
        table_stages=STAGES,
    )
    _validate_boosting_variant(
        staged_output,
        slurm_root="hmda_only_boosting",
        result_root="hmda_only_boosting_selection",
        table_stages=HMDA_ONLY_STAGES,
    )
    _validate_logistic_variant(staged_output, feature_set="core")
    _validate_logistic_variant(staged_output, feature_set="hmda_only")
    _validate_combined_selection(staged_output)
    _validate_final_models(staged_output)


def _validate_artifact_tree(model_root: Path) -> None:
    sidecars = sorted(model_root.rglob(f"*{artifacts.METADATA_SUFFIX}"))
    if not sidecars:
        raise ValueError(f"No model artifacts found under {model_root}")
    for sidecar in sidecars:
        payload = sidecar.with_name(sidecar.name[: -len(artifacts.METADATA_SUFFIX)])
        artifacts.validate_existing_artifact(payload)


def _validate_boosting_variant(
    staged_output: Path,
    *,
    slurm_root: str,
    result_root: str,
    table_stages: dict[str, tuple[str, int, int]],
) -> None:
    for label, (table_dir, expected_candidates, expected_cells) in table_stages.items():
        manifest = (
            staged_output
            / "slurm"
            / slurm_root
            / label
            / "density_ratio_jobs.json"
        )
        planned = read_manifest(manifest)
        shard_root = staged_output / result_root / "shards"
        shard_paths = [shard_root / shard_path(item.job).name for item in planned]
        for path in shard_paths:
            shard = read_shard(path)
            remote_root = Path(shard.job.output_root)
            for model in shard.models:
                relative = Path(model.artifact_path).relative_to(remote_root)
                artifacts.validate_existing_artifact(
                    staged_output / result_root / relative
                )
        aggregated = aggregate_shards(planned, shard_paths)
        copied_dir = staged_output / "tables" / table_dir
        for name in ("cells", "horizons", "summary"):
            copied = pd.read_csv(copied_dir / f"density_ratio_{name}.csv")
            regenerated = getattr(aggregated, name)
            pd.testing.assert_frame_equal(
                copied.reset_index(drop=True),
                regenerated.reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-14,
            )
        summary = aggregated.summary
        if len(summary) != expected_candidates or not (
            summary["n_cells"] == expected_cells
        ).all():
            raise ValueError(f"Incomplete {slurm_root} {label} summary")


def _validate_logistic_variant(staged_output: Path, *, feature_set: str) -> None:
    restricted = feature_set == "hmda_only"
    slurm_root = "hmda_only_logistic_selection" if restricted else "logistic_selection"
    result_root = (
        "hmda_only_raw_logistic_selection" if restricted else "raw_logistic_selection"
    )
    prefix = f"logistic_selection_{feature_set}"
    stage_cells = {}
    for stage in ("coarse", "refinement"):
        manifest = (
            staged_output
            / "slurm"
            / slurm_root
            / stage
            / "logistic_selection_jobs.json"
        )
        jobs = logistic_cluster.read_manifest(manifest)
        parts = []
        for job in jobs:
            path = (
                staged_output
                / result_root
                / "shards"
                / logistic_cluster.shard_path(job).name
            )
            shard = logistic_cluster.read_shard(path)
            logistic_cluster._validate_shard(job, shard, validate_artifacts=False)
            remote_root = Path(job.output_root)
            for artifact_path in shard["artifact_paths"]:
                relative = Path(artifact_path).relative_to(remote_root)
                artifacts.validate_existing_artifact(
                    staged_output / result_root / relative
                )
            parts.append(pd.DataFrame(shard["cells"]))
        cells = logistic_cluster._sort_cells(pd.concat(parts, ignore_index=True))
        horizons, summary = model_selection.aggregate_brier_cells(cells)
        _compare_logistic_tables(
            staged_output,
            f"{prefix}_{stage}",
            cells,
            horizons,
            summary,
        )
        stage_cells[stage] = cells

    combined = logistic_cluster._sort_cells(
        pd.concat([stage_cells["coarse"], stage_cells["refinement"]], ignore_index=True)
    )
    horizons, summary = model_selection.aggregate_brier_cells(combined)
    _compare_logistic_tables(staged_output, prefix, combined, horizons, summary)
    if not (summary["n_cells"] == 45).all():
        raise ValueError(f"Incomplete {feature_set} logistic selection summary")


def _compare_logistic_tables(
    staged_output: Path,
    prefix: str,
    cells: pd.DataFrame,
    horizons: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    for name, regenerated in (
        ("cells", cells),
        ("horizons", horizons),
        ("summary", summary),
    ):
        copied = pd.read_csv(staged_output / "tables" / f"{prefix}_{name}.csv")
        pd.testing.assert_frame_equal(
            copied.reset_index(drop=True),
            regenerated.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-14,
        )

def _validate_combined_selection(staged_output: Path) -> None:
    summary = pd.read_csv(
        staged_output / "tables" / "boosting_challenger_summary.csv"
    ).sort_values(["selection_brier", "parameter_id"])
    if len(summary) != 6 or not (summary["n_cells"] == 45).all():
        raise ValueError("Combined boosting selection must contain six complete candidates")
    winner = summary.iloc[0]
    if winner["parameter_id"] != EXPECTED_WINNER.identifier:
        raise ValueError("Transferred boosting summary has the wrong winner")
    if abs(float(winner["selection_brier"]) - EXPECTED_SELECTION_BRIER) > 5e-7:
        raise ValueError("Transferred boosting winner has an unexpected Brier score")

    decision = pd.read_csv(
        staged_output / "tables" / "boosting_challenger_decision.csv"
    )
    if len(decision) != 1 or decision.iloc[0]["parameter_id"] != winner["parameter_id"]:
        raise ValueError("Transferred boosting decision does not match its summary")

    hmda_summary = pd.read_csv(
        staged_output / "tables" / "hmda_only_boosting_summary.csv"
    ).sort_values(["selection_brier", "configuration_id"])
    hmda_decision = pd.read_csv(
        staged_output / "tables" / "hmda_only_boosting_decision.csv"
    )
    if len(hmda_summary) != 6 or not (hmda_summary["n_cells"] == 45).all():
        raise ValueError("HMDA-only boosting selection is incomplete")
    if len(hmda_decision) != 1 or (
        hmda_decision.iloc[0]["configuration_id"]
        != hmda_summary.iloc[0]["configuration_id"]
    ):
        raise ValueError("HMDA-only boosting decision does not match its summary")


def _validate_final_models(staged_output: Path) -> None:
    _validate_boosting_model(
        staged_output / "model" / "boosting_challenger.pkl",
        staged_output / "tables" / "boosting_challenger_decision.csv",
        specification=SPECIFICATION,
        feature_names=tuple(BOOSTING_FEATURES),
    )
    _validate_boosting_model(
        staged_output / "model" / "boosting_hmda_only_challenger.pkl",
        staged_output / "tables" / "hmda_only_boosting_decision.csv",
        specification=HMDA_ONLY_SPECIFICATION,
        feature_names=tuple(HMDA_ONLY_BOOSTING_FEATURES),
    )
    _validate_logistic_model(
        staged_output / "model" / "logistic_selected.pkl",
        staged_output / "tables" / "logistic_selection_decision.csv",
    )
    _validate_logistic_model(
        staged_output / "model" / "logistic_hmda_only_selected.pkl",
        staged_output / "tables" / "logistic_selection_hmda_only_decision.csv",
    )


def _validate_boosting_model(
    model: Path,
    decision_file: Path,
    *,
    specification: str,
    feature_names: tuple[str, ...],
) -> None:
    metadata = artifacts.validate_existing_artifact(model)
    if metadata is None:
        raise ValueError("Final boosting model is missing metadata")
    if metadata.configuration.specification != specification:
        raise ValueError("Final boosting model has the wrong specification")
    decision = pd.read_csv(decision_file)
    if len(decision) != 1:
        raise ValueError("Boosting decision must contain one row")
    declared = json.loads(decision.iloc[0]["hyperparameters"])
    if metadata.configuration.parameter_dict() != declared:
        raise ValueError("Final boosting model has the wrong hyperparameters")
    if metadata.train_years != tuple(config.TRAIN_YEARS):
        raise ValueError("Final boosting model has the wrong training years")
    if metadata.feature_names != feature_names:
        raise ValueError("Final boosting model has the wrong feature schema")


def _validate_logistic_model(model: Path, decision_file: Path) -> None:
    metadata = artifacts.validate_existing_artifact(model)
    if metadata is None or metadata.configuration.family != "raw_logistic":
        raise ValueError("Final logistic model has invalid metadata")
    decision = pd.read_csv(decision_file)
    if len(decision) != 1:
        raise ValueError("Logistic decision must contain one row")
    row = decision.iloc[0]
    if metadata.configuration.specification != row["specification"] or float(
        metadata.configuration.parameter_dict()["C"]
    ) != float(row["regularization_c"]):
        raise ValueError("Final logistic model does not match its decision")
    if metadata.train_years != tuple(config.TRAIN_YEARS):
        raise ValueError("Final logistic model has the wrong training years")


def promote_results(staged_output: Path, output_dir: Path) -> Path:
    """Back up conflicts and promote a completely validated staging tree."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = output_dir / "sync_backups" / timestamp
    moved_to_backup: list[tuple[Path, Path]] = []
    promoted: list[tuple[Path, Path]] = []
    try:
        for relative in PROMOTION_PATHS:
            source = staged_output / relative
            destination = output_dir / relative
            if destination.exists():
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(destination, saved)
                moved_to_backup.append((saved, destination))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(source, destination)
            promoted.append((destination, source))
    except Exception:
        for destination, source in reversed(promoted):
            source.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.move(destination, source)
        for saved, destination in reversed(moved_to_backup):
            destination.parent.mkdir(parents=True, exist_ok=True)
            if saved.exists():
                shutil.move(saved, destination)
        raise
    return backup


def request_from_environment(
    *,
    user: str | None = None,
    remote_repo: str | None = None,
    host: str | None = None,
    output_dir: str | Path | None = None,
) -> SyncRequest:
    """Resolve CLI overrides over environment-specific settings."""
    resolved_user = user or os.environ.get("HMDA_SECONDS_CLUSTER_USER", "")
    resolved_repo = remote_repo or os.environ.get("HMDA_SECONDS_CLUSTER_REPO", "")
    if not resolved_repo:
        raise ValueError(
            "Set HMDA_SECONDS_CLUSTER_REPO or pass --remote-repo"
        )
    return SyncRequest(
        user=resolved_user,
        remote_repo=PurePosixPath(resolved_repo),
        host=host or os.environ.get("HMDA_SECONDS_CLUSTER_HOST", DEFAULT_DTN_HOST),
        output_dir=Path(output_dir) if output_dir is not None else config.OUTPUT_DIR,
    )
