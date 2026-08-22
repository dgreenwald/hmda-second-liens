from pathlib import Path, PurePosixPath

import pandas as pd
import pytest

from hmda_seconds import cluster_result_sync
from hmda_seconds.cluster_result_sync import (
    DEFAULT_DTN_HOST,
    EXPECTED_WINNER,
    PROMOTION_PATHS,
    SyncRequest,
    promote_results,
    sync_results,
    transfer_command,
)


def _request(tmp_path):
    return SyncRequest(
        user="dlg340",
        host=DEFAULT_DTN_HOST,
        remote_repo=PurePosixPath("/home/dlg340/research/hmda-second-liens"),
        output_dir=tmp_path / "output",
    )


def test_transfer_command_uses_one_torch_dtn_source(tmp_path):
    request = _request(tmp_path)
    command = transfer_command(request, tmp_path / "files", tmp_path / "staging")

    assert command[0] == "rsync"
    assert command.count("rsync") == 1
    assert "--recursive" in command
    assert "--files-from=" in " ".join(command)
    assert command[-2] == (
        "dlg340@dtn.torch.hpc.nyu.edu:"
        "/home/dlg340/research/hmda-second-liens/"
    )


def test_transfer_scope_contains_all_four_workflows_but_no_microdata():
    paths = "\n".join(cluster_result_sync.TRANSFER_PATHS)
    assert "raw_logistic_selection/" in paths
    assert "hmda_only_raw_logistic_selection/" in paths
    assert "boosting_selection/" in paths
    assert "hmda_only_boosting_selection/" in paths
    assert "logistic_hmda_only_selected.pkl" in paths
    assert "boosting_hmda_only_challenger.pkl" in paths
    assert "selection_data" not in paths
    assert "data/raw" not in paths


def test_plan_is_network_free(tmp_path, monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        raise AssertionError("plan mode must not invoke subprocesses")

    monkeypatch.setattr(cluster_result_sync.subprocess, "run", unexpected)
    assert sync_results(_request(tmp_path), apply=False) is None
    output = capsys.readouterr().out
    assert "one authenticated rsync invocation" in output
    assert DEFAULT_DTN_HOST in output


def test_apply_invokes_rsync_once_then_validates_and_promotes(tmp_path, monkeypatch):
    calls = []
    events = []

    def fake_run(command, check):
        calls.append((command, check))

    monkeypatch.setattr(cluster_result_sync.subprocess, "run", fake_run)
    monkeypatch.setattr(
        cluster_result_sync,
        "validate_staged_results",
        lambda path: events.append(("validate", path)),
    )
    backup = tmp_path / "output" / "sync_backups" / "test"
    monkeypatch.setattr(
        cluster_result_sync,
        "promote_results",
        lambda staged, output: events.append(("promote", staged, output)) or backup,
    )

    assert sync_results(_request(tmp_path), apply=True) == backup
    assert len(calls) == 1
    assert calls[0][0][0] == "rsync"
    assert calls[0][1] is True
    assert [event[0] for event in events] == ["validate", "promote"]


def test_failed_transfer_preserves_existing_output_and_staging(tmp_path, monkeypatch):
    request = _request(tmp_path)
    existing = request.output_dir / "tables" / "keep.csv"
    existing.parent.mkdir(parents=True)
    existing.write_text("old\n")

    def fail(*args, **kwargs):
        raise cluster_result_sync.subprocess.CalledProcessError(23, "rsync")

    monkeypatch.setattr(cluster_result_sync.subprocess, "run", fail)
    with pytest.raises(cluster_result_sync.subprocess.CalledProcessError):
        sync_results(request, apply=True)

    assert existing.read_text() == "old\n"
    assert list(request.output_dir.glob(".cluster-sync-staging-*"))


def test_apply_can_resume_retained_staging(tmp_path, monkeypatch):
    request = _request(tmp_path)
    request.output_dir.mkdir(parents=True)
    staging = request.output_dir / ".cluster-sync-staging-retained"
    staging.mkdir()
    calls = []
    monkeypatch.setattr(
        cluster_result_sync.subprocess,
        "run",
        lambda command, check: calls.append(command),
    )
    monkeypatch.setattr(cluster_result_sync, "validate_staged_results", lambda path: None)
    monkeypatch.setattr(
        cluster_result_sync,
        "promote_results",
        lambda staged, output: output / "sync_backups" / "test",
    )

    sync_results(request, apply=True, staging_dir=staging)

    assert len(calls) == 1
    assert not staging.exists()


def test_resume_rejects_arbitrary_directory(tmp_path):
    request = _request(tmp_path)
    arbitrary = tmp_path / "unrelated"
    arbitrary.mkdir()

    with pytest.raises(ValueError, match="directly under"):
        sync_results(request, apply=True, staging_dir=arbitrary)


def test_promote_backs_up_conflicts(tmp_path):
    output = tmp_path / "output"
    staged = tmp_path / "staged"
    for relative in PROMOTION_PATHS:
        source = staged / relative
        if Path(relative).suffix:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("new")
        else:
            source.mkdir(parents=True)
            (source / "result").write_text("new")

    conflict = output / "tables" / "boosting_challenger_decision.csv"
    conflict.parent.mkdir(parents=True)
    conflict.write_text("old")

    backup = promote_results(staged, output)

    assert conflict.read_text() == "new"
    assert (backup / "tables" / "boosting_challenger_decision.csv").read_text() == "old"


def test_promote_rolls_back_after_move_failure(tmp_path, monkeypatch):
    output = tmp_path / "output"
    staged = tmp_path / "staged"
    for relative in PROMOTION_PATHS:
        source = staged / relative
        if Path(relative).suffix:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("new")
        else:
            source.mkdir(parents=True)
            (source / "result").write_text("new")
    conflict = output / PROMOTION_PATHS[0]
    conflict.mkdir(parents=True)
    (conflict / "result").write_text("old")

    real_move = cluster_result_sync.shutil.move
    calls = 0

    def fail_third_move(source, destination):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("simulated promotion failure")
        return real_move(source, destination)

    monkeypatch.setattr(cluster_result_sync.shutil, "move", fail_third_move)
    with pytest.raises(OSError, match="simulated"):
        promote_results(staged, output)

    assert (conflict / "result").read_text() == "old"


def test_validation_rejects_missing_transfer_paths(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        cluster_result_sync.validate_staged_results(tmp_path)


def test_combined_selection_rejects_wrong_winner(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    wrong = "leaves_15__lr_0p05__iter_200__l2_1__minleaf_1000"
    pd.DataFrame(
        {
            "parameter_id": [wrong, EXPECTED_WINNER.identifier, "a", "b", "c", "d"],
            "selection_brier": [0.01, 0.026648, 0.03, 0.04, 0.05, 0.06],
            "n_cells": [45] * 6,
        }
    ).to_csv(tables / "boosting_challenger_summary.csv", index=False)
    pd.DataFrame(
        {"parameter_id": [EXPECTED_WINNER.identifier]}
    ).to_csv(tables / "boosting_challenger_decision.csv", index=False)

    with pytest.raises(ValueError, match="wrong winner"):
        cluster_result_sync._validate_combined_selection(tmp_path)


def test_request_requires_absolute_remote_repo(tmp_path):
    with pytest.raises(ValueError, match="must be absolute"):
        SyncRequest(
            user="dlg340",
            remote_repo=PurePosixPath("relative/repo"),
            output_dir=tmp_path,
        )
