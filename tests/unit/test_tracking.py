"""Unit tests for :mod:`kaggle_driver.tracking`."""

from __future__ import annotations

import json
import logging
import os
import re
import stat
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from immutabledict import immutabledict

from kaggle_driver.config import load_yaml_config
from kaggle_driver.tracking import (
    RUN_SCHEMA_VERSION,
    RunCommand,
    RunDirectoryTracker,
    RunRecord,
    RunStatus,
    list_runs,
    load_run,
)

_RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{4}$")


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    """Return a temporary runs root path that has not yet been created."""
    return tmp_path / "runs"


@pytest.fixture
def tracker(runs_root: Path) -> RunDirectoryTracker:
    """Return a fresh, unstarted tracker rooted at a temporary runs directory."""
    return RunDirectoryTracker(runs_root)


def start_tracked_run(
    tracker_under_test: RunDirectoryTracker,
    *,
    command: RunCommand = RunCommand.TRAIN,
    model_name: str = "dummy_model",
    configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Start a run on a tracker with sensible defaults for the optional fields."""
    tracker_under_test.start_run(
        command=command,
        model_name=model_name,
        configs=configs if configs is not None else {},
    )


def complete_tracked_run(tracker_under_test: RunDirectoryTracker) -> str:
    """Start and immediately complete a run, returning its run id."""
    start_tracked_run(tracker_under_test)
    tracker_under_test.complete_run()
    return tracker_under_test.run_id


def call_record_metrics(tracker_under_test: RunDirectoryTracker) -> None:
    """Invoke record_metrics with a throwaway phase and metrics mapping."""
    tracker_under_test.record_metrics("train", {"sample_count": 1})


def call_record_artifact(tracker_under_test: RunDirectoryTracker) -> None:
    """Invoke record_artifact with a throwaway name and path."""
    tracker_under_test.record_artifact("model", Path("model.joblib"))


def call_complete_run(tracker_under_test: RunDirectoryTracker) -> None:
    """Invoke complete_run."""
    tracker_under_test.complete_run()


def call_fail_run(tracker_under_test: RunDirectoryTracker) -> None:
    """Invoke fail_run with a throwaway error summary."""
    tracker_under_test.fail_run("boom")


_LIFECYCLE_METHOD_CALLS = [
    call_record_metrics,
    call_record_artifact,
    call_complete_run,
    call_fail_run,
]
_LIFECYCLE_METHOD_IDS = ["record_metrics", "record_artifact", "complete_run", "fail_run"]


# RunDirectoryTracker construction and properties


def test_run_id_raises_before_start(tracker: RunDirectoryTracker) -> None:
    """Test reading run_id before start_run raises RuntimeError."""
    with pytest.raises(RuntimeError):
        _ = tracker.run_id


def test_run_directory_raises_before_start(tracker: RunDirectoryTracker) -> None:
    """Test reading run_directory before start_run raises RuntimeError."""
    with pytest.raises(RuntimeError):
        _ = tracker.run_directory


def test_run_id_matches_timestamp_and_suffix_shape(tracker: RunDirectoryTracker) -> None:
    """Test the run id matches the UTC-timestamp-plus-hex-suffix shape."""
    start_tracked_run(tracker)

    assert _RUN_ID_PATTERN.fullmatch(tracker.run_id)


def test_run_directory_is_runs_root_child_named_by_run_id(
    tracker: RunDirectoryTracker,
    runs_root: Path,
) -> None:
    """Test the run directory sits directly under the runs root, named for the run id."""
    start_tracked_run(tracker)

    assert tracker.run_directory == runs_root / tracker.run_id


def test_run_ids_are_unique_across_rapid_starts(runs_root: Path) -> None:
    """Test five trackers started back-to-back against one root get distinct run ids."""
    trackers = [RunDirectoryTracker(runs_root) for _ in range(5)]

    for candidate in trackers:
        start_tracked_run(candidate)

    run_ids = {candidate.run_id for candidate in trackers}
    assert len(run_ids) == 5


# start_run


def test_start_run_creates_running_record(tracker: RunDirectoryTracker) -> None:
    """Test start_run writes a running record with the expected fields."""
    start_tracked_run(tracker, command=RunCommand.TRAIN, model_name="dummy_model")

    record = json.loads((tracker.run_directory / "run.json").read_text(encoding="utf-8"))

    assert record["schema_version"] == RUN_SCHEMA_VERSION
    assert record["status"] == "running"
    assert record["command"] == "train"
    assert record["model_name"] == "dummy_model"
    assert record["finished_at"] is None
    assert record["error"] is None
    started_at = datetime.fromisoformat(record["started_at"])
    assert started_at.tzinfo is not None


def test_start_run_snapshots_configs_as_yaml(tracker: RunDirectoryTracker) -> None:
    """Test config mappings passed to start_run round-trip through their YAML snapshots."""
    model_config = {"learning_rate": 0.01, "layers": 3}
    train_config = {"epochs": 5, "batch_size": 32}

    start_tracked_run(
        tracker,
        configs={"model_config": model_config, "train_config": train_config},
    )

    loaded_model_config = load_yaml_config(
        tracker.run_directory / "configs" / "model_config.yaml",
    )
    loaded_train_config = load_yaml_config(
        tracker.run_directory / "configs" / "train_config.yaml",
    )
    assert loaded_model_config == model_config
    assert loaded_train_config == train_config


def test_start_run_snapshots_empty_config_as_empty_mapping(tracker: RunDirectoryTracker) -> None:
    """Test an empty config mapping still snapshots to a file that loads back empty."""
    start_tracked_run(tracker, configs={"train_config": {}})

    config_path = tracker.run_directory / "configs" / "train_config.yaml"

    assert config_path.is_file()
    assert len(load_yaml_config(config_path)) == 0


def test_start_run_creates_nested_runs_root_on_demand(tmp_path: Path) -> None:
    """Test start_run creates a multi-level runs root that does not yet exist."""
    runs_root = tmp_path / "a" / "b" / "c"
    tracker_instance = RunDirectoryTracker(runs_root)

    start_tracked_run(tracker_instance)

    assert tracker_instance.run_directory.is_dir()
    assert tracker_instance.run_directory.parent == runs_root


def test_start_run_twice_raises_runtime_error(tracker: RunDirectoryTracker) -> None:
    """Test calling start_run a second time on the same tracker raises."""
    start_tracked_run(tracker)

    with pytest.raises(RuntimeError):
        start_tracked_run(tracker)


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory permission checks")
def test_start_run_unwritable_root_raises_oserror(tmp_path: Path) -> None:
    """Test an unwritable runs root parent raises OSError from start_run."""
    parent = tmp_path / "parent"
    parent.mkdir()
    runs_root = parent / "runs"
    tracker_instance = RunDirectoryTracker(runs_root)
    original_mode = stat.S_IMODE(parent.stat().st_mode)
    parent.chmod(0o500)

    try:
        with pytest.raises(PermissionError):
            start_tracked_run(tracker_instance)
    finally:
        parent.chmod(original_mode)


# Lifecycle misuse


@pytest.mark.parametrize(
    "call_lifecycle_method",
    _LIFECYCLE_METHOD_CALLS,
    ids=_LIFECYCLE_METHOD_IDS,
)
def test_lifecycle_methods_before_start_raise_runtime_error(
    tracker: RunDirectoryTracker,
    call_lifecycle_method: Callable[[RunDirectoryTracker], None],
) -> None:
    """Test lifecycle methods raise RuntimeError when called before start_run."""
    with pytest.raises(RuntimeError):
        call_lifecycle_method(tracker)


@pytest.mark.parametrize(
    "call_lifecycle_method",
    _LIFECYCLE_METHOD_CALLS,
    ids=_LIFECYCLE_METHOD_IDS,
)
def test_lifecycle_methods_after_finalize_raise_runtime_error(
    tracker: RunDirectoryTracker,
    call_lifecycle_method: Callable[[RunDirectoryTracker], None],
) -> None:
    """Test lifecycle methods raise RuntimeError once a run has completed."""
    start_tracked_run(tracker)
    tracker.complete_run()

    with pytest.raises(RuntimeError):
        call_lifecycle_method(tracker)


def test_fail_run_after_fail_run_raises_runtime_error(tracker: RunDirectoryTracker) -> None:
    """Test calling fail_run a second time after a run has already failed raises."""
    start_tracked_run(tracker)
    tracker.fail_run("first failure")

    with pytest.raises(RuntimeError):
        tracker.fail_run("second failure")


# Finalization


def test_complete_run_finalizes_record(tracker: RunDirectoryTracker) -> None:
    """Test complete_run finalizes the record with status, timestamp, and metrics."""
    start_tracked_run(tracker)
    tracker.record_metrics("train", {"sample_count": 3})

    tracker.complete_run()

    record = json.loads((tracker.run_directory / "run.json").read_text(encoding="utf-8"))
    started_at = datetime.fromisoformat(record["started_at"])
    finished_at = datetime.fromisoformat(record["finished_at"])
    assert record["status"] == "completed"
    assert finished_at.tzinfo is not None
    assert finished_at >= started_at
    assert record["metrics"]["train"] == {"sample_count": 3}


def test_complete_run_without_metrics_yields_empty_metrics_mapping(
    tracker: RunDirectoryTracker,
) -> None:
    """Test completing a run with no recorded metrics yields an empty metrics mapping."""
    start_tracked_run(tracker)

    tracker.complete_run()

    record = json.loads((tracker.run_directory / "run.json").read_text(encoding="utf-8"))
    assert record["metrics"] == {}


def test_fail_run_records_error_and_failed_status(tracker: RunDirectoryTracker) -> None:
    """Test fail_run finalizes the record with status failed and the error verbatim."""
    start_tracked_run(tracker)

    tracker.fail_run("RuntimeError: boom")

    record = json.loads((tracker.run_directory / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["error"] == "RuntimeError: boom"


def test_metrics_are_keyed_by_phase(tracker: RunDirectoryTracker) -> None:
    """Test metrics recorded under different phases both appear, keyed by phase."""
    start_tracked_run(tracker)
    tracker.record_metrics("train", {"sample_count": 3})
    tracker.record_metrics("test", {"sample_count": 2})

    tracker.complete_run()

    record = json.loads((tracker.run_directory / "run.json").read_text(encoding="utf-8"))
    assert record["metrics"]["train"] == {"sample_count": 3}
    assert record["metrics"]["test"] == {"sample_count": 2}


def test_record_artifact_appears_in_record(
    tracker: RunDirectoryTracker,
    tmp_path: Path,
) -> None:
    """Test a recorded artifact appears in the record keyed by name."""
    start_tracked_run(tracker)
    artifact_path = tmp_path / "submission.csv"

    tracker.record_artifact("submission", artifact_path)
    tracker.complete_run()

    record = json.loads((tracker.run_directory / "run.json").read_text(encoding="utf-8"))
    assert record["artifacts"]["submission"] == str(artifact_path)


def test_non_json_metric_values_are_stringified(tracker: RunDirectoryTracker) -> None:
    """Test metric values without a native JSON representation are stringified."""
    start_tracked_run(tracker)
    odd_value = complex(1, 2)

    tracker.record_metrics("train", {"weight": odd_value})
    tracker.complete_run()

    record = json.loads((tracker.run_directory / "run.json").read_text(encoding="utf-8"))
    loaded = load_run(tracker.run_directory.parent, tracker.run_id)
    assert record["metrics"]["train"]["weight"] == str(odd_value)
    assert loaded.metrics["train"]["weight"] == str(odd_value)


# list_runs


def test_list_runs_missing_root_returns_empty_tuple(tmp_path: Path) -> None:
    """Test list_runs on a nonexistent root returns an empty tuple."""
    missing_root = tmp_path / "runs"

    result = list_runs(missing_root)

    assert result == ()


def test_list_runs_empty_root_returns_empty_tuple(tmp_path: Path) -> None:
    """Test list_runs on an existing but empty root returns an empty tuple."""
    empty_root = tmp_path / "runs"
    empty_root.mkdir()

    result = list_runs(empty_root)

    assert result == ()


def test_list_runs_returns_records_sorted_by_run_id(runs_root: Path) -> None:
    """Test list_runs returns completed runs sorted by run id."""
    run_ids = [complete_tracked_run(RunDirectoryTracker(runs_root)) for _ in range(3)]

    result = list_runs(runs_root)

    assert isinstance(result, tuple)
    assert all(isinstance(record, RunRecord) for record in result)
    listed_ids = [record.run_id for record in result]
    assert listed_ids == sorted(run_ids)
    assert len(set(listed_ids)) == len(listed_ids)


def test_list_runs_ignores_non_run_entries(
    tracker: RunDirectoryTracker,
    runs_root: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a stray file is silently ignored while a run-less directory warns."""
    runs_root.mkdir()
    (runs_root / "notes.txt").write_text("not a run", encoding="utf-8")
    (runs_root / "not-a-run-id").mkdir()
    run_id = complete_tracked_run(tracker)

    with caplog.at_level(logging.WARNING):
        result = list_runs(runs_root)

    assert [record.run_id for record in result] == [run_id]
    assert not any("notes.txt" in message for message in caplog.messages)
    assert any("not-a-run-id" in message for message in caplog.messages)


def test_list_runs_skips_corrupt_record_with_warning(
    runs_root: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a corrupt run.json is skipped with one warning naming its directory."""
    good_run_id = complete_tracked_run(RunDirectoryTracker(runs_root))
    corrupt_run_id = complete_tracked_run(RunDirectoryTracker(runs_root))
    (runs_root / corrupt_run_id / "run.json").write_text("not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        result = list_runs(runs_root)

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert corrupt_run_id in warnings[0].getMessage()
    assert [record.run_id for record in result] == [good_run_id]


def test_list_runs_includes_running_and_failed_runs(runs_root: Path) -> None:
    """Test list_runs includes runs that are still running or have failed."""
    running_tracker = RunDirectoryTracker(runs_root)
    start_tracked_run(running_tracker)
    failed_tracker = RunDirectoryTracker(runs_root)
    start_tracked_run(failed_tracker)
    failed_tracker.fail_run("boom")

    result = list_runs(runs_root)

    statuses_by_id = {record.run_id: record.status for record in result}
    assert statuses_by_id[running_tracker.run_id] == RunStatus.RUNNING
    assert statuses_by_id[failed_tracker.run_id] == RunStatus.FAILED


def test_list_runs_records_have_frozen_mappings(
    tracker: RunDirectoryTracker,
    runs_root: Path,
) -> None:
    """Test records returned by list_runs carry a frozen metrics mapping."""
    start_tracked_run(tracker)
    tracker.record_metrics("train", {"sample_count": 1})
    tracker.complete_run()

    result = list_runs(runs_root)

    metrics = result[0].metrics
    assert isinstance(metrics, immutabledict)
    with pytest.raises(TypeError):
        metrics["train"] = {}  # type: ignore[index]


# load_run


def test_load_run_round_trips_completed_record(
    tracker: RunDirectoryTracker,
    runs_root: Path,
    tmp_path: Path,
) -> None:
    """Test load_run reproduces every field of a completed run written by the tracker."""
    start_tracked_run(
        tracker,
        command=RunCommand.TRAIN,
        model_name="logistic_regression",
        configs={"model_config": {"a": 1}, "train_config": {"b": 2}},
    )
    tracker.record_metrics("train", {"sample_count": 3})
    artifact_path = tmp_path / "model.joblib"
    tracker.record_artifact("model", artifact_path)
    tracker.complete_run()

    record = load_run(runs_root, tracker.run_id)

    assert record.run_id == tracker.run_id
    assert record.command is RunCommand.TRAIN
    assert record.model_name == "logistic_regression"
    assert record.status is RunStatus.COMPLETED
    assert record.started_at.tzinfo is not None
    assert record.finished_at is not None
    assert record.finished_at.tzinfo is not None
    assert record.metrics["train"] == {"sample_count": 3}
    assert record.artifacts["model"] == artifact_path
    assert record.config_paths["model_config"] == Path("configs/model_config.yaml")
    assert record.config_paths["train_config"] == Path("configs/train_config.yaml")


def test_load_run_accepts_string_runs_root(
    tracker: RunDirectoryTracker,
    runs_root: Path,
) -> None:
    """Test load_run accepts a string runs root with the same result as a Path."""
    run_id = complete_tracked_run(tracker)

    from_string_root = load_run(str(runs_root), run_id)
    from_path_root = load_run(runs_root, run_id)

    assert from_string_root == from_path_root


def test_load_run_missing_run_raises_file_not_found(tmp_path: Path) -> None:
    """Test loading a nonexistent run id raises FileNotFoundError."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    with pytest.raises(FileNotFoundError):
        load_run(runs_root, "20260101T000000Z-abcd")


def test_load_run_directory_without_record_raises_file_not_found(tmp_path: Path) -> None:
    """Test a run directory that exists but has no run.json raises FileNotFoundError."""
    runs_root = tmp_path / "runs"
    run_id = "20260101T000000Z-abcd"
    (runs_root / run_id).mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        load_run(runs_root, run_id)


def test_load_run_corrupt_record_raises_value_error(tmp_path: Path) -> None:
    """Test a run.json that is not valid JSON raises ValueError."""
    runs_root = tmp_path / "runs"
    run_id = "20260101T000000Z-abcd"
    run_directory = runs_root / run_id
    run_directory.mkdir(parents=True)
    (run_directory / "run.json").write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError, match=run_id):
        load_run(runs_root, run_id)


@pytest.mark.parametrize(
    "run_json_content",
    ["[]", json.dumps({"schema_version": 1, "run_id": "20260101T000000Z-abcd"})],
    ids=["json-list-root", "missing-status-key"],
)
def test_load_run_valid_json_wrong_shape_raises_value_error(
    tmp_path: Path,
    run_json_content: str,
) -> None:
    """Test valid JSON that does not match the run record shape raises ValueError."""
    runs_root = tmp_path / "runs"
    run_id = "20260101T000000Z-abcd"
    run_directory = runs_root / run_id
    run_directory.mkdir(parents=True)
    (run_directory / "run.json").write_text(run_json_content, encoding="utf-8")

    with pytest.raises(ValueError, match=run_id):
        load_run(runs_root, run_id)
