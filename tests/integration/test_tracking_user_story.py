"""User stories for experiment tracking, driven end-to-end through the public CLI."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from kaggle_driver.cli import build_app
from kaggle_driver.tracking import RunCommand, RunDirectoryTracker
from tests.conftest import _DummyDataset, _DummyModel


class _RaisingTrainModel(_DummyModel):
    """Dummy model whose train always raises, for exercising the failed-run path."""

    def train(
        self,
        train_data: Mapping[str, tuple[int, int]],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del train_data, config
        raise RuntimeError("training exploded")


@pytest.mark.integration
def test_train_then_test_then_inspect_history(
    dummy_dataset: _DummyDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a tracked train-then-test session is visible via runs list/show/compare.

    Also confirms a run's snapshotted train config can be reused verbatim to
    start a later run.
    """
    monkeypatch.chdir(tmp_path)
    app = build_app(dummy_dataset, {"dummy": _DummyModel}, kaggle_info=None)
    runner = CliRunner()
    artifact = tmp_path / "model.joblib"
    train_config = tmp_path / "train.yml"
    train_config.write_text(f"model_path: {artifact}\n", encoding="utf-8")
    submission = tmp_path / "submission.csv"

    train_result = runner.invoke(app, ["train", "dummy", "--train-config", str(train_config)])
    test_result = runner.invoke(app, ["test", "dummy", "--submission", str(submission)])

    assert train_result.exit_code == 0, train_result.stdout
    assert test_result.exit_code == 0, test_result.stdout
    runs_root = tmp_path / "runs"
    assert runs_root.is_dir()
    run_directories = sorted(entry for entry in runs_root.iterdir() if entry.is_dir())
    assert len(run_directories) == 2

    list_result = runner.invoke(app, ["runs", "list"])
    assert list_result.exit_code == 0, list_result.stdout
    directories_by_command = {
        json.loads((directory / "run.json").read_text(encoding="utf-8"))["command"]: directory.name
        for directory in run_directories
    }
    train_id = directories_by_command["train"]
    test_id = directories_by_command["test"]
    assert train_id in list_result.stdout
    assert test_id in list_result.stdout
    assert list_result.stdout.count("completed") == 2

    show_result = runner.invoke(app, ["runs", "show", train_id])
    assert show_result.exit_code == 0, show_result.stdout
    assert str(artifact) in show_result.stdout

    compare_result = runner.invoke(app, ["runs", "compare", train_id, test_id])
    assert compare_result.exit_code == 0, compare_result.stdout
    assert train_id in compare_result.stdout
    assert test_id in compare_result.stdout

    reused_config = runs_root / train_id / "configs" / "train_config.yaml"
    second_train_result = runner.invoke(
        app,
        ["train", "dummy", "--train-config", str(reused_config)],
    )

    assert second_train_result.exit_code == 0, second_train_result.stdout
    run_directories = sorted(entry for entry in runs_root.iterdir() if entry.is_dir())
    assert len(run_directories) == 3
    third_id = next(
        directory.name for directory in run_directories if directory.name not in (train_id, test_id)
    )
    assert (runs_root / third_id / "configs" / "train_config.yaml").read_text(
        encoding="utf-8",
    ) == reused_config.read_text(encoding="utf-8")


@pytest.mark.integration
def test_failed_training_is_recorded_and_visible(
    dummy_dataset: _DummyDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a model failure during train is recorded as a failed run rather than lost."""
    monkeypatch.chdir(tmp_path)
    app = build_app(dummy_dataset, {"raiser": _RaisingTrainModel}, kaggle_info=None)
    runner = CliRunner()

    train_result = runner.invoke(app, ["train", "raiser"])

    assert train_result.exit_code != 0
    runs_root = tmp_path / "runs"
    assert runs_root.is_dir()
    run_directories = list(runs_root.iterdir())
    assert len(run_directories) == 1
    record = json.loads((run_directories[0] / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert "training exploded" in record["error"]

    list_result = runner.invoke(app, ["runs", "list"])

    assert list_result.exit_code == 0, list_result.stdout
    assert run_directories[0].name in list_result.stdout
    assert "failed" in list_result.stdout


@pytest.mark.integration
def test_abandoned_run_appears_as_running(
    dummy_dataset: _DummyDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a run that crashes without finalizing still shows up as running."""
    monkeypatch.chdir(tmp_path)
    app = build_app(dummy_dataset, {"dummy": _DummyModel}, kaggle_info=None)
    runner = CliRunner()
    runs_root = tmp_path / "runs"
    tracker = RunDirectoryTracker(runs_root)
    tracker.start_run(command=RunCommand.TRAIN, model_name="dummy", configs={})

    list_result = runner.invoke(app, ["runs", "list"])

    assert list_result.exit_code == 0, list_result.stdout
    assert tracker.run_id in list_result.stdout
    assert "running" in list_result.stdout
