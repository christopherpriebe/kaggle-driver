"""Unit tests for :mod:`kaggle_driver.cli`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from kaggle_driver import KaggleInfo
from kaggle_driver.cli import build_app
from kaggle_driver.tracking import RunCommand, RunDirectoryTracker
from tests.conftest import _DummyDataset, _DummyModel


def build_dummy_app(
    dataset: _DummyDataset,
    *,
    kaggle_info: KaggleInfo | None = None,
) -> typer.Typer:
    """Build a Typer app wired to the dummy model under the name ``dummy``."""
    return build_app(dataset, {"dummy": _DummyModel}, kaggle_info=kaggle_info)


@pytest.fixture
def cli_runner() -> CliRunner:
    """Return a fresh Typer ``CliRunner`` for invoking a built app."""
    return CliRunner()


def test_build_app_rejects_empty_models(dummy_dataset: _DummyDataset) -> None:
    """Test an empty ``models`` mapping is a programmer error and raises at build time."""
    with pytest.raises(ValueError, match="at least one entry"):
        build_app(dummy_dataset, {}, kaggle_info=None)


def test_help_lists_subcommands(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
) -> None:
    """Test top-level help lists the download, train, and test subcommands."""
    app = build_dummy_app(dummy_dataset)

    result = cli_runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "download" in result.stdout
    assert "train" in result.stdout
    assert "test" in result.stdout


def test_train_dispatches_to_driver(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
) -> None:
    """Test the train subcommand instantiates the named model and calls its train method."""
    app = build_dummy_app(dummy_dataset)

    result = cli_runner.invoke(app, ["train", "dummy"])

    assert result.exit_code == 0, result.stdout


def test_unknown_model_name_is_rejected(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
) -> None:
    """Test an unknown model name surfaces a Typer error pointing to the valid set."""
    app = build_dummy_app(dummy_dataset)

    result = cli_runner.invoke(app, ["train", "bogus"])

    assert result.exit_code != 0
    assert "bogus" in result.stdout or "bogus" in (result.stderr or "")


def test_download_without_kaggle_info_errors(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
) -> None:
    """Test calling download with no KaggleInfo produces a clean error."""
    app = build_dummy_app(dummy_dataset)

    result = cli_runner.invoke(app, ["download"])

    assert result.exit_code != 0
    assert "kaggle_info" in result.stdout or "kaggle_info" in (result.stderr or "")


def test_download_delegates_to_driver(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
) -> None:
    """Test ``download`` delegates to ``driver.download`` when KaggleInfo is provided."""
    info = KaggleInfo(
        competition_name="dummy",
        organize_data_function=lambda *_: None,
    )
    app = build_dummy_app(dummy_dataset, kaggle_info=info)

    with patch("kaggle_driver.cli.driver.download") as download:
        result = cli_runner.invoke(app, ["download"])

    assert result.exit_code == 0, result.stdout
    download.assert_called_once_with(dummy_dataset, info)


def test_test_writes_submission(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Test the test subcommand instantiates the model and writes a submission file."""
    app = build_dummy_app(dummy_dataset)
    submission = tmp_path / "submission.csv"

    result = cli_runner.invoke(app, ["test", "dummy", "--submission", str(submission)])

    assert result.exit_code == 0, result.stdout
    assert submission.read_text(encoding="utf-8").splitlines()[0] == "id,prediction"


def test_train_loads_optional_yaml(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
    tmp_yaml_config: Path,
) -> None:
    """Test providing --train-config loads the YAML into a frozen mapping."""
    app = build_dummy_app(dummy_dataset)

    result = cli_runner.invoke(app, ["train", "dummy", "--train-config", str(tmp_yaml_config)])

    assert result.exit_code == 0, result.stdout


def test_train_records_run_by_default(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a plain train invocation records a completed run under runs/ by default."""
    monkeypatch.chdir(tmp_path)
    app = build_dummy_app(dummy_dataset)

    result = cli_runner.invoke(app, ["train", "dummy"])

    assert result.exit_code == 0, result.stdout
    runs_root = tmp_path / "runs"
    assert runs_root.is_dir()
    run_directories = list(runs_root.iterdir())
    assert len(run_directories) == 1
    record = json.loads((run_directories[0] / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["command"] == "train"
    assert record["model_name"] == "dummy"


def test_runs_root_option_redirects_recording(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test --runs-root redirects recording to the given directory instead of runs/."""
    monkeypatch.chdir(tmp_path)
    app = build_dummy_app(dummy_dataset)
    custom_root = tmp_path / "custom_runs"

    result = cli_runner.invoke(app, ["--runs-root", str(custom_root), "train", "dummy"])

    assert result.exit_code == 0, result.stdout
    assert custom_root.is_dir()
    assert list(custom_root.iterdir())
    assert not (tmp_path / "runs").exists()


def test_no_track_disables_recording(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test --no-track suppresses run recording entirely."""
    monkeypatch.chdir(tmp_path)
    app = build_dummy_app(dummy_dataset)

    result = cli_runner.invoke(app, ["--no-track", "train", "dummy"])

    assert result.exit_code == 0, result.stdout
    assert not (tmp_path / "runs").exists()


def test_no_track_wins_over_runs_root(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test --no-track suppresses recording even when --runs-root is also given."""
    monkeypatch.chdir(tmp_path)
    app = build_dummy_app(dummy_dataset)
    custom_root = tmp_path / "custom_runs"

    result = cli_runner.invoke(
        app,
        ["--runs-root", str(custom_root), "--no-track", "train", "dummy"],
    )

    assert result.exit_code == 0, result.stdout
    assert not custom_root.exists()


def test_test_records_run_with_submission_artifact(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test a tracked test invocation records the submission file as an artifact."""
    monkeypatch.chdir(tmp_path)
    app = build_dummy_app(dummy_dataset)
    submission = tmp_path / "submission.csv"

    result = cli_runner.invoke(app, ["test", "dummy", "--submission", str(submission)])

    assert result.exit_code == 0, result.stdout
    runs_root = tmp_path / "runs"
    assert runs_root.is_dir()
    run_directory = next(runs_root.iterdir())
    record = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    assert "submission" in record["artifacts"]


def test_runs_list_prints_recorded_runs(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test runs list prints every recorded run id along with its status."""
    monkeypatch.chdir(tmp_path)
    app = build_dummy_app(dummy_dataset)
    cli_runner.invoke(app, ["train", "dummy"])
    cli_runner.invoke(app, ["train", "dummy"])
    runs_root = tmp_path / "runs"
    assert runs_root.is_dir()
    run_ids = sorted(path.name for path in runs_root.iterdir())
    assert len(run_ids) == 2

    result = cli_runner.invoke(app, ["runs", "list"])

    assert result.exit_code == 0, result.stdout
    assert all(run_id in result.stdout for run_id in run_ids)
    assert "completed" in result.stdout


def test_runs_list_with_empty_root_succeeds(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test runs list exits cleanly when no runs have been recorded yet."""
    monkeypatch.chdir(tmp_path)
    app = build_dummy_app(dummy_dataset)

    result = cli_runner.invoke(app, ["runs", "list"])

    assert result.exit_code == 0, result.stdout


def test_runs_list_reports_unreadable_entry(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test runs list lists a readable run and omits one with a corrupted record."""
    monkeypatch.chdir(tmp_path)
    app = build_dummy_app(dummy_dataset)
    cli_runner.invoke(app, ["train", "dummy"])
    cli_runner.invoke(app, ["train", "dummy"])
    runs_root = tmp_path / "runs"
    assert runs_root.is_dir()
    run_directories = sorted(runs_root.iterdir())
    assert len(run_directories) == 2
    corrupted_id = run_directories[0].name
    readable_id = run_directories[1].name
    (run_directories[0] / "run.json").write_text("not json", encoding="utf-8")

    result = cli_runner.invoke(app, ["runs", "list"])

    assert result.exit_code == 0, result.stdout
    assert readable_id in result.stdout
    assert corrupted_id not in result.stdout


def test_runs_show_prints_full_record(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test runs show prints the model name, status, a metric key, and the run id."""
    monkeypatch.chdir(tmp_path)
    app = build_dummy_app(dummy_dataset)
    cli_runner.invoke(app, ["train", "dummy"])
    runs_root = tmp_path / "runs"
    assert runs_root.is_dir()
    run_id = next(runs_root.iterdir()).name

    result = cli_runner.invoke(app, ["runs", "show", run_id])

    assert result.exit_code == 0, result.stdout
    assert "dummy" in result.stdout
    assert "completed" in result.stdout
    assert "sample_count" in result.stdout
    assert run_id in result.stdout


def test_runs_show_unknown_id_fails_with_message(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test runs show exits nonzero and names the unknown id when the run does not exist."""
    monkeypatch.chdir(tmp_path)
    app = build_dummy_app(dummy_dataset)

    result = cli_runner.invoke(app, ["runs", "show", "bogus-run-id"])

    assert result.exit_code != 0
    assert "bogus-run-id" in result.stdout or "bogus-run-id" in (result.stderr or "")


def test_runs_compare_aligns_metrics_of_two_runs(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test runs compare aligns shared and missing metric keys across two runs."""
    monkeypatch.chdir(tmp_path)
    app = build_dummy_app(dummy_dataset)
    runs_root = tmp_path / "runs"
    first_tracker = RunDirectoryTracker(runs_root)
    first_tracker.start_run(command=RunCommand.TRAIN, model_name="first", configs={})
    first_tracker.record_metrics("train", {"alpha": 1, "shared": 2})
    first_tracker.complete_run()
    second_tracker = RunDirectoryTracker(runs_root)
    second_tracker.start_run(command=RunCommand.TRAIN, model_name="second", configs={})
    second_tracker.record_metrics("train", {"shared": 5})
    second_tracker.complete_run()

    result = cli_runner.invoke(
        app,
        ["runs", "compare", first_tracker.run_id, second_tracker.run_id],
    )

    assert result.exit_code == 0, result.stdout
    assert first_tracker.run_id in result.stdout
    assert second_tracker.run_id in result.stdout
    assert "shared" in result.stdout
    alpha_line = next(line for line in result.stdout.splitlines() if "alpha" in line)
    assert "1" in alpha_line
    assert "-" in alpha_line


def test_runs_show_displays_error_for_failed_run(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test runs show surfaces the error summary of a failed run."""
    monkeypatch.chdir(tmp_path)
    app = build_dummy_app(dummy_dataset)
    failed_tracker = RunDirectoryTracker(tmp_path / "runs")
    failed_tracker.start_run(command=RunCommand.TRAIN, model_name="dummy", configs={})
    failed_tracker.fail_run("RuntimeError: training exploded")

    result = cli_runner.invoke(app, ["runs", "show", failed_tracker.run_id])

    assert result.exit_code == 0, result.stdout
    assert "failed" in result.stdout
    assert "training exploded" in result.stdout


def test_runs_compare_single_id_fails_with_usage_error(
    dummy_dataset: _DummyDataset,
    cli_runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test runs compare rejects a single run id, reporting at least two are required."""
    monkeypatch.chdir(tmp_path)
    app = build_dummy_app(dummy_dataset)

    result = cli_runner.invoke(app, ["runs", "compare", "some-run-id"])

    assert result.exit_code != 0
    stdout_lower = result.stdout.lower()
    stderr_lower = (result.stderr or "").lower()
    assert "at least two" in stdout_lower or "at least two" in stderr_lower
