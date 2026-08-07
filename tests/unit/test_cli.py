"""Unit tests for :mod:`kaggle_driver.cli`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from kaggle_driver import KaggleInfo
from kaggle_driver.cli import build_app
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
