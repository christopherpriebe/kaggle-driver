"""End-to-end pipeline using :class:`PandasDataset` and :class:`SklearnModel`."""

from __future__ import annotations

from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression
from typer.testing import CliRunner

import kaggle_driver as kd
from kaggle_driver.cli import build_app
from kaggle_driver.integrations.pandas import PandasDataset
from kaggle_driver.integrations.sklearn import SklearnModel


class _LogisticRegressionModel(SklearnModel):
    """SklearnModel subclass with zero constructor args, suitable for kd.run."""

    def __init__(self) -> None:
        super().__init__(LogisticRegression(max_iter=200))


@pytest.mark.integration
def test_full_train_then_test_pipeline(
    tiny_train_csv: Path,
    tiny_test_csv: Path,
    tmp_path: Path,
) -> None:
    """Test a user can drive the full train then test cycle via the public CLI."""
    dataset = PandasDataset(
        raw_train_directory=tiny_train_csv.parent,
        raw_test_directory=tiny_test_csv.parent,
        target_column="target",
        id_column="id",
    )
    app = build_app(dataset, {"logistic_regression": _LogisticRegressionModel}, kaggle_info=None)
    runner = CliRunner()
    artifact = tmp_path / "model.joblib"
    train_config = tmp_path / "train.yml"
    train_config.write_text(f"model_path: {artifact}\n", encoding="utf-8")
    test_config = tmp_path / "test.yml"
    test_config.write_text(f"model_path: {artifact}\n", encoding="utf-8")
    submission = tmp_path / "submission.csv"

    train_result = runner.invoke(
        app, ["train", "logistic_regression", "--train-config", str(train_config)]
    )

    assert train_result.exit_code == 0, train_result.stdout
    assert artifact.is_file()

    test_result = runner.invoke(
        app,
        [
            "test",
            "logistic_regression",
            "--submission",
            str(submission),
            "--test-config",
            str(test_config),
        ],
    )

    assert test_result.exit_code == 0, test_result.stdout
    assert submission.is_file()
    submission_lines = submission.read_text(encoding="utf-8").splitlines()
    assert submission_lines[0] == "id,target"
    header_line_count = 1
    prediction_row_count = 2
    assert len(submission_lines) == header_line_count + prediction_row_count


@pytest.mark.integration
def test_run_dispatch_via_public_entrypoint(
    tiny_train_csv: Path,
    tiny_test_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test ``kd.run`` builds the same app the CLI tests exercise."""
    dataset = PandasDataset(
        raw_train_directory=tiny_train_csv.parent,
        raw_test_directory=tiny_test_csv.parent,
        target_column="target",
        id_column="id",
    )
    monkeypatch.setattr("sys.argv", ["driver", "train", "logistic_regression"])

    with pytest.raises(SystemExit) as excinfo:
        kd.run(dataset, models={"logistic_regression": _LogisticRegressionModel})

    assert excinfo.value.code == 0
