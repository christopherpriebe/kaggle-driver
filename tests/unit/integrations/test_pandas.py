"""Unit tests for :class:`kaggle_driver.integrations.pandas.PandasDataset`."""

from __future__ import annotations

from pathlib import Path

import pytest

from kaggle_driver.integrations.pandas import PandasDataset


@pytest.fixture
def pandas_dataset(tiny_train_csv: Path, tiny_test_csv: Path) -> PandasDataset:
    """A ``PandasDataset`` pointed at the synthetic tabular CSVs."""
    return PandasDataset(
        raw_train_directory=tiny_train_csv.parent,
        raw_test_directory=tiny_test_csv.parent,
        target_column="target",
        id_column="id",
    )


def test_load_train_returns_rows_paired_with_targets(pandas_dataset: PandasDataset) -> None:
    """Test each train example is a (feature_series, target_scalar) pair keyed by id."""
    rows = pandas_dataset.load_train()

    assert set(rows.keys()) == {"1", "2", "3", "4"}
    series, target = rows["1"]
    assert series.tolist() == [0.1, 0.2]
    assert target == 0


def test_load_test_returns_feature_rows(pandas_dataset: PandasDataset) -> None:
    """Test each test example is a feature ``Series`` keyed by id, with no target."""
    rows = pandas_dataset.load_test()

    assert set(rows.keys()) == {"5", "6"}
    assert rows["5"].tolist() == [0.3, 0.3]


def test_store_predictions_writes_two_column_csv(
    pandas_dataset: PandasDataset,
    tmp_path: Path,
) -> None:
    """Test ``store_predictions`` writes ``(id_column, target_column)`` rows in order."""
    submission = tmp_path / "output" / "submission.csv"

    pandas_dataset.store_predictions(submission, {"5": 1, "6": 0})

    text = submission.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "id,target"
    assert "5,1" in text
    assert "6,0" in text


def test_missing_target_column_raises(tiny_test_csv: Path, tmp_path: Path) -> None:
    """Test loading raises when the target column is missing from the train CSV."""
    train_directory = tmp_path / "broken_train"
    train_directory.mkdir(parents=True)
    (train_directory / "train.csv").write_text("id,feature\n1,0.1\n", encoding="utf-8")
    dataset = PandasDataset(
        raw_train_directory=train_directory,
        raw_test_directory=tiny_test_csv.parent,
        target_column="target",
    )

    with pytest.raises(KeyError, match="Target column"):
        dataset.load_train()
