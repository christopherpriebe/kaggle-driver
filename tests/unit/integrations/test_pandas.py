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


def test_default_id_column_falls_back_to_first_column(
    tiny_train_csv: Path,
    tiny_test_csv: Path,
) -> None:
    """Test each CSV is keyed by its own first column when no id column is configured."""
    dataset = PandasDataset(
        raw_train_directory=tiny_train_csv.parent,
        raw_test_directory=tiny_test_csv.parent,
        target_column="target",
    )

    assert set(dataset.load_train().keys()) == {"1", "2", "3", "4"}
    assert set(dataset.load_test().keys()) == {"5", "6"}
    assert dataset.id_column == "id"


def test_store_predictions_without_train_csv_uses_test_header(
    tiny_test_csv: Path,
    tmp_path: Path,
) -> None:
    """Test the default id column for submissions resolves without a train CSV."""
    dataset = PandasDataset(
        raw_train_directory=tmp_path / "missing_train",
        raw_test_directory=tiny_test_csv.parent,
        target_column="target",
    )
    submission = tmp_path / "submission.csv"

    dataset.store_predictions(submission, {"5": 1, "6": 0})

    assert submission.read_text(encoding="utf-8").splitlines()[0] == "id,target"


def test_load_train_missing_id_column_raises(tiny_train_csv: Path, tiny_test_csv: Path) -> None:
    """Test loading train raises a named KeyError when the configured id column is absent."""
    dataset = PandasDataset(
        raw_train_directory=tiny_train_csv.parent,
        raw_test_directory=tiny_test_csv.parent,
        target_column="target",
        id_column="row_identifier",
    )

    with pytest.raises(KeyError, match="Id column"):
        dataset.load_train()


def test_load_test_missing_id_column_raises(tiny_train_csv: Path, tiny_test_csv: Path) -> None:
    """Test loading test raises a named KeyError when the configured id column is absent."""
    dataset = PandasDataset(
        raw_train_directory=tiny_train_csv.parent,
        raw_test_directory=tiny_test_csv.parent,
        target_column="target",
        id_column="row_identifier",
    )

    with pytest.raises(KeyError, match="Id column"):
        dataset.load_test()


def test_duplicate_train_ids_raise(tiny_test_csv: Path, tmp_path: Path) -> None:
    """Test duplicate id values in the train CSV raise instead of dropping rows."""
    train_directory = tmp_path / "duplicated_train"
    train_directory.mkdir(parents=True)
    (train_directory / "train.csv").write_text(
        "id,feature,target\n1,0.1,0\n1,0.2,1\n",
        encoding="utf-8",
    )
    dataset = PandasDataset(
        raw_train_directory=train_directory,
        raw_test_directory=tiny_test_csv.parent,
        target_column="target",
        id_column="id",
    )

    with pytest.raises(ValueError, match="Duplicate id values"):
        dataset.load_train()


def test_duplicate_test_ids_raise(tiny_train_csv: Path, tmp_path: Path) -> None:
    """Test duplicate id values in the test CSV raise instead of dropping rows."""
    test_directory = tmp_path / "duplicated_test"
    test_directory.mkdir(parents=True)
    (test_directory / "test.csv").write_text(
        "id,feature\n5,0.3\n5,0.7\n",
        encoding="utf-8",
    )
    dataset = PandasDataset(
        raw_train_directory=tiny_train_csv.parent,
        raw_test_directory=test_directory,
        target_column="target",
        id_column="id",
    )

    with pytest.raises(ValueError, match="Duplicate id values"):
        dataset.load_test()
