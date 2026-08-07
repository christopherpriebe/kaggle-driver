"""Tests for the public abstract base classes in :mod:`kaggle_driver.core`."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
from immutabledict import immutabledict

from kaggle_driver import Dataset, KaggleInfo, Model
from kaggle_driver.core import freeze_mapping


def test_dataset_is_abstract(tmp_path: Path) -> None:
    """Test instantiating the abstract :class:`Dataset` directly fails."""
    with pytest.raises(TypeError):
        Dataset(tmp_path / "a", tmp_path / "b")  # type: ignore[abstract]


def test_model_is_abstract() -> None:
    """Test instantiating the abstract :class:`Model` directly fails."""
    with pytest.raises(TypeError):
        Model()  # type: ignore[abstract]


def test_dataset_exposes_paths_as_path_objects(tmp_path: Path) -> None:
    """Test the base class normalizes raw directory args to ``Path`` instances."""

    class TrivialDataset(Dataset[int, int]):
        def load_train(self) -> Mapping[str, tuple[int, int]]:
            return {}

        def load_test(self) -> Mapping[str, int]:
            return {}

        def store_predictions(self, path: Path, predictions: Mapping[str, int]) -> None:
            del path, predictions

    dataset = TrivialDataset(
        raw_train_directory=str(tmp_path / "train"),
        raw_test_directory=tmp_path / "test",
    )

    assert isinstance(dataset.raw_train_directory, Path)
    assert isinstance(dataset.raw_test_directory, Path)
    assert dataset.raw_train_directory == tmp_path / "train"
    assert dataset.raw_test_directory == tmp_path / "test"


def test_kaggle_info_is_frozen_dataclass() -> None:
    """Test ``KaggleInfo`` carries the competition name and organize callback immutably."""

    def organizer(unzipped: Path, train: Path, test: Path) -> None:
        del unzipped, train, test

    info = KaggleInfo(competition_name="digit-recognizer", organize_data_function=organizer)

    assert info.competition_name == "digit-recognizer"
    assert isinstance(info.organize_data_function, Callable)
    with pytest.raises(AttributeError):
        info.competition_name = "other"  # type: ignore[misc]


def test_freeze_mapping_rejects_mutation() -> None:
    """Test the frozen result refuses item assignment and deletion."""
    frozen = freeze_mapping({"a": 1, "b": 2})

    assert isinstance(frozen, immutabledict)
    assert frozen == {"a": 1, "b": 2}
    with pytest.raises(TypeError):
        frozen["c"] = 3  # type: ignore[index]
    with pytest.raises(TypeError):
        del frozen["a"]  # type: ignore[attr-defined]


def test_freeze_mapping_preserves_insertion_order() -> None:
    """Test freezing keeps the source order, which drives submission row order."""
    source = {"z": 0, "a": 1, "m": 2}

    assert list(freeze_mapping(source)) == ["z", "a", "m"]


def test_freeze_mapping_does_not_copy_already_frozen_mappings() -> None:
    """Test an ``immutabledict`` is passed through untouched rather than re-wrapped."""
    already_frozen = immutabledict({"a": 1})

    assert freeze_mapping(already_frozen) is already_frozen


def test_freeze_mapping_decouples_from_the_source_dict() -> None:
    """Test later edits to the source dict are not visible through the frozen view."""
    source = {"a": 1}
    frozen = freeze_mapping(source)

    source["b"] = 2

    assert dict(frozen) == {"a": 1}
