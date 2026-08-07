"""Unit tests for :mod:`kaggle_driver.driver` orchestration."""

from __future__ import annotations

import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from immutabledict import immutabledict

from kaggle_driver import KaggleInfo, driver
from kaggle_driver.driver import require_kaggle
from tests.conftest import _DummyDataset, _DummyModel


def test_train_instantiates_and_invokes_model(
    dummy_dataset: _DummyDataset,
) -> None:
    """Test ``train`` constructs the model with model_config kwargs and invokes ``Model.train``."""
    statistics = driver.train(
        dummy_dataset,
        _DummyModel,
        model_config={},
        train_config={"alpha": 1},
    )

    assert statistics == {"sample_count": 3}


def test_test_writes_submission(
    dummy_dataset: _DummyDataset,
    tmp_path: Path,
) -> None:
    """Test ``test`` writes a submission via ``Dataset.store_predictions``."""
    submission = tmp_path / "out" / "submission.csv"

    statistics = driver.test(
        dummy_dataset,
        _DummyModel,
        submission_path=submission,
        model_config={},
        test_config={},
    )

    assert statistics == {"sample_count": 2}
    assert submission.is_file()


def test_train_returns_frozen_statistics(
    dummy_dataset: _DummyDataset,
) -> None:
    """Test the statistics handed back to the caller cannot be edited in place."""
    statistics = driver.train(
        dummy_dataset,
        _DummyModel,
        model_config={},
        train_config={},
    )

    assert isinstance(statistics, immutabledict)
    with pytest.raises(TypeError):
        statistics["sample_count"] = 99  # type: ignore[index]


def test_test_returns_frozen_statistics(
    dummy_dataset: _DummyDataset,
    tmp_path: Path,
) -> None:
    """Test the statistics handed back from ``test`` cannot be edited in place."""
    statistics = driver.test(
        dummy_dataset,
        _DummyModel,
        submission_path=tmp_path / "submission.csv",
        model_config={},
        test_config={},
    )

    assert isinstance(statistics, immutabledict)
    with pytest.raises(TypeError):
        statistics["sample_count"] = 99  # type: ignore[index]


def test_train_freezes_data_and_config_before_the_model_sees_them(
    dummy_dataset: _DummyDataset,
) -> None:
    """Test a dataset returning a plain dict still reaches the model frozen."""
    seen_train_data: list[Mapping[str, tuple[int, int]]] = []
    seen_train_config: list[Mapping[str, Any]] = []

    class CapturingModel(_DummyModel):
        def train(
            self,
            train_data: Mapping[str, tuple[int, int]],
            config: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            seen_train_data.append(train_data)
            seen_train_config.append(config)
            return super().train(train_data, config)

    driver.train(
        dummy_dataset,
        CapturingModel,
        model_config={},
        train_config={"alpha": 1},
    )

    assert isinstance(seen_train_data[0], immutabledict)
    assert isinstance(seen_train_config[0], immutabledict)
    assert dict(seen_train_data[0]) == {"a": (1, 2), "b": (3, 4), "c": (5, 6)}


def test_test_freezes_data_and_config_before_the_model_sees_them(
    dummy_dataset: _DummyDataset,
    tmp_path: Path,
) -> None:
    """Test test inputs and config are frozen on the way into the model."""
    seen_test_data: list[Mapping[str, int]] = []
    seen_test_config: list[Mapping[str, Any]] = []

    class CapturingModel(_DummyModel):
        def test(
            self,
            test_data: Mapping[str, int],
            config: Mapping[str, Any],
        ) -> tuple[Mapping[str, int], Mapping[str, Any]]:
            seen_test_data.append(test_data)
            seen_test_config.append(config)
            return super().test(test_data, config)

    driver.test(
        dummy_dataset,
        CapturingModel,
        submission_path=tmp_path / "submission.csv",
        model_config={},
        test_config={"beta": 2},
    )

    assert isinstance(seen_test_data[0], immutabledict)
    assert isinstance(seen_test_config[0], immutabledict)
    assert dict(seen_test_data[0]) == {"x": 10, "y": 20}


def test_store_predictions_receives_a_frozen_mapping(
    dummy_dataset: _DummyDataset,
    tmp_path: Path,
) -> None:
    """Test the dataset cannot mutate predictions while writing the submission."""
    seen: list[Mapping[str, int]] = []

    class RecordingDataset(_DummyDataset):
        def store_predictions(self, path: Path, predictions: Mapping[str, int]) -> None:
            seen.append(predictions)
            super().store_predictions(path, predictions)

    recording = RecordingDataset(
        raw_train_directory=dummy_dataset.raw_train_directory,
        raw_test_directory=dummy_dataset.raw_test_directory,
    )

    driver.test(
        recording,
        _DummyModel,
        submission_path=tmp_path / "submission.csv",
        model_config={},
        test_config={},
    )

    assert len(seen) == 1
    assert isinstance(seen[0], immutabledict)


def test_download_uses_kaggle_api_and_organize(
    tmp_path: Path,
) -> None:
    """Test ``download`` calls the Kaggle API, unzips, and invokes the organizer."""
    raw_train = tmp_path / "raw_train"
    raw_test = tmp_path / "raw_test"

    class CapturingDataset(_DummyDataset):
        pass

    capturing = CapturingDataset(raw_train_directory=raw_train, raw_test_directory=raw_test)

    organize_calls: list[tuple[Path, Path, Path]] = []

    def organize(
        unzipped: Path,
        train_directory: Path,
        test_directory: Path,
    ) -> None:
        organize_calls.append((unzipped, train_directory, test_directory))

    info = KaggleInfo(competition_name="compy", organize_data_function=organize)

    fake_api = MagicMock()  # kaggle is an optional dependency; no real object to autospec against.

    def fake_download(competition: str, path: str) -> None:
        del competition
        target = Path(path) / "compy.zip"
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("payload.csv", "id,value\n1,42\n")

    fake_api.competition_download_files.side_effect = fake_download

    with (
        patch("kaggle_driver.driver.require_kaggle"),
        patch.dict(
            "sys.modules",
            {"kaggle": MagicMock(api=fake_api)},
        ),
    ):
        driver.download(capturing, info)

    fake_api.authenticate.assert_called_once()
    fake_api.competition_download_files.assert_called_once()
    assert len(organize_calls) == 1
    assert organize_calls[0][1] == raw_train
    assert organize_calls[0][2] == raw_test


def test_download_raises_when_zip_count_unexpected(
    dummy_dataset: _DummyDataset,
) -> None:
    """Test ``download`` aborts when the Kaggle API drops zero or multiple zip files."""
    info = KaggleInfo(competition_name="compy", organize_data_function=lambda *_: None)
    fake_api = MagicMock()  # kaggle is an optional dependency; no real object to autospec against.

    def fake_download(competition: str, path: str) -> None:
        del competition
        for name in ("a.zip", "b.zip"):
            (Path(path) / name).write_bytes(b"")

    fake_api.competition_download_files.side_effect = fake_download

    with (
        patch("kaggle_driver.driver.require_kaggle"),
        patch.dict(
            "sys.modules",
            {"kaggle": MagicMock(api=fake_api)},
        ),
        pytest.raises(RuntimeError, match=r"exactly one \.zip"),
    ):
        driver.download(dummy_dataset, info)


def test_require_kaggle_passes_when_installed() -> None:
    """Test the guard is silent when ``find_spec('kaggle')`` returns a value."""
    with patch("importlib.util.find_spec", return_value=object()):
        require_kaggle()


def test_require_kaggle_raises_when_missing() -> None:
    """Test the guard raises with the install hint when ``find_spec`` returns ``None``."""
    with (
        patch("importlib.util.find_spec", return_value=None),
        pytest.raises(ImportError, match=r"kaggle-driver\[kaggle\]"),
    ):
        require_kaggle()
