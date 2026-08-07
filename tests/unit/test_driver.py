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
from kaggle_driver.tracking import RunCommand
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


class _RecordingHook:
    """Fake ``TrackingHook`` that records every call it receives, in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def start_run(
        self,
        *,
        command: RunCommand,
        model_name: str,
        configs: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.calls.append(
            ("start_run", {"command": command, "model_name": model_name, "configs": configs}),
        )

    def record_metrics(self, phase: str, metrics: Mapping[str, Any]) -> None:
        self.calls.append(("record_metrics", phase, metrics))

    def record_artifact(self, name: str, path: Path) -> None:
        self.calls.append(("record_artifact", name, path))

    def complete_run(self) -> None:
        self.calls.append(("complete_run",))

    def fail_run(self, error: str) -> None:
        self.calls.append(("fail_run", error))


@pytest.fixture
def tracking_hook() -> _RecordingHook:
    """Return a fresh recording tracking hook."""
    return _RecordingHook()


def test_train_with_tracker_calls_hooks_in_order(
    dummy_dataset: _DummyDataset,
    tracking_hook: _RecordingHook,
) -> None:
    """Test a tracked train call invokes start_run, record_metrics, complete_run in order."""
    driver.train(
        dummy_dataset,
        _DummyModel,
        model_config={},
        train_config={"alpha": 1},
        tracker=tracking_hook,
        model_name="dummy",
    )

    call_names = [call[0] for call in tracking_hook.calls]
    assert call_names == ["start_run", "record_metrics", "complete_run"]
    _, start_kwargs = tracking_hook.calls[0]
    assert start_kwargs["command"] is RunCommand.TRAIN
    assert start_kwargs["model_name"] == "dummy"
    assert set(start_kwargs["configs"]) == {"model_config", "train_config"}
    _, phase, metrics = tracking_hook.calls[1]
    assert phase == "train"
    assert dict(metrics) == {"sample_count": 3}


def test_train_hooks_receive_frozen_mappings(
    dummy_dataset: _DummyDataset,
    tracking_hook: _RecordingHook,
) -> None:
    """Test configs and metrics passed to the hook are frozen mappings."""
    driver.train(
        dummy_dataset,
        _DummyModel,
        model_config={},
        train_config={"beta": 2},
        tracker=tracking_hook,
    )

    _, start_kwargs = tracking_hook.calls[0]
    for config in start_kwargs["configs"].values():
        assert isinstance(config, immutabledict)
    _, _, metrics = tracking_hook.calls[1]
    assert isinstance(metrics, immutabledict)


def test_train_default_model_name_is_class_name(
    dummy_dataset: _DummyDataset,
    tracking_hook: _RecordingHook,
) -> None:
    """Test the recorded model name defaults to the model class name when omitted."""
    driver.train(
        dummy_dataset,
        _DummyModel,
        model_config={},
        train_config={},
        tracker=tracking_hook,
    )

    _, start_kwargs = tracking_hook.calls[0]
    assert start_kwargs["model_name"] == "_DummyModel"


def test_train_records_model_path_artifact_when_config_declares_it(
    dummy_dataset: _DummyDataset,
    tracking_hook: _RecordingHook,
    tmp_path: Path,
) -> None:
    """Test a declared model_path in train_config records a model artifact before completion."""
    model_path = tmp_path / "model.joblib"

    driver.train(
        dummy_dataset,
        _DummyModel,
        model_config={},
        train_config={"model_path": str(model_path)},
        tracker=tracking_hook,
    )

    call_names = [call[0] for call in tracking_hook.calls]
    assert ("record_artifact", "model", model_path) in tracking_hook.calls
    assert call_names.index("record_artifact") < call_names.index("complete_run")


def test_train_without_model_path_records_no_artifact(
    dummy_dataset: _DummyDataset,
    tracking_hook: _RecordingHook,
) -> None:
    """Test no artifact is recorded when train_config has no model_path key."""
    driver.train(
        dummy_dataset,
        _DummyModel,
        model_config={},
        train_config={},
        tracker=tracking_hook,
    )

    call_names = [call[0] for call in tracking_hook.calls]
    assert "record_artifact" not in call_names


def test_train_model_exception_fails_run_and_reraises(
    dummy_dataset: _DummyDataset,
    tracking_hook: _RecordingHook,
) -> None:
    """Test a model exception during train fails the run and re-raises unchanged."""

    class _ExplodingModel(_DummyModel):
        def train(
            self,
            train_data: Mapping[str, tuple[int, int]],
            config: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            del train_data, config
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        driver.train(
            dummy_dataset,
            _ExplodingModel,
            model_config={},
            train_config={},
            tracker=tracking_hook,
        )

    call_names = [call[0] for call in tracking_hook.calls]
    assert call_names[-1] == "fail_run"
    assert "boom" in tracking_hook.calls[-1][1]
    assert "record_metrics" not in call_names
    assert "complete_run" not in call_names


def test_train_model_constructor_failure_fails_run(
    dummy_dataset: _DummyDataset,
    tracking_hook: _RecordingHook,
) -> None:
    """Test a model constructor failure still starts and fails the run before re-raising."""

    class _ExplodingConstructorModel(_DummyModel):
        def __init__(self) -> None:
            raise RuntimeError("cannot construct")

    with pytest.raises(RuntimeError, match="cannot construct"):
        driver.train(
            dummy_dataset,
            _ExplodingConstructorModel,
            model_config={},
            train_config={},
            tracker=tracking_hook,
        )

    call_names = [call[0] for call in tracking_hook.calls]
    assert call_names == ["start_run", "fail_run"]


def test_train_returns_statistics_unchanged_when_tracked(
    dummy_dataset: _DummyDataset,
    tracking_hook: _RecordingHook,
) -> None:
    """Test tracking does not alter the statistics returned from train."""
    untracked = driver.train(
        dummy_dataset,
        _DummyModel,
        model_config={},
        train_config={},
    )

    tracked = driver.train(
        dummy_dataset,
        _DummyModel,
        model_config={},
        train_config={},
        tracker=tracking_hook,
    )

    assert tracked == untracked


def test_test_default_model_name_is_class_name(
    dummy_dataset: _DummyDataset,
    tracking_hook: _RecordingHook,
    tmp_path: Path,
) -> None:
    """Test the recorded model name for test defaults to the model class name when omitted."""
    driver.test(
        dummy_dataset,
        _DummyModel,
        submission_path=tmp_path / "submission.csv",
        model_config={},
        test_config={},
        tracker=tracking_hook,
    )

    _, start_kwargs = tracking_hook.calls[0]
    assert start_kwargs["model_name"] == "_DummyModel"


def test_test_with_tracker_calls_hooks_in_order(
    dummy_dataset: _DummyDataset,
    tracking_hook: _RecordingHook,
    tmp_path: Path,
) -> None:
    """Test a tracked test call invokes hooks in order: start, metrics, artifact, complete."""
    submission = tmp_path / "submission.csv"

    driver.test(
        dummy_dataset,
        _DummyModel,
        submission_path=submission,
        model_config={},
        test_config={},
        tracker=tracking_hook,
        model_name="dummy",
    )

    call_names = [call[0] for call in tracking_hook.calls]
    assert call_names == ["start_run", "record_metrics", "record_artifact", "complete_run"]
    _, start_kwargs = tracking_hook.calls[0]
    assert start_kwargs["command"] is RunCommand.TEST
    assert set(start_kwargs["configs"]) == {"model_config", "test_config"}
    _, phase, _metrics = tracking_hook.calls[1]
    assert phase == "test"
    assert tracking_hook.calls[2] == ("record_artifact", "submission", submission)


def test_test_records_model_path_artifact_from_test_config(
    dummy_dataset: _DummyDataset,
    tracking_hook: _RecordingHook,
    tmp_path: Path,
) -> None:
    """Test a declared model_path in test_config adds a model artifact with the submission."""
    submission = tmp_path / "submission.csv"
    model_path = tmp_path / "model.joblib"

    driver.test(
        dummy_dataset,
        _DummyModel,
        submission_path=submission,
        model_config={},
        test_config={"model_path": str(model_path)},
        tracker=tracking_hook,
    )

    artifact_calls = {
        call[1]: call[2] for call in tracking_hook.calls if call[0] == "record_artifact"
    }
    assert artifact_calls == {"submission": submission, "model": model_path}


def test_test_model_exception_fails_run_and_reraises(
    dummy_dataset: _DummyDataset,
    tracking_hook: _RecordingHook,
    tmp_path: Path,
) -> None:
    """Test a model exception during test fails the run, re-raises, and skips the submission."""
    submission = tmp_path / "submission.csv"

    class _ExplodingModel(_DummyModel):
        def test(
            self,
            test_data: Mapping[str, int],
            config: Mapping[str, Any],
        ) -> tuple[Mapping[str, int], Mapping[str, Any]]:
            del test_data, config
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        driver.test(
            dummy_dataset,
            _ExplodingModel,
            submission_path=submission,
            model_config={},
            test_config={},
            tracker=tracking_hook,
        )

    call_names = [call[0] for call in tracking_hook.calls]
    assert call_names[-1] == "fail_run"
    assert not submission.exists()
