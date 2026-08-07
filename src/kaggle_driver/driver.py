"""Orchestration for the download, train, and test subcommands.

These functions are called by the Typer CLI in :mod:`kaggle_driver.cli`.
They are kept as plain functions rather than methods so the CLI layer and
the test suite can drive them directly.

Every mapping handed to user code, and every mapping handed back to the
caller, is frozen here. User implementations may return ordinary dicts;
this module is the boundary that makes them immutable.
"""

import importlib.util
import logging
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from immutabledict import immutabledict

from kaggle_driver.core import Dataset, KaggleInfo, Model, freeze_mapping
from kaggle_driver.tracking import TrackingHook

__all__ = ["download", "require_kaggle", "test", "train"]

_logger = logging.getLogger(__name__)


def _extract_single_archive(workspace: Path) -> Path:
    """Find the single Kaggle archive in ``workspace`` and unzip it.

    Args:
        workspace: Directory containing the downloaded archive.

    Returns:
        Directory the archive was extracted into.

    Raises:
        RuntimeError: ``workspace`` does not contain exactly one .zip file.
    """
    zip_files = sorted(workspace.glob("*.zip"))
    if len(zip_files) != 1:
        raise RuntimeError(
            f"Expected exactly one .zip file in the Kaggle download workspace, "
            f"found {len(zip_files)}: {[zip_file.name for zip_file in zip_files]}",
        )

    unzipped_directory = workspace / "unzipped"
    unzipped_directory.mkdir()
    with zipfile.ZipFile(zip_files[0]) as archive:
        archive.extractall(unzipped_directory)

    return unzipped_directory


def require_kaggle() -> None:
    """Raise a helpful ImportError if the optional kaggle extra is missing.

    The Kaggle Python client is heavy and reads ``~/.kaggle/kaggle.json`` at
    import time. It is therefore an optional dependency, and the framework
    only imports it inside this guard.

    Raises:
        ImportError: The ``kaggle`` package is not installed.
    """
    if importlib.util.find_spec("kaggle") is None:
        raise ImportError(
            "The Kaggle API is not installed. Install with 'pip install kaggle-driver[kaggle]'.",
        )


def download(dataset: Dataset[Any, Any], kaggle_info: KaggleInfo) -> None:
    """Download and organize the competition dataset via the Kaggle API.

    Creates a temporary workspace, asks the Kaggle API for the competition
    archive, unzips it, and delegates to ``kaggle_info.organize_data_function``
    to move the files into the dataset's raw directories.

    Args:
        dataset: User's ``Dataset`` instance. The raw directories on the
            dataset are passed to the organize callback.
        kaggle_info: Kaggle competition metadata.

    Raises:
        ImportError: The Kaggle extra is not installed.
        RuntimeError: The download produced an unexpected number of zip
            files in the workspace.
    """
    require_kaggle()
    # The kaggle import is deferred because the kaggle package is an optional
    # extra and reading kaggle.json at import time would break offline users.
    from kaggle import api

    api.authenticate()

    with tempfile.TemporaryDirectory(prefix="kaggle-driver-") as workspace_name:
        workspace = Path(workspace_name)
        _logger.info("Downloading competition %s to %s", kaggle_info.competition_name, workspace)
        api.competition_download_files(kaggle_info.competition_name, path=str(workspace))

        unzipped_directory = _extract_single_archive(workspace)

        dataset.raw_train_directory.mkdir(parents=True, exist_ok=True)
        dataset.raw_test_directory.mkdir(parents=True, exist_ok=True)
        try:
            kaggle_info.organize_data_function(
                unzipped_directory,
                dataset.raw_train_directory,
                dataset.raw_test_directory,
            )
        except Exception:
            _logger.exception("organize_data_function failed; raw directories may be partial")
            raise

    _logger.info("Download complete: %s", kaggle_info.competition_name)


def train(
    dataset: Dataset[Any, Any],
    model_class: type[Model[Any, Any]],
    *,
    model_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
    tracker: TrackingHook | None = None,
    model_name: str | None = None,
) -> immutabledict[str, Any]:
    """Instantiate the model and fit it on the dataset's training data.

    When a tracker is given, the invocation is recorded as a run:
    ``start_run`` before the model is instantiated, ``record_metrics``
    under the ``"train"`` phase, the conventional ``model_path`` config
    key as the ``model`` artifact when present, and ``complete_run`` on
    success. If the model raises, ``fail_run`` records the error and the
    exception propagates unchanged.

    Args:
        dataset: User's ``Dataset`` instance.
        model_class: Model class to instantiate. Constructor receives
            ``model_config`` as kwargs.
        model_config: Kwargs forwarded to ``model_class``.
        train_config: Free-form training configuration passed to
            ``Model.train``. Frozen before it reaches the model.
        tracker: Optional tracking hook. ``None`` disables tracking.
        model_name: Name recorded in the run record. Defaults to
            ``model_class.__name__``; the CLI passes the CLI-facing name.

    Returns:
        Frozen training statistics returned by ``Model.train``.
    """
    if tracker is not None:
        message = f"Run tracking is not implemented yet ({model_name or model_class.__name__})."
        raise NotImplementedError(message)
    _logger.info("Training model %s", model_class.__name__)
    model = model_class(**model_config)
    train_data = freeze_mapping(dataset.load_train())
    statistics = freeze_mapping(model.train(train_data, freeze_mapping(train_config)))
    _logger.info("Training complete: %s", statistics)
    return statistics


def test(
    dataset: Dataset[Any, Any],
    model_class: type[Model[Any, Any]],
    *,
    submission_path: Path,
    model_config: Mapping[str, Any],
    test_config: Mapping[str, Any],
    tracker: TrackingHook | None = None,
    model_name: str | None = None,
) -> immutabledict[str, Any]:
    """Instantiate the model, generate predictions, and write a submission.

    When a tracker is given, the invocation is recorded as a run:
    ``start_run`` before the model is instantiated, ``record_metrics``
    under the ``"test"`` phase, the submission file as the ``submission``
    artifact, the conventional ``model_path`` config key as the ``model``
    artifact when present, and ``complete_run`` on success. If the model
    raises, ``fail_run`` records the error and the exception propagates
    unchanged.

    Args:
        dataset: User's ``Dataset`` instance.
        model_class: Model class to instantiate. Constructor receives
            ``model_config`` as kwargs.
        submission_path: File path to write the submission to.
        model_config: Kwargs forwarded to ``model_class``.
        test_config: Free-form test configuration passed to ``Model.test``.
            Frozen before it reaches the model.
        tracker: Optional tracking hook. ``None`` disables tracking.
        model_name: Name recorded in the run record. Defaults to
            ``model_class.__name__``; the CLI passes the CLI-facing name.

    Returns:
        Frozen test statistics returned by ``Model.test``.
    """
    if tracker is not None:
        message = f"Run tracking is not implemented yet ({model_name or model_class.__name__})."
        raise NotImplementedError(message)
    _logger.info("Testing model %s, submission -> %s", model_class.__name__, submission_path)
    model = model_class(**model_config)
    test_data = freeze_mapping(dataset.load_test())
    raw_predictions, raw_statistics = model.test(test_data, freeze_mapping(test_config))
    predictions = freeze_mapping(raw_predictions)
    statistics = freeze_mapping(raw_statistics)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.store_predictions(submission_path, predictions)
    _logger.info("Test complete: %s", statistics)
    return statistics
