"""Unit tests for :class:`kaggle_driver.integrations.sklearn.SklearnModel`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from kaggle_driver.integrations.sklearn import SklearnModel


@pytest.fixture
def train_data() -> dict[str, tuple[pd.Series, int]]:
    """Synthetic, linearly separable training rows keyed by example id."""
    rows = [
        ("1", pd.Series({"a": 0.1, "b": 0.1}), 0),
        ("2", pd.Series({"a": 0.9, "b": 0.9}), 1),
        ("3", pd.Series({"a": 0.2, "b": 0.2}), 0),
        ("4", pd.Series({"a": 0.8, "b": 0.8}), 1),
    ]
    return {example_id: (row, target) for example_id, row, target in rows}


@pytest.fixture
def test_data() -> dict[str, pd.Series]:
    """Synthetic test rows keyed by example id."""
    return {
        "5": pd.Series({"a": 0.95, "b": 0.95}),
        "6": pd.Series({"a": 0.05, "b": 0.05}),
    }


@pytest.fixture
def sklearn_model() -> SklearnModel:
    """A fresh :class:`SklearnModel` wrapping a logistic regression."""
    return SklearnModel(LogisticRegression())


def test_train_fits_and_reports_dimensions(
    sklearn_model: SklearnModel,
    train_data: dict[str, tuple[pd.Series, int]],
) -> None:
    """Test ``train`` returns the stacked sample and feature counts."""
    statistics = sklearn_model.train(train_data, config={})

    assert statistics == {"sample_count": 4, "feature_count": 2}


def test_test_returns_predictions_keyed_by_id(
    sklearn_model: SklearnModel,
    train_data: dict[str, tuple[pd.Series, int]],
    test_data: dict[str, pd.Series],
) -> None:
    """Test ``test`` predicts one label per test row, keyed by the test row id."""
    sklearn_model.train(train_data, config={})

    predictions, statistics = sklearn_model.test(test_data, config={})

    assert set(predictions.keys()) == {"5", "6"}
    assert statistics == {"sample_count": 2}


def test_save_load_roundtrip(
    sklearn_model: SklearnModel,
    train_data: dict[str, tuple[pd.Series, int]],
    test_data: dict[str, pd.Series],
    tmp_path: Path,
) -> None:
    """Test ``save`` then ``SklearnModel.load`` returns a model with equivalent predictions."""
    sklearn_model.train(train_data, config={})
    artifact = tmp_path / "model.joblib"
    sklearn_model.save(artifact)

    restored = SklearnModel.load(artifact)
    original_predictions, _ = sklearn_model.test(test_data, config={})
    restored_predictions, _ = restored.test(test_data, config={})

    assert original_predictions == restored_predictions
