"""Shared pytest fixtures and synthetic data builders."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from kaggle_driver import Dataset, Model


class _DummyDataset(Dataset[int, int]):
    """Dataset that returns three fixed examples, useful as a CLI/driver stub."""

    def load_train(self) -> Mapping[str, tuple[int, int]]:
        return {"a": (1, 2), "b": (3, 4), "c": (5, 6)}

    def load_test(self) -> Mapping[str, int]:
        return {"x": 10, "y": 20}

    def store_predictions(self, path: Path, predictions: Mapping[str, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = "\n".join(
            f"{example_id},{prediction}" for example_id, prediction in predictions.items()
        )
        path.write_text(f"id,prediction\n{rows}", encoding="utf-8")


class _DummyModel(Model[int, int]):
    """Model that predicts the input unchanged. Tracks calls for assertions."""

    def __init__(self) -> None:
        self.trained_on: dict[str, tuple[int, int]] | None = None

    def train(
        self,
        train_data: Mapping[str, tuple[int, int]],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del config
        self.trained_on = dict(train_data)
        return {"sample_count": len(train_data)}

    def test(
        self,
        test_data: Mapping[str, int],
        config: Mapping[str, Any],
    ) -> tuple[Mapping[str, int], Mapping[str, Any]]:
        del config
        return dict(test_data), {"sample_count": len(test_data)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("dummy", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> _DummyModel:
        if not path.exists():
            raise FileNotFoundError(path)
        return cls()


@pytest.fixture
def dummy_dataset(tmp_path: Path) -> _DummyDataset:
    """Return a stub :class:`Dataset` with three train rows and two test rows."""
    return _DummyDataset(
        raw_train_directory=tmp_path / "raw" / "train",
        raw_test_directory=tmp_path / "raw" / "test",
    )


@pytest.fixture
def tiny_train_csv(tmp_path: Path) -> Path:
    """Write a tiny tabular train CSV with id, two features, and a target."""
    path = tmp_path / "train" / "tabular" / "train.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "id,feature_a,feature_b,target\n1,0.1,0.2,0\n2,0.5,0.4,1\n3,0.9,0.8,1\n4,0.2,0.1,0\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def tiny_test_csv(tmp_path: Path) -> Path:
    """Write a tiny tabular test CSV with id and two features."""
    path = tmp_path / "test" / "tabular" / "test.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "id,feature_a,feature_b\n5,0.3,0.3\n6,0.7,0.7\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def tmp_yaml_config(tmp_path: Path) -> Path:
    """Write a tiny YAML file with a few scalar keys and return its path."""
    path = tmp_path / "config.yml"
    path.write_text("alpha: 0.5\nbeta: 3\nnote: hello\n", encoding="utf-8")
    return path
