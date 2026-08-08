"""Unit tests for the public :func:`kaggle_driver.run` entry point."""

from __future__ import annotations

import pytest

from kaggle_driver import run
from tests.conftest import _DummyDataset


def test_run_rejects_empty_models(dummy_dataset: _DummyDataset) -> None:
    """Test run raises before dispatching when the models mapping is empty."""
    with pytest.raises(ValueError, match="at least one entry"):
        run(dummy_dataset, {})
