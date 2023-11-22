# pylint: disable=missing-module-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=too-few-public-methods
# pylint: disable=missing-class-docstring

from typing import Any
import pytest
from kaggle_driver.dataset import Input, Target
from kaggle_driver.directory.model_directory import model, ModelDirectory
from kaggle_driver.model import Model


@pytest.fixture
def dummy_model() -> type:
    """Returns a dummy model.

    :return: The dummy model.
    :rtype: type
    """
    class DummyModel(Model):
        def __init__(self) -> None:
            super().__init__("dummy_model")

        def test(self, test_data: dict[str, Input], *args: Any,  # pylint: disable=unused-argument
                    **kwargs: Any) -> dict[str, Target]:
            return {}
    return DummyModel


def test_model_directory_validates_correct_type(dummy_model) -> None:
    """Tests that the model directory validates a correct value type.
    """
    assert ModelDirectory.is_value_valid(dummy_model)


def test_model_directory_invalidates_incorrect_type() -> None:
    """Tests that the model directory invalidates an incorrect value type.
    """
    assert not ModelDirectory.is_value_valid("dummy_model")


def test_add_dummy_model_to_model_directory(dummy_model) -> None:
    """Tests adding a dummy model to the model directory.
    """
    dummy_model = model(dummy_model)
    assert ModelDirectory.get(dummy_model.__name__) == dummy_model
