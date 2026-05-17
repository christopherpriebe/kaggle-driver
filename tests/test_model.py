# pylint: disable=missing-module-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=too-few-public-methods
# pylint: disable=missing-class-docstring

import os
import pytest
from utils import DummyModel, create_empty_temporary_file


@pytest.fixture
def dummy_model() -> DummyModel:
    """Returns a dummy model.

    :return: The dummy model.
    :rtype: DummyModel
    """
    return DummyModel("DeepThought")


def test_model_name(dummy_model):
    """Tests that the model correctly stores a name.
    """
    assert isinstance(dummy_model, DummyModel)

    name: str = "DeepThought"
    assert dummy_model.name == name


def test_store_and_load_model(dummy_model):
    """Tests that a model can be stored and loaded to a file.
    """
    assert isinstance(dummy_model, DummyModel)

    file_path: str = create_empty_temporary_file()
    dummy_model.store_model(file_path)
    new_dummy_model = DummyModel.load_model(file_path)
    assert isinstance(new_dummy_model, DummyModel)
    assert new_dummy_model.name == dummy_model.name
    os.remove(file_path)
