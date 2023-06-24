"""A module that handles registering models.
"""
from typing import Any
from kaggle_driver.model import Model
from .directory import Directory


class ModelDirectory(Directory):
    """A class that handles registering models.
    """
    @classmethod
    def is_value_valid(cls: Directory, value: Any) -> bool:
        """Returns whether a value is valid for this directory.

        :param value: The value to check.
        :type value: Any
        :return: Whether the value is valid.
        :rtype: bool
        """
        return isinstance(value, type) and issubclass(value, Model)


def model(_model: type) -> type:
    """Registers a model.

    :param _model: The model to register.
    :type _model: Model
    :return: The model.
    :rtype: Model
    """
    if not isinstance(_model, type):
        raise TypeError(f"Expected type, got {_model}")
    key: str = _model.__name__
    if ModelDirectory.contains(key):
        raise ValueError(f"Model {key} already registered.")
    ModelDirectory.set(key, _model)
    return _model
