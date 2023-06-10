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
        return isinstance(value, Model)


def model(_model: Model) -> Model:
    """Registers a model.

    :param _model: The model to register.
    :type _model: Model
    :return: The model.
    :rtype: Model
    """
    ModelDirectory.set(_model.name, _model)
    return _model
