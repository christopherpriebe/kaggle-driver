"""A module that contains the configuration classes."""
import abc
from typing import Any


class Config(abc.ABC, dict):
    """The abstract base class for configurations.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.update(kwargs)


class TestConfig(Config):
    """The test configuration class.
    """


class TrainConfig(Config):
    """The training configuration class.
    """
