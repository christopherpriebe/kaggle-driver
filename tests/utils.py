# pylint: disable=missing-module-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=too-few-public-methods
# pylint: disable=missing-class-docstring

from typing import Any
from kaggle_driver.dataset import Input, Target


class DummyInput(Input):
    def __init__(self, value: Any) -> None:
        super().__init__()
        self.value = value


class DummyTarget(Target):
    def __init__(self, value: Any) -> None:
        super().__init__()
        self.value = value
