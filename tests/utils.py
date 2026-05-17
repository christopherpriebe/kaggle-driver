# pylint: disable=missing-module-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=too-few-public-methods
# pylint: disable=missing-class-docstring

import tempfile
from typing import Any
from kaggle_driver.dataset import Input, Target, TestData
from kaggle_driver.model import Model, TestConfig, TestResult

class DummyInput(Input):
    def __init__(self, value: Any) -> None:
        super().__init__()
        self.value = value


class DummyTarget(Target):
    def __init__(self, value: Any) -> None:
        super().__init__()
        self.value = value


class DummyModel(Model):
    def test(self, test_data: TestData, test_config: TestConfig) \
        -> tuple[dict[str, Target], TestResult]:
        return dict(test_data), TestResult()


def create_empty_temporary_file() -> str:
    """Creates an empty temporary file and returns the file path.

    :return: The path of the temporary file.
    :rtype: str
    """
    with tempfile.TemporaryFile("wb") as file:
        file_path: str = str(file.name)
    return file_path
