"""A module that contains the abstract base class for models.
"""
import abc
from kaggle_driver.dataset import Target, TestData, TrainData
from .config import TestConfig, TrainConfig
from .result import TestResult, TrainResult


class Model(abc.ABC):
    """Abstract base class for models.
    """
    _name: str

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        """Returns the name of the model.

        :return: The name of the model.
        :rtype: str
        """
        return self._name

    def train(self, train_data: TrainData,
              train_config: TrainConfig) -> TrainResult:
        """Trains the model.

        :param train_data: The training data.
        :type train_data: TrainData
        :param train_config: The training configuration.
        :type train_config: TrainConfig
        :return: The training results.
        :rtype: TrainResult
        """

    @abc.abstractmethod
    def test(self, test_data: TestData, test_config: TestConfig) \
        -> tuple[dict[str, Target], TestResult]:
        """Tests the model.

        :param test_data: The testing data.
        :type test_data: dict[str, Input]
        :param test_config: The test configuration.
        :type test_config: TestConfig
        :return: The predictions and the testing results.
        :rtype: tuple[dict[str, Target], TestResult]
        """
