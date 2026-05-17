"""A module that contains the abstract base class for models.
"""
import abc
import pickle
from typing import Any
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

    def __getstate__(self) -> dict[str, Any]:
        """Return state values to be pickled.
        """
        state: dict[str, Any] = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state from the unpickled state values.
        """
        self.__dict__.update(state)

    @staticmethod
    def load_model(file_path: str) -> "Model":
        """Load a pickled model.

        :param file_path: The file path to the pickled model.
        :type file_path: str
        :return model: The model.
        :rtype: Model
        """
        with open(file_path, "rb") as file:
            model = pickle.load(file)
            assert isinstance(model, Model)
        return model

    def store_model(self, file_path: str) -> None:
        """Pickle a model and store it.

        :param file_path: The file path to store the pickled model.
        :type file_path: str:
        :rtype: None
        """
        with open(file_path, "wb") as file:
            try:
                pickle.dump(self, file)
            except pickle.PicklingError as pe:
                raise RuntimeError(f"Model \"{self.name}\" could not be \
                                   pickled.\nTry either overwriting the \
                                   \"__getstate__\" and \"__setstate__\" \
                                   methods for {self.__class__} or ensuring \
                                   that all class attributes of \
                                   {self.__class__} implement the \
                                    \"__getstate__\" and \"__setstate__\" \
                                    methods.") from pe

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
