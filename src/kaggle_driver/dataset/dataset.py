"""A module that contains the abstract base class for datasets.
"""
import abc
from collections import OrderedDict
from typing import Optional
from .data_loc_info import DataLocInfo
from .input import Input
from .target import Target
from .test_data import TestData
from .train_data import TrainData


class Dataset(abc.ABC):
    """Abstract base class for datasets.

    :param data_loc_info: Information pertaining to the paths of the dataset
        and further processed versions of that dataset.
    :type data_loc_info: DataLocInfo
    """
    _data_loc_info: DataLocInfo
    _train: Optional[TrainData]
    _test: Optional[TestData]

    def __init__(self, data_loc_info: DataLocInfo) -> None:
        super().__init__()
        self._data_loc_info = data_loc_info
        self._train = None
        self._test = None

    @property
    def raw_train_dir_path(self) -> str:
        """The path to the directory containing the raw training data.

        :return: The path to the directory containing the raw training data.
        :rtype: str
        """
        return self._data_loc_info.raw_train_dir_path

    @property
    def raw_test_dir_path(self) -> str:
        """The path to the directory containing the raw test data.

        :return: The path to the directory containing the raw test data.
        :rtype: str
        """
        return self._data_loc_info.raw_test_dir_path

    @property
    def interim_train_dir_path(self) -> Optional[str]:
        """The path to the directory that will contain the interim training
            data.

        :return: The path to the directory that will contain the interim
            training data.
        :rtype: Optional[str]
        """
        return self._data_loc_info.interim_train_dir_path

    @property
    def interim_test_dir_path(self) -> Optional[str]:
        """The path to the directory that will contain the interim test data.

        :return: The path to the directory that will contain the interim
            test data.
        :rtype: Optional[str]
        """
        return self._data_loc_info.interim_test_dir_path

    @property
    def processed_train_dir_path(self) -> Optional[str]:
        """The path to the directory that will contain the processed training
            data.

        :return: The path to the directory that will contain the processed
            training data.
        :rtype: Optional[str]
        """
        return self._data_loc_info.processed_train_dir_path

    @property
    def processed_test_dir_path(self) -> Optional[str]:
        """The path to the directory that will contain the processed test
            data.

        :return: The path to the directory that will contain the processed
            test data.
        :rtype: Optional[str]
        """
        return self._data_loc_info.processed_test_dir_path

    @property
    def train(self) -> TrainData:
        """The training data.

        :return: The training data.
        :rtype: TrainData
        """
        if self._train is None:
            self._load_train()
        assert self._train is not None, "Something went wrong..."
        return self._train

    @property
    def test(self) -> TestData:
        """The test data.

        :return: The test data.
        :rtype: TestData
        """
        if self._test is None:
            self._load_test()
        assert self._test is not None, "Something went wrong..."
        return self._test

    def _load_train(self) -> None:
        """Loads the training dataset into the internal dictionary of inputs
            and targets.
        """
        self._train = TrainData(self.load_train())

    def _load_test(self) -> None:
        """Loads the test dataset into the internal dictionary of inputs.
        """
        self._test = TestData(self.load_test())

    @abc.abstractmethod
    def load_train(self) -> OrderedDict[str, tuple[Input, Target]]:
        """Loads the raw training dataset and converts it into an ordered
            dictionary of inputs and targets.

        :return: The ordered dictionary of inputs and targets.
        :rtype: OrderedDict[str, tuple[Input, Target]]
        """

    @abc.abstractmethod
    def load_test(self) -> OrderedDict[str, Input]:
        """Loads the raw test dataset and converts it into an ordered
            dictionary of inputs.

        :return: The ordered dictionary of inputs.
        :rtype: OrderedDict[str, Input]
        """

    @abc.abstractmethod
    def store_predictions(self, predictions_file: str,
                          predictions: dict[str, Target]) -> None:
        """Stores the predictions in a file.

        :param predictions_file: The path to the file to store the predictions
            in.
        :type predictions_file: str
        :param predictions: The predictions to store.
        :type predictions: dict[str, Target]
        """
