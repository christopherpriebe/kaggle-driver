"""A module that contains the abstract base class for datasets.
"""
import abc
from typing import Optional
from .data_loc_info import DataLocInfo
from .input import Input
from .target import Target


class Dataset(abc.ABC):
    """Abstract base class for datasets.

    :param data_loc_info: Information pertaining to the paths of the dataset
        and further processed versions of that dataset.
    :type data_loc_info: DataLocInfo
    """
    _data_loc_info: DataLocInfo
    _train: dict[str, tuple[Input, Target]]
    _test: dict[str, tuple[Input]]

    def __init__(self, data_loc_info: DataLocInfo) -> None:
        super().__init__()
        self._data_loc_info = data_loc_info
        self._train = {}
        self._test = {}

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

    def _load_train(self) -> None:
        """Loads the training dataset into the internal dictionary of inputs
            and targets.
        """
        self._train = self.load_train()

    def _load_test(self) -> None:
        """Loads the test dataset into the internal dictionary of inputs.
        """
        self._test = self.load_test()

    @abc.abstractmethod
    def load_train(self) -> dict[str, tuple[Input, Target]]:
        """Loads the raw training dataset and converts it into a dictionary
            of inputs and targets.

        :return: The dictionary of inputs and targets.
        :rtype: dict[str, tuple[Input, Target]]
        """

    @abc.abstractmethod
    def load_test(self) -> dict[str, Input]:
        """Loads the raw test dataset and converts it into a dictionary of
            inputs.

        :return: The dictionary of inputs.
        :rtype: dict[str, Input]
        """
