"""A module that contains the training data class.
"""
from collections import OrderedDict
from .input import Input
from .target import Target


class TrainData:
    """A class that contains the training data.

    :param data: The training data.
    :type data: OrderedDict[str, tuple[Input, Target]]
    """
    _data: OrderedDict[str, tuple[Input, Target]]

    def __init__(self, data: OrderedDict[str, tuple[Input, Target]]) -> None:
        self._data = data

    def data_points(self) -> list[tuple[Input, Target]]:
        """Returns the training data points.

        :return: The training data points.
        :rtype: list[tuple[Input, Target]]
        """
        return list(self._data.values())

    def data_point_ids(self) -> list[str]:
        """Returns the IDs of the training data points.

        :return: The IDs of the training data points.
        :rtype: list[str]
        """
        return list(self._data.keys())
