"""A module that contains the test data class.
"""
from collections import OrderedDict
from .input import Input


class TestData:
    """A class that contains the test data.

    :param data: The test data.
    :type data: OrderedDict[str, Input]
    """
    _data: OrderedDict[str, Input]

    def __init__(self, data: OrderedDict[str, Input]) -> None:
        self._data = data

    def data_points(self) -> list[Input]:
        """Returns the test data points.

        :return: The test data points.
        :rtype: list[Input]
        """
        return list(self._data.values())

    def data_point_ids(self) -> list[str]:
        """Returns the IDs of the test data points.

        :return: The IDs of the test data points.
        :rtype: list[str]
        """
        return list(self._data.keys())
