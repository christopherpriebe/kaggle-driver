"""A module containing the abstract base class for data.
"""
import abc
from collections import OrderedDict
from typing import Any, TypeVar


Datum = TypeVar("Datum")


class Data(abc.ABC):
    """An abstract base class for data.

    :param data: The data.
    :type data: OrderedDict[str, Datum]
    """
    _data: OrderedDict[str, Any]

    _iter_index: int

    def __init__(self, data: OrderedDict[str, Datum]) -> None:
        if not isinstance(data, OrderedDict):
            raise TypeError("The data must be an OrderedDict.")
        self._data = data.copy()
        self._iter_index = 0

    def data_points(self) -> list[Datum]:
        """Returns the data points.

        :return: The data points.
        :rtype: list[Datum]
        """
        return list(self._data.values())

    def data_point_ids(self) -> list[str]:
        """Returns the IDs of the data points.

        :return: The IDs of the data points.
        :rtype: list[str]
        """
        return list(self._data.keys())

    def __iter__(self) -> "Data":
        self._iter_index = 0
        return self

    def __next__(self) -> Datum:
        if self._iter_index >= len(self._data):
            raise StopIteration
        else:
            self._iter_index += 1
            return self.data_points()[self._iter_index - 1]
