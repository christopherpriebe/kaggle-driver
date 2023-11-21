"""A module that contains the test data class.
"""
from collections import OrderedDict
from typing import Any, Union
from .data import Data
from .input import Input


def _check_formatted_data(data: OrderedDict[Any, Any]) -> None:
    for key, value in data:
        if not isinstance(key, str):
            raise TypeError(
                "The keys of the data must be strings."
            )
        if not isinstance(value, Input):
            raise TypeError(
                "The values of the data must be tuples."
            )


class TestData(Data):
    """A class that contains the test data.

    :param data: The test data.
    :type data: Union[
    OrderedDict[str, Input],
    dict[str, Input],
    list[tuple[str, Input]],
    tuple[tuple[str, Input], ...],
    ]
    """

    def __init__(self, data: Union[
        OrderedDict[str, Input],
        dict[str, Input],
        list[tuple[str, Input]],
        tuple[tuple[str, Input], ...],
    ]) -> None:
        if isinstance(data, OrderedDict):
            formatted_data = data
        elif isinstance(data, (dict, list, tuple)):
            formatted_data = OrderedDict(data)
        else:
            raise TypeError(
                "The data must be an OrderedDict, dict, list, or tuple."
            )

        _check_formatted_data(formatted_data)
        super().__init__(formatted_data)
