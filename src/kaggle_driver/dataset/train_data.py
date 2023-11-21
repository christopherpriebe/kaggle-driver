"""A module that contains the training data class.
"""
from collections import OrderedDict
from typing import Any, Union
from .data import Data
from .input import Input
from .target import Target


def _check_formatted_data(data: OrderedDict[Any, Any]) -> None:
    for key, value in data:
        if not isinstance(key, str):
            raise TypeError(
                "The keys of the data must be strings."
            )
        if not isinstance(value, tuple):
            raise TypeError(
                "The values of the data must be tuples."
            )
        if len(value) != 2:
            raise ValueError(
                "The tuples in the data must have two elements."
            )
        if not isinstance(value[0], Input):
            raise TypeError(
                "The first element of the tuples in the data must be an Input."
            )
        if not isinstance(value[1], Target):
            raise TypeError(
                "The second element of the tuples in the data must be a Target."
            )


TrainingExample = tuple[Input, Target]


class TrainData(Data):
    """A class that contains the training data.

    :param data: The training data.
    :type data: Union[
    OrderedDict[str, tuple[Input, Target]],
    dict[str, tuple[Input, Target]],
    list[tuple[str, tuple[Input, Target]]],
    tuple[tuple[str, tuple[Input, Target]], ...],
    ]
    """
    def __init__(self, data: Union[
        OrderedDict[str, TrainingExample],
        dict[str, TrainingExample],
        list[tuple[str, TrainingExample]],
        tuple[tuple[str, TrainingExample], ...],
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
