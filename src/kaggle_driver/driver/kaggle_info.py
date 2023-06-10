"""A module that contains the KaggleInfo class.
"""
import dataclasses
from typing import Callable


@dataclasses.dataclass
class KaggleInfo:
    """A class that contains information about a Kaggle competition.

    :param competition_name: The name of the competition.
    :type competition_name: str
    :param organize_data_fn: A function that organizes the downloaded dataset
        from Kaggle into the correct folders.

        The function should take in the path to the directory containing the
        downloaded dataset, the path to the directory where the train data
        should be stored, and the path to the directory where the test data
        should be stored.
    :type organize_data_fn: Callable[[str, str, str], None]
    """
    competition_name: str
    organize_data_fn: Callable[[str, str, str], None]
