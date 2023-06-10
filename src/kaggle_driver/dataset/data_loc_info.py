"""Information pertaining to the paths of the dataset and further processed
    versions of that dataset.
"""
import dataclasses
from typing import Optional


@dataclasses.dataclass
class DataLocInfo:
    """Information pertaining to the paths of the dataset and further processed
        versions of that dataset.

    :param raw_train_dir_path: The path to the directory containing
        the raw training data.
    :type raw_train_dir_path: str
    :param raw_test_dir_path: The path to the directory containing
        the raw test data.
    :type raw_test_dir_path: str
    :param interim_train_dir_path: The path to the directory
        that will contain the interim training data.
    :type interim_train_dir_path: Optional[str]
    :default interim_train_dir_path: None
    :param interim_test_dir_path: The path to the directory
        that will contain the interim test data.
    :type interim_test_dir_path: Optional[str]
    :default interim_test_dir_path: None
    :param processed_train_dir_path: The path to the directory
        that will contain the processed training data.
    :type processed_train_dir_path: Optional[str]
    :default processed_train_dir_path: None
    :param processed_test_dir_path: The path to the directory
        that will contain the processed test data.
    :type processed_test_dir_path: Optional[str]
    """

    raw_train_dir_path: str
    raw_test_dir_path: str
    interim_train_dir_path: Optional[str] = None
    interim_test_dir_path: Optional[str] = None
    processed_train_dir_path: Optional[str] = None
    processed_test_dir_path: Optional[str] = None
