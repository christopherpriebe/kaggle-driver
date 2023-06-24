"""Utility functions for the kaggle_driver package.
"""
import os
from typing import Any


def make_dir_if_not_exists(path: str) -> None:
    """Makes a directory if it does not exist.

    :param path: The path to the directory.
    :type path: str
    """
    if not os.path.exists(path):
        os.mkdir(path)


def yaml_to_dict(yaml_file_path: str) -> dict[str, Any]:
    """Converts a YAML file to a dictionary.

    :param yaml_file_path: The path to the YAML file.
    :type yaml_file_path: str
    :return: The dictionary representation of the YAML file.
    :rtype: dict
    """
    import yaml  # pylint: disable=import-outside-toplevel
    with open(yaml_file_path, "r", encoding="utf-8") as yaml_file:
        return yaml.load(yaml_file, Loader=yaml.FullLoader)
