"""A module that contains the pretty printer for the kaggle_driver package.
"""
from typing import Any


def pprint_dict(_dict: dict[Any, Any], indent: int = 0) -> str:
    """Pretty prints a dictionary.

    :param _dict: The dictionary.
    :type _dict: dict
    :param indent: The number of spaces to indent the dictionary by. Defaults
        to 0.
    :type indent: int
    :return: The pretty printed dictionary.
    :rtype: str
    """
    res: str = ""
    for key, value in _dict.items():
        res += " " * (indent + 2) + str(key) + ": "
        if isinstance(value, dict):
            res += "\n" + pprint_dict(value, indent + 2)
        else:
            res += str(value) + "\n"
    return res
