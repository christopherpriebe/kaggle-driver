"""A module that defines the abstract base class for all directories.
"""
import abc
from typing import Any


class Directory(abc.ABC):
    """An abstract base class for all directories.
    """

    _dir: dict[str, Any] = {}

    @classmethod
    def get(cls: "Directory", key: str) -> Any:
        """Returns the value for a key.

        :param key: The key.
        :type key: str
        :return: The value.
        :rtype: Any
        """
        return cls._dir[key]

    @classmethod
    def set(cls: "Directory", key: str, value: Any) -> None:
        """Sets the value for a key.

        :param key: The key.
        :type key: str
        :param value: The value.
        :type value: Any
        """
        if not cls.is_value_valid(value):
            raise ValueError(f"Invalid type for value {value}")
        cls._dir[key] = value

    @classmethod
    @abc.abstractmethod
    def is_value_valid(cls: "Directory", value: Any) -> bool:
        """Returns whether a value is valid for this directory.

        :param value: The value to check.
        :type value: Any
        :return: Whether the value is valid.
        :rtype: bool
        """
