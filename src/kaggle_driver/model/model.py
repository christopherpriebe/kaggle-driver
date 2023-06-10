"""A module containing the abstract base class for models.
"""
import abc


class Model(abc.ABC):
    """Abstract base class for models.
    """

    _name: str

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        """Returns the name of the model.

        :return: The name of the model.
        :rtype: str
        """
        return self._name
