"""Core abstract base classes and data carriers for kaggle-driver.

Defines the three primitives a competition pipeline must implement:

* `Dataset`: load training and test data from disk; write predictions back out.
* `Model`: train, test, and persist a model.
* `KaggleInfo`: name plus dataset organizer for the Kaggle competition.

The base classes are generic over the user's input type ``InputT`` and
target type ``TargetT``. The framework imposes no structural requirements
on either; downstream code, mypy, and the user's own contracts decide what
they are.

Mappings that cross a framework boundary are frozen with
:func:`freeze_mapping` so no downstream stage can mutate data another stage
still depends on. Implementations of the abstract methods below may return
an ordinary ``dict``: the framework freezes it before passing it on.
"""

import abc
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from immutabledict import immutabledict

__all__ = ["Dataset", "KaggleInfo", "Model", "OrganizeDataFunction", "freeze_mapping"]

InputT = TypeVar("InputT")
TargetT = TypeVar("TargetT")

KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")

OrganizeDataFunction = Callable[[Path, Path, Path], None]


class Dataset(abc.ABC, Generic[InputT, TargetT]):
    """Abstract base for a Kaggle competition dataset.

    Subclasses load raw inputs and targets from disk and write predictions
    back out. The framework supplies the path locations and consumes the
    returned mappings.

    Args:
        raw_train_directory: Directory holding raw training files.
        raw_test_directory: Directory holding raw test files.
    """

    def __init__(
        self,
        raw_train_directory: Path | str,
        raw_test_directory: Path | str,
    ) -> None:
        self._raw_train_directory = Path(raw_train_directory)
        self._raw_test_directory = Path(raw_test_directory)

    @property
    def raw_train_directory(self) -> Path:
        """Return the directory holding raw training files."""
        return self._raw_train_directory

    @property
    def raw_test_directory(self) -> Path:
        """Return the directory holding raw test files."""
        return self._raw_test_directory

    @abc.abstractmethod
    def load_train(self) -> Mapping[str, tuple[InputT, TargetT]]:
        """Load the raw training data keyed by example id.

        Returns:
            Mapping from example id to an ``(input, target)`` pair. A plain
            ``dict`` is acceptable; the framework freezes the result before
            handing it to the model. Insertion order is preserved
            throughout and drives submission row order.
        """

    @abc.abstractmethod
    def load_test(self) -> Mapping[str, InputT]:
        """Load the raw test inputs keyed by example id.

        Returns:
            Mapping from example id to the input. A plain ``dict`` is
            acceptable; the framework freezes the result before handing it
            to the model. Insertion order is preserved throughout and
            drives submission row order.
        """

    @abc.abstractmethod
    def store_predictions(
        self,
        path: Path,
        predictions: Mapping[str, TargetT],
    ) -> None:
        """Write a submission file at ``path``.

        Args:
            path: File path to write the submission to.
            predictions: Frozen mapping from example id to predicted target.
        """


class Model(abc.ABC, Generic[InputT, TargetT]):
    """Abstract base for a competition model.

    The framework instantiates the class with kwargs taken from the model
    config (a mapping loaded from YAML), then dispatches to ``train`` or
    ``test`` with the resolved data and config. ``save`` and ``load`` handle
    persistence; the base class does not provide a pickle default because
    pickle has too many sharp edges for real ML artifacts.

    Subclasses are responsible for matching the input and target types
    declared on the ``Dataset`` they are paired with.
    """

    # `Any` below is genuinely free-form: config is user-defined YAML data and
    # statistics are whatever the user chooses to report, in both `train` and
    # `test`. The framework does not constrain their shape.
    @abc.abstractmethod
    def train(
        self,
        train_data: Mapping[str, tuple[InputT, TargetT]],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Train the model.

        Args:
            train_data: Frozen mapping from example id to an
                ``(input, target)`` pair.
            config: Frozen training configuration loaded from the train YAML.

        Returns:
            Free-form training statistics (loss curves, metrics, anything
            else the user wants logged). A plain ``dict`` is acceptable; the
            framework freezes it before returning it to the caller.
        """

    @abc.abstractmethod
    def test(
        self,
        test_data: Mapping[str, InputT],
        config: Mapping[str, Any],
    ) -> tuple[Mapping[str, TargetT], Mapping[str, Any]]:
        """Generate predictions for the test set.

        Args:
            test_data: Frozen mapping from example id to the input.
            config: Frozen test configuration loaded from the test YAML.

        Returns:
            A pair ``(predictions, statistics)``. ``predictions`` maps
            example id to the predicted target. ``statistics`` is free-form.
            Plain ``dict`` values are acceptable; the framework freezes both
            before passing them on.
        """

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained model to ``path``.

        Args:
            path: File path to write to.
        """

    @classmethod
    @abc.abstractmethod
    def load(cls, path: Path) -> "Model[InputT, TargetT]":
        """Load a previously saved model from ``path``.

        Args:
            path: File path to read from.

        Returns:
            A model instance ready to call ``test`` on.
        """


@dataclass(frozen=True)
class KaggleInfo:
    """Information needed to download a competition via the Kaggle API.

    ``competition_name`` is the Kaggle URL slug for the competition.
    ``organize_data_function`` is a callback the framework calls as
    ``organize_data_function(unzipped_directory, raw_train_directory,
    raw_test_directory)`` after the archive has been unpacked into a
    temporary workspace.
    """

    competition_name: str
    organize_data_function: OrganizeDataFunction


def freeze_mapping(
    mapping: Mapping[KeyT, ValueT],
) -> immutabledict[KeyT, ValueT]:
    """Return an immutable view of ``mapping``, preserving insertion order.

    Args:
        mapping: Mapping to freeze. Already-frozen mappings are returned
            unchanged rather than copied, so repeated freezing across
            framework boundaries costs nothing.

    Returns:
        An ``immutabledict`` holding the same entries in the same order.
    """
    if isinstance(mapping, immutabledict):
        return mapping
    return immutabledict(mapping)
