"""``PandasDataset``: a CSV-backed :class:`Dataset` for tabular competitions.

Install with ``pip install kaggle-driver[pandas]``.
"""

# Kept: pandas.Series is not subscriptable at runtime (pandas 2.3.3), and this
# module's annotations use pd.Series[Any] directly in module-level function
# signatures (_build_train_examples, _build_test_inputs) and method signatures
# (load_train, load_test). Removing this import raises
# `TypeError: 'type' object is not subscriptable` at import time.
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError as _error:  # pragma: no cover
    raise ImportError(
        "kaggle_driver.integrations.pandas requires pandas. "
        "Install with 'pip install kaggle-driver[pandas]'.",
    ) from _error

from immutabledict import immutabledict

from kaggle_driver.core import Dataset

__all__ = ["PandasDataset"]


def _ensure_unique_example_ids(example_ids: list[str], file_name: str) -> None:
    """Raise when any example id appears more than once.

    A duplicate id would silently overwrite an earlier row in the
    id-keyed mappings this module builds, shrinking the data and the
    submission with no visible error.

    Args:
        example_ids: Id values in row order, already stringified.
        file_name: CSV filename the ids came from, used in the error
            message.

    Raises:
        ValueError: At least one id value appears more than once.
    """
    counts = Counter(example_ids)
    duplicated = sorted(example_id for example_id, count in counts.items() if count > 1)
    if duplicated:
        raise ValueError(
            f"Duplicate id values in {file_name}: {duplicated}. "
            f"Each row must have a unique value in the id column.",
        )


def _build_train_examples(
    features: pd.DataFrame,
    targets: pd.Series[Any],
    id_column: str,
) -> Mapping[str, tuple[pd.Series[Any], Any]]:
    """Pair each feature row with its target, keyed by the resolved id.

    Args:
        features: Training frame with the target column already removed.
        targets: Target values aligned with ``features`` by position.
        id_column: Column in ``features`` whose values become mapping keys.

    Returns:
        Frozen mapping from example id to a ``(row, target)`` pair, in
        ``features`` row order.
    """
    train_examples: dict[str, tuple[pd.Series[Any], Any]] = {}
    for position, index_label in enumerate(features.index):
        example_id = str(features.at[index_label, id_column])
        row = features.loc[index_label].drop(labels=[id_column])
        train_examples[example_id] = (row, targets.iat[position])
    return immutabledict(train_examples)


def _build_test_inputs(frame: pd.DataFrame, id_column: str) -> Mapping[str, pd.Series[Any]]:
    """Build the id-to-row mapping for test rows.

    Args:
        frame: Test frame, including the id column.
        id_column: Column whose values become mapping keys and are dropped
            from each returned row.

    Returns:
        Frozen mapping from example id to the feature row, in ``frame`` row
        order.
    """
    test_inputs: dict[str, pd.Series[Any]] = {}
    for index_label in frame.index:
        example_id = str(frame.at[index_label, id_column])
        row = frame.loc[index_label].drop(labels=[id_column])
        test_inputs[example_id] = row
    return immutabledict(test_inputs)


class PandasDataset(Dataset["pd.Series[Any]", Any]):
    """Dataset over flat ``train.csv`` and ``test.csv`` files.

    Each row in the train CSV is returned as a ``pandas.Series`` (the input)
    paired with the scalar value of the target column. Test rows are
    returned as a ``pandas.Series`` only. The id column drives both the
    mapping keys and the first column of the submission file.

    Args:
        raw_train_directory: Directory containing the train CSV.
        raw_test_directory: Directory containing the test CSV.
        target_column: Name of the target column in the train CSV.
        id_column: Name of the id column in both CSVs. When omitted, each
            CSV falls back to its own first column, and the submission
            header uses the id column resolved from the test CSV.
        train_file: Train CSV filename. Defaults to ``"train.csv"``.
        test_file: Test CSV filename. Defaults to ``"test.csv"``.
        read_csv_kwargs: Extra kwargs forwarded to ``pandas.read_csv``.
    """

    def __init__(
        self,
        raw_train_directory: Path | str,
        raw_test_directory: Path | str,
        *,
        target_column: str,
        id_column: str | None = None,
        train_file: str = "train.csv",
        test_file: str = "test.csv",
        read_csv_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(raw_train_directory, raw_test_directory)
        self._target_column = target_column
        self._id_column = id_column
        self._train_file = train_file
        self._test_file = test_file
        self._read_csv_kwargs = dict(read_csv_kwargs) if read_csv_kwargs else {}

    @property
    def target_column(self) -> str:
        """Return the target column name."""
        return self._target_column

    @property
    def id_column(self) -> str:
        """Return the id column name, resolved against the test CSV header when unset.

        The submission's rows come from the test CSV, so the default
        resolution reads that file rather than the train CSV; a machine
        holding only the test data can still write a submission.
        """
        if self._id_column is not None:
            return self._id_column
        return self._resolve_id_column(self._read_test_csv())

    def _resolve_id_column(self, frame: pd.DataFrame) -> str:
        """Resolve the id column for ``frame``, falling back to its first column.

        Args:
            frame: DataFrame whose first column is treated as the id column
                when no explicit ``id_column`` was configured.

        Returns:
            The configured id column when set, else the name of
            ``frame.columns[0]``.
        """
        if self._id_column is not None:
            return self._id_column
        return str(frame.columns[0])

    def _read_train_csv(self) -> pd.DataFrame:
        """Read the train CSV into a DataFrame."""
        return pd.read_csv(self.raw_train_directory / self._train_file, **self._read_csv_kwargs)

    def _read_test_csv(self) -> pd.DataFrame:
        """Read the test CSV into a DataFrame."""
        return pd.read_csv(self.raw_test_directory / self._test_file, **self._read_csv_kwargs)

    def load_train(self) -> Mapping[str, tuple[pd.Series[Any], Any]]:
        """Load training rows keyed by id, paired with the target scalar.

        Returns:
            Frozen mapping from example id to a ``(row, target)`` pair, in
            train CSV row order.

        Raises:
            KeyError: The configured target column or the resolved id
                column is absent from the CSV.
            ValueError: The id column holds duplicate values.
        """
        frame = self._read_train_csv()
        if self._target_column not in frame.columns:
            raise KeyError(
                f"Target column {self._target_column!r} not found in {self._train_file}; "
                f"columns are {list(frame.columns)}",
            )
        resolved_id_column = self._resolve_id_column(frame)
        if resolved_id_column not in frame.columns:
            raise KeyError(
                f"Id column {resolved_id_column!r} not found in {self._train_file}; "
                f"columns are {list(frame.columns)}",
            )
        _ensure_unique_example_ids(
            [str(value) for value in frame[resolved_id_column]],
            self._train_file,
        )
        features = frame.drop(columns=[self._target_column])
        targets = frame[self._target_column]
        return _build_train_examples(features, targets, resolved_id_column)

    def load_test(self) -> Mapping[str, pd.Series[Any]]:
        """Load test rows keyed by id.

        Returns:
            Frozen mapping from example id to the feature row, in test CSV
            row order.

        Raises:
            KeyError: The resolved id column is absent from the CSV.
            ValueError: The id column holds duplicate values.
        """
        frame = self._read_test_csv()
        resolved_id_column = self._resolve_id_column(frame)
        if resolved_id_column not in frame.columns:
            raise KeyError(
                f"Id column {resolved_id_column!r} not found in {self._test_file}; "
                f"columns are {list(frame.columns)}",
            )
        _ensure_unique_example_ids(
            [str(value) for value in frame[resolved_id_column]],
            self._test_file,
        )
        return _build_test_inputs(frame, resolved_id_column)

    def store_predictions(
        self,
        path: Path,
        predictions: Mapping[str, Any],
    ) -> None:
        """Write a ``(id_column, target_column)`` CSV submission file."""
        submission = pd.DataFrame(
            {
                self.id_column: list(predictions.keys()),
                self._target_column: list(predictions.values()),
            },
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(path, index=False)
