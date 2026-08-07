"""``SklearnModel``: concrete :class:`Model` wrapping a scikit-learn estimator.

Install with ``pip install kaggle-driver[sklearn]``.
"""

# Kept: pandas.Series is not subscriptable at runtime (pandas 2.3.3), and this
# module's annotations use pd.Series[Any] directly in Mapping/tuple types on
# _stack_feature_rows, train, and test. Removing this import raises
# `TypeError: 'type' object is not subscriptable` at import time.
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import joblib
    import pandas as pd
    from sklearn.base import BaseEstimator
except ImportError as _error:  # pragma: no cover
    raise ImportError(
        "kaggle_driver.integrations.sklearn requires scikit-learn, joblib, and pandas. "
        "Install with 'pip install kaggle-driver[sklearn,pandas]'.",
    ) from _error

from immutabledict import immutabledict

from kaggle_driver.core import Model

__all__ = ["SklearnModel"]


def _stack_feature_rows(rows: Mapping[str, pd.Series[Any]]) -> pd.DataFrame:
    """Stack per-row Series into a DataFrame indexed by mapping key.

    Args:
        rows: Mapping from example id to a feature row.

    Returns:
        DataFrame with one row per mapping entry, indexed by the mapping
        keys in insertion order.
    """
    features = pd.DataFrame(list(rows.values()))
    features.index = list(rows.keys())
    return features


class SklearnModel(Model["pd.Series[Any]", Any]):
    """Wrap any scikit-learn estimator into the kaggle-driver ``Model`` API.

    ``train`` stacks the per-row series into a DataFrame and calls
    ``estimator.fit(X, y)``. ``test`` stacks test rows and calls
    ``estimator.predict``. Persistence uses ``joblib``.

    Args:
        estimator: A scikit-learn estimator instance (anything that
            implements ``fit`` and ``predict``).
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        self._estimator = estimator

    @property
    def estimator(self) -> BaseEstimator:
        """Return the underlying sklearn estimator."""
        return self._estimator

    def train(
        self,
        train_data: Mapping[str, tuple[pd.Series[Any], Any]],
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Fit the estimator on the stacked training rows.

        Args:
            train_data: Mapping from example id to ``(row, target)``.
            config: Optional keys:

                * ``fit_kwargs``: mapping forwarded as kwargs to
                  ``estimator.fit``.
                * ``model_path``: if present, the fitted estimator is
                  persisted to this path via :meth:`save`. Required when
                  driving via the CLI so the matching ``test`` invocation
                  can restore the same weights.

        Returns:
            Frozen mapping containing ``sample_count`` and ``feature_count``.
        """
        rows = {example_id: row for example_id, (row, _) in train_data.items()}
        features = _stack_feature_rows(rows)
        targets = pd.Series(
            [target for _, target in train_data.values()],
            index=list(train_data.keys()),
        )
        fit_kwargs: Mapping[str, Any] = config.get("fit_kwargs", {})
        self._estimator.fit(features, targets, **fit_kwargs)
        if "model_path" in config:
            self.save(Path(config["model_path"]))
        return immutabledict(
            {"sample_count": len(features), "feature_count": features.shape[1]},
        )

    def test(
        self,
        test_data: Mapping[str, pd.Series[Any]],
        config: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Predict targets for the stacked test rows.

        Args:
            test_data: Mapping from example id to a feature row.
            config: Optional ``"model_path"`` key. If present, the estimator
                stored at that path replaces the in-memory estimator before
                predicting. Required when running ``test`` via the CLI in a
                fresh process after a separate ``train`` invocation.

        Returns:
            Tuple of (frozen predictions, frozen statistics).
        """
        if "model_path" in config:
            self._estimator = joblib.load(Path(config["model_path"]))
        features = _stack_feature_rows(test_data)
        raw_predictions = self._estimator.predict(features)
        predictions = immutabledict(
            {
                example_id: raw_predictions[position]
                for position, example_id in enumerate(test_data.keys())
            },
        )
        return predictions, immutabledict({"sample_count": len(features)})

    def save(self, path: Path) -> None:
        """Persist the wrapped estimator to ``path`` via joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._estimator, path)

    @classmethod
    def load(cls, path: Path) -> SklearnModel:
        """Restore a :class:`SklearnModel` previously written by :meth:`save`."""
        estimator = joblib.load(path)
        return cls(estimator)
