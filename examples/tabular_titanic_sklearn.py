"""Titanic survival prediction with scikit-learn.

Demonstrates the opinionated, tabular path: pair :class:`PandasDataset`
with thin :class:`SklearnModel` subclasses. The user supplies almost no
glue code; the framework handles CSV reading, stacking, fit/predict, and
submission writing.

Expected layout::

    examples/data/raw/train/titanic/train.csv
    examples/data/raw/test/titanic/test.csv

Each CSV must contain at least the numeric columns this example uses
(``Pclass``, ``Age``, ``SibSp``, ``Parch``, ``Fare``) plus the id column
(``PassengerId``). The train CSV must also include ``Survived``.

Run with::

    python examples/tabular_titanic_sklearn.py train logistic_regression
    python examples/tabular_titanic_sklearn.py test logistic_regression --submission submission.csv
"""

from __future__ import annotations

import shutil
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import kaggle_driver as kd
from kaggle_driver.integrations.pandas import PandasDataset
from kaggle_driver.integrations.sklearn import SklearnModel

COMPETITION_NAME = "titanic"
USABLE_COLUMNS = ["PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare", "Survived"]


class LogisticRegressionSurvival(SklearnModel):
    """Logistic regression with median imputation for missing values."""

    def __init__(self) -> None:
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("classifier", LogisticRegression(max_iter=1000)),
            ],
        )
        super().__init__(pipeline)


class RandomForestSurvival(SklearnModel):
    """Random forest baseline with median imputation."""

    def __init__(self) -> None:
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("classifier", RandomForestClassifier(n_estimators=200, random_state=0)),
            ],
        )
        super().__init__(pipeline)


def organize_titanic_data(
    unzipped_directory: Path,
    raw_train_directory: Path,
    raw_test_directory: Path,
) -> None:
    """Move ``train.csv`` and ``test.csv`` into the dataset's raw directories."""
    shutil.move(unzipped_directory / "train.csv", raw_train_directory / "train.csv")
    shutil.move(unzipped_directory / "test.csv", raw_test_directory / "test.csv")


_HERE = Path(__file__).resolve().parent

dataset = PandasDataset(
    raw_train_directory=_HERE / "data" / "raw" / "train" / "titanic",
    raw_test_directory=_HERE / "data" / "raw" / "test" / "titanic",
    target_column="Survived",
    id_column="PassengerId",
    read_csv_kwargs={"usecols": USABLE_COLUMNS},
)

kaggle_info = kd.KaggleInfo(
    competition_name=COMPETITION_NAME,
    organize_data_function=organize_titanic_data,
)


if __name__ == "__main__":
    kd.run(
        dataset,
        models={
            "logistic_regression": LogisticRegressionSurvival,
            "random_forest": RandomForestSurvival,
        },
        kaggle_info=kaggle_info,
    )
