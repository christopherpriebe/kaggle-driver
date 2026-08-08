==========================================
Tutorial: tabular competition with sklearn
==========================================

This tutorial walks through a complete tabular pipeline using
:class:`~kaggle_driver.integrations.pandas.PandasDataset` and
:class:`~kaggle_driver.integrations.sklearn.SklearnModel`. The example tracks
the Kaggle Titanic competition.

Install the extras
==================

.. code-block:: bash

    pip install "kaggle-driver[pandas,sklearn,kaggle]"

The ``kaggle`` extra is only needed if you want to use the
:doc:`download <usage>` subcommand; manual download works without it.

Driver script
=============

The complete driver is shipped as
``examples/tabular_titanic_sklearn.py``. The interesting parts are
reproduced here.

Define one or more sklearn-backed models:

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from kaggle_driver.integrations.sklearn import SklearnModel


    class LogisticRegressionSurvival(SklearnModel):
        def __init__(self) -> None:
            super().__init__(
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("classifier", LogisticRegression(max_iter=1000)),
                ]),
            )

Define the dataset and wire up the CLI:

.. code-block:: python

    import kaggle_driver as kd
    from kaggle_driver.integrations.pandas import PandasDataset

    dataset = PandasDataset(
        raw_train_directory="data/raw/train/titanic",
        raw_test_directory="data/raw/test/titanic",
        target_column="Survived",
        id_column="PassengerId",
        read_csv_kwargs={"usecols": [
            "PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare", "Survived",
        ]},
    )

    if __name__ == "__main__":
        kd.run(dataset, models={"logistic_regression": LogisticRegressionSurvival})

Configs
=======

Both ``train`` and ``test`` accept optional YAML config files. For
``SklearnModel`` the only meaningful key is ``model_path``: ``train``
saves the fitted estimator there; ``test`` restores it before
predicting.

.. code-block:: yaml

    # configs/train.yml and configs/test.yml
    model_path: artifacts/titanic_logistic_regression.joblib

Run it
======

.. code-block:: bash

    # Optional: have kaggle-driver download the competition for you.
    python tabular_titanic_sklearn.py download

    # Train. Writes artifacts/titanic_logistic_regression.joblib.
    python tabular_titanic_sklearn.py train logistic_regression \
        --train-config configs/train.yml

    # Test. Reads the saved model and writes submission.csv.
    python tabular_titanic_sklearn.py test logistic_regression \
        --submission submission.csv \
        --test-config configs/test.yml

Things to try
=============

* Train both shipped models (``logistic_regression`` and
  ``random_forest``), then put their statistics side by side with
  ``python tabular_titanic_sklearn.py runs compare <first-run-id>
  <second-run-id>`` (get the run ids from ``runs list``).
* Tweak the ``Pipeline`` to include feature engineering. The
  ``SklearnModel`` base wraps any estimator-shaped object, including a
  full ``sklearn.pipeline.Pipeline``.
* Pass ``fit_kwargs`` via the train YAML to forward kwargs to
  ``estimator.fit``.
