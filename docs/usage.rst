=====
Usage
=====

This page walks through the public surface of ``kaggle-driver`` at a
high level. For end-to-end examples see :doc:`tutorial_tabular` and
:doc:`tutorial_mnist`.

The mental model
================

A competition pipeline has three moving parts:

1. A :class:`~kaggle_driver.Dataset` subclass that knows how to read the
   raw competition files from disk and write a submission file back out.
2. One or more :class:`~kaggle_driver.Model` subclasses that own the
   model architecture and implement train, test, save, and load.
3. An optional :class:`~kaggle_driver.KaggleInfo` describing the
   competition for the Kaggle API integration.

Once you have those, you call :func:`kaggle_driver.run` from your driver
script. ``run`` builds a Typer CLI for you with ``download``, ``train``,
and ``test`` subcommands and dispatches based on ``sys.argv``.

Defining a dataset
==================

:class:`~kaggle_driver.Dataset` is generic over an input type ``InputT``
and a target type ``TargetT``. Subclasses pick those types freely. The base
class gives you two read-only properties (``raw_train_directory``,
``raw_test_directory``) and asks you to implement three abstract methods:

.. code-block:: python

    class MyDataset(kd.Dataset[MyInput, MyTarget]):
        def load_train(self) -> Mapping[str, tuple[MyInput, MyTarget]]: ...
        def load_test(self) -> Mapping[str, MyInput]: ...
        def store_predictions(self, path: Path, predictions: Mapping[str, MyTarget]) -> None: ...

Example ids drive both the training pairs and the submission ordering.
The mapping returned by ``load_test`` controls the order of rows in the
submission file.

Return an ordinary ``dict`` if that is convenient. The framework freezes
it into an ``immutabledict`` before handing it to the model, so no later
stage can mutate data an earlier one still holds.

Defining a model
================

:class:`~kaggle_driver.Model` is generic over the same ``InputT`` and
``TargetT`` type variables. Subclasses must implement four methods:

.. code-block:: python

    class MyModel(kd.Model[MyInput, MyTarget]):
        def __init__(self, **hyperparameters: Any) -> None: ...
        def train(self, train_data, config) -> Mapping[str, Any]: ...
        def test(self, test_data, config) -> tuple[Mapping[str, MyTarget], Mapping[str, Any]]: ...
        def save(self, path: Path) -> None: ...

        @classmethod
        def load(cls, path: Path) -> "MyModel": ...

The framework instantiates the model class with kwargs from a model
config YAML file. ``train`` and ``test`` receive the data and a per-call
config, both frozen. They return free-form statistics; the framework logs
them but does not interpret them. Statistics and predictions are frozen on
the way back out, so anything the framework returns to the caller is an
``immutabledict``.

Wiring up the CLI
=================

Your driver script ends with:

.. code-block:: python

    dataset = MyDataset(raw_train_directory=..., raw_test_directory=...)

    if __name__ == "__main__":
        kd.run(
            dataset,
            models={"v1": MyModelV1, "v2": MyModelV2},
            kaggle_info=optional_kaggle_info,
        )

The dict keys passed as ``models`` are the names used on the CLI; users
do not see your Python class names. ``kaggle_info`` is required if you
want the ``download`` subcommand and may be omitted otherwise.

Run it:

.. code-block:: bash

    python driver.py --help
    python driver.py download
    python driver.py train v1 --model-config m.yml --train-config t.yml
    python driver.py test v1 --submission submission.csv --test-config t.yml

Tracking runs
=============

Every ``train`` and ``test`` invocation is recorded as a run by default.
A run is a directory under ``runs/`` (next to wherever you invoked the
CLI) holding a ``run.json`` record: the subcommand, the model name, the
lifecycle status, timestamps, the statistics the model returned, paths
to produced artifacts, and YAML snapshots of the resolved configs. A
run that fails records the error summary and a ``failed`` status, so
failed experiments stay visible.

Inspect recorded runs with the ``runs`` subcommands:

.. code-block:: bash

    python driver.py runs list
    python driver.py runs show 20260807T120301Z-1a2b
    python driver.py runs compare 20260807T120301Z-1a2b 20260807T130502Z-3c4d

Two app-level options control recording:

* ``--runs-root PATH`` records runs under ``PATH`` instead of ``runs/``.
* ``--no-track`` disables recording for the invocation.

Built-in helpers
================

If your competition fits a common shape, you can skip writing the
``Dataset`` and ``Model`` subclasses yourself:

* :class:`kaggle_driver.integrations.pandas.PandasDataset` reads flat
  ``train.csv`` and ``test.csv`` files. See :doc:`tutorial_tabular`.
* :class:`kaggle_driver.integrations.sklearn.SklearnModel` wraps any
  scikit-learn estimator into the ``Model`` API. See
  :doc:`tutorial_tabular`.
* :class:`kaggle_driver.integrations.torch.TorchModel` is an abstract base
  for PyTorch users; it handles persistence and device selection so you
  only implement the architecture and the training loop. See
  :doc:`tutorial_mnist`.

Each helper lives behind its own install extra
(``kaggle-driver[pandas]``, ``[sklearn]``, ``[torch]``).
