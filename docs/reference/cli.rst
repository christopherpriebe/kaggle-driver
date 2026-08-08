============================
Command-line interface (CLI)
============================

:func:`kaggle_driver.run` builds a Typer application from your dataset
and model set. The application exposes three subcommands.

``download``
============

.. code-block:: text

    python driver.py download

Authenticates with the Kaggle API, downloads the competition archive,
unzips it into a temporary workspace, and calls your
``KaggleInfo.organize_data_function`` to move the files into the dataset's
``raw_train_directory`` and ``raw_test_directory``.

Requires:

* The ``kaggle`` extra (``pip install kaggle-driver[kaggle]``).
* A valid ``~/.kaggle/kaggle.json`` credential file.
* A :class:`~kaggle_driver.KaggleInfo` passed to
  :func:`~kaggle_driver.run`.

``train``
=========

.. code-block:: text

    python driver.py train MODEL [--model-config PATH] [--train-config PATH]

Where ``MODEL`` is one of the dict keys you passed as ``models=`` to
:func:`~kaggle_driver.run`. The framework instantiates the model class
with the YAML at ``--model-config`` as kwargs, calls
``dataset.load_train()``, and dispatches to ``Model.train`` with the
YAML at ``--train-config``.

``test``
========

.. code-block:: text

    python driver.py test MODEL --submission FILE \
        [--model-config PATH] [--test-config PATH]

Same model instantiation as ``train``. Calls ``dataset.load_test()`` and
dispatches to ``Model.test``. The returned predictions are then written
to the file passed as ``--submission`` via
``dataset.store_predictions``.

Top-level options
=================

* ``--verbose``/``-v`` raises the Python ``logging`` level from
  ``WARNING`` to ``INFO``. Useful when running ``download`` to see what
  the Kaggle API is doing.
