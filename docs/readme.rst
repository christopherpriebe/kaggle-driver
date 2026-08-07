========
Overview
========

``kaggle-driver`` is scaffolding for Kaggle competition workflows.

You implement a :class:`~kaggle_driver.Dataset` and one or more
:class:`~kaggle_driver.Model` classes, then hand them to
:func:`kaggle_driver.run` and get a Typer-based CLI for free with
``download``, ``train``, and ``test`` subcommands.

Install
=======

.. code-block:: bash

    pip install kaggle-driver

Optional extras enable the built-in helpers:

.. code-block:: bash

    pip install "kaggle-driver[kaggle]"
    pip install "kaggle-driver[pandas]"
    pip install "kaggle-driver[sklearn]"
    pip install "kaggle-driver[torch]"
    pip install "kaggle-driver[all]"

License
=======

MIT.
