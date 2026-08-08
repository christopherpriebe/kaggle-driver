"""Opinionated, optional helpers for common Kaggle competition patterns.

Each submodule here is gated behind a ``pip`` extra and imports its heavy
dependency at module load time. Import only what you need:

* :mod:`kaggle_driver.integrations.pandas` (extra ``pandas``)
* :mod:`kaggle_driver.integrations.sklearn` (extra ``sklearn``)
* :mod:`kaggle_driver.integrations.torch` (extra ``torch``)
"""
