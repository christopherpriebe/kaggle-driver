========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/kaggle-driver/badge/?style=flat
    :target: https://kaggle-driver.readthedocs.io/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/christopherpriebe/kaggle-driver/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/christopherpriebe/kaggle-driver/actions

.. |codecov| image:: https://codecov.io/gh/christopherpriebe/kaggle-driver/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://app.codecov.io/github/christopherpriebe/kaggle-driver

.. |version| image:: https://img.shields.io/pypi/v/kaggle-driver.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/kaggle-driver

.. |wheel| image:: https://img.shields.io/pypi/wheel/kaggle-driver.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/kaggle-driver

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/kaggle-driver.svg
    :alt: Supported versions
    :target: https://pypi.org/project/kaggle-driver

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/kaggle-driver.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/kaggle-driver

.. |commits-since| image:: https://img.shields.io/github/commits-since/christopherpriebe/kaggle-driver/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/christopherpriebe/kaggle-driver/compare/v0.0.0...master



.. end-badges

A package to support model development for Kaggle competitions.

* Free software: MIT license

Installation
============

::

    pip install kaggle-driver

You can also install the in-development version with::

    pip install https://github.com/christopherpriebe/kaggle-driver/archive/master.zip


Documentation
=============


https://kaggle-driver.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
