Changelog
=========

0.1.0 (2026-05-16)
------------------

This release is a near-total rewrite. The 0.0.0 placeholder uploaded to
PyPI in 2023 reserved the name; this is the first release with a real
public surface.

Added
~~~~~

* Generic abstract base classes :class:`kaggle_driver.Dataset` and
  :class:`kaggle_driver.Model`, parameterized by user-supplied input and
  target types.
* :class:`kaggle_driver.KaggleInfo` dataclass for the Kaggle API
  integration.
* :func:`kaggle_driver.run` entry point: pass a dataset and a
  ``{"name": ModelClass}`` dict and get a Typer-based CLI for free.
* ``kaggle_driver.integrations.pandas.PandasDataset``: concrete
  ``Dataset`` over flat ``train.csv`` and ``test.csv`` files.
* ``kaggle_driver.integrations.sklearn.SklearnModel``: concrete ``Model``
  wrapping any scikit-learn estimator, with joblib persistence.
* ``kaggle_driver.integrations.torch.TorchModel``: abstract ``Model`` base
  for PyTorch users, with ``state_dict``-based save/load and a sensible
  default device.
* Optional install extras: ``kaggle``, ``pandas``, ``sklearn``,
  ``torch``, ``all``, ``dev``, ``docs``.
* New tabular Titanic example using ``PandasDataset`` and
  ``SklearnModel``.
* ``kaggle_driver.core.freeze_mapping``: wraps a mapping in an
  ``immutabledict`` (passing already-frozen mappings through untouched).
  The framework applies it at every boundary, so configs, training and
  test data, predictions, and statistics cannot be mutated by a later
  stage. ``immutabledict`` is a new required dependency.
* ``py.typed`` marker; the library now ships type information.
* Strict ``mypy``, ``ruff``, and ``pylint`` configurations enforced in
  CI.
* Pytest test suite split into unit and integration jobs with explicit
  markers (``unit``, ``integration``, ``slow``).
* Modern packaging: ``pyproject.toml`` only (PEP 621), hatchling build
  backend, ``uv`` for environment management.

Breaking changes
~~~~~~~~~~~~~~~~

* Removed the global ``ModelDirectory`` and the ``@kd.model`` decorator.
  Pass models explicitly to
  ``kd.run(models={"name": Cls}, ...)`` instead.
* Removed ``Input`` and ``Target`` abstract base classes.
  ``Dataset[InputT, TargetT]`` and ``Model[InputT, TargetT]`` are now
  generic; pick any input/target type you like.
* Removed the ``TrainConfig``, ``TestConfig``, ``TrainResult``, and
  ``TestResult`` dict subclasses. Configs and results are mappings
  loaded from YAML, frozen by the framework before they reach user code.
* The abstract methods on ``Dataset`` and ``Model`` are declared in
  terms of ``Mapping``, not ``dict``. Implementations may still return a
  plain ``dict``; the framework freezes it. Anything the framework hands
  back to the caller is an ``immutabledict``.
* :meth:`Model.save` and :meth:`Model.load` are now abstract. The
  pickle-based default is gone.
* ``Dataset.__init__`` now takes ``raw_train_directory`` and
  ``raw_test_directory`` directly. The ``DataLocInfo`` dataclass and the
  ``interim_*`` and ``processed_*`` directory knobs were removed.
* Identifiers are spelled out rather than abbreviated across the public
  surface: ``KaggleInfo.organize_data_fn`` is now
  ``organize_data_function``, and ``SklearnModel`` reports
  ``sample_count`` and ``feature_count`` instead of ``n_samples`` and
  ``n_features``. Its ``fit_params`` config key is now ``fit_kwargs``.
* The optional helpers moved from ``kaggle_driver.contrib`` to
  ``kaggle_driver.integrations``. They are first-party adapters gated
  behind install extras, not third-party contributed code, so ``contrib``
  named them inaccurately.
* ``kaggle_driver.io`` is now ``kaggle_driver.config`` and holds only
  ``load_yaml_config``. It no longer shadows the stdlib ``io`` module.
  ``require_kaggle`` moved to ``kaggle_driver.driver``, its only caller.
* The CLI moved from ``argparse`` to Typer. Help text and the names of
  some options changed.
* The Kaggle API client is now an optional dependency. Install it with
  ``pip install kaggle-driver[kaggle]``.
* The ``python -m kaggle_driver`` entry point was removed. The library
  is driven by the user's own script that calls :func:`kaggle_driver.run`.

Removed
~~~~~~~

* Python 3.9 support.
* PyPy support and the Windows CI matrix entries.
* ``setup.py``, ``tox.ini``, ``pytest.ini``, ``.coveragerc``,
  ``.bumpversion.cfg``, the cookiecutter-pylibrary ``ci/`` scaffolding,
  and the black formatter. All replaced by single-source
  ``pyproject.toml`` config and ``ruff`` for lint plus format.

0.0.0 (2023-11-21)
------------------

* Initial command-line interface implemented.
