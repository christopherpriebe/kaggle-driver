# Roadmap

kaggle-driver is growing from thin scaffolding into a batteries-included
framework for the full Kaggle competition loop: download, train, validate,
predict, track. This document records where it is going and, just as
deliberately, where it is not.

Releases are milestone-based, not date-based. The package is developed in
focused bursts, so each release is one coherent battery that can ship
complete rather than a long train of half-done features.

## Guiding constraints

- **One battery per release.** Each 0.x release delivers a single major
  capability, finished, with docs and tests.
- **Severable batteries.** Anything with a heavy dependency lives behind a
  pip extra. The core stays importable with nothing but pyyaml, typer, and
  immutabledict.
- **The core contract stays small.** `Dataset`, `Model`, `KaggleInfo`, and
  `run` are the whole mental model. Batteries build on the contract; they
  do not bloat it.
- **Docs-first community.** Every battery lands with a tutorial. Users
  self-serve; contribution infrastructure waits until contributors exist.
- **Breaking changes are allowed until 1.0** and always documented in the
  changelog. After 1.0 the core contract is frozen and semver applies.

## 0.1.0 - The rewrite and experiment tracking (current)

The near-total rewrite that establishes the public surface: generic
`Dataset`/`Model` abstract base classes, the Typer CLI via `kd.run`,
frozen-mapping boundaries, and the pandas/sklearn/torch integrations.
Ships to PyPI as the first real release.

It also ships the first battery, experiment tracking:

- Every `train`/`test` invocation writes a timestamped run directory under
  a configurable root (default `runs/`): resolved config snapshots, the
  statistics mapping, and paths to produced artifacts (model file,
  submission file).
- A `runs` subcommand to list runs and show or compare their statistics.
- Zero new dependencies: the run store is plain files, works offline.
- A tracking hook protocol underneath, designed so external adapters
  (MLflow, Weights & Biases) can be added post-1.0 without breaking the
  run format or the hooks.
- The run schema anticipates per-fold metrics so 0.2.0 slots in without a
  format break.

## 0.2.0 - Validation and metrics

Every user currently reimplements the split-train-score loop. This release
makes the framework own it.

- A `validate` subcommand running a framework-owned loop: split the
  training data, train a fresh model per fold, predict on the held-out
  fold, score.
- Holdout and k-fold strategies, configurable via the validation YAML.
- A metric protocol: users supply named metric callables; the framework
  reports them per fold and aggregated.
- Fold-level and aggregate metrics recorded into the 0.1.0 runs directory.
- Tutorial: validating a model before submitting.

## 0.3.0 - Boosted trees

The Kaggle tabular workhorses join the integrations package.

- `LightGBMModel` and `XGBoostModel` behind `[lightgbm]` and `[xgboost]`
  extras, following the `SklearnModel` pattern, with early stopping wired
  to the validation machinery from 0.2.0.
- Tutorial: a tabular competition with gradient boosting.

## 0.4.0 - Deepen the existing integrations

No new names; more value from the ones that exist.

- `TorchModel`: opt-in training-loop helpers (batching, device placement,
  epoch loop) so subclasses stop hand-rolling the same fit loop.
- `PandasDataset`: cache the resolved id column instead of re-reading the
  test CSV per access; dtype and column-selection controls.
- `SklearnModel`: probability predictions and pipeline-friendly docs.

## 1.0.0 - Feature complete

1.0 means the checklist is done, not that a date arrived:

- [x] Experiment tracking (0.1.0)
- [ ] Validation and metrics (0.2.0)
- [ ] LightGBM and XGBoost integrations (0.3.0)
- [ ] Deepened pandas/sklearn/torch integrations (0.4.0)
- [ ] A tutorial for every battery and complete API reference docs
- [ ] Core contract unchanged across the two releases before 1.0

From 1.0 on: semver, deprecation cycles for any public change, and the
runs-directory format treated as a stable interface.

## Post-1.0 horizon

Legitimate batteries, deliberately sequenced after the core loop is solid:

- **Submission upload**: a `submit` subcommand via the Kaggle API,
  closing the last manual step in the loop.
- **Hyperparameter search**: sweep support over model configs, most
  likely as an optuna adapter rather than a homegrown engine.
- **External tracking adapters**: MLflow and Weights & Biases adapters on
  the 0.1.0 hook protocol, each behind an extra.

## Non-goals through 1.0

- **Keras/TensorFlow integration.** One deep-learning path (torch) keeps
  the maintenance surface honest.
- **Polars dataset.** Revisit if users ask.
- **Active community infrastructure.** Issue templates, CONTRIBUTING, and
  promotion wait until there are users to serve; docs come first.
- **Experiment-manager pivot.** Tracking serves the competition loop; the
  package is not becoming a general ML experiment platform.
