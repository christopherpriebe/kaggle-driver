# kaggle-driver

Scaffolding for Kaggle competition workflows.

[![PyPI](https://img.shields.io/pypi/v/kaggle-driver.svg)](https://pypi.org/project/kaggle-driver/)
[![Python versions](https://img.shields.io/pypi/pyversions/kaggle-driver.svg)](https://pypi.org/project/kaggle-driver/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://readthedocs.org/projects/kaggle-driver/badge/?version=latest)](https://kaggle-driver.readthedocs.io/)
[![CI](https://github.com/christopherpriebe/kaggle-driver/actions/workflows/github-actions.yml/badge.svg)](https://github.com/christopherpriebe/kaggle-driver/actions)

`kaggle-driver` gives a Kaggle competition workflow a small, opinionated
shape. You implement a `Dataset` and one or more `Model` classes, then
hand them to `kd.run` and get a Typer CLI for free with `download`,
`train`, and `test` subcommands. The framework wires together the Kaggle
API integration, the train/test orchestration, and submission writing.

## Install

```bash
pip install kaggle-driver
```

Optional extras pull in the dependencies for the built-in helpers:

```bash
pip install "kaggle-driver[kaggle]"   # Kaggle API integration
pip install "kaggle-driver[pandas]"   # PandasDataset
pip install "kaggle-driver[sklearn]"  # SklearnModel
pip install "kaggle-driver[torch]"    # TorchModel
pip install "kaggle-driver[all]"      # everything
```

## Quickstart: tabular competition with sklearn

```python
"""titanic.py"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import kaggle_driver as kd
from kaggle_driver.integrations.pandas import PandasDataset
from kaggle_driver.integrations.sklearn import SklearnModel


class LogReg(SklearnModel):
    def __init__(self) -> None:
        super().__init__(
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]),
        )


dataset = PandasDataset(
    raw_train_directory="data/raw/train",
    raw_test_directory="data/raw/test",
    target_column="Survived",
    id_column="PassengerId",
)


if __name__ == "__main__":
    kd.run(dataset, models={"logistic_regression": LogReg})
```

Then on the command line:

```bash
python titanic.py train logistic_regression --train-config train.yml
python titanic.py test logistic_regression --submission submission.csv --test-config test.yml
```

Where `train.yml` and `test.yml` both contain something like
`model_path: model.joblib`.

Each `train` and `test` invocation is recorded as a run under `runs/`
next to wherever you invoked the CLI. Inspect recorded runs with
`python titanic.py runs list`, or pass `--no-track` to skip recording.

## What 0.1.0 ships

- `Dataset[I, T]` and `Model[I, T]` generic abstract bases.
- `KaggleInfo` dataclass for the Kaggle API integration.
- `kd.run(dataset, models={"name": Cls}, kaggle_info=None)` entry point.
- Typer-based CLI with `download`, `train`, `test`.
- `kaggle_driver.integrations.pandas.PandasDataset` for tabular competitions.
- `kaggle_driver.integrations.sklearn.SklearnModel` for sklearn estimators.
- `kaggle_driver.integrations.torch.TorchModel` abstract base for PyTorch
  users (handles state_dict save/load and default device).
- Experiment tracking, on by default: every `train`/`test` invocation
  records a run directory under `runs/` with config snapshots,
  statistics, and artifact paths (`--runs-root` moves it, `--no-track`
  disables it).
- `runs list`, `runs show`, and `runs compare` subcommands for
  inspecting recorded runs.

Validation and metrics are planned for the 0.2.0 release; see
[ROADMAP.md](ROADMAP.md) for the path to 1.0.

## Documentation

Full documentation lives at
[kaggle-driver.readthedocs.io](https://kaggle-driver.readthedocs.io/),
including the tabular and MNIST tutorials.

## License

MIT. See [LICENSE](LICENSE).
