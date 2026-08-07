"""Public ``run`` entry point that wires a dataset and model set into a CLI."""

from collections.abc import Mapping
from typing import Any

from kaggle_driver.cli import build_app
from kaggle_driver.core import Dataset, KaggleInfo, Model, freeze_mapping

__all__ = ["run"]


def run(
    dataset: Dataset[Any, Any],
    models: Mapping[str, type[Model[Any, Any]]],
    *,
    kaggle_info: KaggleInfo | None = None,
) -> None:
    """Build a CLI for this dataset and model set, then dispatch.

    The user's driver script typically ends with::

        if __name__ == "__main__":
            kd.run(dataset, models={"multilayer_perceptron": MyModel}, kaggle_info=kaggle_info)

    Args:
        dataset: An instantiated ``Dataset``.
        models: Mapping from CLI-facing model name (used as the ``MODEL``
            positional argument on the ``train`` and ``test`` subcommands)
            to model class.
        kaggle_info: Optional ``KaggleInfo``. Required if the user invokes
            the ``download`` subcommand; otherwise unused.

    Raises:
        ValueError: ``models`` is empty.
    """
    app = build_app(dataset, freeze_mapping(models), kaggle_info)
    app()
