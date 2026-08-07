"""Typer-based CLI that exposes ``download``, ``train``, and ``test`` subcommands.

The CLI is built fresh for each ``kd.run`` invocation so it can close over
the user's specific ``Dataset``, model classes, and optional ``KaggleInfo``.

This module deliberately avoids ``from __future__ import annotations``:
Typer relies on ``inspect.get_annotations`` to drive Click options, and
stringified annotations break when ``Annotated[...]`` references closure
variables built inside ``build_app``.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer
from immutabledict import immutabledict

from kaggle_driver import driver
from kaggle_driver.config import load_yaml_config
from kaggle_driver.core import Dataset, KaggleInfo, Model, freeze_mapping

__all__ = ["build_app"]

_logger = logging.getLogger(__name__)


def _resolve_model(
    models: Mapping[str, type[Model[Any, Any]]],
    name: str,
) -> type[Model[Any, Any]]:
    """Look up a model class by its CLI-facing name.

    Args:
        models: Mapping from CLI-facing name to model class.
        name: Name requested on the command line.

    Returns:
        The matching model class.

    Raises:
        typer.BadParameter: ``name`` is not a key of ``models``.
    """
    try:
        return models[name]
    except KeyError as error:
        valid = ", ".join(sorted(models))
        raise typer.BadParameter(
            f"Unknown model {name!r}. Valid choices: {valid}.",
        ) from error


def _load_optional_yaml(path: Path | None) -> immutabledict[str, Any]:
    """Load a YAML config file if ``path`` is given, otherwise return an empty mapping.

    Args:
        path: Optional path to a YAML file.

    Returns:
        Parsed YAML as a frozen mapping, or an empty one when ``path`` is
        ``None``.
    """
    if path is None:
        return immutabledict()
    return load_yaml_config(path)


def build_app(
    dataset: Dataset[Any, Any],
    models: Mapping[str, type[Model[Any, Any]]],
    kaggle_info: KaggleInfo | None,
) -> typer.Typer:
    """Build a Typer application bound to a specific dataset and model set.

    Args:
        dataset: User's ``Dataset`` instance.
        models: Mapping from CLI-facing model name to model class. Frozen
            on entry so the built app cannot be reconfigured behind its own
            back by later edits to the caller's mapping.
        kaggle_info: Optional Kaggle competition metadata.

    Returns:
        A Typer ``app`` ready to be called.

    Raises:
        ValueError: ``models`` is empty.
    """
    if not models:
        raise ValueError("`models` must contain at least one entry.")

    resolved_models = freeze_mapping(models)

    app = typer.Typer(
        help="Driver for a Kaggle competition.",
        no_args_is_help=True,
        add_completion=False,
    )
    valid_models_help = "One of: " + ", ".join(sorted(resolved_models)) + "."

    @app.callback()
    def main(
        verbose: Annotated[
            bool,
            typer.Option("--verbose", "-v", help="Enable verbose logging."),
        ] = False,
        runs_root: Annotated[
            Path,
            typer.Option("--runs-root", help="Directory that holds recorded runs."),
        ] = Path("runs"),
        no_track: Annotated[
            bool,
            typer.Option("--no-track", help="Disable run recording for this invocation."),
        ] = False,
    ) -> None:
        """Configure logging and run tracking for the subcommand."""
        # Tracking options are wired to the driver during implementation.
        del runs_root, no_track
        level = logging.INFO if verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )

    runs_app = typer.Typer(
        help="Inspect recorded runs.",
        no_args_is_help=True,
    )
    app.add_typer(runs_app, name="runs")

    @runs_app.command(name="list")
    def list_command() -> None:
        """List recorded runs, one line per run."""
        raise NotImplementedError

    @runs_app.command(name="show")
    def show_command(run_id: Annotated[str, typer.Argument()]) -> None:
        """Show the full record of one run."""
        raise NotImplementedError

    @runs_app.command(name="compare")
    def compare_command(run_ids: Annotated[list[str], typer.Argument()]) -> None:
        """Compare the metrics of two or more runs side by side."""
        raise NotImplementedError

    @app.command(name="download")
    def download_command() -> None:
        """Download and organize the competition dataset via the Kaggle API."""
        if kaggle_info is None:
            raise typer.BadParameter(
                "`kaggle_info` was not provided to kd.run; cannot download.",
            )
        driver.download(dataset, kaggle_info)

    @app.command(name="train", help=f"Train a model. {valid_models_help}")
    def train_command(
        model: Annotated[str, typer.Argument()],
        model_config: Annotated[
            Path | None,
            typer.Option("--model-config", help="YAML file of constructor kwargs."),
        ] = None,
        train_config: Annotated[
            Path | None,
            typer.Option("--train-config", help="YAML file of training settings."),
        ] = None,
    ) -> None:
        """Train a model on the dataset's training data."""
        driver.train(
            dataset,
            _resolve_model(resolved_models, model),
            model_config=_load_optional_yaml(model_config),
            train_config=_load_optional_yaml(train_config),
        )

    @app.command(
        name="test",
        help=f"Generate predictions and write a submission. {valid_models_help}",
    )
    def test_command(
        model: Annotated[str, typer.Argument()],
        submission: Annotated[
            Path,
            typer.Option("--submission", help="Path to write the submission file."),
        ],
        model_config: Annotated[
            Path | None,
            typer.Option("--model-config", help="YAML file of constructor kwargs."),
        ] = None,
        test_config: Annotated[
            Path | None,
            typer.Option("--test-config", help="YAML file of test-time settings."),
        ] = None,
    ) -> None:
        """Generate predictions on the test set and write a submission file."""
        driver.test(
            dataset,
            _resolve_model(resolved_models, model),
            submission_path=submission,
            model_config=_load_optional_yaml(model_config),
            test_config=_load_optional_yaml(test_config),
        )

    return app
