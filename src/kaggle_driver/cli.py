"""Typer-based CLI that exposes ``download``, ``train``, and ``test`` subcommands.

The CLI is built fresh for each ``kd.run`` invocation so it can close over
the user's specific ``Dataset``, model classes, and optional ``KaggleInfo``.

This module deliberately avoids ``from __future__ import annotations``:
Typer relies on ``inspect.get_annotations`` to drive Click options, and
stringified annotations break when ``Annotated[...]`` references closure
variables built inside ``build_app``.
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from immutabledict import immutabledict

from kaggle_driver import driver
from kaggle_driver.config import load_yaml_config
from kaggle_driver.core import Dataset, KaggleInfo, Model, freeze_mapping
from kaggle_driver.tracking import RunDirectoryTracker, RunRecord, list_runs, load_run

__all__ = ["build_app"]

_logger = logging.getLogger(__name__)

_RUN_JSON_FILENAME = "run.json"
_MINIMUM_COMPARE_RUN_COUNT = 2
_LIST_COMMAND_WIDTH = 5
_LIST_MODEL_WIDTH = 15
_LIST_STATUS_WIDTH = 9
_COMPARE_LABEL_WIDTH = 20


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


@dataclass
class _TrackingOptions:
    """Run-tracking options captured by the app callback for the subcommands to read."""

    runs_root: Path = Path("runs")
    no_track: bool = False


def _build_tracker(options: _TrackingOptions) -> RunDirectoryTracker | None:
    """Build the tracker a train or test invocation should use, if any.

    Args:
        options: Tracking options captured by the app callback.

    Returns:
        A fresh :class:`RunDirectoryTracker` rooted at ``options.runs_root``,
        or ``None`` when ``options.no_track`` is set.
    """
    if options.no_track:
        return None
    return RunDirectoryTracker(options.runs_root)


def _format_metrics_summary(metrics: Mapping[str, Mapping[str, Any]]) -> str:
    """Build a compact one-line summary of a run's phase metrics.

    Args:
        metrics: Statistics mappings keyed by phase.

    Returns:
        A summary such as ``train: sample_count=3``, with phases joined by
        ``"; "``. Empty when ``metrics`` is empty.
    """
    phase_summaries = []
    for phase, phase_metrics in metrics.items():
        rendered_metrics = ", ".join(f"{key}={value}" for key, value in phase_metrics.items())
        phase_summaries.append(f"{phase}: {rendered_metrics}")
    return "; ".join(phase_summaries)


def _format_run_summary(record: RunRecord) -> str:
    """Build the one-line summary of a run shown by ``runs list``.

    Args:
        record: Run record to summarize.

    Returns:
        A line containing the run id, command, model name, status, and a
        compact metrics summary.
    """
    metrics_summary = _format_metrics_summary(record.metrics)
    return (
        f"{record.run_id}  {record.command.value:<{_LIST_COMMAND_WIDTH}}  "
        f"{record.model_name:<{_LIST_MODEL_WIDTH}}  "
        f"{record.status.value:<{_LIST_STATUS_WIDTH}}  {metrics_summary}"
    )


def _metric_keys_for_phase(records: list[RunRecord], phase: str) -> list[str]:
    """Return the sorted union of metric keys recorded under ``phase``.

    Args:
        records: Runs being compared.
        phase: Phase to collect metric keys for.

    Returns:
        Sorted, deduplicated metric key names.
    """
    keys: set[str] = set()
    for record in records:
        keys.update(record.metrics.get(phase, {}))
    return sorted(keys)


def _format_compare_table(records: list[RunRecord]) -> str:
    """Build the aligned plain-text comparison table for two or more runs.

    Args:
        records: Runs being compared, in the order they should appear as
            columns.

    Returns:
        A multi-line table: a header row of run ids, then one row per
        metric key (grouped by phase, unioned across ``records``), with
        ``"-"`` where a run lacks that key.
    """
    lines = [
        "  ".join(["metric".ljust(_COMPARE_LABEL_WIDTH), *(record.run_id for record in records)]),
    ]
    phases = sorted({phase for record in records for phase in record.metrics})
    for phase in phases:
        for key in _metric_keys_for_phase(records, phase):
            row_label = f"{phase}.{key}".ljust(_COMPARE_LABEL_WIDTH)
            values = [str(record.metrics.get(phase, {}).get(key, "-")) for record in records]
            lines.append("  ".join([row_label, *values]))
    return "\n".join(lines)


def _load_run_or_exit(runs_root: Path, run_id: str) -> RunRecord:
    """Load a run record, converting an unreadable run into a Typer error.

    Args:
        runs_root: Directory that holds run directories.
        run_id: Identifier of the run to load.

    Returns:
        The loaded record.

    Raises:
        typer.BadParameter: No run matches ``run_id``, or its record cannot
            be parsed. The message names ``run_id``.
    """
    try:
        return load_run(runs_root, run_id)
    except (FileNotFoundError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error


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
    tracking_options = _TrackingOptions()

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
        tracking_options.runs_root = runs_root
        tracking_options.no_track = no_track
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
        for record in list_runs(tracking_options.runs_root):
            typer.echo(_format_run_summary(record))

    @runs_app.command(name="show")
    def show_command(run_id: Annotated[str, typer.Argument()]) -> None:
        """Show the full record of one run."""
        # Validate via _load_run_or_exit for the shape/error checks, but print the
        # stored record verbatim rather than reserializing the parsed dataclass.
        _load_run_or_exit(tracking_options.runs_root, run_id)
        run_json_path = tracking_options.runs_root / run_id / _RUN_JSON_FILENAME
        payload = json.loads(run_json_path.read_text(encoding="utf-8"))
        typer.echo(json.dumps(payload, indent=2))

    @runs_app.command(name="compare")
    def compare_command(run_ids: Annotated[list[str], typer.Argument()]) -> None:
        """Compare the metrics of two or more runs side by side."""
        if len(run_ids) < _MINIMUM_COMPARE_RUN_COUNT:
            raise typer.BadParameter("runs compare requires at least two run ids.")
        records = [_load_run_or_exit(tracking_options.runs_root, run_id) for run_id in run_ids]
        typer.echo(_format_compare_table(records))

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
            tracker=_build_tracker(tracking_options),
            model_name=model,
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
            tracker=_build_tracker(tracking_options),
            model_name=model,
        )

    return app
