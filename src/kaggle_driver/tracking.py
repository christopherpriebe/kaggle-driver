"""Experiment tracking: run records, the tracking hook protocol, and the run store.

Every tracked ``train`` or ``test`` invocation is recorded as a run: a
directory under a configurable root holding a ``run.json`` record and YAML
snapshots of the resolved configs. :class:`TrackingHook` is the protocol the
driver speaks; :class:`RunDirectoryTracker` is its built-in file-backed
implementation. :func:`list_runs` and :func:`load_run` read recorded runs
back as :class:`RunRecord` values.

The full design, including the run directory layout and the ``run.json``
schema, is documented in ``docs/design/experiment_tracking.md``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

# Statistics and config values are free-form user data (the framework never
# interprets them), so their value type is deliberately Any.
from typing import Any, Protocol

__all__ = [
    "RUN_SCHEMA_VERSION",
    "RunCommand",
    "RunDirectoryTracker",
    "RunRecord",
    "RunStatus",
    "TrackingHook",
    "list_runs",
    "load_run",
]

RUN_SCHEMA_VERSION = 1


class RunCommand(Enum):
    """Subcommand a run records."""

    TRAIN = "train"
    TEST = "test"


class RunStatus(Enum):
    """Lifecycle state of a recorded run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class RunRecord:
    """One recorded run, as read back from a run directory.

    All mapping attributes are frozen mappings; callers cannot mutate a
    loaded record.

    Attributes:
        run_id: Unique, chronologically sortable run identifier.
        command: Subcommand that produced the run.
        model_name: CLI-facing model name the run was invoked with.
        status: Lifecycle state the record was last written with.
        started_at: Timezone-aware UTC start time.
        finished_at: Timezone-aware UTC finalization time, or ``None`` for
            a run that has not finalized.
        error: Error summary for a failed run, or ``None``.
        metrics: Statistics mappings keyed by phase (``"train"``,
            ``"test"``).
        artifacts: Artifact paths keyed by artifact name.
        config_paths: Config snapshot paths keyed by config name, relative
            to the run directory.
    """

    run_id: str
    command: RunCommand
    model_name: str
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None
    error: str | None
    metrics: Mapping[str, Mapping[str, Any]]
    artifacts: Mapping[str, Path]
    config_paths: Mapping[str, Path]


class TrackingHook(Protocol):
    """Protocol the driver uses to record a single run.

    A hook instance is single-use and stateful: the driver calls
    :meth:`start_run` exactly once, then :meth:`record_metrics` and
    :meth:`record_artifact` as results appear, and finally exactly one of
    :meth:`complete_run` or :meth:`fail_run`.
    """

    def start_run(
        self,
        *,
        command: RunCommand,
        model_name: str,
        configs: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Begin recording a run.

        Args:
            command: Subcommand being recorded.
            model_name: CLI-facing model name.
            configs: Resolved config mappings keyed by config name
                (for example ``"model_config"``, ``"train_config"``).
        """

    def record_metrics(self, phase: str, metrics: Mapping[str, Any]) -> None:
        """Attach a statistics mapping under a phase key.

        Args:
            phase: Phase the metrics belong to (``"train"``, ``"test"``).
            metrics: Free-form statistics mapping.
        """

    def record_artifact(self, name: str, path: Path) -> None:
        """Attach a named artifact path to the run.

        Args:
            name: Artifact name (for example ``"submission"``).
            path: Path of the produced artifact, recorded as given.
        """

    def complete_run(self) -> None:
        """Finalize the run as completed."""

    def fail_run(self, error: str) -> None:
        """Finalize the run as failed.

        Args:
            error: Summary of the failure, typically the exception type
                and message.
        """


class RunDirectoryTracker:
    """File-backed :class:`TrackingHook` writing a run directory.

    ``start_run`` creates ``<runs_root>/<run_id>/`` containing ``run.json``
    (status ``running``) and YAML snapshots of the resolved configs under
    ``configs/``. Finalization rewrites ``run.json`` atomically with the
    final status, metrics, artifacts, and timestamps.

    Args:
        runs_root: Directory that holds run directories. Created on demand,
            parents included.
    """

    def __init__(self, runs_root: Path | str) -> None:
        raise NotImplementedError

    @property
    def run_id(self) -> str:
        """Return the identifier of the started run.

        Raises:
            RuntimeError: ``start_run`` has not been called.
        """
        raise NotImplementedError

    @property
    def run_directory(self) -> Path:
        """Return the directory of the started run.

        Raises:
            RuntimeError: ``start_run`` has not been called.
        """
        raise NotImplementedError

    def start_run(
        self,
        *,
        command: RunCommand,
        model_name: str,
        configs: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Create the run directory and write the initial record.

        Args:
            command: Subcommand being recorded.
            model_name: CLI-facing model name.
            configs: Resolved config mappings keyed by config name; each is
                snapshotted to ``configs/<name>.yaml``.

        Raises:
            RuntimeError: ``start_run`` was already called on this instance.
            OSError: The run directory or its files cannot be written.
        """
        raise NotImplementedError

    def record_metrics(self, phase: str, metrics: Mapping[str, Any]) -> None:
        """Attach a statistics mapping under a phase key.

        Args:
            phase: Phase the metrics belong to.
            metrics: Free-form statistics mapping. Values without a native
                JSON representation are serialized via ``str``.

        Raises:
            RuntimeError: The run has not started or is already finalized.
        """
        raise NotImplementedError

    def record_artifact(self, name: str, path: Path) -> None:
        """Attach a named artifact path to the run.

        Args:
            name: Artifact name.
            path: Path of the produced artifact, recorded as given.

        Raises:
            RuntimeError: The run has not started or is already finalized.
        """
        raise NotImplementedError

    def complete_run(self) -> None:
        """Rewrite the record with status completed and the finish time.

        Raises:
            RuntimeError: The run has not started or is already finalized.
            OSError: The record cannot be written.
        """
        raise NotImplementedError

    def fail_run(self, error: str) -> None:
        """Rewrite the record with status failed and the error summary.

        Args:
            error: Summary of the failure.

        Raises:
            RuntimeError: The run has not started or is already finalized.
            OSError: The record cannot be written.
        """
        raise NotImplementedError


def list_runs(runs_root: Path | str) -> tuple[RunRecord, ...]:
    """Load every readable run under ``runs_root``.

    Args:
        runs_root: Directory that holds run directories. A missing root
            yields an empty tuple.

    Returns:
        Records sorted by ``run_id``, which sorts chronologically. Entries
        that are not run directories and run directories whose record is
        unreadable are skipped.
    """
    raise NotImplementedError


def load_run(runs_root: Path | str, run_id: str) -> RunRecord:
    """Load a single run record.

    Args:
        runs_root: Directory that holds run directories.
        run_id: Identifier of the run to load.

    Returns:
        The parsed record.

    Raises:
        FileNotFoundError: No run directory named ``run_id`` exists under
            ``runs_root``, or it has no ``run.json``.
        ValueError: The record exists but cannot be parsed.
    """
    raise NotImplementedError
