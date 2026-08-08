"""Experiment tracking: run records, the tracking hook protocol, and the run store.

Every tracked ``train`` or ``test`` invocation is recorded as a run: a
directory under a configurable root holding a ``run.json`` record and YAML
snapshots of the resolved configs. :class:`TrackingHook` is the protocol the
driver speaks; :class:`RunDirectoryTracker` is its built-in file-backed
implementation. :func:`list_runs` and :func:`load_run` read recorded runs
back as :class:`RunRecord` values.

A run directory is ``<runs_root>/<run_id>/`` holding ``run.json`` (the
record: schema version, command, model name, status, timestamps, error,
phase-keyed metrics, artifact paths, and config snapshot paths) and a
``configs/`` directory of YAML snapshots.
"""

import json
import logging
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

# Statistics and config values are free-form user data (the framework never
# interprets them), so their value type is deliberately Any.
from typing import Any, Protocol

import yaml

from kaggle_driver.core import freeze_mapping

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

_RUN_ID_SUFFIX_BYTES = 2
_RUN_JSON_NAME = "run.json"
_CONFIGS_DIRECTORY_NAME = "configs"

_logger = logging.getLogger(__name__)


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


@dataclass
class _RunState:
    """Mutable in-memory state for a run a :class:`RunDirectoryTracker` owns."""

    run_id: str
    run_directory: Path
    command: RunCommand
    model_name: str
    started_at: datetime
    config_paths: dict[str, Path]
    status: RunStatus = RunStatus.RUNNING
    finished_at: datetime | None = None
    error: str | None = None
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)


def _generate_run_id() -> str:
    """Return a new, chronologically sortable run identifier.

    The identifier is the current UTC time formatted as
    ``%Y%m%dT%H%M%SZ``, a hyphen, and four lowercase hex characters.

    Returns:
        The generated run identifier.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(_RUN_ID_SUFFIX_BYTES)
    return f"{timestamp}-{suffix}"


def _create_run_directory(runs_root: Path) -> tuple[str, Path]:
    """Create a new, uniquely named run directory under ``runs_root``.

    Args:
        runs_root: Directory that holds run directories. Must already
            exist.

    Returns:
        The generated run identifier and the created run directory.
    """
    run_id = _generate_run_id()
    run_directory = runs_root / run_id
    while run_directory.exists():
        run_id = _generate_run_id()
        run_directory = runs_root / run_id
    run_directory.mkdir()
    return run_id, run_directory


def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` to ``path`` as JSON, atomically.

    The payload is first written to a temporary file in the same directory
    as ``path``, then moved into place so readers never observe a torn
    write.

    Args:
        path: Destination file path.
        payload: JSON-serializable mapping. Values without a native JSON
            representation are serialized via ``str``.
    """
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary_path.replace(path)


def _snapshot_configs(configs_directory: Path, configs: Mapping[str, Mapping[str, Any]]) -> None:
    """Write each config mapping to ``configs_directory`` as a YAML file.

    Args:
        configs_directory: Directory to write the snapshots into. Created
            if it does not already exist.
        configs: Config mappings keyed by config name; each is written to
            ``<name>.yaml``.
    """
    configs_directory.mkdir(parents=True, exist_ok=True)
    for name, config in configs.items():
        config_path = configs_directory / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(dict(config)), encoding="utf-8")


def _run_record_to_payload(state: _RunState) -> dict[str, Any]:
    """Build the JSON-serializable payload for a run record.

    Args:
        state: Current in-memory run state.

    Returns:
        A plain ``dict`` matching the ``run.json`` schema.
    """
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": state.run_id,
        "command": state.command.value,
        "model_name": state.model_name,
        "status": state.status.value,
        "started_at": state.started_at.isoformat(),
        "finished_at": state.finished_at.isoformat() if state.finished_at is not None else None,
        "error": state.error,
        "metrics": {phase: dict(metrics) for phase, metrics in state.metrics.items()},
        "artifacts": {name: str(path) for name, path in state.artifacts.items()},
        "configs": {name: str(path) for name, path in state.config_paths.items()},
    }


def _load_run_payload(run_directory: Path, run_id: str) -> object:
    """Read and JSON-decode the ``run.json`` file in ``run_directory``.

    Args:
        run_directory: Directory expected to hold ``run.json``.
        run_id: Identifier of the run, used only for error messages.

    Returns:
        The decoded JSON value.

    Raises:
        FileNotFoundError: ``run_directory`` has no ``run.json``.
        ValueError: ``run.json`` is not valid JSON.
    """
    run_json_path = run_directory / _RUN_JSON_NAME
    if not run_json_path.is_file():
        raise FileNotFoundError(f"No run.json found for run {run_id!r} at {run_json_path}.")
    try:
        return json.loads(run_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Run record for {run_id!r} is not valid JSON.") from error


def _parse_run_payload(run_id: str, payload: object) -> RunRecord:
    """Parse a decoded ``run.json`` payload into a :class:`RunRecord`.

    Args:
        run_id: Identifier of the run directory the payload was read from.
            The run directory name is authoritative for identity; the
            payload's own ``run_id`` field, if any, is not consulted.
        payload: Object decoded from ``run.json``.

    Returns:
        The parsed record.

    Raises:
        ValueError: ``payload`` does not match the run record shape.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"Run record for {run_id!r} is not a JSON object.")
    try:
        finished_at_raw = payload["finished_at"]
        return RunRecord(
            run_id=run_id,
            command=RunCommand(payload["command"]),
            model_name=payload["model_name"],
            status=RunStatus(payload["status"]),
            started_at=datetime.fromisoformat(payload["started_at"]),
            finished_at=(
                datetime.fromisoformat(finished_at_raw) if finished_at_raw is not None else None
            ),
            error=payload["error"],
            metrics=freeze_mapping(
                {
                    phase: freeze_mapping(phase_metrics)
                    for phase, phase_metrics in payload["metrics"].items()
                },
            ),
            artifacts=freeze_mapping(
                {name: Path(path) for name, path in payload["artifacts"].items()},
            ),
            config_paths=freeze_mapping(
                {name: Path(path) for name, path in payload["configs"].items()},
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Run record for {run_id!r} has an unexpected shape.") from error


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
        self._runs_root = Path(runs_root)
        self._state: _RunState | None = None

    @property
    def run_id(self) -> str:
        """Return the identifier of the started run.

        Raises:
            RuntimeError: ``start_run`` has not been called.
        """
        return self._require_started().run_id

    @property
    def run_directory(self) -> Path:
        """Return the directory of the started run.

        Raises:
            RuntimeError: ``start_run`` has not been called.
        """
        return self._require_started().run_directory

    def _require_started(self) -> _RunState:
        """Return the run state, requiring that a run has started.

        Returns:
            The current run state.

        Raises:
            RuntimeError: ``start_run`` has not been called.
        """
        if self._state is None:
            raise RuntimeError("start_run has not been called yet.")
        return self._state

    def _require_running(self) -> _RunState:
        """Return the run state, requiring that the run is still running.

        Returns:
            The current run state.

        Raises:
            RuntimeError: The run has not started or is already finalized.
        """
        state = self._require_started()
        if state.status is not RunStatus.RUNNING:
            raise RuntimeError(f"Run {state.run_id!r} has already finalized.")
        return state

    def _write_record(self, state: _RunState) -> None:
        """Write the current state of ``state`` to its ``run.json``, atomically.

        Args:
            state: Run state to serialize.
        """
        payload = _run_record_to_payload(state)
        _write_json_atomically(state.run_directory / _RUN_JSON_NAME, payload)

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
        if self._state is not None:
            raise RuntimeError("start_run has already been called on this tracker.")

        self._runs_root.mkdir(parents=True, exist_ok=True)
        run_id, run_directory = _create_run_directory(self._runs_root)

        state = _RunState(
            run_id=run_id,
            run_directory=run_directory,
            command=command,
            model_name=model_name,
            started_at=datetime.now(timezone.utc),
            config_paths={name: Path(_CONFIGS_DIRECTORY_NAME) / f"{name}.yaml" for name in configs},
        )
        self._state = state

        _snapshot_configs(run_directory / _CONFIGS_DIRECTORY_NAME, configs)
        self._write_record(state)

    def record_metrics(self, phase: str, metrics: Mapping[str, Any]) -> None:
        """Attach a statistics mapping under a phase key.

        Args:
            phase: Phase the metrics belong to.
            metrics: Free-form statistics mapping. Values without a native
                JSON representation are serialized via ``str``.

        Raises:
            RuntimeError: The run has not started or is already finalized.
        """
        state = self._require_running()
        state.metrics[phase] = dict(metrics)

    def record_artifact(self, name: str, path: Path) -> None:
        """Attach a named artifact path to the run.

        Args:
            name: Artifact name.
            path: Path of the produced artifact, recorded as given.

        Raises:
            RuntimeError: The run has not started or is already finalized.
        """
        state = self._require_running()
        state.artifacts[name] = path

    def complete_run(self) -> None:
        """Rewrite the record with status completed and the finish time.

        Raises:
            RuntimeError: The run has not started or is already finalized.
            OSError: The record cannot be written.
        """
        state = self._require_running()
        state.status = RunStatus.COMPLETED
        state.finished_at = datetime.now(timezone.utc)
        self._write_record(state)

    def fail_run(self, error: str) -> None:
        """Rewrite the record with status failed and the error summary.

        Args:
            error: Summary of the failure.

        Raises:
            RuntimeError: The run has not started or is already finalized.
            OSError: The record cannot be written.
        """
        state = self._require_running()
        state.status = RunStatus.FAILED
        state.error = error
        state.finished_at = datetime.now(timezone.utc)
        self._write_record(state)


def list_runs(runs_root: Path | str) -> tuple[RunRecord, ...]:
    """Load every readable run under ``runs_root``.

    Args:
        runs_root: Directory that holds run directories. A missing root
            yields an empty tuple.

    Returns:
        Records sorted by ``run_id``, which sorts chronologically. Entries
        that are not run directories and run directories whose record is
        unreadable are skipped; each skipped run directory logs one
        warning.
    """
    resolved_root = Path(runs_root)
    if not resolved_root.is_dir():
        return ()

    records = []
    for entry in resolved_root.iterdir():
        if not entry.is_dir():
            continue
        try:
            records.append(load_run(resolved_root, entry.name))
        except FileNotFoundError:
            _logger.warning("Skipping run directory %r: no run.json found.", entry.name)
        except ValueError:
            _logger.warning("Skipping run directory %r: unreadable run.json.", entry.name)

    return tuple(sorted(records, key=lambda record: record.run_id))


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
    run_directory = Path(runs_root) / run_id
    if not run_directory.is_dir():
        raise FileNotFoundError(f"No run directory found for run {run_id!r} at {run_directory}.")

    payload = _load_run_payload(run_directory, run_id)
    return _parse_run_payload(run_id, payload)
