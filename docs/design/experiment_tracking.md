# Experiment tracking (0.2.0)

## Summary

Every CLI `train` and `test` invocation records itself as a run: a
timestamped directory under a configurable root (default `runs/`) holding a
`run.json` record, YAML snapshots of the resolved configs, and references to
produced artifacts. A new `runs` subcommand lists, shows, and compares
recorded runs. Underneath sits a small tracking hook protocol; the local run
directory is its first implementation, and post-1.0 adapters (MLflow,
Weights & Biases) will be alternative implementations of the same protocol.
Zero new dependencies: the store is plain JSON and YAML files.

## Motivation

`driver.train` and `driver.test` return statistics mappings that are logged
at INFO level and then dropped. A user comparing two configurations today
copies numbers out of terminal scrollback. This is the first battery from
ROADMAP.md: make every invocation a durable, comparable record, and shape
the record so that 0.3.0 validation metrics slot in without a format break.

## Public interface

All new symbols live in a new module, `kaggle_driver.tracking`. The full
signatures with docstrings are in the interface stub
(`src/kaggle_driver/tracking.py`); this section summarizes the contract.

### New module `kaggle_driver.tracking`

- `RUN_SCHEMA_VERSION: int` - current `run.json` schema version, `1`.
- `class RunCommand(Enum)` - `TRAIN`, `TEST`. The subcommand a run records.
  0.3.0 adds `VALIDATE`.
- `class RunStatus(Enum)` - `RUNNING`, `COMPLETED`, `FAILED`.
- `@dataclass(frozen=True) class RunRecord` - one recorded run:
  `run_id`, `command`, `model_name`, `status`, `started_at`,
  `finished_at | None`, `error: str | None`,
  `metrics: Mapping[phase, Mapping[str, Any]]`,
  `artifacts: Mapping[str, Path]`, `config_paths: Mapping[str, Path]`.
  Mappings are frozen at construction (house `freeze_mapping` rule).
- `class TrackingHook(Protocol)` - the adapter seam. Single-use, stateful,
  one instance per run:
  - `start_run(*, command, model_name, configs)` - begin recording;
    `configs` maps config name to its resolved mapping.
  - `record_metrics(phase, metrics)` - attach a statistics mapping under a
    phase key (`"train"`, `"test"`; `"validation"` arrives in 0.3.0).
  - `record_artifact(name, path)` - attach a named artifact path.
  - `complete_run()` - finalize as `COMPLETED`.
  - `fail_run(error)` - finalize as `FAILED` with an error summary.
    Separate from `complete_run` per the no-flag-arguments rule.
- `class RunDirectoryTracker` - the built-in `TrackingHook` writing the run
  directory layout below. Constructed with just `runs_root`; exposes
  `run_id` and `run_directory` properties (raise `RuntimeError` before
  `start_run`).
- `list_runs(runs_root) -> tuple[RunRecord, ...]` - all readable runs,
  sorted by `run_id` (which sorts chronologically). Unreadable run
  directories are skipped; the `runs list` command prints one warning line
  per skipped entry.
- `load_run(runs_root, run_id) -> RunRecord` - one run.
  `FileNotFoundError` if absent, `ValueError` if unparseable.

### Changed signatures

- `driver.train(..., tracker: TrackingHook | None = None)` and
  `driver.test(..., tracker: TrackingHook | None = None)`: additive
  keyword-only parameter. `None` means no tracking, so programmatic
  callers and the test suite are unaffected. The driver calls
  `start_run` before instantiating the model, `record_metrics` /
  `record_artifact` as results appear, `complete_run` on success, and
  `fail_run` followed by a re-raise when the model raises.
- CLI: the app callback gains `--runs-root PATH` (default `runs/`) and
  `--no-track`. `train` and `test` construct a `RunDirectoryTracker` unless
  `--no-track` is passed. A new `runs` sub-app provides `runs list`,
  `runs show RUN_ID`, and `runs compare RUN_ID RUN_ID [RUN_ID...]`.
- `build_app` and `kd.run` signatures are unchanged.

## Behavior

Run identity: `<UTC timestamp>Z-<4 hex>` such as `20260806T142530Z-1a2b`.
Lexicographic order equals chronological order; the random suffix breaks
same-second collisions (regenerated if the directory already exists).

Directory layout after a tracked `train`:

```
runs/
  20260806T142530Z-1a2b/
    run.json
    configs/
      model_config.yaml
      train_config.yaml
```

`run.json` after that run completes:

```json
{
  "schema_version": 1,
  "run_id": "20260806T142530Z-1a2b",
  "command": "train",
  "model_name": "logistic_regression",
  "status": "completed",
  "started_at": "2026-08-06T14:25:30+00:00",
  "finished_at": "2026-08-06T14:25:41+00:00",
  "error": null,
  "metrics": {"train": {"sample_count": 891, "feature_count": 5}},
  "artifacts": {"model": "artifacts/titanic.joblib"},
  "configs": {
    "model_config": "configs/model_config.yaml",
    "train_config": "configs/train_config.yaml"
  }
}
```

Decisions this commits to:

- `run.json` is written at `start_run` with status `running` and rewritten
  at finalization. A crash leaves a `running` record behind, which is
  honest: the run never finished. Writes are atomic (temp file + rename) so
  `runs list` never reads a torn record.
- Timestamps are timezone-aware UTC, ISO 8601 in JSON.
- `metrics` is keyed by phase precisely so 0.3.0 can add a `"validation"`
  entry (fold metrics plus aggregate) without a schema break.
- Statistics values are free-form; JSON-incompatible values (numpy scalars,
  arrays) are serialized via `default=str`. The record is a faithful log,
  not a lossless round-trip of arbitrary objects.
- Config snapshots are YAML so a previous run's exact config can be reused
  directly: `train ... --train-config runs/<id>/configs/train_config.yaml`.
  Omitted configs (user passed no `--train-config`) snapshot as empty YAML
  files, keeping the layout uniform.
- The framework records artifacts it knows about: the submission file for
  `test` runs, and the conventional `model_path` config key (recorded as
  the `model` artifact whenever present in the train or test config). The
  key is a documented convention rather than part of the core contract;
  the tutorial states that naming it buys run records that point at the
  produced model file.
- Tracking write failures raise. A `start_run` failure aborts the
  invocation before any training happens; a finalization failure
  propagates rather than leaving a run silently unrecorded.
- `runs list` prints one line per run: id, command, model name, status, and
  a compact metrics summary. `runs show` pretty-prints the full record.
  `runs compare` prints an aligned plain-text table: one row per metric
  key (unioned across the selected runs, grouped by phase), one column per
  run, `-` for missing values. Plain text only; no new dependencies.

## Non-goals

- External tracker adapters (MLflow, Weights & Biases). The protocol is the
  seam; adapters are post-1.0 per ROADMAP.md.
- Run deletion, pruning, tagging, or searching. `runs` is read-only in
  0.2.0.
- Default-on tracking for programmatic `driver.train`/`driver.test` calls.
  The CLI is the recording surface; library callers opt in by passing a
  tracker.
- A `runs_root` parameter on `kd.run`. The CLI flag covers it; revisit if
  embedding users ask.
- Capturing git state (commit, dirty flag) into the record. Attractive, but
  scope creep for this release; noted for the post-1.0 horizon.
- Remote or shared storage of any kind.

## Test plan

Categories only; the test-writing phase produces the tests.

- Unit tests for `kaggle_driver.tracking`:
  - Run id shape, uniqueness, chronological sortability, collision retry.
  - `RunDirectoryTracker` lifecycle: `start_run` creates the directory and
    a `running` record; `complete_run` finalizes with metrics and
    timestamps; `fail_run` records the error and `failed` status;
    lifecycle misuse (`record_metrics` before `start_run`, double
    `start_run`) raises.
  - Config snapshots: YAML round-trips through `load_yaml_config`; empty
    configs produce empty files.
  - `list_runs`: empty or missing root; multiple runs sorted; non-run
    entries ignored; unreadable `run.json` skipped, with the CLI warning
    per skipped entry.
  - `load_run`: missing id raises `FileNotFoundError`; corrupt record
    raises `ValueError`.
  - `RunRecord` mappings are frozen.
- Unit tests for driver integration (fake tracker implementing the
  protocol): call ordering; metrics and artifacts forwarded; `fail_run`
  called and exception re-raised when the model raises; `tracker=None`
  produces no calls.
- Unit tests for the CLI: default tracking writes under `runs/`;
  `--runs-root` redirects; `--no-track` writes nothing; `runs list`,
  `runs show`, `runs compare` output shapes; `runs show` unknown id exits
  nonzero with a clear message.
- User-story test: train then test via the CLI, `runs list` shows both,
  `runs compare` aligns their metrics, and re-running train with the
  snapshotted config from the first run reproduces its configuration.
- Edge cases: empty statistics mapping; numpy values in statistics;
  two runs within one second; deeply nested runs root created on demand.
- Adversarial cases: model raises mid-train (record says `failed`, error
  preserved, exception propagates unchanged); `run.json` deleted or
  corrupted behind a live tracker; run directory with no `run.json` in the
  root; unwritable runs root (tracking raises and aborts).

## Resolved questions

Decided at design sign-off (2026-08-06):

1. The tracker records the conventional `model_path` config key as the
   `model` artifact whenever present. Documented convention over purity.
2. `runs list` skips unreadable run directories and warns once per skipped
   entry. Nothing silent, nothing bricked.
3. Tracking write failures raise and abort the invocation. Errors never
   pass silently; the cost of a rare late failure is accepted.
4. `runs compare` ships in 0.2.0 alongside `list` and `show`. Comparison
   is the payoff of tracking.
