"""Unit tests for :mod:`kaggle_driver.config`."""

from __future__ import annotations

from pathlib import Path

import pytest
from immutabledict import immutabledict

from kaggle_driver.config import load_yaml_config


def test_load_yaml_config_returns_mapping(tmp_yaml_config: Path) -> None:
    """Test the loader parses a flat YAML file into a frozen mapping."""
    config = load_yaml_config(tmp_yaml_config)

    assert config == {"alpha": 0.5, "beta": 3, "note": "hello"}


def test_load_yaml_config_result_rejects_mutation(tmp_yaml_config: Path) -> None:
    """Test a loaded config cannot be edited in place by the caller."""
    config = load_yaml_config(tmp_yaml_config)

    assert isinstance(config, immutabledict)
    with pytest.raises(TypeError):
        config["alpha"] = 1.0  # type: ignore[index]


def test_load_yaml_config_unpacks_as_keyword_arguments(tmp_yaml_config: Path) -> None:
    """Test the frozen config still splats into a call, as the driver relies on."""

    def accept(alpha: float, beta: int, note: str) -> tuple[float, int, str]:
        return alpha, beta, note

    result = accept(**load_yaml_config(tmp_yaml_config))

    assert result == (0.5, 3, "hello")


def test_load_yaml_config_accepts_string_paths(tmp_yaml_config: Path) -> None:
    """Test string paths are accepted in addition to ``Path`` objects."""
    from_string_path = load_yaml_config(str(tmp_yaml_config))
    from_path_object = load_yaml_config(tmp_yaml_config)

    assert from_string_path == from_path_object


def test_load_yaml_config_returns_empty_mapping_for_empty_file(tmp_path: Path) -> None:
    """Test an empty YAML file yields an empty frozen mapping, not ``None``."""
    empty = tmp_path / "empty.yml"
    empty.write_text("", encoding="utf-8")

    config = load_yaml_config(empty)

    assert isinstance(config, immutabledict)
    assert len(config) == 0


def test_load_yaml_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    """Test a YAML root that is a list (not a mapping) raises :class:`TypeError`."""
    list_rooted_config = tmp_path / "list.yml"
    list_rooted_config.write_text("- 1\n- 2\n", encoding="utf-8")

    with pytest.raises(TypeError, match="YAML mapping"):
        load_yaml_config(list_rooted_config)


def test_load_yaml_config_rejects_missing_path(tmp_path: Path) -> None:
    """Test a path that does not exist raises :class:`FileNotFoundError`."""
    missing = tmp_path / "nope.yml"

    with pytest.raises(FileNotFoundError):
        load_yaml_config(missing)
