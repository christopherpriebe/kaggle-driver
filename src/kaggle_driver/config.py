"""Loading of the YAML configuration files the CLI passes to models."""

from pathlib import Path
from typing import Any

import yaml
from immutabledict import immutabledict

__all__ = ["load_yaml_config"]


def load_yaml_config(path: Path | str) -> immutabledict[str, Any]:
    """Load a YAML file into a frozen mapping.

    Args:
        path: Path to a YAML file.

    Returns:
        The parsed YAML as an ``immutabledict``. Empty files yield an empty
        mapping. Nested containers are left as parsed; only the top level is
        frozen.

    Raises:
        FileNotFoundError: The given path does not exist.
        TypeError: The YAML root is not a mapping.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return immutabledict()
    if not isinstance(data, dict):
        raise TypeError(
            f"Config file must contain a YAML mapping at the root, got {type(data).__name__}",
        )
    return immutabledict(data)
