#!/usr/bin/env bash
# Fail if any committed text file contains a literal em dash (U+2014).
# Rewrite with periods, commas, parens, or semicolons.
set -euo pipefail

cd "$(dirname "$0")/.."

python3 - <<'PY'
import sys
from pathlib import Path

EXTS = {".py", ".md", ".rst", ".toml", ".yaml", ".yml", ".sh", ".cfg"}
SKIP_DIRS = {".venv", ".git", "build", "dist", "docs/_build", ".tox", ".ruff_cache", ".mypy_cache", ".pytest_cache"}
EM_DASH = chr(0x2014)

failures = []
for path in Path(".").rglob("*"):
    if not path.is_file() or path.suffix not in EXTS:
        continue
    rel = path.as_posix()
    if any(rel == d or rel.startswith(d + "/") for d in SKIP_DIRS):
        continue
    try:
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                if EM_DASH in line:
                    failures.append((rel, lineno, line.rstrip()))
    except UnicodeDecodeError:
        continue

if failures:
    print("Em dash (U+2014) found in committed text. Rewrite or use alternative punctuation:", file=sys.stderr)
    for rel, lineno, line in failures:
        print(f"  {rel}:{lineno}: {line}", file=sys.stderr)
    sys.exit(1)
PY
