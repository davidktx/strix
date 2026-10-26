#!/usr/bin/env python3
"""Fail if litellm appears in dependencies or application imports."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_IMPORT = re.compile(r"(?:^|\s)(?:import\s+litellm|from\s+litellm\b)")


def _check_pyproject() -> list[str]:
    errors: list[str] = []
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data.get("project", {}).get("dependencies", [])
    errors.extend(
        f"pyproject.toml dependency: {dep}"
        for dep in deps
        if isinstance(dep, str) and dep.split("[", 1)[0].strip().startswith("litellm")
    )
    lock = ROOT / "uv.lock"
    if lock.exists() and 'name = "litellm"' in lock.read_text(encoding="utf-8"):
        errors.append("uv.lock still contains litellm")
    return errors


def _check_imports() -> list[str]:
    errors: list[str] = []
    for path in (ROOT / "strix").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if FORBIDDEN_IMPORT.search(text):
            errors.append(f"forbidden import in {path.relative_to(ROOT)}")
    return errors


def main() -> int:
    errors = _check_pyproject() + _check_imports()
    if errors:
        for err in errors:
            print(err, file=sys.stderr)  # noqa: T201
        return 1
    print("ok: litellm not present in dependencies or strix imports")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
