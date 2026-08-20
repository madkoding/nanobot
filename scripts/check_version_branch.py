#!/usr/bin/env python3
"""Guard: version bumps must originate on `develop`.

develop is the single source of truth for version numbers. A release is just a
propagation of whatever version develop already carries, so:

- On any feature/PR branch, changing the version is a mistake -> fail.
- On `develop`, bumps are allowed (they are the only place a new version is set).
- On `main`, the only acceptable version change is the release merge that
  brings in the version already present on `develop`. If `main`'s version
  differs from `develop`'s *other* than by exactly adopting it, that is a
  drift -> fail.

Run as a pre-commit hook on pyproject.toml / webui/package.json changes.
"""
from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
PACKAGE_JSON = ROOT / "webui" / "package.json"


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def current_branch() -> str:
    try:
        return _git(["branch", "--show-current"])
    except subprocess.CalledProcessError:
        # Detached HEAD (CI). Fall back to the env var.
        return __import__("os").environ.get("GITHUB_REF_NAME", "")


def local_version() -> str:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data.get("project", {}).get("version", "")


def remote_version(ref: str) -> str:
    try:
        blob = _git(["show", f"{ref}:pyproject.toml"])
    except subprocess.CalledProcessError:
        return ""
    data = tomllib.loads(blob)
    return data.get("project", {}).get("version", "")


def main() -> int:
    branch = current_branch()
    version = local_version()

    if branch in ("", "HEAD"):
        print("check_version_branch.py: unknown branch; skipping (CI should set GITHUB_REF_NAME).")
        return 0

    # Feature / PR / other branches: no version changes.
    if branch not in ("develop", "main"):
        print(
            f"check_version_branch.py: version change on branch '{branch}' — "
            "bumps must originate on 'develop'. Run `git checkout develop` and "
            "use `python scripts/bump_version.py <part>`."
        )
        return 1

    if branch == "develop":
        # develop is the source of truth; any bump here is legitimate.
        return 0

    # branch == "main": allow only adopting the version develop already has.
    develop_version = remote_version("origin/develop")
    if develop_version and version == develop_version:
        return 0
    print(
        f"check_version_branch.py: version on 'main' is {version!r} but develop "
        f"has {develop_version!r}. main must only carry the version released "
        "from develop — align by merging develop, not by bumping here."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
