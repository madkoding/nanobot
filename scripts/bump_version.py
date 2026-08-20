#!/usr/bin/env python3
"""Bump nanobot's canonical version (pyproject.toml + webui/package.json).

develop is the single source of truth for version numbers. A release only
propagates whatever version develop already carries — it is never recomputed
by CI. This script is the only intended way to change the version, so a bump
can't happen by accident.

Usage:
    python scripts/bump_version.py <part>          # patch | minor | major
    python scripts/bump_version.py 0.6.1           # explicit exact version
    python scripts/bump_version.py --check 0.6.1   # verify current == 0.6.1

Examples:
    python scripts/bump_version.py patch   # 0.6.0 -> 0.6.1
    python scripts/bump_version.py minor   # 0.6.0 -> 0.7.0
    python scripts/bump_version.py major   # 0.6.0 -> 1.0.0
"""
from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
PACKAGE_JSON = ROOT / "webui" / "package.json"

_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def read_pyproject_version() -> str:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    if not version:
        raise SystemExit(f"scripts/bump_version.py: no version found in {PYPROJECT}")
    return version


def read_package_json_version() -> str:
    return json.loads(PACKAGE_JSON.read_text(encoding="utf-8")).get("version", "")


def parse(version: str) -> tuple[int, int, int]:
    match = _SEMVER.match(version.strip())
    if not match:
        raise SystemExit(
            f"scripts/bump_version.py: invalid semver '{version}'. Expected MAJOR.MINOR.PATCH."
        )
    return tuple(int(g) for g in match.groups())  # type: ignore[return-value]


def next_version(current: str, part: str) -> str:
    major, minor, patch = parse(current)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise SystemExit(f"scripts/bump_version.py: unknown part '{part}'")


def write_versions(version: str) -> None:
    # Rewrite only the canonical `version = "..."` line under [project]; the
    # rest of pyproject.toml (inline tables, arrays of tables) is left untouched.
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    new_project = re.sub(
        r"(?m)^version\s*=\s*\".*?\"$",
        f'version = "{version}"',
        pyproject,
        count=1,
    )
    if new_project == pyproject:
        raise SystemExit(
            "scripts/bump_version.py: could not locate the `version = ...` line in pyproject.toml"
        )
    PYPROJECT.write_text(new_project, encoding="utf-8")

    pkg = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    pkg["version"] = version
    PACKAGE_JSON.write_text(json.dumps(pkg, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bump nanobot version (develop is canonical).")
    parser.add_argument("part", help="patch | minor | major | <exact semver>")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail unless current pyproject version equals the given exact version",
    )
    args = parser.parse_args()

    current = read_pyproject_version()
    pkg_current = read_package_json_version()

    if pkg_current != current:
        raise SystemExit(
            f"scripts/bump_version.py: webui/package.json={pkg_current} != "
            f"pyproject.toml={current}. Run scripts/sync_versions.py first."
        )

    if args.check:
        if args.part == current:
            print(f"scripts/bump_version.py: OK — version is {current}")
            return 0
        raise SystemExit(
            f"scripts/bump_version.py: expected version {args.part}, found {current}"
        )

    if args.part in ("patch", "minor", "major"):
        target = next_version(current, args.part)
    else:
        target = args.part
        # Validate the target is a strict bump forward from current, so an
        # explicit version can never be set to something equal or lower.
        if parse(target) <= parse(current):
            raise SystemExit(
                f"scripts/bump_version.py: target {target} is not newer than current {current}."
            )

    write_versions(target)
    print(f"scripts/bump_version.py: {current} -> {target} (pyproject.toml + webui/package.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
