"""Self-update helpers for nanobot.

Provides detection of the current install (editable vs PyPI/zip), remote
version comparison against ``madkoding/nanobot`` main, and orchestration of a
full update (Python + WebUI rebuild + gateway restart). All network calls are
plain ``urllib`` so the module has no extra dependencies.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GITHUB_REPO = "madkoding/nanobot"
MAIN_ZIP_URL = f"https://github.com/{GITHUB_REPO}/archive/refs/heads/main.zip"
MAIN_COMMITS_API = f"https://api.github.com/repos/{GITHUB_REPO}/commits/main"
RAW_PYPROJECT_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/pyproject.toml"
GATEWAY_SERVICE = "nanobot-gateway"

# Marker written next to the package source so `update` can tell a source
# checkout from a plain PyPI install even when dist-info is missing.
_SOURCE_MARKER = ".nanobot-source"


@dataclass(frozen=True)
class InstallInfo:
    """Detected installation state."""

    kind: str  # "editable" | "pypi" | "unknown"
    repo_path: Path | None  # source dir for editable installs
    venv_python: str  # python executable to use for pip
    version: str  # local version string


@dataclass(frozen=True)
class RemoteInfo:
    """Remote main-branch state."""

    sha: str | None
    version: str | None
    date: str | None


def _run(
    cmd: list[str], *, check: bool = False, capture: bool = True, cwd: str | None = None
) -> subprocess.CompletedProcess:
    """Run a subprocess, returning the result without raising on failure."""
    try:
        return subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=True,
            cwd=cwd,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(cmd, 127, "", "command not found")


def _http_get(url: str, timeout: float = 15.0) -> str | None:
    """Fetch a URL as text, returning None on any network/HTTP error."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return None


def _read_direct_url(dist: Any) -> dict[str, Any] | None:
    """Read ``direct_url.json`` from a distribution's dist-info, if present."""
    try:
        dist_path = Path(dist._path)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        return None
    direct = dist_path / "direct_url.json"
    if not direct.is_file():
        return None
    try:
        return json.loads(direct.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _find_repo_path() -> Path | None:
    """Locate the source checkout for an editable install."""
    try:
        import importlib.metadata as md

        dist = md.distribution("nanobot-ai")
    except (md.PackageNotFoundError, Exception):
        return None
    direct = _read_direct_url(dist)
    if not direct:
        return None
    url = direct.get("url", "")
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("file://"):
        return Path(url[len("file://") :]).resolve(strict=False)
    return None


def _local_version() -> str:
    from nanobot import __version__

    return __version__


def _local_sha(repo_path: Path | None) -> str | None:
    """Return the short commit SHA of a source checkout, if it is a git repo."""
    if repo_path is None:
        return None
    result = _run(["git", "-C", str(repo_path), "rev-parse", "--short", "HEAD"])
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def detect_install() -> InstallInfo:
    """Detect the current install kind, source path, and venv python."""
    repo_path = _find_repo_path()
    kind = "editable" if repo_path is not None else "pypi"
    return InstallInfo(
        kind=kind,
        repo_path=repo_path,
        venv_python=sys.executable,
        version=_local_version(),
    )


def get_remote_main_sha() -> str | None:
    """Return the latest main-branch commit SHA via the GitHub API."""
    data = _http_get(MAIN_COMMITS_API)
    if not data:
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    sha = payload.get("sha")
    return sha if isinstance(sha, str) and sha else None


def get_remote_pyproject_version() -> str | None:
    """Return the version from the remote main ``pyproject.toml``."""
    data = _http_get(RAW_PYPROJECT_URL)
    if not data:
        return None
    try:
        import tomllib

        parsed = tomllib.loads(data)
    except (tomllib.TOMLDecodeError, Exception):
        return None
    version = parsed.get("project", {}).get("version")
    return version if isinstance(version, str) and version else None


def get_remote_info() -> RemoteInfo:
    """Return remote main-branch SHA, version, and commit date."""
    sha = get_remote_main_sha()
    version = get_remote_pyproject_version()
    date = None
    if sha:
        data = _http_get(MAIN_COMMITS_API)
        if data:
            try:
                payload = json.loads(data)
                commit = payload.get("commit", {})
                date = commit.get("committer", {}).get("date")
            except (json.JSONDecodeError, AttributeError):
                date = None
    return RemoteInfo(sha=sha, version=version, date=date)


def _download_main_zip(dest: Path) -> Path:
    """Download and extract the main-branch zip into *dest*; return the source dir."""
    zip_path = dest / "nanobot-main.zip"
    with urllib.request.urlopen(MAIN_ZIP_URL, timeout=60) as resp:
        zip_path.write_bytes(resp.read())
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    # The zip extracts to <dest>/nanobot-main/
    source = dest / "nanobot-main"
    if not source.is_dir():
        # Fallback: locate the single top-level dir.
        dirs = [p for p in dest.iterdir() if p.is_dir()]
        if len(dirs) == 1:
            source = dirs[0]
        else:
            raise RuntimeError("could not locate extracted nanobot source")
    return source


def _pip_install(source: str, venv_python: str) -> tuple[bool, str]:
    """Install/upgrade nanobot from *source* into the venv, without deps.

    Returns (ok, message) so callers can surface the underlying pip error
    instead of a bare "pip install failed".
    """
    result = _run(
        [
            venv_python,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            "--upgrade",
            source,
        ],
    )
    if result.returncode == 0:
        return True, ""
    detail = (result.stderr or result.stdout or "").strip()
    return False, detail[:2000]


def _git_update(repo_path: Path) -> bool:
    """Fetch and hard-reset a source checkout to origin/main."""
    fetch = _run(["git", "-C", str(repo_path), "fetch", "origin"])
    if fetch.returncode != 0:
        return False
    reset = _run(["git", "-C", str(repo_path), "reset", "--hard", "origin/main"])
    return reset.returncode == 0


def _find_webui_dir(source_dir: Path) -> Path | None:
    """Locate the webui directory under a source checkout."""
    for candidate in (source_dir / "webui", source_dir / "nanobot" / "webui"):
        if (candidate / "package.json").is_file():
            return candidate
    return None


def rebuild_webui(source_dir: Path) -> tuple[bool, str]:
    """Rebuild the WebUI bundle. Returns (ok, message)."""
    webui_dir = _find_webui_dir(source_dir)
    if webui_dir is None:
        return False, "webui directory not found in source"
    bun = shutil.which("bun")
    npm = shutil.which("npm")
    if bun:
        install = _run(["bun", "install", "--frozen-lockfile"], cwd=str(webui_dir))
        if install.returncode != 0:
            return False, "bun install failed"
        build = _run(["bun", "run", "build"], cwd=str(webui_dir))
        if build.returncode != 0:
            return False, "bun run build failed"
        return True, "WebUI rebuilt with bun"
    if npm:
        install = _run(["npm", "ci"], cwd=str(webui_dir))
        if install.returncode != 0:
            return False, "npm ci failed"
        build = _run(["npm", "run", "build"], cwd=str(webui_dir))
        if build.returncode != 0:
            return False, "npm run build failed"
        return True, "WebUI rebuilt with npm"
    return False, "neither bun nor npm found; WebUI not rebuilt"


def _gateway_active() -> bool:
    result = _run(["systemctl", "--user", "is-active", GATEWAY_SERVICE])
    return result.returncode == 0 and result.stdout.strip() == "active"


def restart_gateway() -> tuple[bool, str]:
    """Restart the user systemd gateway service. Returns (ok, message)."""
    if not _gateway_active():
        return False, "gateway service not active; not restarted"
    result = _run(["systemctl", "--user", "restart", GATEWAY_SERVICE])
    if result.returncode != 0:
        return False, f"gateway restart failed: {result.stderr.strip()}"
    return True, "gateway restarted"


def _reinstall_channel_deps(source_dir: Path, venv_python: str) -> bool:
    """Reinstall channel dependencies if the helper script exists."""
    script = source_dir / "scripts" / "install_channel_dependencies.py"
    if not script.is_file():
        return False
    result = _run([venv_python, "-m", "scripts.install_channel_dependencies", "--all-channels"])
    return result.returncode == 0


def _format_remote(remote: RemoteInfo) -> str:
    parts = []
    if remote.sha:
        parts.append(f"main@{remote.sha[:7]}")
    if remote.version:
        parts.append(f"v{remote.version}")
    if remote.date:
        parts.append(remote.date[:10])
    return " ".join(parts) if parts else "unknown"


def perform_update(
    *,
    check: bool = False,
    yes: bool = False,
    no_restart: bool = False,
    no_webui: bool = False,
    confirm: Any | None = None,
) -> int:
    """Run the update flow. Returns a process exit code.

    *confirm* is an optional callable ``(prompt: str) -> bool`` used to ask the
    user before destructive steps; defaults to ``input``-based confirmation.
    """
    info = detect_install()
    remote = get_remote_info()

    local_desc = f"v{info.version}"
    if info.repo_path:
        sha = _local_sha(info.repo_path)
        if sha:
            local_desc += f" ({sha})"

    print(f"Local:  {local_desc}  [{info.kind}]")
    print(f"Remote: {_format_remote(remote)}")

    if check:
        return 0

    if remote.sha is None and remote.version is None:
        print("Could not reach the remote repository; aborting.")
        return 1

    if confirm is None:

        def confirm(prompt: str) -> bool:
            if yes:
                return True
            try:
                return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
            except (EOFError, KeyboardInterrupt):
                return False

    if not confirm("Update nanobot to the latest main branch?"):
        print("Aborted.")
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="nanobot-update-"))
    try:
        if info.kind == "editable" and info.repo_path is not None:
            # Source checkout: fast git update, then refresh editable metadata.
            if not _git_update(info.repo_path):
                print("git update failed; aborting.")
                return 1
            source_dir = info.repo_path
            ok, err = _pip_install(str(source_dir), info.venv_python)
            if not ok:
                print(f"pip editable refresh failed: {err}")
                return 1
        else:
            # PyPI/zip install: download main and reinstall.
            source_dir = _download_main_zip(tmp)
            ok, err = _pip_install(str(source_dir), info.venv_python)
            if not ok:
                print(f"pip install failed: {err}")
                return 1

        if not no_webui:
            ok, msg = rebuild_webui(source_dir)
            print(f"WebUI: {msg}")
        else:
            print("WebUI: skipped (--no-webui)")

        _reinstall_channel_deps(source_dir, info.venv_python)

        if not no_restart:
            ok, msg = restart_gateway()
            print(f"Gateway: {msg}")
        else:
            print("Gateway: skipped (--no-restart)")

        print(
            f"Updated nanobot to {_format_remote(remote)}. Hard-refresh the browser (Ctrl+Shift+R)."
        )
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
