"""Tests for the self-update helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from nanobot.utils.update import (
    InstallInfo,
    RemoteInfo,
    _find_repo_path,
    detect_install,
    get_remote_info,
    get_remote_main_sha,
    get_remote_pyproject_version,
    perform_update,
    rebuild_webui,
    restart_gateway,
)


class _FakeDist:
    def __init__(self, direct_url: dict | None):
        self._path = None
        self._direct_url = direct_url

    @property
    def _path(self):  # noqa: N802
        return self._path_value

    @_path.setter
    def _path(self, value):
        self._path_value = value


def _make_dist(tmp_path: Path, direct_url: dict | None) -> _FakeDist:
    dist = _FakeDist(direct_url)
    if direct_url is not None:
        dist_info = tmp_path / "nanobot_ai-0.3.0.dist-info"
        dist_info.mkdir(parents=True, exist_ok=True)
        (dist_info / "direct_url.json").write_text(json.dumps(direct_url), encoding="utf-8")
        dist._path = dist_info
    return dist


def test_detect_install_editable(tmp_path):
    _make_dist(tmp_path, {"dir_info": {"editable": True}, "url": f"file://{tmp_path}"})
    with patch("nanobot.utils.update._find_repo_path", return_value=tmp_path):
        info = detect_install()
    assert info.kind == "editable"
    assert info.repo_path == tmp_path


def test_detect_install_pypi(tmp_path):
    with patch("nanobot.utils.update._find_repo_path", return_value=None):
        info = detect_install()
    assert info.kind == "pypi"
    assert info.repo_path is None


def test_find_repo_path_prefers_direct_url_when_valid(tmp_path):
    direct_url_path = tmp_path / "from-direct-url"
    direct_url_path.mkdir()
    (direct_url_path / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n', encoding="utf-8")
    with (
        patch("nanobot.utils.update._find_repo_path_from_direct_url", return_value=direct_url_path),
        patch("nanobot.utils.update._find_repo_path_from_package", return_value=tmp_path),
    ):
        found = _find_repo_path()
    assert found == direct_url_path


def test_find_repo_path_validates_direct_url_has_pyproject(tmp_path):
    stale_path = tmp_path / "stale"
    stale_path.mkdir()
    # No pyproject.toml → stale path is rejected, fallback to package path.
    with (
        patch("nanobot.utils.update._find_repo_path_from_direct_url", return_value=stale_path),
        patch("nanobot.utils.update._find_repo_path_from_package", return_value=tmp_path),
    ):
        found = _find_repo_path()
    assert found == tmp_path


def test_find_repo_path_falls_back_to_package_location(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n', encoding="utf-8")
    with (
        patch("nanobot.utils.update._find_repo_path_from_direct_url", return_value=None),
        patch("nanobot.utils.update._find_repo_path_from_package", return_value=tmp_path),
    ):
        found = _find_repo_path()
    assert found == tmp_path


def test_get_remote_main_sha():
    with patch("nanobot.utils.update._http_get", return_value=json.dumps({"sha": "abc123"})):
        assert get_remote_main_sha() == "abc123"


def test_get_remote_main_sha_network_error():
    with patch("nanobot.utils.update._http_get", return_value=None):
        assert get_remote_main_sha() is None


def test_get_remote_pyproject_version():
    body = '[project]\nname = "nanobot-ai"\nversion = "0.4.0"\n'
    with patch("nanobot.utils.update._http_get", return_value=body):
        assert get_remote_pyproject_version() == "0.4.0"


def test_get_remote_info_combines_sha_and_version():
    body = json.dumps({"sha": "abc123", "commit": {"committer": {"date": "2026-08-01T00:00:00Z"}}})
    with patch("nanobot.utils.update._http_get", side_effect=[body, '[project]\nversion = "0.4.0"\n', body]):
        remote = get_remote_info()
    assert remote.sha == "abc123"
    assert remote.version == "0.4.0"
    assert remote.date == "2026-08-01T00:00:00Z"


def test_perform_update_check_only_returns_zero(capsys):
    with patch("nanobot.utils.update.detect_install", return_value=InstallInfo("pypi", None, "python", "0.3.0")), \
         patch("nanobot.utils.update.get_remote_info", return_value=RemoteInfo("abc123", "0.4.0", None)):
        code = perform_update(check=True)
    assert code == 0
    out = capsys.readouterr().out
    assert "Local:" in out
    assert "Remote:" in out


def test_perform_update_aborts_when_remote_unreachable(capsys):
    with patch("nanobot.utils.update.detect_install", return_value=InstallInfo("pypi", None, "python", "0.3.0")), \
         patch("nanobot.utils.update.get_remote_info", return_value=RemoteInfo(None, None, None)):
        code = perform_update()
    assert code == 1
    assert "Could not reach" in capsys.readouterr().out


def test_perform_update_full_mocked(tmp_path, capsys):
    source = tmp_path / "nanobot-main"
    source.mkdir()
    (source / "webui").mkdir()
    (source / "webui" / "package.json").write_text("{}", encoding="utf-8")

    with patch("nanobot.utils.update.detect_install", return_value=InstallInfo("pypi", None, "python", "0.3.0")), \
         patch("nanobot.utils.update.get_remote_info", return_value=RemoteInfo("abc123", "0.4.0", None)), \
         patch("nanobot.utils.update._download_main_zip", return_value=source), \
         patch("nanobot.utils.update._pip_install", return_value=(True, "")), \
         patch("nanobot.utils.update.rebuild_webui", return_value=(True, "WebUI rebuilt")), \
         patch("nanobot.utils.update._reinstall_channel_deps", return_value=False), \
         patch("nanobot.utils.update.restart_gateway", return_value=(True, "gateway restarted")):
        code = perform_update(yes=True)
    assert code == 0
    out = capsys.readouterr().out
    assert "WebUI: WebUI rebuilt" in out
    assert "Gateway: gateway restarted" in out


def test_rebuild_webui_no_bun_npm(tmp_path):
    webui_dir = tmp_path / "webui"
    webui_dir.mkdir()
    (webui_dir / "package.json").write_text("{}", encoding="utf-8")
    with patch("nanobot.utils.update.shutil.which", return_value=None):
        ok, msg = rebuild_webui(tmp_path)
    assert ok is False
    assert "neither bun nor npm" in msg


def test_restart_gateway_not_active():
    with patch("nanobot.utils.update._gateway_active", return_value=False):
        ok, msg = restart_gateway()
    assert ok is False
    assert "not active" in msg
