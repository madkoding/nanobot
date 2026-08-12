"""Regression tests for collision-resistant session filenames."""

import json
from datetime import datetime
from pathlib import Path

from nanobot.session.manager import Session, SessionManager


def _manager(tmp_path: Path, monkeypatch) -> SessionManager:
    monkeypatch.setattr(
        "nanobot.session.manager.get_legacy_sessions_dir",
        lambda: tmp_path / "legacy_sessions",
    )
    return SessionManager(tmp_path / "workspace")


def _write_session_file(path: Path, key: str, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "_type": "metadata",
        "key": key,
        "created_at": datetime(2025, 1, 1).isoformat(),
        "updated_at": datetime(2025, 1, 1).isoformat(),
        "metadata": {"source": "test"},
        "last_consolidated": 0,
    }
    message = {"role": "user", "content": content}
    path.write_text(
        json.dumps(metadata) + "\n" + json.dumps(message) + "\n",
        encoding="utf-8",
    )


def test_distinct_keys_have_distinct_filenames(tmp_path: Path, monkeypatch) -> None:
    sm = _manager(tmp_path, monkeypatch)

    first = sm._get_session_path("telegram:a_b")
    second = sm._get_session_path("telegram:a:b")

    assert first.name != second.name
    assert sm.safe_key("telegram:a_b") == sm.safe_key("telegram:a:b")
    assert sm._storage_key("telegram:a_b") != sm._storage_key("telegram:a:b")


def test_safe_key_is_lossy() -> None:
    assert SessionManager.safe_key("telegram:a_b") == SessionManager.safe_key("telegram:a:b")


def test_storage_key_is_collision_resistant() -> None:
    encoded = {
        SessionManager._storage_key("a:b"),
        SessionManager._storage_key("a_b"),
        SessionManager._storage_key("a:b:c"),
    }

    assert len(encoded) == 3
    assert SessionManager._storage_key("telegram:a_b") != SessionManager._storage_key(
        "telegram:a:b"
    )


def test_storage_paths_are_distinct_when_keys_collide_under_safe_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sm = _manager(tmp_path, monkeypatch)
    first = Session(key="telegram:a_b")
    first.add_message("user", "underscore history")
    second = Session(key="telegram:a:b")
    second.add_message("user", "colon history")

    sm.save(first)
    sm.save(second)

    assert sm.safe_key(first.key) == sm.safe_key(second.key)
    assert sm._get_session_path(first.key).exists()
    assert sm._get_session_path(second.key).exists()
    assert sm._get_session_path(first.key) != sm._get_session_path(second.key)

    sm.invalidate(first.key)
    sm.invalidate(second.key)
    loaded_first = sm._load(first.key)
    loaded_second = sm._load(second.key)

    assert loaded_first is not None
    assert loaded_second is not None
    assert loaded_first.messages[0]["content"] == "underscore history"
    assert loaded_second.messages[0]["content"] == "colon history"


def test_migrate_legacy_sessions_moves_global_legacy_file(tmp_path: Path, monkeypatch) -> None:
    sm = _manager(tmp_path, monkeypatch)
    legacy_dir = tmp_path / "legacy_sessions"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_file = legacy_dir / f"{sm.safe_key('telegram:12345')}.jsonl"
    _write_session_file(legacy_file, "telegram:12345", "legacy history")

    assert sm.migrate_legacy_sessions() == 1
    assert not legacy_file.exists()
    assert sm._get_session_path("telegram:12345").exists()
    loaded = sm._load("telegram:12345")
    assert loaded is not None
    assert loaded.messages[0]["content"] == "legacy history"


def test_migrate_legacy_sessions_moves_lossy_workspace_file(tmp_path: Path, monkeypatch) -> None:
    sm = _manager(tmp_path, monkeypatch)
    lossy_file = sm.sessions_dir / f"{sm.safe_key('websocket:abc')}.jsonl"
    _write_session_file(lossy_file, "websocket:abc", "lossy history")

    assert sm.migrate_legacy_sessions() == 1
    assert not lossy_file.exists()
    assert sm._get_session_path("websocket:abc").exists()


def test_migrate_legacy_sessions_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    sm = _manager(tmp_path, monkeypatch)
    legacy_dir = tmp_path / "legacy_sessions"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_file = legacy_dir / f"{sm.safe_key('telegram:12345')}.jsonl"
    _write_session_file(legacy_file, "telegram:12345", "legacy history")

    assert sm.migrate_legacy_sessions() == 1
    assert sm.migrate_legacy_sessions() == 0
    assert sm._get_session_path("telegram:12345").exists()


def test_migrate_legacy_sessions_skips_when_target_exists(tmp_path: Path, monkeypatch) -> None:
    sm = _manager(tmp_path, monkeypatch)
    legacy_dir = tmp_path / "legacy_sessions"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_file = legacy_dir / f"{sm.safe_key('telegram:12345')}.jsonl"
    _write_session_file(legacy_file, "telegram:12345", "legacy history")
    _write_session_file(sm._get_session_path("telegram:12345"), "telegram:12345", "newer history")

    assert sm.migrate_legacy_sessions() == 0
    assert legacy_file.exists()
    loaded = sm._load("telegram:12345")
    assert loaded is not None
    assert loaded.messages[0]["content"] == "newer history"
