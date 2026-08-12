"""Tests for ``compile_project_context`` and the project context provider."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import pytest

from nanobot.runtime_context import (
    PROJECT_CONTEXT_SOURCE,
    compile_project_context,
)
from nanobot.webui.project_context_provider import make_project_context_provider
from nanobot.webui.projects import WebUIProjectsController


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_compile_project_context_returns_block_with_name_and_files(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("alpha", "Always respond in Spanish")
    c.add_file(
        s.id, "notes.md", "data:text/markdown;base64," + base64.b64encode(b"# notes").decode()
    )
    block = compile_project_context(c, s.id)
    assert block is not None
    assert block.source == PROJECT_CONTEXT_SOURCE
    assert "<name>alpha</name>" in block.content
    assert "Always respond in Spanish" in block.content
    assert 'name="notes.md"' in block.content
    assert "Runtime Context" in block.content


def test_compile_project_context_returns_none_for_unknown_project(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    assert compile_project_context(c, "missing") is None


def test_compile_project_context_returns_none_for_empty_project(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("blank", "")
    assert compile_project_context(c, s.id) is None


def test_compile_project_context_includes_folders(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("alpha", "")
    c.add_folder(s.id, "/tmp/alpha")
    block = compile_project_context(c, s.id)
    assert block is not None
    assert '<folder path="/tmp/alpha" />' in block.content
    assert "<folders>" in block.content


def test_compile_project_context_includes_file_paths(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("alpha", "instructions")
    f = c.add_file(
        s.id, "notes.md", "data:text/markdown;base64," + base64.b64encode(b"# notes").decode()
    )
    files_dir = c.files_dir_for(s.id)
    block = compile_project_context(c, s.id)
    assert block is not None
    # Paths are rendered through json.dumps, which escapes backslashes on Windows.
    assert f"<files_dir path={json.dumps(str(files_dir))} />" in block.content
    assert f'id="{f.id}"' in block.content
    assert f"path={json.dumps(str(files_dir / f'{f.id}.bin'))}" in block.content


def test_compile_project_context_truncates_to_budget(data_dir: Path) -> None:
    c = WebUIProjectsController(data_dir)
    s = c.create_project("huge", "x" * 20_000)
    block = compile_project_context(c, s.id, token_budget=200)
    assert block is not None
    assert len(block.content) < 5_000
    assert "(truncated)" in block.content


def test_provider_skips_when_no_session_key() -> None:
    c = WebUIProjectsController(Path("/tmp/nope"))
    provider = make_project_context_provider(object(), c)
    block = asyncio.run(provider(_FakeRequest(session_key=None)))
    assert block is None


def test_provider_skips_when_unbound() -> None:
    sm = _FakeSessionManager({"websocket:1": {"metadata": {}}})
    c = WebUIProjectsController(Path("/tmp/nope"))
    provider = make_project_context_provider(sm, c)
    block = asyncio.run(provider(_FakeRequest(session_key="websocket:1")))
    assert block is None


def test_provider_returns_block_and_marks_injected() -> None:
    data_dir = Path("/tmp/nope")
    c = WebUIProjectsController(data_dir)
    s = c.create_project("proj", "instructions here")
    session = _FakeSession(metadata={"project_id": s.id})
    sm = _FakeSessionManager({"websocket:1": {"metadata": {}}})
    sm._sessions = session
    provider = make_project_context_provider(sm, c)
    block = asyncio.run(provider(_FakeRequest(session_key="websocket:1")))
    assert block is not None
    assert "<name>proj</name>" in block.content
    assert session.metadata["_project_context_injected"] is True


def test_provider_skips_after_inject() -> None:
    data_dir = Path("/tmp/nope")
    c = WebUIProjectsController(data_dir)
    s = c.create_project("proj", "")
    session = _FakeSession(metadata={"project_id": s.id, "_project_context_injected": True})
    sm = _FakeSessionManager({})
    sm._sessions = session
    provider = make_project_context_provider(sm, c)
    block = asyncio.run(provider(_FakeRequest(session_key="websocket:1")))
    assert block is None


class _FakeRequest:
    def __init__(self, session_key: str | None) -> None:
        self.session_key = session_key


class _FakeSession:
    def __init__(self, metadata: dict | None = None) -> None:
        self.metadata = metadata or {}


class _FakeSessionManager:
    def __init__(self, by_key: dict[str, dict[str, Any]]) -> None:
        self._by_key = by_key
        self._sessions: _FakeSession | None = None
        self.saved: list[_FakeSession] = []

    def read_session_metadata(self, key: str) -> dict[str, Any] | None:
        if self._sessions is not None:
            return {"metadata": self._sessions.metadata}
        return self._by_key.get(key)

    def get_or_create(self, key: str) -> _FakeSession:
        if self._sessions is None:
            self._sessions = _FakeSession()
        return self._sessions

    def save(self, session: _FakeSession) -> None:
        self.saved.append(session)
