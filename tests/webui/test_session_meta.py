"""Tests for ``nanobot.webui.session_meta`` (chat project binding)."""

from __future__ import annotations

from nanobot.webui.session_meta import (
    CHAT_PROJECT_ID_METADATA_KEY,
    CHAT_PROJECT_INJECTED_FLAG,
    chat_project_id_from_metadata,
    mark_project_context_injected,
    set_chat_project_id,
)


class _FakeSession:
    def __init__(self, metadata: dict | None = None) -> None:
        self.metadata = metadata or {}


def test_chat_project_id_from_metadata_returns_clean_id() -> None:
    assert chat_project_id_from_metadata({"project_id": "  demo  "}) == "demo"


def test_chat_project_id_from_metadata_handles_missing() -> None:
    assert chat_project_id_from_metadata(None) is None
    assert chat_project_id_from_metadata({}) is None
    assert chat_project_id_from_metadata({"project_id": ""}) is None
    assert chat_project_id_from_metadata({"project_id": 123}) is None


def test_set_chat_project_id_writes_and_resets_flag() -> None:
    session = _FakeSession({"_other": "keep"})
    set_chat_project_id(session, "alpha")
    assert session.metadata[CHAT_PROJECT_ID_METADATA_KEY] == "alpha"
    assert session.metadata[CHAT_PROJECT_INJECTED_FLAG] is False
    assert session.metadata["_other"] == "keep"


def test_set_chat_project_id_rebind_resets_flag() -> None:
    session = _FakeSession(
        {CHAT_PROJECT_ID_METADATA_KEY: "alpha", CHAT_PROJECT_INJECTED_FLAG: True}
    )
    set_chat_project_id(session, "beta")
    assert session.metadata[CHAT_PROJECT_ID_METADATA_KEY] == "beta"
    assert session.metadata[CHAT_PROJECT_INJECTED_FLAG] is False


def test_set_chat_project_id_unbind_clears_both() -> None:
    session = _FakeSession(
        {CHAT_PROJECT_ID_METADATA_KEY: "alpha", CHAT_PROJECT_INJECTED_FLAG: True}
    )
    set_chat_project_id(session, None)
    assert CHAT_PROJECT_ID_METADATA_KEY not in session.metadata
    assert CHAT_PROJECT_INJECTED_FLAG not in session.metadata


def test_set_chat_project_id_blank_clears_both() -> None:
    session = _FakeSession(
        {CHAT_PROJECT_ID_METADATA_KEY: "alpha", CHAT_PROJECT_INJECTED_FLAG: True}
    )
    set_chat_project_id(session, "   ")
    assert CHAT_PROJECT_ID_METADATA_KEY not in session.metadata
    assert CHAT_PROJECT_INJECTED_FLAG not in session.metadata


def test_set_chat_project_id_handles_missing_metadata() -> None:
    """A session without a metadata dict is a no-op rather than a crash."""

    class _Broken:
        pass

    set_chat_project_id(_Broken(), "alpha")  # no metadata attribute -> no exception


def test_mark_project_context_injected_sets_flag() -> None:
    session = _FakeSession({CHAT_PROJECT_ID_METADATA_KEY: "alpha"})
    mark_project_context_injected(session)
    assert session.metadata[CHAT_PROJECT_INJECTED_FLAG] is True
    assert session.metadata[CHAT_PROJECT_ID_METADATA_KEY] == "alpha"
