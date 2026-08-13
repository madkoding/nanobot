"""Tests for nanobot.providers.base.ToolIdRemapper."""

from __future__ import annotations

from nanobot.providers.base import ToolIdRemapper


def _normalize(tid: str) -> str:
    return tid.replace(".", "_").replace("|", "_")[:9]


def test_unique_tool_id_generates_provider_safe_id():
    r = ToolIdRemapper(_normalize)
    assert r.unique_tool_id("call.123") != "call.123"
    assert isinstance(r.unique_tool_id("call.456"), str)


def test_repeated_calls_for_same_raw_return_different_ids():
    r = ToolIdRemapper(_normalize)
    first = r.unique_tool_id("raw")
    second = r.unique_tool_id("raw")
    assert first != second


def test_map_tool_result_id_matches_paired_tool_call():
    r = ToolIdRemapper(_normalize)
    mapped = r.unique_tool_id("raw")
    assert r.map_tool_result_id("raw") == mapped
    assert r.map_tool_result_id("raw") == mapped


def test_map_tool_result_id_without_pending_falls_back_to_normalize():
    r = ToolIdRemapper(_normalize)
    assert r.map_tool_result_id("call.1") == "call_1"


def test_empty_raw_id_uses_short_id():
    r = ToolIdRemapper(_normalize)
    generated = r.unique_tool_id(None)
    assert len(generated) == 9


def test_non_string_tool_result_id_passed_through():
    r = ToolIdRemapper(_normalize)
    assert r.map_tool_result_id(42) == 42
