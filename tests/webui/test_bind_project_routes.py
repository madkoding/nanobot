"""Tests for the WebUI session project binding routes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from loguru import logger
from websockets.datastructures import Headers
from websockets.http11 import Request as WsRequest

from nanobot.channels.websocket.runtime import WebSocketConfig
from nanobot.channels.websocket.tests.ws_test_client import _HttpConnection
from nanobot.session.manager import SessionManager
from nanobot.webui.gateway_tokens import GatewayTokenStore
from nanobot.webui.ingress_policy import DEFAULT_WEBUI_INGRESS_POLICY
from nanobot.webui.media_gateway import WebUIMediaGateway
from nanobot.webui.projects import WebUIProjectsController
from nanobot.webui.workspaces import WebUIWorkspaceController
from nanobot.webui.ws_http import GatewayHTTPHandler

pytestmark = pytest.mark.asyncio


@pytest.fixture
def runtime_dir(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    data = tmp_path / "data"
    workspace.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    return workspace, data


def _build_handler(
    workspace: Path,
    data: Path,
) -> tuple[GatewayHTTPHandler, SessionManager, WebUIProjectsController]:
    config = WebSocketConfig()
    sessions = SessionManager(workspace=workspace)
    tokens = GatewayTokenStore()
    media = WebUIMediaGateway(workspace_path=workspace, logger=logger)
    ingress = DEFAULT_WEBUI_INGRESS_POLICY
    workspaces = WebUIWorkspaceController(
        session_manager=sessions,
        default_workspace=workspace,
        default_restrict_to_workspace=True,
    )
    projects = WebUIProjectsController(data_dir=data)
    handler = GatewayHTTPHandler(
        config=config,
        session_manager=sessions,
        static_dist_path=None,
        runtime_model_name=None,
        runtime_surface="browser",
        runtime_capabilities_overrides=None,
        bus=None,
        tokens=tokens,
        media=media,
        ingress=ingress,
        workspaces=workspaces,
        projects=projects,
        skills_workspace_path=workspace,
        disabled_skills=set(),
    )
    return handler, sessions, projects


def _make_request(path: str, headers: dict[str, str]) -> WsRequest:
    return WsRequest(path, Headers(list(headers.items())))


def _issue_token(handler: GatewayHTTPHandler) -> str:
    return handler.tokens.issue_api_token(handler.config.token_ttl_s)


async def _get(handler: GatewayHTTPHandler, token: str, path: str) -> tuple[int, str]:
    request = _make_request(
        path,
        {
            "Authorization": f"Bearer {token}",
            "Host": "localhost:8765",
        },
    )
    response = await handler.dispatch(_HttpConnection(), request)
    body = response.body.decode("utf-8") if response is not None and response.body else ""
    return response.status_code if response is not None else 0, body


async def _make_chat(sessions: SessionManager, chat_id: str = "chat-1") -> str:
    key = f"websocket:{chat_id}"
    sessions.save(sessions.get_or_create(key))
    return key


async def test_get_project_returns_null_when_unbound(runtime_dir: tuple[Path, Path]) -> None:
    workspace, data = runtime_dir
    handler, sessions, _ = _build_handler(workspace, data)
    key = await _make_chat(sessions)
    token = _issue_token(handler)
    status, body = await _get(handler, token, f"/api/sessions/{key}/project")
    assert status == 200
    assert json.loads(body) == {"session_key": key, "project_id": None}


async def test_bind_then_get_round_trip(runtime_dir: tuple[Path, Path]) -> None:
    workspace, data = runtime_dir
    handler, sessions, projects = _build_handler(workspace, data)
    key = await _make_chat(sessions)
    token = _issue_token(handler)
    summary = projects.create_project("alpha", "do the thing")
    status, body = await _get(
        handler, token, f"/api/sessions/{key}/project/bind?project_id={summary.id}"
    )
    assert status == 200
    assert json.loads(body) == {"session_key": key, "project_id": summary.id}
    status, body = await _get(handler, token, f"/api/sessions/{key}/project")
    assert status == 200
    assert json.loads(body)["project_id"] == summary.id


async def test_bind_unknown_project_returns_404(runtime_dir: tuple[Path, Path]) -> None:
    workspace, data = runtime_dir
    handler, sessions, _ = _build_handler(workspace, data)
    key = await _make_chat(sessions)
    token = _issue_token(handler)
    status, body = await _get(
        handler, token, f"/api/sessions/{key}/project/bind?project_id=missing"
    )
    assert status == 404
    assert "missing" in body


async def test_bind_missing_project_id_returns_400(runtime_dir: tuple[Path, Path]) -> None:
    workspace, data = runtime_dir
    handler, sessions, _ = _build_handler(workspace, data)
    key = await _make_chat(sessions)
    token = _issue_token(handler)
    status, _ = await _get(handler, token, f"/api/sessions/{key}/project/bind")
    assert status == 400


async def test_unbind_clears_binding(runtime_dir: tuple[Path, Path]) -> None:
    workspace, data = runtime_dir
    handler, sessions, projects = _build_handler(workspace, data)
    key = await _make_chat(sessions)
    token = _issue_token(handler)
    summary = projects.create_project("alpha", "")
    await _get(handler, token, f"/api/sessions/{key}/project/bind?project_id={summary.id}")
    status, body = await _get(handler, token, f"/api/sessions/{key}/project/unbind")
    assert status == 200
    assert json.loads(body) == {"session_key": key, "project_id": None}


async def test_unauthorized_without_token(runtime_dir: tuple[Path, Path]) -> None:
    workspace, data = runtime_dir
    handler, sessions, _ = _build_handler(workspace, data)
    key = await _make_chat(sessions)
    request = _make_request(
        f"/api/sessions/{key}/project",
        {"Host": "localhost:8765"},
    )
    response = await handler.dispatch(_HttpConnection(), request)
    assert response is not None
    assert response.status_code == 401
