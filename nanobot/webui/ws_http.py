"""HTTP API handler extracted from WebSocketChannel.

Handles all non-WebSocket HTTP routes: bootstrap, sessions, settings,
media, commands, sidebar state, static file serving, and token management.

Also houses shared HTTP utility functions used by both this module and
``websocket.py`` to avoid circular imports.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote

from loguru import logger
from websockets.http11 import Request as WsRequest
from websockets.http11 import Response

from nanobot.command.builtin import builtin_command_palette
from nanobot.cron.session_turns import is_bound_cron_job
from nanobot.cron.types import CronJob, CronSchedule
from nanobot.runtime_context import public_history_messages
from nanobot.security.workspace_access import WorkspaceScope
from nanobot.triggers.local_types import LocalTrigger
from nanobot.utils.subagent_channel_display import scrub_subagent_messages_for_channel
from nanobot.webui.agenda_api import (
    create_appointment as agenda_create_appointment,
)
from nanobot.webui.agenda_api import (
    delete_appointment as agenda_delete_appointment,
)
from nanobot.webui.agenda_api import (
    fetch_appointment as agenda_fetch_appointment,
)
from nanobot.webui.agenda_api import (
    list_appointments as agenda_list_appointments,
)
from nanobot.webui.agenda_api import (
    update_appointment as agenda_update_appointment,
)
from nanobot.webui.clawhub_api import (
    ClawhubError,
    clawhub_browse,
    clawhub_install,
    clawhub_search,
    clawhub_trending,
    clawhub_update_all,
)
from nanobot.webui.file_preview import (
    WebUIFilePreviewError,
    file_download_bytes,
    file_preview_availability_payload,
    file_preview_payload,
)
from nanobot.webui.gateway_tokens import GatewayTokenStore, token_response_payload
from nanobot.webui.http_utils import (
    case_insensitive_header as _case_insensitive_header,
)
from nanobot.webui.http_utils import (
    host_for_url as _host_for_url,
)
from nanobot.webui.http_utils import (
    http_error as _http_error,
)
from nanobot.webui.http_utils import (
    http_json_response as _http_json_response,
)
from nanobot.webui.http_utils import (
    http_response as _http_response,
)
from nanobot.webui.http_utils import (
    is_local_browser_request as _is_local_browser_request,
)
from nanobot.webui.http_utils import (
    is_localhost as _is_localhost,
)
from nanobot.webui.http_utils import (
    issue_route_secret_matches as _issue_route_secret_matches,
)
from nanobot.webui.http_utils import (
    normalize_config_path as _normalize_config_path,
)
from nanobot.webui.http_utils import (
    parse_query as _parse_query,
)
from nanobot.webui.http_utils import (
    parse_request_path as _parse_request_path,
)
from nanobot.webui.http_utils import (
    query_first as _query_first,
)
from nanobot.webui.http_utils import (
    read_json_request_header as read_json_request_header,
)
from nanobot.webui.http_utils import (
    safe_host_header as _safe_host_header,
)
from nanobot.webui.ingress_policy import WebUIIngressPolicy
from nanobot.webui.media_gateway import WebUIMediaGateway
from nanobot.webui.projects import (
    ProjectError,
    WebUIProjectsController,
    board_payload,
    project_detail_payload,
    project_file_payload,
    projects_list_payload,
)
from nanobot.webui.research_api import share_research_article
from nanobot.webui.rlaif_api import (
    approve_proposal as _rlaif_approve_proposal,
)
from nanobot.webui.rlaif_api import (
    get_proposal as _rlaif_get_proposal,
)
from nanobot.webui.rlaif_api import (
    list_proposals as _rlaif_list_proposals,
)
from nanobot.webui.rlaif_api import (
    read_log as _rlaif_read_log,
)
from nanobot.webui.rlaif_api import (
    read_preferences as _rlaif_read_preferences,
)
from nanobot.webui.rlaif_api import (
    reject_proposal as _rlaif_reject_proposal,
)
from nanobot.webui.session_automations import (
    all_automations_payload,
    serialize_automation_jobs,
    session_automation_jobs,
    session_automations_payload,
)
from nanobot.webui.session_list_index import list_webui_sessions
from nanobot.webui.session_meta import (
    chat_project_id_from_metadata,
    chat_todo_list_from_metadata,
    set_chat_project_id,
    set_chat_todo_list,
)
from nanobot.webui.sidebar_state import (
    read_webui_sidebar_state,
    write_webui_sidebar_state,
)
from nanobot.webui.skills_api import webui_skill_detail_payload, webui_skills_payload
from nanobot.webui.thread_disk import delete_webui_thread
from nanobot.webui.todos_api import (
    create_item as todo_create_item,
)
from nanobot.webui.todos_api import (
    create_todo_list as todo_create_todo_list,
)
from nanobot.webui.todos_api import (
    delete_item as todo_delete_item,
)
from nanobot.webui.todos_api import (
    delete_todo_list as todo_delete_todo_list,
)
from nanobot.webui.todos_api import (
    fetch_todo_list as todo_fetch_todo_list,
)
from nanobot.webui.todos_api import (
    fetch_users as todo_fetch_users,
)
from nanobot.webui.todos_api import (
    list_todo_lists as todo_list_todo_lists,
)
from nanobot.webui.todos_api import (
    migrate_legacy as todo_migrate_legacy,
)
from nanobot.webui.todos_api import (
    update_item as todo_update_item,
)
from nanobot.webui.todos_api import (
    update_users as todo_update_users,
)
from nanobot.webui.transcript import build_webui_thread_response
from nanobot.webui.workspace_browser_api import (
    workspace_copy,
    workspace_create_directory,
    workspace_delete,
    workspace_file_bytes,
    workspace_list_files,
    workspace_move,
    workspace_read_file,
    workspace_rename,
    workspace_write_file,
)
from nanobot.webui.workspaces import WebUIWorkspaceController

# ponytail: module-level store for in-flight RLAIF approve jobs. The
# state persists across HTTP requests and across WebUI page reloads,
# so the frontend can resume a progress view after a refresh.
_RLAIF_APPROVE_JOBS: dict[str, dict] = {}

_SLOW_WEBUI_HTTP_LOG_MS = 1_000
_AUTOMATION_VALUES_HEADER = "X-Nanobot-Automation-Values"
_PROJECT_DATA_HEADER = "X-Nanobot-Project-Data"
_PROJECT_FILE_HEADER = "X-Nanobot-Project-File"
_PROJECT_DATA_HEADER_MAX_BYTES = 256 * 1024
_PROJECT_FILE_HEADER_MAX_BYTES = 16 * 1024 * 1024
_WORKSPACE_BROWSER_DATA_HEADER = "X-Nanobot-Workspace-Browser-Data"
_WORKSPACE_BROWSER_DATA_MAX_BYTES = 512 * 1024

if TYPE_CHECKING:
    from nanobot.bus.queue import MessageBus
    from nanobot.cron.service import CronService
    from nanobot.session.manager import SessionManager
    from nanobot.triggers.local_store import LocalTriggerStore


def _decode_api_key(raw_key: str) -> str | None:
    key = unquote(raw_key)
    _api_key_re = re.compile(r"^[A-Za-z0-9_:.-]{1,128}$")
    if _api_key_re.match(key) is None:
        return None
    return key


def _default_model_name_from_config() -> str | None:
    try:
        from nanobot.config.loader import load_config
        model = load_config().resolve_preset().model.strip()
        return model or None
    except Exception as e:
        logger.debug("bootstrap model_name could not load from config: {}", e)
        return None


def _resolve_bootstrap_model_name(
    runtime_name: Callable[[], str | None] | None,
) -> str:
    if runtime_name is not None:
        try:
            raw = runtime_name()
        except Exception as e:
            logger.debug("bootstrap runtime model resolver failed: {}", e)
        else:
            if isinstance(raw, str):
                stripped = raw.strip()
                if stripped:
                    return stripped
    return _default_model_name_from_config() or ""


# ---------------------------------------------------------------------------
# GatewayHTTPHandler
# ---------------------------------------------------------------------------


class GatewayHTTPHandler:
    """Handles all HTTP routes served alongside the WebSocket endpoint.

    Routes HTTP requests and delegates stateful work to explicit gateway
    services owned by the composition layer.
    """

    def __init__(
        self,
        *,
        config: Any,  # WebSocketConfig
        session_manager: SessionManager | None,
        static_dist_path: Path | None,
        runtime_model_name: Callable[[], str | None] | None,
        runtime_surface: str,
        runtime_capabilities_overrides: dict[str, Any] | None,
        bus: MessageBus,
        tokens: GatewayTokenStore,
        media: WebUIMediaGateway,
        ingress: WebUIIngressPolicy,
        workspaces: WebUIWorkspaceController,
        projects: WebUIProjectsController,
        skills_workspace_path: Path,
        disabled_skills: set[str] | None = None,
        cron_service: CronService | None = None,
        local_trigger_store: LocalTriggerStore | None = None,
        cron_pending_job_ids: Callable[[str], set[str]] | None = None,
        local_trigger_pending_ids: Callable[[str], set[str]] | None = None,
        channel_feature_action: Callable[..., Any] | None = None,
        channel_runtime_status: Callable[[], dict[str, Any]] | None = None,
        subagent_manager: Any | None = None,
        runtime_resolver: Callable[[str | None], Any] | None = None,
        log: Any = logger,
    ) -> None:
        self.config = config
        self.session_manager = session_manager
        self.static_dist_path = static_dist_path
        self.runtime_model_name = runtime_model_name
        self.bus = bus
        self.tokens = tokens
        self.media = media
        self.ingress = ingress
        self.workspaces = workspaces
        self.projects = projects
        self.skills_workspace_path = skills_workspace_path
        self.disabled_skills = disabled_skills or set()
        self.cron_service = cron_service
        self.local_trigger_store = local_trigger_store
        self.cron_pending_job_ids = cron_pending_job_ids
        self.local_trigger_pending_ids = local_trigger_pending_ids
        self.subagent_manager = subagent_manager
        self.runtime_resolver = runtime_resolver
        self._log = log
        self._runtime_surface = runtime_surface

        from nanobot.webui.settings_api import runtime_capabilities as _rc
        from nanobot.webui.settings_routes import WebUISettingsRouter

        self._capabilities = _rc(runtime_surface, runtime_capabilities_overrides or {})
        self.settings_routes = WebUISettingsRouter(
            bus=bus,
            logger=self._log,
            check_api_token=self.check_api_token,
            parse_query=_parse_query,
            json_response=_http_json_response,
            error_response=_http_error,
            runtime_surface=runtime_surface,
            runtime_capabilities=self._capabilities,
            channel_feature_action=channel_feature_action,
            channel_runtime_status=channel_runtime_status,
        )

    def workspace_controls_available(self, connection: Any) -> bool:
        return self._runtime_surface == "native" or _is_localhost(connection)

    # -- Token management ---------------------------------------------------

    def check_api_token(self, request: WsRequest) -> bool:
        return self.tokens.check_api_token(request)

    # -- Main dispatch ------------------------------------------------------

    async def dispatch(self, connection: Any, request: WsRequest) -> Any | None:
        """Route an HTTP request. Returns Response or None."""
        got, _ = _parse_request_path(request.path)
        started = time.perf_counter()
        response: Any | None = None

        try:
            response = await self._dispatch_resolved(connection, request, got)
            return response
        finally:
            self._log_slow_http(got, response, started)

    async def _dispatch_resolved(
        self,
        connection: Any,
        request: WsRequest,
        got: str,
    ) -> Any | None:
        # Token issue endpoint
        if self.config.token_issue_path:
            issue_expected = _normalize_config_path(self.config.token_issue_path)
            if got == issue_expected:
                return self._handle_token_issue(connection, request)

        # Bootstrap
        if got == "/webui/bootstrap":
            return self._handle_bootstrap(connection, request)

        # Settings routes (delegated)
        response = await self.settings_routes.dispatch(connection, request, got)
        if response is not None:
            return response

        # Project routes
        response = self._dispatch_project_routes(request, got)
        if response is not None:
            return response

        # Workspace browser routes
        response = self._dispatch_workspace_browser_routes(request, got)
        if response is not None:
            return response

        # Agenda and Todos routes
        response = self._dispatch_agenda_routes(request, got)
        if response is not None:
            return response
        response = self._dispatch_todos_routes(request, got)
        if response is not None:
            return response

        # Research routes
        response = self._dispatch_research_routes(request, got)
        if response is not None:
            return response

        # RLAIF watch routes (preferences + filtered gateway log)
        response = self._dispatch_rlaif_routes(request, got)
        if response is not None:
            return response

        # Session routes
        response = await self._dispatch_session_routes(request, got)
        if response is not None:
            return response

        # Media routes
        response = self._dispatch_media_routes(request, got)
        if response is not None:
            return response

        # Automation routes
        response = await self._dispatch_automation_routes(request, got)
        if response is not None:
            return response

        # Misc routes
        response = await self._dispatch_misc_routes(connection, request, got)
        if response is not None:
            return response

        # API 404 (never serve SPA for /api/ routes)
        if got.startswith("/api/"):
            return _http_error(404, "API route not found")

        # Static SPA serving
        if self.static_dist_path is not None:
            response = self._serve_static(got)
            if response is not None:
                return response

        return connection.respond(404, "Not Found")

    def _log_slow_http(self, path: str, response: Any | None, started: float) -> None:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if elapsed_ms < _SLOW_WEBUI_HTTP_LOG_MS:
            return
        if not (path.startswith("/api/") or path == "/webui/bootstrap"):
            return
        status = getattr(response, "status_code", None)
        self._log.warning(
            "slow webui http route path={} status={} duration_ms={}",
            path,
            status if status is not None else "none",
            elapsed_ms,
        )

    # -- Token issue --------------------------------------------------------

    def _handle_token_issue(self, connection: Any, request: Any) -> Any:
        secret = self.config.token_issue_secret.strip() or self.config.token.strip()
        if secret:
            if not _issue_route_secret_matches(request.headers, secret):
                return connection.respond(401, "Unauthorized")
        else:
            self._log.warning(
                "token_issue_path is set but token_issue_secret is empty; "
                "any client can obtain connection tokens — set token_issue_secret for production."
            )
        if not self.tokens.can_issue():
            self._log.error(
                "too many outstanding issued tokens ({}), rejecting issuance",
                len(self.tokens.issued_tokens),
            )
            return _http_json_response({"error": "too many outstanding tokens"}, status=429)
        token_value = self.tokens.issue_token(self.config.token_ttl_s)
        return _http_json_response(token_response_payload(token_value, self.config.token_ttl_s))

    # -- Bootstrap ----------------------------------------------------------

    def _handle_bootstrap(self, connection: Any, request: Any) -> Response:
        secret = self.config.token_issue_secret.strip() or self.config.token.strip()
        is_local_browser = _is_local_browser_request(connection, request.headers)
        if secret:
            if not _issue_route_secret_matches(request.headers, secret):
                return _http_error(401, "Unauthorized")
        elif not is_local_browser:
            return _http_error(403, "bootstrap is localhost-only")

        api_token_allowed = bool(secret) or is_local_browser
        if not self.tokens.can_issue(include_api_token=api_token_allowed):
            return _http_response(
                json.dumps({"error": "too many outstanding tokens"}).encode("utf-8"),
                status=429,
                content_type="application/json; charset=utf-8",
            )
        token = self.tokens.issue_token(self.config.token_ttl_s, audience="webui")
        api_token = (
            self.tokens.issue_api_token(self.config.token_ttl_s)
            if api_token_allowed
            else None
        )

        ws_url = self._bootstrap_ws_url(request)
        expected_path = _normalize_config_path(self.config.path)
        payload = {
            "token": token,
            "ws_path": expected_path,
            "ws_url": ws_url,
            "expires_in": self.config.token_ttl_s,
            "limits": self.ingress.bootstrap_limits(
                max_frame_bytes=self.config.max_message_bytes,
            ),
            "model_name": _resolve_bootstrap_model_name(self.runtime_model_name),
            "runtime_surface": self._runtime_surface,
            "runtime_capabilities": self._capabilities,
        }
        if api_token is not None:
            payload["api_token"] = api_token
        return _http_json_response(payload)

    def _bootstrap_ws_url(self, request: Any) -> str:
        headers = getattr(request, "headers", {}) or {}
        host = _safe_host_header(_case_insensitive_header(headers, "Host"))
        if not host:
            host = _host_for_url(self.config.host, self.config.port)
        proto = _case_insensitive_header(headers, "X-Forwarded-Proto")
        proto = proto.split(",", 1)[0].strip().lower()
        secure = proto in {"https", "wss"} or bool(self.config.ssl_certfile.strip())
        scheme = "wss" if secure else "ws"
        expected_path = _normalize_config_path(self.config.path)
        return f"{scheme}://{host}{expected_path}"

    # -- Session routes -----------------------------------------------------

    async def _dispatch_session_routes(self, request: WsRequest, got: str) -> Response | None:
        m = re.match(r"^/api/sessions/([^/]+)/messages$", got)
        if m:
            return self._handle_session_messages(request, m.group(1))

        m = re.match(r"^/api/sessions/([^/]+)/webui-thread$", got)
        if m:
            return self._handle_webui_thread_get(request, m.group(1))

        m = re.match(r"^/api/sessions/([^/]+)/file-preview$", got)
        if m:
            return self._handle_file_preview(request, m.group(1))

        m = re.match(r"^/api/sessions/([^/]+)/file-download$", got)
        if m:
            return self._handle_file_download(request, m.group(1))

        m = re.match(r"^/api/sessions/([^/]+)/subagents/([^/]+)$", got)
        if m:
            return self._handle_session_subagent(request, m.group(1), m.group(2))

        m = re.match(r"^/api/sessions/([^/]+)/automations$", got)
        if m:
            return self._handle_session_automations(request, m.group(1))

        m = re.match(r"^/api/sessions/([^/]+)/delete$", got)
        if m:
            return await self._handle_session_delete(request, m.group(1))

        m = re.match(r"^/api/sessions/([^/]+)/project$", got)
        if m:
            return self._handle_session_project_get(request, m.group(1))

        m = re.match(r"^/api/sessions/([^/]+)/project/bind$", got)
        if m:
            return self._handle_session_project_bind(request, m.group(1))

        m = re.match(r"^/api/sessions/([^/]+)/project/unbind$", got)
        if m:
            return self._handle_session_project_unbind(request, m.group(1))

        m = re.match(r"^/api/sessions/([^/]+)/todo/bind$", got)
        if m:
            return self._handle_session_todo_bind(request, m.group(1))

        m = re.match(r"^/api/sessions/([^/]+)/todo/unbind$", got)
        if m:
            return self._handle_session_todo_unbind(request, m.group(1))

        return None

    # -- Project routes -----------------------------------------------------

    def _dispatch_project_routes(self, request: WsRequest, got: str) -> Response | None:
        if got == "/api/projects":
            return self._handle_projects_list(request)
        if got == "/api/projects/create":
            return self._handle_projects_create(request)
        m = re.match(r"^/api/projects/([^/]+)$", got)
        if m:
            return self._handle_projects_detail(request, m.group(1))
        m = re.match(r"^/api/projects/([^/]+)/update$", got)
        if m:
            return self._handle_projects_update(request, m.group(1))
        m = re.match(r"^/api/projects/([^/]+)/delete$", got)
        if m:
            return self._handle_projects_delete(request, m.group(1))
        m = re.match(r"^/api/projects/([^/]+)/files$", got)
        if m:
            return self._handle_projects_files_list(request, m.group(1))
        if got.startswith("/api/projects/") and "/files/upload" in got:
            m = re.match(r"^/api/projects/([^/]+)/files/upload$", got)
            if m:
                return self._handle_projects_files_upload(request, m.group(1))
        m = re.match(r"^/api/projects/([^/]+)/files/([^/]+)$", got)
        if m:
            return self._handle_projects_file_get(request, m.group(1), m.group(2))
        m = re.match(r"^/api/projects/([^/]+)/files/([^/]+)/delete$", got)
        if m:
            return self._handle_projects_file_delete(request, m.group(1), m.group(2))
        m = re.match(r"^/api/projects/([^/]+)/folders/add$", got)
        if m:
            return self._handle_projects_folder_add(request, m.group(1))
        m = re.match(r"^/api/projects/([^/]+)/folders/remove$", got)
        if m:
            return self._handle_projects_folder_remove(request, m.group(1))
        m = re.match(r"^/api/projects/([^/]+)/board$", got)
        if m:
            return self._handle_board_get(request, m.group(1))
        m = re.match(r"^/api/projects/([^/]+)/board/setup$", got)
        if m:
            return self._handle_board_setup(request, m.group(1))
        m = re.match(r"^/api/projects/([^/]+)/board/columns/add$", got)
        if m:
            return self._handle_board_column_add(request, m.group(1))
        m = re.match(r"^/api/projects/([^/]+)/board/columns/([^/]+)/remove$", got)
        if m:
            return self._handle_board_column_remove(request, m.group(1), m.group(2))
        m = re.match(r"^/api/projects/([^/]+)/board/columns/([^/]+)/rename$", got)
        if m:
            return self._handle_board_column_rename(request, m.group(1), m.group(2))
        m = re.match(r"^/api/projects/([^/]+)/board/cards/add$", got)
        if m:
            return self._handle_board_card_add(request, m.group(1))
        m = re.match(r"^/api/projects/([^/]+)/board/cards/([^/]+)/move$", got)
        if m:
            return self._handle_board_card_move(request, m.group(1), m.group(2))
        m = re.match(r"^/api/projects/([^/]+)/board/cards/([^/]+)/chat$", got)
        if m:
            return self._handle_board_card_chat(request, m.group(1), m.group(2))
        m = re.match(r"^/api/projects/([^/]+)/board/cards/([^/]+)/delete$", got)
        if m:
            return self._handle_board_card_delete(request, m.group(1), m.group(2))
        m = re.match(r"^/api/projects/([^/]+)/board/cards/([^/]+)/merge$", got)
        if m:
            return self._handle_board_card_merge(request, m.group(1), m.group(2))
        m = re.match(r"^/api/projects/([^/]+)/board/cards/([^/]+)/spawn$", got)
        if m:
            return self._handle_board_card_spawn(request, m.group(1), m.group(2))
        m = re.match(r"^/api/projects/([^/]+)/board/cards/([^/]+)/plan$", got)
        if m:
            return self._handle_board_card_plan(request, m.group(1), m.group(2))
        m = re.match(r"^/api/projects/([^/]+)/board/cards/([^/]+)/build$", got)
        if m:
            return self._handle_board_card_build(request, m.group(1), m.group(2))
        m = re.match(r"^/api/projects/([^/]+)/board/cards/([^/]+)/validate$", got)
        if m:
            return self._handle_board_card_validate(request, m.group(1), m.group(2))
        m = re.match(r"^/api/projects/([^/]+)/board/cards/([^/]+)/subagent$", got)
        if m:
            return self._handle_board_card_subagent(request, m.group(1), m.group(2))
        return None

    def _handle_projects_list(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        try:
            payload = projects_list_payload(self.projects)
        except ProjectError as exc:
            return _http_error(500, str(exc))
        return _http_json_response(payload)

    def _handle_projects_create(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(
            request, _PROJECT_DATA_HEADER, _PROJECT_DATA_HEADER_MAX_BYTES
        )
        if err is not None:
            return err
        name = (body.get("name") or "").strip() if isinstance(body, dict) else ""
        instructions = (
            (body.get("instructions_md") or "") if isinstance(body, dict) else ""
        )
        try:
            summary = self.projects.create_project(name, instructions)
        except ProjectError as exc:
            return _http_error(400, str(exc))
        return _http_json_response(project_detail_payload(self.projects, summary.id))

    def _handle_projects_detail(
        self, request: WsRequest, project_id: str
    ) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        try:
            payload = project_detail_payload(self.projects, project_id)
        except ProjectError as exc:
            return _http_error(404, str(exc))
        return _http_json_response(payload)

    def _handle_projects_update(
        self, request: WsRequest, project_id: str
    ) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(
            request, _PROJECT_DATA_HEADER, _PROJECT_DATA_HEADER_MAX_BYTES
        )
        if err is not None:
            return err
        name = (body.get("name") or "").strip() if isinstance(body, dict) else ""
        instructions = (
            (body.get("instructions_md") or "") if isinstance(body, dict) else ""
        )
        try:
            summary = self.projects.update_project(project_id, name, instructions)
        except ProjectError as exc:
            return _http_error(404, str(exc))
        return _http_json_response(project_detail_payload(self.projects, summary.id))

    def _handle_projects_delete(
        self, request: WsRequest, project_id: str
    ) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        try:
            self.projects.delete_project(project_id)
        except ProjectError as exc:
            return _http_error(404, str(exc))
        return _http_json_response({"ok": True, "id": project_id})

    def _handle_projects_files_list(
        self, request: WsRequest, project_id: str
    ) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        try:
            files = self.projects.list_files(project_id)
        except ProjectError as exc:
            return _http_error(404, str(exc))
        return _http_json_response(
            {
                "files": [
                    {
                        "id": f.id,
                        "project_id": f.project_id,
                        "name": f.name,
                        "mime_type": f.mime_type,
                        "size": f.size,
                        "created_at_ms": f.created_at_ms,
                    }
                    for f in files
                ]
            }
        )

    def _handle_projects_files_upload(
        self, request: WsRequest, project_id: str
    ) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(
            request, _PROJECT_FILE_HEADER, _PROJECT_FILE_HEADER_MAX_BYTES
        )
        if err is not None:
            return err
        name = (body.get("name") or "").strip() if isinstance(body, dict) else ""
        data_url = body.get("data_url") if isinstance(body, dict) else None
        if not isinstance(data_url, str):
            return _http_error(400, "file data_url is required")
        try:
            f = self.projects.add_file(project_id, name, data_url)
        except ProjectError as exc:
            return _http_error(400, str(exc))
        return _http_json_response(
            {
                "id": f.id,
                "project_id": f.project_id,
                "name": f.name,
                "mime_type": f.mime_type,
                "size": f.size,
                "created_at_ms": f.created_at_ms,
            }
        )

    def _handle_projects_file_get(
        self, request: WsRequest, project_id: str, file_id: str
    ) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        try:
            payload = project_file_payload(self.projects, project_id, file_id)
        except ProjectError as exc:
            return _http_error(404, str(exc))
        return _http_json_response(payload)

    def _handle_projects_file_delete(
        self, request: WsRequest, project_id: str, file_id: str
    ) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        try:
            self.projects.delete_file(project_id, file_id)
        except ProjectError as exc:
            return _http_error(404, str(exc))
        return _http_json_response({"ok": True, "id": file_id})

    def _handle_projects_folder_add(
        self, request: WsRequest, project_id: str
    ) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(
            request, _PROJECT_DATA_HEADER, _PROJECT_DATA_HEADER_MAX_BYTES
        )
        if err is not None:
            return err
        path = (body.get("path") or "") if isinstance(body, dict) else ""
        try:
            folder = self.projects.add_folder(project_id, path)
        except ProjectError as exc:
            return _http_error(400, str(exc))
        return _http_json_response(
            {"path": folder.path, "created_at_ms": folder.created_at_ms}
        )

    def _handle_projects_folder_remove(
        self, request: WsRequest, project_id: str
    ) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(
            request, _PROJECT_DATA_HEADER, _PROJECT_DATA_HEADER_MAX_BYTES
        )
        if err is not None:
            return err
        path = (body.get("path") or "") if isinstance(body, dict) else ""
        try:
            self.projects.remove_folder(project_id, path)
        except ProjectError as exc:
            return _http_error(404, str(exc))
        return _http_json_response({"ok": True, "path": path})

    # -- Board (kanban of worktrees) routes --------------------------------

    def _handle_board_get(self, request: WsRequest, project_id: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        try:
            payload = board_payload(self.projects, project_id)
        except ProjectError as exc:
            return _http_error(404, str(exc))
        return _http_json_response(payload)

    def _handle_board_setup(self, request: WsRequest, project_id: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(
            request, _PROJECT_DATA_HEADER, _PROJECT_DATA_HEADER_MAX_BYTES
        )
        if err is not None:
            return err
        repo_path = (body.get("repo_path") or "") if isinstance(body, dict) else ""
        try:
            self.projects.setup_board(project_id, repo_path)
        except ProjectError as exc:
            return _http_error(400, str(exc))
        return _http_json_response(board_payload(self.projects, project_id))

    def _handle_board_column_add(self, request: WsRequest, project_id: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(
            request, _PROJECT_DATA_HEADER, _PROJECT_DATA_HEADER_MAX_BYTES
        )
        if err is not None:
            return err
        name = (body.get("name") or "") if isinstance(body, dict) else ""
        try:
            col = self.projects.add_column(project_id, name)
        except ProjectError as exc:
            return _http_error(400, str(exc))
        return _http_json_response(col)

    def _handle_board_column_remove(
        self, request: WsRequest, project_id: str, column_id: str
    ) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        try:
            self.projects.remove_column(project_id, column_id)
        except ProjectError as exc:
            return _http_error(404, str(exc))
        return _http_json_response({"ok": True, "id": column_id})

    def _handle_board_column_rename(
        self, request: WsRequest, project_id: str, column_id: str
    ) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(
            request, _PROJECT_DATA_HEADER, _PROJECT_DATA_HEADER_MAX_BYTES
        )
        if err is not None:
            return err
        name = (body.get("name") or "") if isinstance(body, dict) else ""
        try:
            col = self.projects.rename_column(project_id, column_id, name)
        except ProjectError as exc:
            return _http_error(404, str(exc))
        return _http_json_response(col)

    def _handle_board_card_add(self, request: WsRequest, project_id: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(
            request, _PROJECT_DATA_HEADER, _PROJECT_DATA_HEADER_MAX_BYTES
        )
        if err is not None:
            return err
        title = (body.get("title") or "") if isinstance(body, dict) else ""
        brief = (body.get("brief") or "") if isinstance(body, dict) else ""
        column_id = (body.get("column_id") or "") if isinstance(body, dict) else ""
        try:
            card = self.projects.create_card(project_id, brief, column_id, title=title)
        except ProjectError as exc:
            return _http_error(400, str(exc))
        return _http_json_response(card)

    def _handle_board_card_move(
        self, request: WsRequest, project_id: str, card_id: str
    ) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(
            request, _PROJECT_DATA_HEADER, _PROJECT_DATA_HEADER_MAX_BYTES
        )
        if err is not None:
            return err
        column_id = (body.get("column_id") or "") if isinstance(body, dict) else ""
        try:
            card = self.projects.move_card(project_id, card_id, column_id)
        except ProjectError as exc:
            return _http_error(404, str(exc))
        return _http_json_response(card)

    def _handle_board_card_chat(
        self, request: WsRequest, project_id: str, card_id: str
    ) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(
            request, _PROJECT_DATA_HEADER, _PROJECT_DATA_HEADER_MAX_BYTES
        )
        if err is not None:
            return err
        session_key = (body.get("session_key") or "") if isinstance(body, dict) else ""
        if not session_key:
            return _http_error(400, "session_key is required")
        try:
            card = self.projects.set_card_chat(project_id, card_id, session_key)
        except ProjectError as exc:
            return _http_error(404, str(exc))
        return _http_json_response(card)

    def _handle_board_card_delete(
        self, request: WsRequest, project_id: str, card_id: str
    ) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        try:
            self.projects.delete_card(project_id, card_id)
        except ProjectError as exc:
            return _http_error(404, str(exc))
        return _http_json_response({"ok": True, "id": card_id})

    def _handle_board_card_merge(
        self, request: WsRequest, project_id: str, card_id: str
    ) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        into = request.query.get("into", "main")
        try:
            output = self.projects.merge_card(project_id, card_id, into)
        except ProjectError as exc:
            return _http_error(400, str(exc))
        return _http_json_response({"ok": True, "output": output})

    def _handle_board_card_spawn(
        self, request: WsRequest, project_id: str, card_id: str
    ) -> Response:
        return self._run_card_phase(request, project_id, card_id, "build")

    def _handle_board_card_plan(
        self, request: WsRequest, project_id: str, card_id: str
    ) -> Response:
        return self._run_card_phase(request, project_id, card_id, "plan")

    def _handle_board_card_build(
        self, request: WsRequest, project_id: str, card_id: str
    ) -> Response:
        return self._run_card_phase(request, project_id, card_id, "build")

    def _handle_board_card_validate(
        self, request: WsRequest, project_id: str, card_id: str
    ) -> Response:
        return self._run_card_phase(request, project_id, card_id, "validate")

    def _run_card_phase(
        self, request: WsRequest, project_id: str, card_id: str, phase: str
    ) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        manager = getattr(self, "subagent_manager", None)
        if manager is None:
            return _http_error(503, "subagent manager unavailable")
        try:
            card = self.projects.run_card_phase(
                project_id,
                card_id,
                phase,
                subagent_manager=manager,
                runtime_resolver=self.runtime_resolver,
            )
        except ProjectError as exc:
            return _http_error(400, str(exc))
        return _http_json_response(card)

    def _handle_board_card_subagent(
        self, request: WsRequest, project_id: str, card_id: str
    ) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        manager = getattr(self, "subagent_manager", None)
        if manager is None:
            return _http_error(503, "subagent manager unavailable")
        try:
            status = self.projects.card_subagent_status(project_id, card_id, manager)
        except ProjectError as exc:
            return _http_error(404, str(exc))
        if status is None:
            return _http_json_response({"status": None})
        return _http_json_response(status)

    # -- Research routes -----------------------------------------------------

    def _dispatch_research_routes(self, request: WsRequest, got: str) -> Response | None:
        if got == "/api/research/share":
            return self._handle_research_share(request)
        return None

    def _handle_research_share(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        query = _parse_query(request.path)
        path = _query_first(query, "path") or ""
        scope = self._workspace_browser_scope(request)
        payload = share_research_article(path, scope)
        if not payload.get("ok"):
            return _http_error(400, payload.get("error") or "Failed to share")
        return _http_json_response(payload)

    # -- RLAIF watch routes --------------------------------------------------

    def _dispatch_rlaif_routes(self, request: WsRequest, got: str) -> Response | None:
        if got == "/api/rlaif/preferences":
            return self._handle_rlaif_preferences(request)
        if got == "/api/rlaif/log":
            return self._handle_rlaif_log(request)
        if got == "/api/rlaif/proposals":
            return self._handle_rlaif_proposals_list(request)
        if got.startswith("/api/rlaif/jobs/"):
            return self._handle_rlaif_job_status(request, got)
        if got.startswith("/api/rlaif/proposals/"):
            return self._handle_rlaif_proposal_action(request, got)
        return None

    def _handle_rlaif_preferences(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        query = _parse_query(request.path)
        try:
            offset = int(_query_first(query, "offset") or 0)
            limit_raw = _query_first(query, "limit")
            limit = int(limit_raw) if limit_raw else None
            since_raw = _query_first(query, "since_index")
            since_index = int(since_raw) if since_raw is not None else None
        except ValueError:
            return _http_error(400, "invalid query parameter")
        payload = _rlaif_read_preferences(
            offset=max(0, offset),
            limit=limit,
            since_index=since_index,
        )
        return _http_json_response(payload)

    def _handle_rlaif_log(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        query = _parse_query(request.path)
        try:
            since_raw = _query_first(query, "since_line")
            since_line = int(since_raw) if since_raw is not None else None
            max_lines = int(_query_first(query, "max_lines") or 200)
        except ValueError:
            return _http_error(400, "invalid query parameter")
        payload = _rlaif_read_log(
            since_line=since_line,
            max_lines=max(1, min(max_lines, 1000)),
        )
        return _http_json_response(payload)

    def _handle_rlaif_job_status(self, request: WsRequest, got: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        job_id = got[len("/api/rlaif/jobs/"):]
        state = _RLAIF_APPROVE_JOBS.get(job_id)
        if state is None:
            return _http_error(404, "job not found")
        return _http_json_response({"job_id": job_id, **state})

    def _handle_rlaif_proposals_list(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        return _http_json_response(_rlaif_list_proposals())

    def _handle_rlaif_proposal_action(self, request: WsRequest, got: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        # /api/rlaif/proposals/<id>/<action>
        rest = got[len("/api/rlaif/proposals/"):]
        parts = rest.split("/", 1)
        if len(parts) != 2:
            return _http_error(400, "expected /api/rlaif/proposals/<id>/<approve|reject>")
        try:
            proposal_id = int(parts[0])
        except ValueError:
            return _http_error(400, "proposal id must be an integer")
        action = parts[1]

        if action == "view":
            prop = _rlaif_get_proposal(proposal_id)
            if prop is None:
                return _http_error(404, "proposal not found")
            return _http_json_response(prop)
        if action == "approve":
            # ponytail: never block the WebSocket event loop on a
            # long-running approve. The work runs in a worker thread,
            # the handler responds immediately with a 202 + a job id,
            # and the frontend polls /api/rlaif/jobs/<job_id> to read
            # the result back. The state lives on a module-level dict
            # so it survives across HTTP requests and across WebUI
            # page reloads.
            import threading
            job_id = f"approve-{proposal_id}-{int(time.time() * 1000)}"
            _RLAIF_APPROVE_JOBS[job_id] = {
                "status": "running",
                "proposal_id": proposal_id,
                "started_at": time.time(),
                "result": None,
                "error": None,
            }

            def _run_approve() -> None:
                try:
                    result = asyncio.run(
                        _rlaif_approve_proposal(proposal_id)
                    )
                    _RLAIF_APPROVE_JOBS[job_id]["result"] = result
                    _RLAIF_APPROVE_JOBS[job_id]["status"] = "done"
                except Exception as exc:  # noqa: BLE001
                    _RLAIF_APPROVE_JOBS[job_id]["error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    _RLAIF_APPROVE_JOBS[job_id]["status"] = "error"
                finally:
                    _RLAIF_APPROVE_JOBS[job_id]["finished_at"] = time.time()

            thread = threading.Thread(
                target=_run_approve,
                name=f"rlaif-approve-{proposal_id}",
                daemon=True,
            )
            thread.start()
            return _http_json_response(
                {
                    "ok": True,
                    "status": "running",
                    "job_id": job_id,
                    "message": "approve started; poll /api/rlaif/jobs/<id>",
                },
                status=202,
            )
        if action == "reject":
            return _http_json_response(
                {"ok": True, "result": _rlaif_reject_proposal(proposal_id)}
            )
        return _http_error(404, f"unknown action {action!r}")

    # -- Workspace browser routes ------------------------------------------

    def _dispatch_workspace_browser_routes(
        self, request: WsRequest, got: str
    ) -> Response | None:
        if got == "/api/workspace-browser/list":
            return self._handle_workspace_browser_list(request)
        if got == "/api/workspace-browser/read":
            return self._handle_workspace_browser_read(request)
        if got == "/api/workspace-browser/write":
            return self._handle_workspace_browser_write(request)
        if got == "/api/workspace-browser/rename":
            return self._handle_workspace_browser_rename(request)
        if got == "/api/workspace-browser/move":
            return self._handle_workspace_browser_move(request)
        if got == "/api/workspace-browser/delete":
            return self._handle_workspace_browser_delete(request)
        if got == "/api/workspace-browser/mkdir":
            return self._handle_workspace_browser_mkdir(request)
        if got == "/api/workspace-browser/copy":
            return self._handle_workspace_browser_copy(request)
        if got == "/api/workspace-browser/raw":
            return self._handle_workspace_browser_raw(request)
        return None

    def _workspace_browser_scope(self, request: WsRequest) -> WorkspaceScope:
        """Resolve the workspace scope for a browser request."""
        query = _parse_query(request.path)
        chat_id = _query_first(query, "chat_id")
        if chat_id:
            scope = self.workspaces.scope_for_session_key(f"websocket:{chat_id}")
        else:
            scope = self.workspaces.default_scope()
        return scope

    def _handle_workspace_browser_list(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        query = _parse_query(request.path)
        subpath = _query_first(query, "path") or ""
        scope = self._workspace_browser_scope(request)
        payload = workspace_list_files(scope=scope, subpath=subpath)
        if payload.get("error"):
            return _http_error(400, payload["error"])
        return _http_json_response(payload)

    def _handle_workspace_browser_read(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        query = _parse_query(request.path)
        path = _query_first(query, "path") or ""
        scope = self._workspace_browser_scope(request)
        payload = workspace_read_file(path, scope=scope)
        if payload.get("error"):
            return _http_error(400, payload["error"])
        return _http_json_response(payload)

    def _handle_workspace_browser_write(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(
            request, _WORKSPACE_BROWSER_DATA_HEADER, _WORKSPACE_BROWSER_DATA_MAX_BYTES
        )
        if err is not None:
            return err
        path = (body.get("path") or "") if isinstance(body, dict) else ""
        content = (body.get("content") or "") if isinstance(body, dict) else ""
        scope = self._workspace_browser_scope(request)
        payload = workspace_write_file(path, content, scope=scope)
        if payload.get("error"):
            return _http_error(400, payload["error"])
        return _http_json_response(payload)

    def _handle_workspace_browser_rename(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(
            request, _WORKSPACE_BROWSER_DATA_HEADER, _WORKSPACE_BROWSER_DATA_MAX_BYTES
        )
        if err is not None:
            return err
        old_path = (body.get("old_path") or "") if isinstance(body, dict) else ""
        new_name = (body.get("new_name") or "") if isinstance(body, dict) else ""
        scope = self._workspace_browser_scope(request)
        payload = workspace_rename(old_path, new_name, scope=scope)
        if payload.get("error"):
            return _http_error(400, payload["error"])
        return _http_json_response(payload)

    def _handle_workspace_browser_move(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(
            request, _WORKSPACE_BROWSER_DATA_HEADER, _WORKSPACE_BROWSER_DATA_MAX_BYTES
        )
        if err is not None:
            return err
        source_path = (body.get("source_path") or "") if isinstance(body, dict) else ""
        dest_path = (body.get("dest_path") or "") if isinstance(body, dict) else ""
        scope = self._workspace_browser_scope(request)
        payload = workspace_move(source_path, dest_path, scope=scope)
        if payload.get("error"):
            return _http_error(400, payload["error"])
        return _http_json_response(payload)

    def _handle_workspace_browser_delete(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(
            request, _WORKSPACE_BROWSER_DATA_HEADER, _WORKSPACE_BROWSER_DATA_MAX_BYTES
        )
        if err is not None:
            return err
        path = (body.get("path") or "") if isinstance(body, dict) else ""
        scope = self._workspace_browser_scope(request)
        payload = workspace_delete(path, scope=scope)
        if payload.get("error"):
            return _http_error(400, payload["error"])
        return _http_json_response(payload)

    def _handle_workspace_browser_mkdir(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(
            request, _WORKSPACE_BROWSER_DATA_HEADER, _WORKSPACE_BROWSER_DATA_MAX_BYTES
        )
        if err is not None:
            return err
        path = (body.get("path") or "") if isinstance(body, dict) else ""
        scope = self._workspace_browser_scope(request)
        payload = workspace_create_directory(path, scope=scope)
        if payload.get("error"):
            return _http_error(400, payload["error"])
        return _http_json_response(payload)

    def _handle_workspace_browser_copy(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(
            request, _WORKSPACE_BROWSER_DATA_HEADER, _WORKSPACE_BROWSER_DATA_MAX_BYTES
        )
        if err is not None:
            return err
        source_path = (body.get("source_path") or "") if isinstance(body, dict) else ""
        dest_path = (body.get("dest_path") or "") if isinstance(body, dict) else ""
        scope = self._workspace_browser_scope(request)
        payload = workspace_copy(source_path, dest_path, scope=scope)
        if payload.get("error"):
            return _http_error(400, payload["error"])
        return _http_json_response(payload)

    def _handle_workspace_browser_raw(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        query = _parse_query(request.path)
        path = _query_first(query, "path") or ""
        scope = self._workspace_browser_scope(request)
        payload = workspace_file_bytes(path, scope=scope)
        if payload.get("error"):
            return _http_error(400, payload["error"])
        return _http_response(
            payload["data"],
            content_type=payload["mime_type"],
            extra_headers=[
                ("Content-Disposition", f"inline; filename*=UTF-8''{quote(payload['name'])}"),
            ],
        )

    async def _handle_sessions_list(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        if self.session_manager is None:
            return _http_error(503, "session manager unavailable")
        payload = await asyncio.to_thread(self._sessions_list_payload)
        return _http_json_response(payload)

    def _sessions_list_payload(self) -> dict[str, Any]:
        assert self.session_manager is not None
        sessions = list_webui_sessions(self.session_manager)
        from nanobot.session.webui_turns import websocket_turn_wall_started_at

        cleaned = []
        for s in sessions:
            key = s.get("key")
            if not (isinstance(key, str) and key.startswith("websocket:")):
                continue
            row = {k: v for k, v in s.items() if k != "path"}
            chat_id = key.split(":", 1)[1]
            started_at = websocket_turn_wall_started_at(chat_id)
            if started_at is not None:
                row["run_started_at"] = started_at
            scope = self.workspaces.scope_for_session_key(key)
            row["workspace_scope"] = scope.payload()
            cleaned.append(row)
        return {"sessions": cleaned}

    def _handle_session_messages(self, request: WsRequest, key: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        if self.session_manager is None:
            return _http_error(503, "session manager unavailable")
        decoded_key = _decode_api_key(key)
        if decoded_key is None:
            return _http_error(400, "invalid session key")
        if not _is_websocket_channel_session_key(decoded_key):
            return _http_error(404, "session not found")
        data = self.session_manager.read_session_file(decoded_key)
        if data is None:
            return _http_error(404, "session not found")
        messages = data.get("messages")
        if isinstance(messages, list):
            scrub_subagent_messages_for_channel(messages)
            data["messages"] = public_history_messages(
                message for message in messages if isinstance(message, dict)
            )
        self.media.augment_media_urls(data)
        return _http_json_response(data)

    def _handle_webui_thread_get(self, request: WsRequest, key: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        decoded_key = _decode_api_key(key)
        if decoded_key is None:
            return _http_error(400, "invalid session key")
        if not _is_websocket_channel_session_key(decoded_key):
            return _http_error(404, "session not found")
        scope = self.workspaces.scope_for_session_key(decoded_key)
        session_messages: list[dict[str, Any]] | None = None
        if self.session_manager is not None:
            session_data = self.session_manager.read_session_file(decoded_key)
            raw_messages = session_data.get("messages") if isinstance(session_data, dict) else None
            if isinstance(raw_messages, list):
                session_messages = [m for m in raw_messages if isinstance(m, dict)]
        query = _parse_query(request.path)
        raw_limit = _query_first(query, "limit")
        limit: int | None = None
        if raw_limit is not None and raw_limit.strip():
            try:
                limit = int(raw_limit)
            except ValueError:
                return _http_error(400, "invalid limit")
        direction = _query_first(query, "direction")
        if direction is not None and direction not in {"latest"}:
            return _http_error(400, "invalid direction")
        before = _query_first(query, "before")
        data = build_webui_thread_response(
            decoded_key,
            augment_user_media=self.media.augment_transcript_media,
            augment_assistant_media=self.media.augment_transcript_media,
            augment_assistant_text=lambda text: self.media.rewrite_local_markdown_images(
                text,
                workspace_path=scope.project_path,
            ),
            session_messages=session_messages,
            limit=limit,
            direction=direction,
            before=before,
        )
        if data is None:
            return _http_error(404, "webui thread not found")
        data["workspace_scope"] = scope.payload()
        return _http_json_response(data)

    def _handle_file_preview(self, request: WsRequest, key: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        decoded_key = _decode_api_key(key)
        if decoded_key is None:
            return _http_error(400, "invalid session key")
        if not _is_websocket_channel_session_key(decoded_key):
            return _http_error(404, "session not found")
        query = _parse_query(request.path)
        path = _query_first(query, "path")
        is_probe = _query_first(query, "probe") == "1"
        try:
            scope = self.workspaces.scope_for_session_key(decoded_key)
            if is_probe:
                payload = file_preview_availability_payload(path, scope=scope)
            else:
                payload = file_preview_payload(path, scope=scope)
        except WebUIFilePreviewError as e:
            if is_probe and e.status in {400, 403, 404, 415}:
                return _http_json_response({"available": False})
            return _http_error(e.status, e.message)
        return _http_json_response(payload)

    def _handle_file_download(self, request: WsRequest, key: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        decoded_key = _decode_api_key(key)
        if decoded_key is None:
            return _http_error(400, "invalid session key")
        if not _is_websocket_channel_session_key(decoded_key):
            return _http_error(404, "session not found")
        query = _parse_query(request.path)
        path = _query_first(query, "path")
        try:
            scope = self.workspaces.scope_for_session_key(decoded_key)
            data, name = file_download_bytes(path, scope=scope)
        except WebUIFilePreviewError as e:
            return _http_error(e.status, e.message)
        content_type, _ = mimetypes.guess_type(name)
        return _http_response(
            data,
            content_type=content_type or "application/octet-stream",
            extra_headers=[
                ("Content-Disposition", f"attachment; filename*=UTF-8''{quote(name)}")
            ],
        )

    def _handle_session_subagent(
        self,
        request: WsRequest,
        key: str,
        task_id: str,
    ) -> Response:
        logger.debug(
            "subagent HTTP fetch key={} task_id={}",
            key,
            task_id,
        )
        if not self.check_api_token(request):
            logger.debug("subagent HTTP fetch rejected: unauthorized")
            return _http_error(401, "Unauthorized")
        decoded_key = _decode_api_key(key)
        if decoded_key is None:
            logger.debug("subagent HTTP fetch rejected: invalid session key")
            return _http_error(400, "invalid session key")
        if not _is_websocket_channel_session_key(decoded_key):
            logger.debug("subagent HTTP fetch rejected: not a websocket session")
            return _http_error(404, "session not found")
        manager = getattr(self, "subagent_manager", None)
        if manager is None:
            logger.debug("subagent HTTP fetch rejected: manager unavailable")
            return _http_error(503, "subagent manager unavailable")
        status = manager.get_status(task_id)
        if status is None:
            logger.debug("subagent HTTP fetch rejected: task_id={} not found", task_id)
            return _http_error(404, "subagent not found")
        logger.debug("subagent HTTP fetch returned task_id={} phase={}", task_id, status.phase)
        return _http_json_response(status.to_payload())

    def _handle_session_automations(self, request: WsRequest, key: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        decoded_key = _decode_api_key(key)
        if decoded_key is None:
            return _http_error(400, "invalid session key")
        if not _is_websocket_channel_session_key(decoded_key):
            return _http_error(404, "session not found")
        pending_job_ids = self._pending_automation_ids_for_session(decoded_key)
        return _http_json_response(
            session_automations_payload(
                self.cron_service,
                decoded_key,
                local_trigger_store=self.local_trigger_store,
                pending_job_ids=pending_job_ids,
            )
        )

    async def _handle_session_delete(self, request: WsRequest, key: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        if self.session_manager is None:
            return _http_error(503, "session manager unavailable")
        decoded_key = _decode_api_key(key)
        if decoded_key is None:
            return _http_error(400, "invalid session key")
        if not _is_websocket_channel_session_key(decoded_key):
            return _http_error(404, "session not found")
        query = _parse_query(request.path)
        delete_automations = (_query_first(query, "delete_automations") or "").lower()
        automation_jobs = session_automation_jobs(
            self.cron_service,
            decoded_key,
            local_trigger_store=self.local_trigger_store,
        )
        if automation_jobs and delete_automations not in {"1", "true", "yes"}:
            return _http_json_response(
                {
                    "deleted": False,
                    "blocked_by_automations": True,
                    "automations": serialize_automation_jobs(automation_jobs),
                }
            )
        if automation_jobs:
            for job in automation_jobs:
                if isinstance(job, LocalTrigger):
                    if self.local_trigger_store is not None:
                        self.local_trigger_store.delete(job.id)
                elif self.cron_service is not None:
                    self.cron_service.remove_job(job.id)
        deleted = self.session_manager.delete_session(decoded_key)
        if deleted and self.bus is not None:
            purge = getattr(self.bus, "purge_inbound_for_session", None)
            if purge is not None:
                await purge(decoded_key)
        delete_webui_thread(decoded_key)
        # ponytail: invalidate the WebUI session list index cache so deleted
        # sessions don't reappear in the sidebar before the next reconciliation.
        try:
            from nanobot.webui.session_list_index import _index_path
            idx = _index_path(self.session_manager.sessions_dir)
            if idx.is_file():
                idx.unlink()
        except OSError:
            pass
        return _http_json_response({"deleted": bool(deleted)})

    def _resolve_session_for_project(self, key: str) -> tuple[Any | None, str | None, Response | None]:
        """Validate a session key and return ``(session, decoded_key, error)``."""
        if self.session_manager is None:
            return None, None, _http_error(503, "session manager unavailable")
        decoded_key = _decode_api_key(key)
        if decoded_key is None:
            return None, None, _http_error(400, "invalid session key")
        if not _is_websocket_channel_session_key(decoded_key):
            return None, None, _http_error(404, "session not found")
        session = self.session_manager.get_or_create(decoded_key)
        return session, decoded_key, None

    def _handle_session_project_get(self, request: WsRequest, key: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        session, decoded_key, err = self._resolve_session_for_project(key)
        if err is not None:
            return err
        project_id = chat_project_id_from_metadata(session.metadata)
        return _http_json_response({
            "session_key": decoded_key,
            "project_id": project_id,
        })

    def _handle_session_project_bind(self, request: WsRequest, key: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        session, decoded_key, err = self._resolve_session_for_project(key)
        if err is not None:
            return err
        query = _parse_query(request.path)
        project_id = (_query_first(query, "project_id") or "").strip()
        if not project_id:
            return _http_error(400, "missing project_id")
        try:
            self.projects.get_project(project_id)
        except ProjectError as exc:
            return _http_error(404, str(exc))
        set_chat_project_id(session, project_id)
        self.session_manager.save(session)
        return _http_json_response({
            "session_key": decoded_key,
            "project_id": chat_project_id_from_metadata(session.metadata),
        })

    def _handle_session_project_unbind(self, request: WsRequest, key: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        session, decoded_key, err = self._resolve_session_for_project(key)
        if err is not None:
            return err
        set_chat_project_id(session, None)
        self.session_manager.save(session)
        return _http_json_response({
            "session_key": decoded_key,
            "project_id": None,
        })

    def _handle_session_todo_bind(self, request: WsRequest, key: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        if self.session_manager is None:
            return _http_error(503, "session manager unavailable")
        decoded_key = _decode_api_key(key)
        if decoded_key is None:
            return _http_error(400, "invalid session key")
        if not _is_websocket_channel_session_key(decoded_key):
            return _http_error(404, "session not found")
        session = self.session_manager.get_or_create(decoded_key)
        query = _parse_query(request.path)
        slug = (_query_first(query, "slug") or "").strip()
        if not slug:
            return _http_error(400, "missing slug")
        set_chat_todo_list(session, slug)
        self.session_manager.save(session)
        return _http_json_response({
            "session_key": decoded_key,
            "todo_list": chat_todo_list_from_metadata(session.metadata),
        })

    def _handle_session_todo_unbind(self, request: WsRequest, key: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        if self.session_manager is None:
            return _http_error(503, "session manager unavailable")
        decoded_key = _decode_api_key(key)
        if decoded_key is None:
            return _http_error(400, "invalid session key")
        if not _is_websocket_channel_session_key(decoded_key):
            return _http_error(404, "session not found")
        session = self.session_manager.get_or_create(decoded_key)
        set_chat_todo_list(session, None)
        self.session_manager.save(session)
        return _http_json_response({
            "session_key": decoded_key,
            "todo_list": None,
        })

    # -- Automation routes --------------------------------------------------

    async def _dispatch_automation_routes(
        self,
        request: WsRequest,
        got: str,
    ) -> Response | None:
        if got == "/api/webui/automations":
            return self._handle_webui_automations(request)
        m = re.match(r"^/api/webui/automations/(enable|disable|delete|run|update)$", got)
        if m:
            return await self._handle_webui_automation_action(request, m.group(1))
        return None

    def _pending_cron_job_ids_for_all(self) -> set[str]:
        if self.cron_service is None or self.cron_pending_job_ids is None:
            return set()
        pending: set[str] = set()
        for job in self.cron_service.list_jobs(include_disabled=True):
            session_key = job.payload.session_key
            if not session_key and job.payload.origin_channel and job.payload.origin_chat_id:
                session_key = f"{job.payload.origin_channel}:{job.payload.origin_chat_id}"
            if session_key:
                pending.update(self.cron_pending_job_ids(session_key))
        return pending

    def _pending_local_trigger_ids_for_all(self) -> set[str]:
        if self.local_trigger_store is None or self.local_trigger_pending_ids is None:
            return set()
        pending: set[str] = set()
        for trigger in self.local_trigger_store.list_triggers(include_disabled=True):
            session_key = trigger.session_key
            if not session_key and trigger.channel and trigger.chat_id:
                session_key = f"{trigger.channel}:{trigger.chat_id}"
            if session_key:
                pending.update(self.local_trigger_pending_ids(session_key))
        return pending

    def _pending_automation_ids_for_session(self, session_key: str) -> set[str]:
        pending: set[str] = set()
        if self.cron_pending_job_ids is not None:
            pending.update(self.cron_pending_job_ids(session_key))
        if self.local_trigger_pending_ids is not None:
            pending.update(self.local_trigger_pending_ids(session_key))
        return pending

    def _handle_webui_automations(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        pending_job_ids = self._pending_cron_job_ids_for_all()
        pending_job_ids.update(self._pending_local_trigger_ids_for_all())
        return _http_json_response(
            all_automations_payload(
                self.cron_service,
                local_trigger_store=self.local_trigger_store,
                session_manager=self.session_manager,
                pending_job_ids=pending_job_ids,
            )
        )

    async def _handle_webui_automation_action(
        self,
        request: WsRequest,
        action: str,
    ) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        if self.cron_service is None and self.local_trigger_store is None:
            return _http_error(503, "automation service unavailable")

        query = _parse_query(request.path)
        job_id = (_query_first(query, "id") or _query_first(query, "job_id") or "").strip()
        if not job_id:
            return _http_error(400, "missing automation id")
        trigger = self.local_trigger_store.get(job_id) if self.local_trigger_store else None
        if trigger is not None:
            return self._handle_local_trigger_action(request, action, trigger)

        if self.cron_service is None:
            return _http_error(404, "automation not found")
        job = self.cron_service.get_job(job_id)
        if job is None:
            return _http_error(404, "automation not found")
        if job.payload.kind == "system_event":
            return _http_error(403, "system automation is protected")
        if action in {"enable", "run"} and not is_bound_cron_job(job):
            return _http_error(409, "automation has no linked chat")

        if action == "enable":
            if self.cron_service.enable_job(job_id, enabled=True) is None:
                return _http_error(404, "automation not found")
        elif action == "disable":
            if self.cron_service.enable_job(job_id, enabled=False) is None:
                return _http_error(404, "automation not found")
        elif action == "delete":
            result = self.cron_service.remove_job(job_id)
            if result == "not_found":
                return _http_error(404, "automation not found")
            if result == "protected":
                return _http_error(403, "system automation is protected")
        elif action == "run":
            if not job.enabled:
                return _http_error(409, "automation is disabled")
            task = asyncio.create_task(self.cron_service.run_job(job_id, force=False))
            task.add_done_callback(self._log_automation_run_result)
        elif action == "update":
            values = _automation_values_from_request(request)
            if values is None:
                return _http_error(400, "invalid automation update payload")
            parsed = _parse_automation_update(values, current_job=job)
            if isinstance(parsed, str):
                return _http_error(400, parsed)
            try:
                result = self.cron_service.update_job(job_id, **parsed)
            except ValueError as exc:
                return _http_error(400, str(exc))
            if result == "not_found":
                return _http_error(404, "automation not found")
            if result == "protected":
                return _http_error(403, "system automation is protected")
        else:
            return _http_error(404, "unknown automation action")

        return self._handle_webui_automations(request)

    def _handle_local_trigger_action(
        self,
        request: WsRequest,
        action: str,
        trigger: LocalTrigger,
    ) -> Response:
        if self.local_trigger_store is None:
            return _http_error(503, "trigger service unavailable")
        if action == "enable":
            if self.local_trigger_store.enable(trigger.id, enabled=True) is None:
                return _http_error(404, "automation not found")
        elif action == "disable":
            if self.local_trigger_store.enable(trigger.id, enabled=False) is None:
                return _http_error(404, "automation not found")
        elif action == "delete":
            if not self.local_trigger_store.delete(trigger.id):
                return _http_error(404, "automation not found")
        elif action == "run":
            return _http_error(409, "local trigger requires a CLI message")
        elif action == "update":
            values = _automation_values_from_request(request)
            if values is None:
                return _http_error(400, "invalid automation update payload")
            parsed = _parse_local_trigger_update(values)
            if isinstance(parsed, str):
                return _http_error(400, parsed)
            if parsed:
                if self.local_trigger_store.update(trigger.id, **parsed) is None:
                    return _http_error(404, "automation not found")
        else:
            return _http_error(404, "unknown automation action")

        return self._handle_webui_automations(request)

    @staticmethod
    def _log_automation_run_result(task: asyncio.Task[bool]) -> None:
        try:
            ran = task.result()
        except Exception:
            logger.exception("WebUI automation run-now task failed")
            return
        if not ran:
            logger.warning("WebUI automation run-now task did not execute")

    # -- Media routes -------------------------------------------------------

    def _dispatch_media_routes(self, request: WsRequest, got: str) -> Response | None:
        m = re.match(r"^/api/media/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)$", got)
        if m:
            return self._handle_media_fetch(m.group(1), m.group(2), request)
        return None

    def _handle_media_fetch(
        self, sig: str, payload: str, request: WsRequest | None = None
    ) -> Response:
        return self.media.serve_signed_media(
            sig,
            payload,
            request=request,
        )

    # -- Misc routes --------------------------------------------------------

    async def _dispatch_misc_routes(
        self, connection: Any, request: WsRequest, got: str
    ) -> Response | None:
        if got == "/api/sessions":
            return await self._handle_sessions_list(request)
        if got == "/api/commands":
            return self._handle_commands(request)
        if got == "/api/workspaces":
            return self._handle_workspaces(connection, request)
        if got == "/api/webui/skills":
            return self._handle_webui_skills(request)
        if got == "/api/webui/skills/toggle":
            return self._handle_webui_skill_toggle(request)
        m = re.match(r"^/api/webui/skills/([^/]+)$", got)
        if m:
            return self._handle_webui_skill_detail(request, m.group(1))
        if got == "/api/webui/clawhub/search":
            return await self._handle_clawhub_search(request)
        if got == "/api/webui/clawhub/trending":
            return await self._handle_clawhub_trending(request)
        if got == "/api/webui/clawhub/browse":
            return await self._handle_clawhub_browse(request)
        if got == "/api/webui/clawhub/install":
            return await self._handle_clawhub_install(request)
        if got == "/api/webui/clawhub/delete":
            return await self._handle_clawhub_delete(request)
        if got == "/api/webui/clawhub/update-all":
            return await self._handle_clawhub_update_all(request)
        if got == "/api/webui/sidebar-state":
            return self._handle_webui_sidebar_state(request)
        if got == "/api/webui/sidebar-state/update":
            return self._handle_webui_sidebar_state_update(request)
        return None

    # -- Agenda and Todos routes -------------------------------------------
    #
    # The WebUI's WS+HTTP transport issues every request as HTTP GET and
    # carries mutation payloads in headers. Route shapes and header names
    # must stay in sync with webui/src/lib/agenda-api.ts and todos-api.ts.

    _AGENDA_DATA_HEADER = "X-Nanobot-Agenda-Data"
    _TODO_DATA_HEADER = "X-Nanobot-Todo-Data"

    def _dispatch_agenda_routes(
        self, request: WsRequest, got: str
    ) -> Response | None:
        if got == "/api/agenda" or got == "/api/agenda/appointments":
            return self._handle_agenda_list(request)
        if got == "/api/agenda/create" or got == "/api/agenda/appointments/create":
            return self._handle_agenda_create(request)
        m = re.match(r"^/api/agenda/([^/]+)/update$", got)
        if m:
            return self._handle_agenda_update(request, m.group(1))
        m = re.match(r"^/api/agenda/([^/]+)/delete$", got)
        if m:
            return self._handle_agenda_delete(request, m.group(1))
        m = re.match(r"^/api/agenda/([^/]+)$", got)
        if m:
            return self._handle_agenda_get(request, m.group(1))
        return None

    def _handle_agenda_list(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        scope = self._agenda_todos_scope(request)
        return _http_json_response(agenda_list_appointments(scope))

    def _handle_agenda_get(self, request: WsRequest, appointment_id: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        scope = self._agenda_todos_scope(request)
        result = agenda_fetch_appointment(appointment_id, scope)
        if result.get("error"):
            return _http_error(404, result["error"])
        return _http_json_response(result)

    def _handle_agenda_create(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        scope = self._agenda_todos_scope(request)
        body, err = read_json_request_header(request, self._AGENDA_DATA_HEADER, 256_000)
        if err is not None:
            return err
        payload = body if isinstance(body, dict) else {}
        result = agenda_create_appointment(payload, scope)
        if result.get("error"):
            return _http_error(400, result["error"])
        return _http_json_response(result, status=201)

    def _handle_agenda_update(self, request: WsRequest, appointment_id: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        scope = self._agenda_todos_scope(request)
        body, err = read_json_request_header(request, self._AGENDA_DATA_HEADER, 256_000)
        if err is not None:
            return err
        changes = body if isinstance(body, dict) else {}
        result = agenda_update_appointment(appointment_id, changes, scope)
        if result.get("error"):
            return _http_error(400, result["error"])
        return _http_json_response(result)

    def _handle_agenda_delete(self, request: WsRequest, appointment_id: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        scope = self._agenda_todos_scope(request)
        result = agenda_delete_appointment(appointment_id, scope)
        if result.get("error"):
            return _http_error(404, result["error"])
        return _http_json_response(result)

    def _dispatch_todos_routes(
        self, request: WsRequest, got: str
    ) -> Response | None:
        if got == "/api/todos" or got == "/api/todos/lists":
            return self._handle_todo_list_index(request)
        if got == "/api/todos/create" or got == "/api/todos/lists/create":
            return self._handle_todo_list_create(request)
        if got == "/api/todos/_users" or got == "/api/todos/users":
            return self._handle_todo_users(request)
        if got == "/api/todos/migrate":
            return self._handle_todo_migrate(request)
        m = re.match(r"^/api/todos/([^/]+)/items/([^/]+)/delete$", got)
        if m:
            return self._handle_todo_item_delete(request, m.group(1), m.group(2))
        m = re.match(r"^/api/todos/([^/]+)/items/([^/]+)$", got)
        if m:
            return self._handle_todo_item_update(request, m.group(1), m.group(2))
        m = re.match(r"^/api/todos/([^/]+)/items$", got)
        if m:
            return self._handle_todo_item_create(request, m.group(1))
        m = re.match(r"^/api/todos/([^/]+)$", got)
        if m:
            return self._handle_todo_list_get_or_delete(request, m.group(1))
        return None

    def _handle_todo_list_index(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        scope = self._agenda_todos_scope(request)
        return _http_json_response(todo_list_todo_lists(scope))

    def _handle_todo_list_create(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        scope = self._agenda_todos_scope(request)
        body, err = read_json_request_header(request, self._TODO_DATA_HEADER, 256_000)
        if err is not None:
            return err
        payload = body if isinstance(body, dict) else {}
        name = payload.get("name") or ""
        result = todo_create_todo_list(name, scope, slug=payload.get("slug"))
        if result.get("error"):
            return _http_error(400, result["error"])
        return _http_json_response(result, status=201)

    def _handle_todo_list_get_or_delete(self, request: WsRequest, slug: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        scope = self._agenda_todos_scope(request)
        # The frontend uses /api/todos/<slug> for both fetch (no data header)
        # and delete (no data header). We distinguish by checking whether the
        # list exists: GET returns it, DELETE removes it.
        result = todo_fetch_todo_list(slug, scope)
        if result.get("error"):
            return _http_error(404, result["error"])
        # If the caller provided a mutation payload, treat it as an explicit
        # delete request (future-proofing) otherwise just return the list.
        if request.headers.get(self._TODO_DATA_HEADER):
            result = todo_delete_todo_list(slug, scope)
            if result.get("error"):
                return _http_error(404, result["error"])
        return _http_json_response(result)

    def _handle_todo_item_create(self, request: WsRequest, slug: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        scope = self._agenda_todos_scope(request)
        body, err = read_json_request_header(request, self._TODO_DATA_HEADER, 256_000)
        if err is not None:
            return err
        payload = body if isinstance(body, dict) else {}
        result = todo_create_item(slug, payload, scope)
        if result.get("error"):
            return _http_error(400, result["error"])
        return _http_json_response(result, status=201)

    def _handle_todo_item_update(self, request: WsRequest, slug: str, item_id: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        scope = self._agenda_todos_scope(request)
        body, err = read_json_request_header(request, self._TODO_DATA_HEADER, 256_000)
        if err is not None:
            return err
        changes = body if isinstance(body, dict) else {}
        result = todo_update_item(slug, item_id, changes, scope)
        if result.get("error"):
            return _http_error(400, result["error"])
        return _http_json_response(result)

    def _handle_todo_item_delete(self, request: WsRequest, slug: str, item_id: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        scope = self._agenda_todos_scope(request)
        result = todo_delete_item(slug, item_id, scope)
        if result.get("error"):
            return _http_error(404, result["error"])
        return _http_json_response(result)

    def _handle_todo_users(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        scope = self._agenda_todos_scope(request)
        body, err = read_json_request_header(request, self._TODO_DATA_HEADER, 256_000)
        if err is not None:
            # No data header => GET; present => PATCH.
            return _http_json_response(todo_fetch_users(scope))
        users = body if isinstance(body, dict) else {}
        result = todo_update_users(users, scope)
        if result.get("error"):
            return _http_error(400, result["error"])
        return _http_json_response(result)

    def _handle_todo_migrate(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        scope = self._agenda_todos_scope(request)
        return _http_json_response(todo_migrate_legacy(scope))

    def _agenda_todos_scope(self, request: WsRequest) -> WorkspaceScope:
        """Resolve the workspace scope for agenda/todos requests."""
        query = _parse_query(request.path)
        chat_id = _query_first(query, "chat_id")
        if chat_id:
            scope = self.workspaces.scope_for_session_key(f"websocket:{chat_id}")
        else:
            scope = self.workspaces.default_scope()
        return scope

    def _handle_commands(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        return _http_json_response({"commands": builtin_command_palette()})

    def _handle_workspaces(self, connection: Any, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        return _http_json_response(
            self.workspaces.payload(
                controls_available=self.workspace_controls_available(connection)
            )
        )

    def _handle_webui_skills(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        return _http_json_response(
            webui_skills_payload(
                self.skills_workspace_path,
                disabled_skills=self.disabled_skills,
            )
        )

    def _handle_webui_skill_toggle(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(request, "X-Nanobot-Clawhub-Data", 16_384)
        if err is not None:
            return err
        name = (body.get("name") or "").strip() if isinstance(body, dict) else ""
        enabled = bool(body.get("enabled")) if isinstance(body, dict) else False
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            return _http_error(400, "invalid skill name")
        from nanobot.config.loader import load_config, save_config

        config = load_config()
        disabled = list(config.agents.defaults.disabled_skills)
        if enabled:
            disabled = [item for item in disabled if item != name]
        elif name not in disabled:
            disabled.append(name)
        config.agents.defaults.disabled_skills = disabled
        save_config(config)
        # Keep the in-memory snapshot in sync so the skills list reflects
        # the change immediately; the agent loop still needs a restart.
        self.disabled_skills = set(disabled)
        return _http_json_response({"name": name, "enabled": enabled})

    def _handle_webui_skill_detail(self, request: WsRequest, raw_name: str) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")

        name = unquote(raw_name)
        if not name or "/" in name or "\\" in name:
            return _http_error(400, "invalid skill name")
        payload = webui_skill_detail_payload(
            self.skills_workspace_path,
            name,
            disabled_skills=self.disabled_skills,
        )
        if payload is None:
            return _http_error(404, "skill not found")
        return _http_json_response(payload)

    async def _handle_clawhub_search(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        query = _query_first(_parse_query(request.path), "q") or ""
        try:
            results = await asyncio.to_thread(clawhub_search, query)
        except ClawhubError as exc:
            return _http_error(502, str(exc))
        return _http_json_response({"results": results})

    async def _handle_clawhub_trending(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        try:
            results = await asyncio.to_thread(clawhub_trending)
        except ClawhubError as exc:
            return _http_error(502, str(exc))
        return _http_json_response({"results": results})

    async def _handle_clawhub_browse(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        query = _parse_query(request.path)
        try:
            page = int(_query_first(query, "page") or 1)
        except ValueError:
            page = 1
        try:
            page_size = int(_query_first(query, "page_size") or 50)
        except ValueError:
            page_size = 50
        try:
            payload = await asyncio.to_thread(clawhub_browse, page, page_size)
        except ClawhubError as exc:
            return _http_error(502, str(exc))
        return _http_json_response(payload)

    async def _handle_clawhub_install(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(request, "X-Nanobot-Clawhub-Data", 16_384)
        if err is not None:
            return err
        reference = (body.get("reference") or "").strip() if isinstance(body, dict) else ""
        if not reference:
            return _http_error(400, "missing reference")
        try:
            result = await asyncio.to_thread(
                clawhub_install, reference, self.skills_workspace_path / "skills"
            )
        except ClawhubError as exc:
            return _http_error(502, str(exc))
        return _http_json_response(result)

    async def _handle_clawhub_delete(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        body, err = read_json_request_header(request, "X-Nanobot-Clawhub-Data", 16_384)
        if err is not None:
            return err
        name = (body.get("name") or "").strip() if isinstance(body, dict) else ""
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            return _http_error(400, "invalid skill name")
        skills_root = (self.skills_workspace_path / "skills").resolve()
        target = (skills_root / name).resolve()
        if not target.is_relative_to(skills_root) or not (target / "SKILL.md").exists():
            return _http_error(404, "skill not found")
        try:
            await asyncio.to_thread(shutil.rmtree, target)
        except OSError as exc:
            return _http_error(500, f"could not delete skill: {exc}")
        return _http_json_response({"name": name, "deleted": True})

    async def _handle_clawhub_update_all(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        try:
            result = await asyncio.to_thread(
                clawhub_update_all, self.skills_workspace_path / "skills"
            )
        except ClawhubError as exc:
            return _http_error(502, str(exc))
        return _http_json_response(result)

    def _handle_webui_sidebar_state(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        return _http_json_response(read_webui_sidebar_state())

    def _handle_webui_sidebar_state_update(self, request: WsRequest) -> Response:
        if not self.check_api_token(request):
            return _http_error(401, "Unauthorized")
        query = _parse_query(request.path)
        raw_state = _query_first(query, "state")
        if raw_state is None:
            return _http_error(400, "missing state")
        try:
            decoded = json.loads(raw_state)
        except json.JSONDecodeError:
            return _http_error(400, "state must be JSON")
        if not isinstance(decoded, dict):
            return _http_error(400, "state must be an object")
        try:
            state = write_webui_sidebar_state(decoded)
        except ValueError as e:
            return _http_error(400, str(e))
        except OSError:
            self._log.exception("failed to write webui sidebar state")
            return _http_error(500, "failed to write sidebar state")
        return _http_json_response(state)

    # -- Static file serving ------------------------------------------------

    def _serve_static(self, request_path: str) -> Response | None:
        assert self.static_dist_path is not None
        rel = request_path.lstrip("/")
        if not rel:
            rel = "index.html"
        if ".." in rel.split("/") or rel.startswith("/"):
            return _http_error(403, "Forbidden")
        candidate = (self.static_dist_path / rel).resolve()
        try:
            candidate.relative_to(self.static_dist_path)
        except ValueError:
            return _http_error(403, "Forbidden")
        if not candidate.is_file():
            index = self.static_dist_path / "index.html"
            if index.is_file():
                candidate = index
            else:
                return None
        try:
            body = candidate.read_bytes()
        except OSError as e:
            self._log.warning("static: failed to read {}: {}", candidate, e)
            return _http_error(500, "Internal Server Error")
        ctype, _ = mimetypes.guess_type(candidate.name)
        if ctype is None:
            ctype = "application/octet-stream"
        if ctype.startswith("text/") or ctype in {"application/javascript", "application/json"}:
            ctype = f"{ctype}; charset=utf-8"
        if candidate.name == "index.html":
            cache = "no-cache"
        else:
            cache = "public, max-age=31536000, immutable"
        return _http_response(
            body,
            status=200,
            content_type=ctype,
            extra_headers=[("Cache-Control", cache)],
        )


def _automation_values_from_request(request: WsRequest) -> dict[str, Any] | None:
    raw = _case_insensitive_header(request.headers, _AUTOMATION_VALUES_HEADER)
    if not raw:
        return {}
    try:
        values = json.loads(raw)
    except Exception:
        try:
            values = json.loads(unquote(raw))
        except Exception:
            return None
    return values if isinstance(values, dict) else None


def _parse_automation_update(
    values: dict[str, Any],
    *,
    current_job: CronJob | None = None,
) -> dict[str, Any] | str:
    update: dict[str, Any] = {}
    if "name" in values:
        raw_name = values.get("name")
        if not isinstance(raw_name, str):
            return "name must be a string"
        name = raw_name.strip()
        if not name:
            return "name cannot be empty"
        update["name"] = name
    if "message" in values:
        raw_message = values.get("message")
        if not isinstance(raw_message, str):
            return "message must be a string"
        message = raw_message.strip()
        if not message:
            return "message cannot be empty"
        update["message"] = message
    if "schedule" in values:
        raw_schedule = values.get("schedule")
        if not isinstance(raw_schedule, dict):
            return "schedule must be an object"
        parsed_schedule = _parse_automation_schedule(raw_schedule)
        if isinstance(parsed_schedule, str):
            return parsed_schedule
        if current_job is not None and _schedule_matches_job(parsed_schedule, current_job):
            return update
        schedule_error = _validate_automation_schedule(parsed_schedule)
        if schedule_error:
            return schedule_error
        update["schedule"] = parsed_schedule
        update["delete_after_run"] = parsed_schedule.kind == "at"
    return update


def _parse_local_trigger_update(values: dict[str, Any]) -> dict[str, Any] | str:
    update: dict[str, Any] = {}
    if "name" in values:
        raw_name = values.get("name")
        if not isinstance(raw_name, str):
            return "name must be a string"
        name = raw_name.strip()
        if not name:
            return "name cannot be empty"
        update["name"] = name
    forbidden = [key for key in ("message", "schedule") if key in values]
    if forbidden:
        return "local trigger updates only support name"
    return update


def _parse_automation_schedule(values: dict[str, Any]) -> CronSchedule | str:
    raw_kind = values.get("kind")
    if not isinstance(raw_kind, str):
        return "schedule kind must be a string"
    kind = raw_kind.strip()
    if kind == "every":
        every_ms = _positive_int(values.get("every_ms"))
        if every_ms is None:
            return "every schedule requires positive every_ms"
        return CronSchedule(kind="every", every_ms=every_ms)
    if kind == "cron":
        raw_expr = values.get("expr")
        if not isinstance(raw_expr, str):
            return "cron schedule requires expr"
        expr = raw_expr.strip()
        if not expr:
            return "cron schedule requires expr"
        raw_tz = values.get("tz")
        if raw_tz is not None and not isinstance(raw_tz, str):
            return "cron schedule timezone must be a string"
        tz = raw_tz.strip() if isinstance(raw_tz, str) else ""
        return CronSchedule(kind="cron", expr=expr, tz=tz or None)
    if kind == "at":
        at_ms = _positive_int(values.get("at_ms"))
        if at_ms is None:
            return "one-time schedule requires positive at_ms"
        return CronSchedule(kind="at", at_ms=at_ms)
    return "unknown schedule kind"


def _schedule_matches_job(schedule: CronSchedule, job: CronJob) -> bool:
    current = job.schedule
    if schedule.kind != current.kind:
        return False
    if schedule.kind == "at":
        return schedule.at_ms == current.at_ms
    if schedule.kind == "every":
        return schedule.every_ms == current.every_ms
    if schedule.kind == "cron":
        return (schedule.expr or "") == (current.expr or "") and (
            schedule.tz or None
        ) == (current.tz or None)
    return False


def _validate_automation_schedule(schedule: CronSchedule) -> str | None:
    if schedule.kind == "at":
        if not schedule.at_ms or schedule.at_ms <= int(time.time() * 1000):
            return "one-time schedule must be in the future"
        return None
    if schedule.kind != "cron":
        return None

    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from croniter import croniter

        tz = ZoneInfo(schedule.tz) if schedule.tz else datetime.now().astimezone().tzinfo
        base = datetime.now(tz=tz)
        croniter(schedule.expr, base).get_next(datetime)
    except Exception:
        return "cron schedule is invalid"
    return None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _is_websocket_channel_session_key(key: str) -> bool:
    return key.startswith("websocket:")
