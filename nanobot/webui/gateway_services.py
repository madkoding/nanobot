"""Composition helpers for the embedded WebUI gateway."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from loguru import logger as default_logger

from nanobot.config.paths import get_data_dir
from nanobot.webui.gateway_tokens import GatewayTokenStore
from nanobot.webui.ingress_policy import DEFAULT_WEBUI_INGRESS_POLICY, WebUIIngressPolicy
from nanobot.webui.media_gateway import WebUIMediaGateway
from nanobot.webui.project_context_provider import make_project_context_provider
from nanobot.webui.projects import WebUIProjectsController
from nanobot.webui.session_meta import chat_project_id_from_metadata as _chat_project_id
from nanobot.webui.transcript import WebUITranscriptRecorder
from nanobot.webui.workspaces import WebUIWorkspaceController
from nanobot.webui.ws_http import GatewayHTTPHandler


@dataclass(frozen=True)
class GatewayServices:
    """Explicit dependencies shared by WebSocket transport and HTTP routes."""

    http: GatewayHTTPHandler
    tokens: GatewayTokenStore
    media: WebUIMediaGateway
    ingress: WebUIIngressPolicy
    transcripts: WebUITranscriptRecorder
    workspaces: WebUIWorkspaceController
    projects: WebUIProjectsController
    session_manager: Any | None
    cron_service: Any | None
    local_trigger_store: Any | None
    cron_pending_job_ids: Callable[[str], set[str]] | None
    local_trigger_pending_ids: Callable[[str], set[str]] | None
    subagent_manager: Any | None
    owner_id: str | None = None


def build_gateway_services(
    *,
    config: Any,
    bus: Any,
    session_manager: Any | None,
    static_dist_path: Path | None,
    workspace_path: Path,
    worktree_root: Path | None = None,
    default_restrict_to_workspace: bool,
    runtime_model_name: Any | None,
    runtime_surface: str,
    runtime_capabilities_overrides: dict[str, Any] | None,
    disabled_skills: set[str] | None = None,
    cron_service: Any | None = None,
    local_trigger_store: Any | None = None,
    cron_pending_job_ids: Callable[[str], set[str]] | None = None,
    local_trigger_pending_ids: Callable[[str], set[str]] | None = None,
    channel_feature_action: Callable[..., Any] | None = None,
    channel_runtime_status: Callable[[], dict[str, Any]] | None = None,
    agent_loop: Any | None = None,
    subagent_manager: Any | None = None,
    runtime_resolver: Callable[[str | None], Any] | None = None,
    owner_id: str | None = None,
    logger: Any = default_logger,
) -> GatewayServices:
    tokens = GatewayTokenStore()
    ingress = DEFAULT_WEBUI_INGRESS_POLICY
    minimum_frame_bytes = ingress.minimum_full_policy_frame_bytes()
    if config.max_message_bytes < minimum_frame_bytes:
        logger.warning(
            "WebSocket maxMessageBytes={} is below the WebUI ingress policy capacity={}; "
            "policy-valid messages may still hit the transport frame guard",
            config.max_message_bytes,
            minimum_frame_bytes,
        )
    media = WebUIMediaGateway(
        workspace_path=workspace_path,
        logger=logger,
        attachment_limits=ingress.attachments,
    )
    transcripts = WebUITranscriptRecorder(log=logger)
    workspaces = WebUIWorkspaceController(
        session_manager=session_manager,
        default_workspace=workspace_path,
        default_restrict_to_workspace=default_restrict_to_workspace,
    )
    projects = WebUIProjectsController(
        data_dir=get_data_dir(),
        worktree_root=worktree_root,
    )
    try:
        result = projects.migrate_worktrees()
        if result.get("moved") or result.get("skipped"):
            logger.info(
                "Worktree migration: moved={} skipped={}", result.get("moved"), result.get("skipped")
            )
    except Exception as exc:
        logger.warning("Worktree migration failed: {}", exc)

    def _project_extra_read_dirs(session_metadata: Any) -> tuple[Path, ...]:
        """Read-only roots for the project bound to the session: uploaded files + folders."""
        project_id = _chat_project_id(session_metadata)
        if not project_id:
            return ()
        try:
            return projects.extra_read_dirs_for(project_id)
        except Exception:
            return ()

    if agent_loop is not None:
        if (
            session_manager is not None
            and callable(getattr(agent_loop, "register_runtime_context_provider", None))
        ):
            try:
                agent_loop.register_runtime_context_provider(
                    make_project_context_provider(session_manager, projects)
                )
            except Exception as exc:
                logger.warning("failed to register project context provider: {}", exc)
        if callable(getattr(agent_loop, "set_workspace_extra_read_dirs", None)):
            try:
                agent_loop.set_workspace_extra_read_dirs(_project_extra_read_dirs)
            except Exception as exc:
                logger.warning("failed to register project folder read access: {}", exc)
    http = GatewayHTTPHandler(
        config=config,
        session_manager=session_manager,
        static_dist_path=static_dist_path,
        runtime_model_name=runtime_model_name,
        runtime_surface=runtime_surface,
        runtime_capabilities_overrides=runtime_capabilities_overrides,
        bus=bus,
        tokens=tokens,
        media=media,
        ingress=ingress,
        workspaces=workspaces,
        projects=projects,
        skills_workspace_path=workspace_path,
        disabled_skills=disabled_skills,
        cron_service=cron_service,
        local_trigger_store=local_trigger_store,
        cron_pending_job_ids=cron_pending_job_ids,
        local_trigger_pending_ids=local_trigger_pending_ids,
        channel_feature_action=channel_feature_action,
        channel_runtime_status=channel_runtime_status,
        subagent_manager=subagent_manager,
        runtime_resolver=runtime_resolver,
        log=logger,
    )
    return GatewayServices(
        http=http,
        tokens=tokens,
        media=media,
        ingress=ingress,
        transcripts=transcripts,
        workspaces=workspaces,
        projects=projects,
        session_manager=session_manager,
        cron_service=cron_service,
        local_trigger_store=local_trigger_store,
        cron_pending_job_ids=cron_pending_job_ids,
        local_trigger_pending_ids=local_trigger_pending_ids,
        subagent_manager=subagent_manager,
        owner_id=owner_id,
    )
