# WebUI backend (`nanobot/webui/`)

Backend helpers that serve the bundled React WebUI over the gateway's
WebSocket HTTP surface. This package is **not** the frontend — the React
source lives in `webui/src/` and builds to `nanobot/web/dist/`.

When you need to change a WebUI feature, start from its domain below and read
the two dispatch modules that wire it up.

## The two dispatch modules (read these first)

| Module | What it does |
|---|---|
| `ws_http.py` | `GatewayHTTPHandler` — routes every non-WebSocket HTTP request (bootstrap, sessions, projects, media, settings, commands, static files, tokens). Imports most of this package and shapes their responses. |
| `settings_routes.py` | `WebUISettingsRouter` — maps the Settings request surface onto the Settings APIs, plus channel/provider/feature actions. |

Anything user-visible in the WebUI flows through one of these two.

## Domains

| Domain | Modules |
|---|---|
| **Chat / messages** | `transcript.py` (display JSONL), `thread_disk.py`, `session_list_index.py`, `session_meta.py`, `forking.py` |
| **Settings** | `settings_api.py`, `settings_routes.py`, `nanobot_features_api.py`, `mcp_presets_api.py`, `cli_apps_api.py`, `skills_api.py`, `version_check.py` |
| **Projects / workspaces** | `projects.py`, `workspaces.py`, `workspace_browser_api.py`, `worktrees.py`, `project_context_provider.py`, `sidebar_state.py` |
| **Todos / agenda / research** | `todos_api.py`, `agenda_api.py`, `research_api.py` |
| **Media / attachments** | `media_api.py`, `media_gateway.py`, `attachment_ingress.py`, `file_preview.py`, `ingress_policy.py` |
| **Tokens / auth / gateway** | `gateway_tokens.py`, `gateway_services.py`, `http_utils.py` |
| **Automations / telemetry** | `session_automations.py`, `token_usage.py`, `websocket_logging.py`, `transcription_ws.py` |
| **Build / packaging** | `build.py` (bundle sync), `metadata.py` (shared WebUI metadata keys) |

## Notes

- `ws_http.py` is large on purpose (a single HTTP dispatcher). It reads more
  easily as a routing table than as a framework of abstract handlers; keep new
  route handlers as methods here and put business logic in the domain modules.
- `todos_api.py` and `agenda_api.py` are business state that the agent tools
  (`agent/tools/todos.py`, `agent/tools/agenda.py`) also depend on. That is a
  known, flagged inversion — see the `TODO(architecture)` comments there.
