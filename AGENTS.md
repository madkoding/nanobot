This file provides guidance to AI coding agents working with this repository.

## Project Overview

nanobot is a lightweight, open-source AI agent framework written in Python with a React/TypeScript WebUI. It centers around a small agent loop that receives messages from chat channels, invokes an LLM provider, executes tools, and manages session memory.

## Workspace Setup

This repository lives at `/home/madkoding/repos/nanobot`. The companion agent workspace (sessions, logs, runtime state) lives at `~/.nanobot/workspace/` and is configured at `~/.nanobot/config.json`.

```
/home/madkoding/repos/nanobot/      # <- this repo (source of truth for code + WebUI)
/home/madkoding/.nanobot/            # <- runtime workspace (config, sessions, venv)
  ├── config.json                    # gateway configuration
  ├── workspace/                     # session memory + logs
  └── venv/                          # Python venv with nanobot-ai installed editable
```

The gateway runs as a user systemd service (`nanobot-gateway.service`, see `~/.config/systemd/user/`). The venv has `nanobot-ai` installed in **editable** mode (`pip install -e .`) so the gateway reads `.py` files directly from this repo. The WebUI bundle ships in `nanobot/web/dist/` (gitignored) and is served from the same path the gateway finds via `import nanobot.web`.

```
~/.local/bin/bun                   # bun runtime for webui build/dev (fallback: npm)
```

## Development Commands

```bash
# Python: run single test / lint
pytest tests/test_openai_api.py::test_function -v
ruff check nanobot/

# WebUI: dev server, build, test (bun; npm also works)
cd webui && bun run dev             # Vite dev server with HMR
cd webui && bun run build           # outputs to ../nanobot/web/dist
cd webui && bun run test            # vitest

# Gateway: managed by systemd, but can be run manually for debugging
~/.nanobot/venv/bin/python -m nanobot gateway --foreground \
  --port 18790 \
  --workspace /home/madkoding/.nanobot/workspace \
  --config /home/madkoding/.nanobot/config.json

# Service control
systemctl --user status nanobot-gateway
systemctl --user restart nanobot-gateway
journalctl --user -u nanobot-gateway -n 50 --no-pager
```

## Updating the Gateway

The gateway runs from the venv as `nanobot-gateway.service` (user systemd). Because `nanobot-ai` is installed editable, **Python changes in this repo are picked up on the next process restart** — no `pip install` needed. WebUI bundle changes require a rebuild.

1. **Python changes** (`nanobot/*.py`):
   ```bash
   # Just edit, then restart the service
   systemctl --user restart nanobot-gateway
   tail -f /home/madkoding/.nanobot/logs/gateway.log
   ```

2. **WebUI changes** (`webui/src/*`):
   ```bash
   cd /home/madkoding/repos/nanobot/webui
   bun run build
   # Vite writes to ../nanobot/web/dist. Since the venv is editable, the
   # gateway will pick up the new dist on the next request — but you still
   # need to restart the gateway because it caches the path at startup.
   systemctl --user restart nanobot-gateway
   # User then hard-refreshes the browser (Ctrl+Shift+R / Cmd+Shift+R).
   ```

3. **Verify the running version** matches the source: open the WebUI at `http://localhost:8765/` and check the version shown next to the sidebar logo and in Settings → About. Both read `pyproject.toml`'s `version` field (the canonical semver source).

> If the user reports the gateway is "stale" (old code still running), the cause is almost always missing service restart or missing browser hard-refresh.

> If `pip install -e .` ever gets out of sync (e.g. after moving the repo), re-run:
> ```bash
> ~/.nanobot/venv/bin/pip install -e /home/madkoding/repos/nanobot --no-deps --quiet
> ```

## WebUI Architecture

The WebUI is a Vite + React 18 + TypeScript SPA. As of the `refactor(webui): extract hooks, dialogs, and shell components` commit, the structure is:

```
webui/src/
  App.tsx                          # thin entry — bootstrap + Provider + AppShell render
  components/
    shell/                         # shell-level composites
      AppShell.tsx                 # composition root (ThemedShell + providers)
      MainView.tsx                 # view switch (chat/settings/projects/workspace/todos/agenda/research)
      SidebarLayout.tsx            # 3 Sidebar renders (host, host-preview, mobile sheet)
      Overlays.tsx                 # 2 lazy dialogs (delete/rename) + restart toast + pairing popup
      ShellNativeHeader.tsx        # native chrome (HostChrome + theme toggle)
    HostChrome.tsx                 # native host wrapper
    PairingCodePopup.tsx           # pairing code UI
    Sidebar.tsx, ChatList.tsx, ... # feature components
  hooks/                           # one hook per concern (see below)
  lib/                             # pure modules (no React)
    routing.ts                     # hash routing + pushState race fix
    dialogs.ts                     # useDialogsState reducer
    bootstrap.ts                   # auth + token refresh
    sidebar-state-keys.ts          # localStorage helpers
  tests/                           # vitest, App layout etc.
```

### Key hooks

| Hook | Responsibility |
| --- | --- |
| `useBootstrap` | auth/loading/error views + token refresh |
| `useShellRoute` | routing state (activeKey, view, settingsSection) + pushState navigation |
| `useHostSidebarLayout` | host sidebar open/preview/mobile state |
| `useChatActions` | 22 callbacks for chat + utility actions |
| `useRunTracker` | running + updated chat ids (active session tracking) |
| `useEngineRestart` | restart state, toast, command |
| `useWorkspaceScope` | workspaces/draft/overrides + error |
| `usePairing` | pairing code UI state + polling |
| `useSettingsSnapshot` | fetchSettings + cache |
| `useChatActions` | chat + utility callbacks (openApps/Automations/Skills/Settings live under `chatActions.utility`) |
| `useShellShortcuts` | keyboard shortcuts (Cmd+K, Cmd+Shift+O) |
| `useDocumentTitle` | `document.title` effect per view |
| `useThreadSessionSync` | active chat ref + thread session updates |
| `useRuntimeModelSync` | model name sync from gateway |
| `useMissingSessionRedirect` | redirect when activeKey disappears |
| `useNativeHostClass` | toggles `native-host` body class |

### Routing race condition

`lib/routing.ts#writeShellRoute` uses `history.pushState` (not `window.location.hash =`) to update the URL, so the `hashchange` listener is not re-triggered. The listener is only fired on user-driven back/forward.

## High-Level Architecture

### Core Data Flow

Messages flow through an async `MessageBus` (`nanobot/bus/queue.py`) that decouples chat channels from the agent core:

1. **Channels** (`nanobot/channels/`) receive messages from external platforms and publish `InboundMessage` events to the bus.
2. **`AgentLoop`** (`nanobot/agent/loop.py`) consumes inbound messages, builds context, and coordinates the turn.
3. **`AgentRunner`** (`nanobot/agent/runner.py`) handles the actual LLM conversation loop: send messages to the provider, receive tool calls, execute tools, and stream responses.
4. Responses are published as `OutboundMessage` events back to the appropriate channel.

### Key Subsystems

- **Agent Loop** (`nanobot/agent/loop.py`, `runner.py`): The core processing engine. `AgentLoop` manages session keys, hooks, and context building. `AgentRunner` executes the multi-turn LLM conversation with tool execution.
- **LLM Providers** (`nanobot/providers/`): Provider implementations (Anthropic, OpenAI-compatible, OpenAI Responses API, Azure, Bedrock, GitHub Copilot, OpenAI Codex, etc.) built on a common base (`base.py`). Includes image generation (`image_generation.py`) and audio transcription (`transcription.py`). `factory.py` and `registry.py` handle instantiation and model discovery.
- **Channels** (`nanobot/channels/`): Platform integrations (Telegram, Discord, Slack, Feishu, Matrix, WhatsApp, QQ, WeChat, WeCom, DingTalk, Email, MoChat, MS Teams, WebSocket, Mattermost). `manager.py` discovers and coordinates them. Channels are self-contained packages auto-discovered via `pkgutil` scanning.
- **Tools** (`nanobot/agent/tools/`): Agent capabilities exposed to the LLM: filesystem (read/write/edit/list), shell execution (with sandbox backends), web search/fetch, MCP servers, cron, notebook editing, subagent spawning, long-running tasks / sustained goals (`long_task.py`), image generation, and self-modification. Tools are auto-discovered via `pkgutil` scan + entry-point plugins.
- **Memory** (`nanobot/agent/memory.py`): Session history persistence with Dream two-phase memory consolidation. Uses atomic writes with fsync for durability.
- **Session Management** (`nanobot/session/`): Per-session history, context compaction, TTL-based auto-compaction (`manager.py`), and sustained goal state tracking (`goal_state.py`).
- **Config** (`nanobot/config/schema.py`, `loader.py`): Pydantic-based configuration loaded from `~/.nanobot/config.json`. Supports camelCase aliases for JSON compatibility.
- **WebUI** (`webui/`): Vite-based React SPA that talks to the gateway over a WebSocket multiplex protocol. The dev server proxies `/api`, `/webui`, `/auth`, and WebSocket traffic to the gateway.
- **API Server** (`nanobot/api/server.py`): OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/models`) for programmatic access.
- **Command Router** (`nanobot/command/`): Slash command routing and built-in command handlers.
- **Heartbeat** (`nanobot/templates/HEARTBEAT.md`): Periodic task list checked via `cron` jobs (legacy dedicated service removed).
- **Pairing** (`nanobot/pairing/`): DM sender approval store with persistent pairing codes per channel.
- **Skills** (`nanobot/skills/`): Built-in skill definitions (cron, github, image-generation, etc.) loaded into agent context.
- **Security** (`nanobot/security/`): PTH file guard and other security measures activated at CLI entry.

### Entry Points

- **CLI**: `nanobot/cli/commands.py`
- **Python SDK**: `nanobot/nanobot.py`

## Project-Specific Notes

- Architecture constraints: [`.agent/design.md`](.agent/design.md)
- Security boundaries: [`.agent/security.md`](.agent/security.md)
- Common gotchas: [`.agent/gotchas.md`](.agent/gotchas.md)

## Telegram Rich Messages (formatting rules)

When sending rich messages (Bot API 10.1+) over Telegram:

- **Tables always at block level** — never inside blockquotes (`>`). Rich markdown does not render pipe tables as tables inside a blockquote; they degrade to plain text with raw pipes. Keep tables at the top level of the message.
- **Clear a stuck reply keyboard**: send `reply_keyboard=[]` (explicit empty list) with the next message to dismiss a previously shown reply keyboard (sends `ReplyKeyboardRemove`). `reply_keyboard=None` (default) sends no markup.

## Contribution Flow

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for contribution flow and PR guidelines.

## Code Style

- Python 3.11+, asyncio throughout.
- Line length: 100.
- Linting: `ruff` with rules E, F, I, N, W (E501 ignored).
- pytest with `asyncio_mode = "auto"`.

## Common File Locations

- Config schema: `nanobot/config/schema.py`
- Provider base / new provider template: `nanobot/providers/base.py`
- Channel base / new channel template: `nanobot/channels/base.py`
- Tool registry: `nanobot/agent/tools/registry.py`
- WebUI dev proxy config: `webui/vite.config.ts`
- Tests mirror the `nanobot/` package structure.
