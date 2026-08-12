<div align="center">
  <img src="images/nanobot_logo.svg" alt="nanobot" width="120">
  <h1>nanobot</h1>
  <p><strong>Ultra-lightweight, self-hosted personal AI agent runtime</strong></p>
  <p>
    <img src="https://img.shields.io/badge/python-≥3.11-blue" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
    <img src="https://img.shields.io/badge/version-0.3.12-blueviolet" alt="Version">
    <a href="https://github.com/madkoding/nanobot/graphs/commit-activity"><img src="https://img.shields.io/github/commit-activity/m/madkoding/nanobot" alt="Commits last month"></a>
    <a href="https://github.com/madkoding/nanobot/issues?q=is%3Aissue%20is%3Aclosed"><img src="https://img.shields.io/github/issues-search?query=repo%3Amadkoding%2Fnanobot%20is%3Aissue%20is%3Aclosed&label=issues%20closed" alt="Issues closed"></a>
  </p>
</div>

**nanobot** is a self-hosted personal AI agent runtime. It keeps the agent core small and readable while providing the practical pieces for real, long-running work: a browser WebUI, chat channels, tools, memory, MCP, model routing, automation, and deployment.

This repository is a fork of the [nanobot](https://github.com/re-bin/nanobot) project, maintained by [madkoding](https://github.com/madkoding) with production-focused additions for WhatsApp and the WebUI.

---

## Table of Contents

- [Features](#features)
- [What can nanobot do?](#what-can-nanobot-do)
- [Install](#install)
- [Quick Start](#quick-start)
- [WebUI](#webui)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [Differences from upstream](#differences-from-upstream)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Persistent workflows** — goals, memory, tools, and chat context survive long-running work.
- **Chat-native reach** — WebUI, API, Telegram, Feishu, Slack, Discord, Teams, email, Mattermost, WhatsApp, and more.
- **Model freedom** — OpenAI-compatible APIs, local LLMs, image generation, search, and fallbacks.
- **Small core** — readable internals with MCP, memory, deployment, and automation built in.
- **Own your stack** — inspect, customize, self-host, and extend without a giant platform.

## What can nanobot do?

nanobot is a self-hosted personal AI agent runtime. It can:

- run in a browser WebUI or terminal
- connect to Telegram, Discord, Slack, WeChat, Email, Mattermost, WhatsApp, and other chat apps
- use tools such as files, shell, web search, web fetch, MCP, cron, image generation, and subagents
- keep session history and long-term memory through Dream
- run long-horizon goals and scheduled automations
- expose a Python SDK and OpenAI-compatible API for integrations
- deploy as a long-running local or server-side agent gateway

---

## Install

> [!IMPORTANT]
> This fork is installed directly from the `madkoding/nanobot` repository.

**Prerequisites:** Python 3.11 or newer.

### One-command setup

macOS / Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/madkoding/nanobot/main/scripts/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/madkoding/nanobot/main/scripts/install.ps1 | iex
```

The default command installs or upgrades this fork from its `main` branch on GitHub. On a fresh local desktop, it then starts `nanobot webui` so you can configure the first provider and model in **Settings → Models**. The installer avoids system-wide pip installs by using an active virtual environment, `uv`, `pipx`, or a managed venv under `~/.nanobot/venv`.

To preview the plan without changing your environment, pass `--dry-run`:

```bash
curl -fsSL https://raw.githubusercontent.com/madkoding/nanobot/main/scripts/install.sh | sh -s -- --dry-run
```

To install the published upstream release from PyPI instead, pass `--pypi`:

```bash
curl -fsSL https://raw.githubusercontent.com/madkoding/nanobot/main/scripts/install.sh | sh -s -- --pypi
```

### Install from source

`bun` or `npm` must be available to build the WebUI. From an activated virtual environment:

```bash
git clone https://github.com/madkoding/nanobot.git
cd nanobot
python -m pip install .
```

Verify the install:

```bash
nanobot --version
```

### Update

Update nanobot to the latest `main` branch. The command detects your install
(editable source checkout vs GitHub zip), updates the Python package, rebuilds
the WebUI bundle, and restarts the gateway service automatically:

```bash
nanobot update
```

Check for updates without applying anything:

```bash
nanobot update --check
```

Useful flags:

- `--yes` / `-y` — skip confirmation prompts
- `--no-restart` — do not restart the gateway service
- `--no-webui` — skip the WebUI rebuild

---

## Quick Start

**Open nanobot in your browser**

```bash
nanobot webui
```

This is the recommended first run. The launcher creates the config and workspace when needed, safely enables the local WebSocket channel after confirmation, starts the gateway, and opens [`http://127.0.0.1:8765`](http://127.0.0.1:8765). The first-run WebUI binds to localhost by default and is not exposed to your LAN.

**Your first three steps**

1. Open **Settings → Models** and choose a provider, credential, and model.
2. Start a new topic and send `Hello!` to verify the connection.
3. Before project work, choose the intended workspace and access mode from the composer.

**Keep nanobot running after you close the terminal**

```bash
nanobot webui --background
```

```bash
nanobot gateway status
nanobot gateway logs
nanobot gateway restart
nanobot gateway stop
```

**Prefer a gateway-first workflow?**

```bash
nanobot gateway
```

This skips WebUI setup and browser opening, then runs the same complete gateway in the current terminal. Use `nanobot gateway --background` for the same direct entry point without keeping the terminal attached.

**Prefer to work entirely in the terminal?**

```bash
nanobot agent
```

This opens an interactive terminal chat with the same configured model, workspace, and tools. Type `exit` or press `Ctrl+C` when you are done.

For one request and an immediate exit:

```bash
nanobot agent -m "Hello!"
```

---

## WebUI

The WebUI ships **inside the published wheel** with no separate frontend build. It is the browser workbench for persistent topics, visible agent activity, workspace controls, Apps, Skills, Automations, and settings.

<p align="center">
  <img src="images/nanobot_webui.png" alt="nanobot webui preview" width="900">
</p>

Use it to:

- keep separate topics for different tasks and projects;
- inspect reasoning, tool calls, file edits, diffs, command output, and generated artifacts;
- switch models and workspaces without leaving the conversation;
- configure providers, chat channels, Apps, Skills, and Automations from one place.

---

## Architecture

<p align="center">
  <img src="images/nanobot_arch.png" alt="nanobot architecture" width="800">
</p>

nanobot stays lightweight by centering everything around a small agent loop: messages come in from chat apps, the LLM decides when tools are needed, and memory or skills are pulled in only as context instead of becoming a heavy orchestration layer. That keeps the core path readable and easy to extend, while still letting you add channels, tools, memory, and deployment options without turning the system into a monolith.

---

## Documentation

Browse the [repo docs](./docs/README.md) for the latest features and GitHub development version.

- [Guides](./docs/guides/README.md) — task-oriented guides
- [Start Without Technical Background](./docs/start-without-technical-background.md)
- [Install and Quick Start](./docs/quick-start.md)
- [Concepts](./docs/concepts.md)
- [Architecture](./docs/architecture.md)
- [Providers and Models](./docs/providers.md)
- [Provider Cookbook](./docs/provider-cookbook.md)
- [Troubleshooting](./docs/troubleshooting.md)
- [Chat Apps](./docs/chat-apps.md)
- [Automations](./docs/automations.md)
- [Configuration](./docs/configuration.md)
- [OpenAI-Compatible API](./docs/openai-api.md) · [Python SDK](./docs/python-sdk.md)
- [Deployment](./docs/deployment.md)

---

## Differences from upstream

This fork builds on the upstream [nanobot](https://github.com/re-bin/nanobot) project and adds production-focused changes:

**WhatsApp**
- Per-group / per-sender context isolation with turn queuing and safe mentions
- Sender identity injection into agent context for group turns
- Contact display-name persistence and recall per chat
- Outbound allowlist to prevent bans when messaging unknown numbers
- Circuit-breaker for the WhatsApp 463 throttle with configurable cooldown
- Removed auto-reconnect — manual re-link only
- Outage prevention ("live but silent") with locks, watchdog, and persistence
- Typing indicator on the correct chat JID
- LID JID resolution in the outbound allowlist check

**WebUI**
- Live subagent panel (event hook + status TTL + HTTP/WS fan-out)
- Subagent spawn chips that open the live panel
- Automation chips + clickable markdown media
- Projects CRUD with chat binding and context injection
- Responsive layout for mobile (full-screen sidebar on phones, drawer on tablets, support ≤380px)

**Core / tools**
- Text-to-speech tool using edge-tts
- Native Gemma 4 tool-call and thinking-tag parsing
- Identity-separation guardrails + `users.json` template
- Gateway fix: foreground startup no longer aborts on its own pid

**Docs / install**
- Linux installer script and updated documentation links

---

## Contributing

PRs welcome! The codebase is intentionally small and readable. 🤗

See [CONTRIBUTING.md](./CONTRIBUTING.md) for setup, review, and contribution guidelines.

**Roadmap** — pick an item and [open a PR](https://github.com/madkoding/nanobot/pulls)!

- **Multi-modal** — see and hear (images, voice, video)
- **Long-term memory** — never forget important context
- **Better reasoning** — multi-step planning and reflection
- **More integrations** — calendar and more
- **Self-improvement** — learn from feedback and mistakes

---

## License

nanobot is released under the [MIT License](./LICENSE). Third-party notices are listed in [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

### Contributors

<a href="https://github.com/madkoding/nanobot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=madkoding/nanobot&max=100&columns=12&updated=20260210" alt="Contributors" />
</a>
