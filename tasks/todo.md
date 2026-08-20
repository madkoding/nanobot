# Tasks: Telegram — Message effects opt-in (sin confeti por defecto)

Spec: `docs/spec-telegram-message-effects-opt-in.md` · Plan: `tasks/plan.md`

## Tareas

### T1: TDD (RED) — test de regresión nuevo

- [x] **T1.1 Test de regresión: sin effect y sin config → sin message_effect_id**
  - Acceptance: `send()` con `OutboundMessage` sin `effect` y `TelegramConfig` sin `message_effect_id` no incluye `message_effect_id` en ningún payload (rich y legacy); el test falla (RED) por el default actual
  - Verify: `uv run pytest nanobot/channels/telegram/tests/test_telegram_channel.py -q -k "without_effect"` falla (RED)
  - Files: `nanobot/channels/telegram/tests/test_telegram_channel.py`

### T2: Implementación (GREEN)

- [x] **T2.1 `_resolve_message_effect` sin default confeti**
  - Acceptance: falsy/`None` → `None`; `confeti`/`confetti` → id; id crudo → passthrough; REQ-001..005 cumplidos
  - Verify: T1.1 pasa (GREEN) + los 4 tests de effect existentes pasan
  - Files: `nanobot/channels/telegram/runtime.py`

- [x] **T2.2 Actualizar test de default por config**
  - Acceptance: `test_send_effect_config_default_applies_when_no_override` setea `message_effect_id="confeti"` explícito en `TelegramConfig` (caso REQ-003) y sigue pasando
  - Verify: `uv run pytest nanobot/channels/telegram/tests/test_telegram_channel.py -q -k "effect"` verde
  - Files: `nanobot/channels/telegram/tests/test_telegram_channel.py`

### T3: Docs

- [x] **T3.1 Actualizar spec UX existente (D3/REQ-007)**
  - Acceptance: `docs/spec-telegram-ux-checklists-polls-effects.md` ya no dice "default confeti"; REQ-007 refleja opt-in con config
  - Verify: `grep -rn "default.*confeti" docs/ nanobot/ | grep -v "sin confeti\|no-default\|opt-in"` sin hits
  - Files: `docs/spec-telegram-ux-checklists-polls-effects.md`

### T4: Verificación y release

- [x] **T4.1 Suite completa + ruff**
  - Acceptance: suite completa sin regresiones nuevas (5759+ passed / 16 failed WhatsApp preexistentes); `ruff check` limpio
  - Verify: `uv run pytest -q` y `uv run ruff check`
  - Files: —

- [x] **T4.2 Commit, fork y PR a madkoding**
  - Acceptance: commit conventional en `feature/telegram-no-default-confetti`, push al fork `jpyunism/nanobot`, PR a `madkoding/nanobot` con test de regresión
  - Verify: PR abierto y mergeable
  - Files: —

- [x] **T4.3 Sync site-packages + aviso reinicio**
  - Acceptance: `runtime.py` copiado a site-packages pyenv 3.13.3 y uv tool con md5 idénticos; aviso al usuario de reinicio manual
  - Verify: `md5sum` de los 3 archivos iguales
  - Files: `/home/jyunis/.pyenv/versions/3.13.3/lib/python3.13/site-packages/nanobot/channels/telegram/runtime.py`, `/home/jyunis/.local/share/uv/tools/nanobot-ai/lib/python3.13/site-packages/nanobot/channels/telegram/runtime.py`
