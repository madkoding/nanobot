# Tasks: Telegram UX — Task lists rich administradas por el agente, polls y efectos

Spec: `docs/spec-telegram-ux-checklists-polls-effects.md` · Plan: `tasks/plan.md`

## Tareas

### T1: Tests primero (TDD) — `nanobot/channels/telegram/tests/test_telegram_channel.py`

- [x] **T1.1 Task list rich**
  - Acceptance: `send()` con `checklist={title, tasks}` → `sendRichMessage` con markdown `# title` + `- [ ] task` por tarea; `message_id` guardado en el registro del canal
  - Verify: `pytest nanobot/channels/telegram/tests/test_telegram_channel.py -v` verde
  - Files: `nanobot/channels/telegram/tests/test_telegram_channel.py`

- [x] **T1.2 Actualización de progreso**
  - Acceptance: `send()` con `checklist_update={message_id, done:[0,2]}` → `editMessageText(rich_message=...)` con `- [x]` en las marcadas y resumen "✅ 2/3 tareas completadas"; no marcadas siguen `- [ ]`
  - Verify: `pytest nanobot/channels/telegram/tests/test_telegram_channel.py -v` verde
  - Files: `nanobot/channels/telegram/tests/test_telegram_channel.py`

- [x] **T1.3 Poll de aprobación**
  - Acceptance: `send()` con `poll={question, options}` → `send_poll` con `is_anonymous=False`, `allows_multiple_answers=False`; poll cacheado
  - Verify: `pytest nanobot/channels/telegram/tests/test_telegram_channel.py -v` verde
  - Files: `nanobot/channels/telegram/tests/test_telegram_channel.py`

- [x] **T1.4 PollAnswer → contexto**
  - Acceptance: `_on_poll_answer` → `InboundMessage` con prefijo "🗳️ El usuario votó: APROBAR"; opción resuelta vía cache; sin cache → poll_id crudo
  - Verify: `pytest nanobot/channels/telegram/tests/test_telegram_channel.py -v` verde
  - Files: `nanobot/channels/telegram/tests/test_telegram_channel.py`

- [x] **T1.5 Efectos de mensaje**
  - Acceptance: `effect` → `message_effect_id` en payload rich y legacy; config default aplicado; BadRequest → reintento sin efecto
  - Verify: `pytest nanobot/channels/telegram/tests/test_telegram_channel.py -v` verde
  - Files: `nanobot/channels/telegram/tests/test_telegram_channel.py`

- [x] **T1.6 Validación del tool message**
  - Acceptance: `checklist` {title, tasks 1-30}; `poll` {question, options 2-10}; `checklist_update` {message_id, done}; errores → ToolResult.error
  - Verify: `pytest nanobot/agent/tools/ -v` verde
  - Files: `nanobot/agent/tools/tests/` (o test existente del tool message)

### T2: Implementación en `nanobot/channels/telegram/runtime.py`

- [x] **T2.1 `_send_task_list`**
  - Acceptance: markdown rich `# title` + `- [ ] task`; `sendRichMessage`; message_id en `self._task_lists[chat_id]`
  - Files: `nanobot/channels/telegram/runtime.py`

- [x] **T2.2 `_update_task_list`**
  - Acceptance: `editMessageText(rich_message=...)` con `- [x]`/`- [ ]` + resumen "✅ N/M tareas completadas"
  - Files: `nanobot/channels/telegram/runtime.py`

- [x] **T2.3 `_on_poll_answer` + `_polls_cache`**
  - Acceptance: resuelve opción vía cache, publica `InboundMessage` con prefijo, encola como turno normal
  - Files: `nanobot/channels/telegram/runtime.py`

- [x] **T2.4 `send()` orquesta checklist/checklist_update/poll/effect**
  - Acceptance: los 4 campos del OutboundMessage se procesan; prioridad: checklist > checklist_update > poll > texto normal
  - Files: `nanobot/channels/telegram/runtime.py`

- [x] **T2.5 `message_effect_id` en payloads**
  - Acceptance: config default o override por mensaje en `sendRichMessage` y `send_message`; BadRequest → reintento sin efecto
  - Files: `nanobot/channels/telegram/runtime.py`

- [x] **T2.6 `TelegramConfig` + `message_effect_id`**
  - Acceptance: campo nuevo `message_effect_id: str | None = None`; default confeti resuelto en runtime
  - Files: `nanobot/channels/telegram/runtime.py`

### T3: Bus + tool

- [x] **T3.1 `OutboundMessage` + campos**
  - Acceptance: `checklist`, `checklist_update`, `poll`, `effect` (dicts/str opcionales)
  - Files: `nanobot/bus/events.py`

- [x] **T3.2 Tool `message` + parámetros**
  - Acceptance: `checklist`, `checklist_update`, `poll`, `effect` con validación (1-30 tasks, 2-10 options, done indices)
  - Files: `nanobot/agent/tools/message.py`

### T4: Verificación final + PR

- [x] **T4.1 Suite completa**
  - Acceptance: `pytest nanobot/channels/telegram/tests/ -q` verde + `ruff check` limpio + smoke `pytest tests/ -q`
  - Verify: comandos de verificación
  - Files: —

- [x] **T4.2 Sync + commit + push**
  - Acceptance: sync a los 3 site-packages; commit conventional (`feat(telegram): task lists rich, polls y efectos`); push a `feature/telegram-generative-ui`
  - Verify: `git log --oneline -3` + `git push origin feature/telegram-generative-ui`
  - Files: —
