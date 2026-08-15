# Plan: Telegram UX — Task lists rich administradas por el agente, polls y efectos

Spec: `docs/spec-telegram-ux-checklists-polls-effects.md`

## Componentes y dependencias

```
TelegramChannel (runtime.py)
   ├── _send_task_list()        → nuevo: sendRichMessage con task list rich (- [ ] / - [x])
   ├── _update_task_list()      → nuevo: editMessageText(rich_message=...) marcando tareas + progreso
   ├── _on_poll_answer()        → nuevo: handler PollAnswer → InboundMessage al agente
   ├── _polls_cache             → nuevo: poll_id → (chat_id, options) para resolver poll_answer
   ├── send()                  → + checklist, checklist_update, poll, effect
   └── _try_send_rich()         → + message_effect_id en payload

OutboundMessage (bus/events.py) → + checklist, checklist_update, poll, effect
MessageTool (agent/tools/message.py) → + checklist, checklist_update, poll, effect (validación)
TelegramConfig (runtime.py) → + message_effect_id: str | None (default confeti)
```

## Orden de implementación

### T1: Tests primero (TDD) — `nanobot/channels/telegram/tests/test_telegram_channel.py`

1. **T1.1 Task list rich**
   - `send()` con `checklist={title, tasks}` → `sendRichMessage` con markdown
     `# title\n\n- [ ] tarea1\n- [ ] tarea2` (checkboxes nativos)
   - `message_id` de la task list guardado en el registro del canal
2. **T1.2 Actualización de progreso**
   - `send()` con `checklist_update={message_id, done:[0,2]}` →
     `editMessageText(rich_message=...)` con `- [x]` en las tareas marcadas y
     resumen "✅ 2/3 tareas completadas"
   - Tareas no marcadas siguen `- [ ]`
3. **T1.3 Poll de aprobación**
   - `send()` con `poll={question, options}` → `send_poll` con
     `is_anonymous=False`, `allows_multiple_answers=False`
   - Poll cacheado (poll_id → chat_id + options)
4. **T1.4 PollAnswer → contexto**
   - `_on_poll_answer` con update → `InboundMessage` publicado al bus con
     prefijo "🗳️ El usuario votó: APROBAR"
   - Opción resuelta vía cache; sin cache → poll_id crudo
5. **T1.5 Efectos de mensaje**
   - `send()` con `effect="confeti"` → `message_effect_id` en payload de
     `sendRichMessage` y `send_message`
   - Config `message_effect_id` default aplicado si no hay override
   - BadRequest por efecto no soportado → reintento sin efecto (best-effort)
6. **T1.6 Validación del tool message**
   - `checklist` debe ser {title: str, tasks: [str]} (1-30 tasks)
   - `poll` debe ser {question: str, options: [str]} (2-10 options)
   - `checklist_update` debe ser {message_id: int, done: [int]}
   - Errores de validación → ToolResult.error

### T2: Implementación en `nanobot/channels/telegram/runtime.py`

1. `_send_task_list(chat_id, title, tasks, ...)`: construye markdown rich con
   `# title` + `- [ ] task` por tarea; `sendRichMessage`; guarda message_id en
   `self._task_lists[chat_id] = message_id`
2. `_update_task_list(chat_id, message_id, done, ...)`: `editMessageText` con
   `rich_message.markdown` = título + tareas con `- [x]`/`- [ ]` + resumen
   "✅ N/M tareas completadas"
3. `_on_poll_answer(update, context)`: resuelve opción vía `_polls_cache`,
   publica `InboundMessage` con prefijo, encola como turno normal
4. `_polls_cache: dict[str, dict]` (poll_id → {chat_id, options}); limpieza
   básica (TTL o tamaño máx)
5. `send()`: orquesta `checklist` / `checklist_update` / `poll` / `effect`
6. `_try_send_rich` + `_send_text`: `message_effect_id` en payload (config
   default o override por mensaje); BadRequest → reintento sin efecto
7. `TelegramConfig` + `message_effect_id: str | None = None` (default confeti
   resuelto en runtime si None)

### T3: Bus + tool

1. `nanobot/bus/events.py`: `OutboundMessage` + `checklist: dict | None`,
   `checklist_update: dict | None`, `poll: dict | None`, `effect: str | None`
2. `nanobot/agent/tools/message.py`: parámetros `checklist`, `checklist_update`,
   `poll`, `effect` con validación (1-30 tasks, 2-10 options, done indices)

### T4: Verificación final + PR

1. `pytest nanobot/channels/telegram/tests/test_telegram_channel.py -v` verde
2. `ruff check nanobot/channels/telegram/ nanobot/agent/tools/message.py nanobot/bus/events.py`
3. Smoke: `pytest tests/ -q` (suite completa)
4. Sync a los 3 site-packages del gateway
5. Commit conventional (`feat(telegram): task lists rich, polls y efectos`),
   push al fork, PR a madkoding/nanobot con tests (requisito del PR Guardian)

## Riesgos y mitigaciones

- **Task lists rich vs checklists nativas**: las rich no envían updates al bot
  cuando el usuario tilda (solo el agente las edita). Aceptado: el agente es el
  administrador del progreso (requisito del usuario).
- **Efecto no soportado (grupos/servidor viejo)**: BadRequest → reintento sin
  efecto (best-effort, sin latch).
- **Poll sin cache**: si el poll_answer llega sin cache (gateway reiniciado),
  se publica con poll_id crudo — el agente puede pedir contexto.
- **Flood control**: `_call_with_retry` con backoff (ya existe).
- **Límite 30 tasks**: validación en el tool (error claro al agente).
