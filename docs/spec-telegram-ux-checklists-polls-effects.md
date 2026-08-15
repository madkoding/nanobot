# Spec: Telegram UX — Task lists administradas por el agente, polls de aprobación y efectos de mensaje

## Contexto

El canal Telegram de nanobot ya tiene rich messages, thinking blocks, reply
keyboards con cleanup, ephemeral y streaming. Para mejorar la experiencia del
usuario (flujo SDD en el chat), se agregan tres features:

1. **Task lists rich administradas por el agente**: checkboxes nativos
   (`- [ ]` / `- [x]` en Rich Markdown) que el agente actualiza in-place a medida
   que avanza, indicando el progreso al usuario. Ideal para trackear
   `tasks/todo.md` en el chat.
2. **Polls de aprobación** (`sendPoll`): decisiones con resultado visible
   (APROBAR/CAMBIOS/RECHAZAR como opciones de poll). El bot recibe `poll_answer`.
3. **Efectos de mensaje** (`message_effect_id`): animaciones (confeti 🎉) al
   celebrar aprobaciones o completar tareas.

## Investigación

### Checklists (Bot API 10.2) — ⚠️ restricción clave

- `sendChecklist` **requiere `business_connection_id`** — solo disponible para
  bots conectados a business accounts, **no para bots regulares** (verificado en
  core.telegram.org y GramIO: "Business connection is mandatory").
- El bot de nanobot (@jyunis_nanobot) es un bot normal → `sendChecklist` **no
  aplica**.
- **Alternativa nativa disponible**: Rich Markdown soporta **task lists**
  (`- [ ]` / `- [x]`) que se renderizan como checkboxes reales en el cliente
  (bloque `InputRichBlockList` con items). El bot las edita con
  `editMessageText(rich_message=...)` — el patrón de edición rich in-place ya
  implementado en `_finalize_stream`.
- Límites rich: 32.768 chars, 500 bloques, 16 niveles de anidación.

### Polls (Bot API clásica, sin versión)

- `sendPoll(chat_id, question, options, is_anonymous=False, ...)` — soportado
  nativamente por PTB 22.8 (`bot.send_poll`, tipos `Poll`, `PollAnswer`).
- El bot recibe `poll_answer` (poll_id + option_ids) cuando el usuario vota.
- Para aprobaciones en chat privado: poll de 1 usuario funciona (muestra su
  selección); en grupos muestra el resultado en vivo.

### Efectos de mensaje (Bot API 10.2)

- Parámetro `message_effect_id` en `sendMessage` (y otros send methods) y en
  `sendRichMessage`.
- IDs documentados en core.telegram.org (sección "Message effects"); ejemplos
  conocidos: confeti, fuegos artificiales, corazones, fuego, like.
- El efecto se aplica al mensaje enviado; el usuario lo ve animado una vez.
- Verificar: disponibilidad en grupos (los efectos están pensados para chats
  privados; en grupos puede fallar → best-effort sin latch).

## Estado actual en nanobot

- `TelegramChannel` envía rich vía `do_api_request` (payloads dict) — patrón
  reutilizable para task lists rich.
- El tool `message` ya expone `rich`, `reply_keyboard`, `menu_commands`,
  `ephemeral` — se extiende con `checklist`, `checklist_update`, `poll` y
  `effect`.
- Los updates entrantes se manejan en `_on_message` / handlers de PTB
  (`CallbackQueryHandler`, `MessageHandler`) — se agrega handler para
  `PollAnswer`.
- `show_reasoning` / `rich_messages` ya controlan features por config.

## Objetivo

Que el agente pueda, desde el tool `message` (o un tool dedicado):

- Enviar una **task list rich** (checkboxes nativos `- [ ]` / `- [x]`) con las
  tareas del plan SDD; el agente la **administra**: a medida que avanza, edita el
  mensaje in-place marcando las tareas completadas y le indica al usuario el
  progreso (ej. "✅ 3/5 tareas completadas").
- Enviar un **poll de aprobación** (APROBAR/CAMBIOS/RECHAZAR) con resultado
  visible; el `poll_answer` llega al agente como contexto del turno.
- Aplicar un **efecto de mensaje** (confeti por defecto) a mensajes de
  celebración (aprobación de spec, tarea completada).

**Usuario**: operador del gateway (Telegram). **Éxito**: el flujo SDD en el chat
usa task lists rich administradas por el agente (progreso visible en vivo),
polls para decisiones, y efectos para celebrar — todo sin salir de Telegram.

## Requisitos

| ID | Requisito |
|---|---|
| REQ-001 | El tool `message` acepta `checklist` (title + tasks) y envía una **task list rich** (`- [ ]` / `- [x]`) vía `sendRichMessage` |
| REQ-002 | El agente **administra** la task list: edita el mensaje in-place (`editMessageText(rich_message=...)`) marcando tareas completadas a medida que avanza |
| REQ-003 | El agente indica el progreso al usuario (ej. "✅ 3/5 tareas completadas") al actualizar la lista |
| REQ-004 | El tool `message` acepta `poll` (question + options) y envía `sendPoll` nativo |
| REQ-005 | El canal recibe `poll_answer` y lo publica al agente como contexto del turno |
| REQ-006 | El tool `message` acepta `effect` (id de efecto) y lo aplica al mensaje (sendMessage y sendRichMessage) |
| REQ-007 | Config `message_effect_id` por canal (default confeti) para celebración automática de aprobaciones |
| REQ-008 | Fallbacks best-effort: task list/poll/effect no soportados → error claro sin romper el envío |
| REQ-009 | Tests de regresión: envío de task list rich, edición de progreso, poll, effect, handlers de updates entrantes |

## Decisiones de diseño

### D1: Task list rich administrada por el agente (no sendChecklist)

- `sendChecklist` **requiere business account** — no aplica a bots regulares.
- En su lugar: el tool `message` acepta `checklist: {title, tasks}` y el canal
  envía una **task list rich** (`- [ ]` / `- [x]`) vía `sendRichMessage`.
- El agente **administra el progreso**: guarda el `message_id` de la task list
  (en `_StreamBuf` o un registro por chat) y la edita in-place con
  `editMessageText(rich_message=...)` marcando tareas completadas + un resumen
  de progreso ("✅ 3/5").
- El tool `message` acepta `checklist_update: {message_id, done: [indices]}`
  para actualizar una task list existente.
- Alternativa descartada: `sendChecklist` nativo — requiere business account.

### D2: Updates entrantes → contexto del turno

- `PollAnswer` se convierte en `InboundMessage` con prefijo descriptivo
  (ej. "🗳️ El usuario votó: APROBAR") y se encola como un turno normal del
  agente (mismo session key del chat).
- El agente decide qué hacer (confirmar la decisión, avanzar el flujo SDD).
- No se responde automáticamente (evita spam); el agente responde si corresponde.

### D3: Efectos — config + param

- `message_effect_id` en `TelegramConfig` (default: confeti).
- El tool `message` acepta `effect: str | None` (override por mensaje).
- Se aplica en `send_message` y `sendRichMessage` (payload `message_effect_id`).
- Best-effort: BadRequest → reintento sin efecto (sin latch).

### D4: Polls — nativo PTB

- `sendPoll` con `is_anonymous=False` (decisiones visibles), `allows_multiple_answers=False`.
- El `poll_answer` se publica al agente con el texto de la opción elegida
  (resuelto vía el poll cacheado en el canal).

## Alcance

**Dentro**:
- `TelegramChannel`: envío de task list rich (`sendRichMessage` con `- [ ]`),
  edición de progreso (`editMessageText` rich), handler `PollAnswer`,
  `message_effect_id` en payloads
- `OutboundMessage` + `checklist`, `checklist_update`, `poll`, `effect`
- Tool `message` + `checklist`, `checklist_update`, `poll`, `effect` (validación)
- `TelegramConfig` + `message_effect_id`
- Tests de regresión

**Fuera**:
- `sendChecklist` nativo (requiere business account — no aplica a bots regulares)
- Sincronización automática de `tasks/todo.md` (la decide el agente vía tool
  `todos` existente)
- Otros canales
- Mini App / WebUI

## Notas

- PTB 22.8 no tiene tipos para task lists rich → payloads dict (patrón existente).
- Los IDs de efectos están documentados en core.telegram.org; el default (confeti)
  se verifica contra la API real antes del release.
- Límite de tasks por task list: 30 tareas máx (límite de checklists nativas,
  aplicado también a task lists rich para consistencia).
