# Spec: Telegram Generative UI — Rich Messages, streaming nativo y teclados generados por el agente

## Objective

Mejorar el canal Telegram de nanobot para aprovechar **todas las capacidades de UI que
ofrece un bot de Telegram**, con foco en lo que la comunidad llama **Generative UI**: que
el modelo (el agente) genere interfaces interactivas dinámicamente, no solo texto.

Capacidades reales de la Bot API (verificadas en core.telegram.org, jul 2026):

1. **Rich Messages** (Bot API 10.1, 11 jun 2026): `sendRichMessage` — mensajes altamente
   estructurados: headings (`#`…`######`), tablas nativas con alineación/colspan/rowspan,
   listas y todo-lists (`- [ ]`), blockquotes y pull quotes, `details` colapsables,
   fórmulas LaTeX (`<tg-math-block>`), footnotes con anclas, divisores, footers, mapas,
   collages, slideshows, media embebida (foto/video/audio/voice), bloques `thinking`.
   Límites: 32.768 chars UTF-8, 500 bloques, 16 niveles de anidación, 50 media, 20
   columnas de tabla. Acepta **Rich Markdown** (GFM + tags HTML) o **Rich HTML**.
   Es el formato ideal para respuestas de IA (reportes, documentación, menús, resúmenes).
2. **Streaming de Rich Messages** (`sendRichMessageDraft`, 10.1): el bot muestra un
   borrador **efímero** (~30 s) que se anima con cada update del mismo `draft_id`; al
   terminar se "fija" con `sendRichMessage`. Reemplaza el patrón actual "send + edit"
   con algo nativo: sin parpadeos, sin mensajes intermedios, sin basura si el stream se
   corta. `editMessageText` acepta `rich_message` para editar rich ya enviados.
3. **Inline keyboards** (ya implementado en nanobot, config `inline_keyboards`):
   botones callback bajo el mensaje. Se extiende: el agente ya puede pedir `buttons` en
   el tool `message`; se mantiene y se documenta como parte de la UI generativa.
4. **Reply keyboards** (`ReplyKeyboardMarkup`): teclado que reemplaza el teclado del
   usuario con opciones predefinidas; soporta `one_time_keyboard`,
   `input_field_placeholder`, `request_contact`, `request_location`, `request_poll`,
   `request_user`, `request_chat`. El agente puede ofrecer un menú de opciones sin
   escribir texto.
5. **Menu button + comandos dinámicos** (`setChatMenuButton`, `setMyCommands` con
   scopes): el bot ya registra comandos globales; se agrega la posibilidad de que el
   agente ajuste el menú por chat (p.ej. acciones contextuales del momento).
6. **Ephemeral messages** (Bot API 10.2, 14 jul 2026): mensajes visibles solo para un
   usuario específico en grupos (`is_ephemeral`, `receiver_user_id`,
   `editEphemeralMessageText`, `deleteEphemeralMessage`). Útil para respuestas privadas
   en grupos sin spam.
7. **Deep linking** (`/start <param>`): parámetros de arranque; útil para pairing y
   atajos (ya soportado por el handler `/start`).
8. **Reacciones y typing** (ya implementado): `set_message_reaction`, `send_chat_action`.

**No existe** "App Actions" ni `setMyActions` en la Bot API (verificado jul 2026); la
spec anterior los incluía por error. La UI generativa se logra con Rich Messages +
teclados + comandos dinámicos + ephemeral.

**Usuario**: operador del gateway (Telegram/WebUI). **Éxito**: las respuestas del bot en
Telegram usan Rich Messages cuando aportan (tablas/estructura), el streaming usa drafts
nativos, el agente puede ofrecer teclados (inline y reply) y comandos dinámicos, y puede
responder de forma privada en grupos con ephemeral messages.

## Assumptions

1. El canal Telegram ya tiene un fast-path `sendRichMessage` (config `rich_messages`,
   runtime.py:673-726) que envía markdown crudo vía `do_api_request` y hace latch-off si
   el servidor no lo soporta. Se mantiene y se extiende.
2. `python-telegram-bot` 22.8 **no** tiene tipos para Rich Messages (issue #5261 abierta
   upstream); el envío se hace vía `bot.do_api_request("sendRichMessage", ...)` con
   payloads dict — patrón ya usado en el repo.
3. El streaming actual (`send_delta`) usa el patrón "send + edit_message_text" con
   previews en texto plano. Los drafts nativos son superiores (efímeros, animados, sin
   parpadeo) y se integran en `send_delta` sin cambiar el contrato del bus.
4. El tool `message` ya expone `buttons` (inline keyboards). Se agregan parámetros
   nuevos sin romper los existentes: `rich` (bool o markdown explícito), `reply_keyboard`
   (list[list[str]]), `menu_commands` (list[dict]) y `ephemeral` (bool).
5. Los mensajes rich se envían con `reply_parameters` (no `reply_to_message_id`) y
   soportan `reply_markup` (inline keyboards) — ya implementado en `_try_send_rich`.
6. El límite de 32.768 chars de Rich Messages es mayor que el de 4.096 de sendMessage;
   el split actual (4000/4096) se mantiene para el path legacy, y el path rich puede
   enviar chunks más grandes (30.000 chars por chunk rich).
7. Ephemeral messages requieren Bot API 10.2+; si el servidor no lo soporta, se cae a
   mensaje normal (best-effort, sin latch).

## Tech Stack

- Python 3.11+, asyncio
- `python-telegram-bot` 22.8 (sin tipos rich; payloads dict vía `do_api_request`)
- pytest (tests de regresión), ruff (lint)

## Commands

```bash
# Tests nuevos + regresión
pytest nanobot/channels/telegram/tests/test_telegram_channel.py -v

# Lint
ruff check nanobot/channels/telegram/ nanobot/agent/tools/message.py nanobot/bus/events.py

# Smoke general
pytest tests/ -q  # suite completa (excl. whatsapp/neonize si fallan por deps)
```

## Project Structure

- `nanobot/channels/telegram/runtime.py` → `TelegramChannel`:
  - `_try_send_rich()`: extiende para soportar payload completo (markdown + media +
    blocks), `reply_markup`, `is_ephemeral`/`receiver_user_id`, y chunks > 4096.
  - `send_delta()`: nuevo path de streaming con `sendRichMessageDraft` (draft_id por
    stream) + fix final con `sendRichMessage`.
  - `_send_reply_keyboard()`: envía `ReplyKeyboardMarkup` (one_time, placeholder).
  - `_set_chat_menu_commands()`: `setMyCommands` con scope por chat.
  - `_send_ephemeral()`: `sendMessage` con `is_ephemeral` + `receiver_user_id`.
- `nanobot/bus/events.py` → `OutboundMessage`: campos `rich: bool | None`,
  `reply_keyboard: list[list[str]]`, `menu_commands: list[dict]`, `ephemeral: bool`.
- `nanobot/agent/tools/message.py` → `MessageTool`: parámetros `rich`, `reply_keyboard`,
  `menu_commands`, `ephemeral` (con validación).
- `nanobot/channels/telegram/manifest.py` → `SETUP_SPEC`: campo `richMessages` (ya
  existe en config; se agrega al setup del canal).
- `webui/src/components/settings/channels/...` → campo `richMessages` en el setup.

## Behavior

### 1. Rich Messages (extensión del fast-path existente)

- `_try_send_rich()` ya envía `{rich_message: {markdown: content}}`. Se extiende:
  - Si `msg.rich` es `True` o el contenido tiene estructura (tablas/headings), se usa
    el path rich (ya es el comportamiento con `rich_messages=True`).
  - Si el contenido excede 30.000 chars, se parte en chunks rich (límite 32.768).
  - `reply_markup` se pasa como parámetro (ya soportado).
  - `is_ephemeral` + `receiver_user_id` se pasan si `msg.ephemeral` (10.2).
- El latch `_rich_send_disabled` se mantiene: si el servidor no soporta
  `sendRichMessage`, se cae al path legacy (HTML) sin degradar.

### 2. Streaming con drafts nativos

- En `send_delta()` con `streaming=True` y `rich_messages=True`:
  - Primer delta: `sendRichMessageDraft(chat_id, draft_id=<random>, rich_message={markdown: preview})`.
  - Deltas siguientes: mismo `draft_id`, contenido acumulado (con throttle por
    `stream_edit_interval`).
  - `stream_end`: `sendRichMessage(chat_id, rich_message={markdown: texto_final})` y
    Telegram reemplaza el draft automáticamente (no hay que borrar nada).
  - Si `sendRichMessageDraft` falla (servidor viejo), se cae al path actual
    (send + edit).
- El draft es efímero: si el stream se corta (crash), no queda mensaje basura.
- `_StreamBuf` gana `draft_id: int | None` para el streaming rich.

### 3. Reply keyboards (teclado de respuesta)

- El tool `message` acepta `reply_keyboard: list[list[str]]`.
- `OutboundMessage.reply_keyboard` se propaga al canal.
- El canal envía `ReplyKeyboardMarkup(keyboard, one_time_keyboard=True,
  input_field_placeholder="Elige una opción…")` en el mensaje final (no en previews).
- El tap de un botón envía el texto como mensaje normal → el agente lo recibe y
  responde (sin handler especial).

### 4. Comandos dinámicos por chat (menu button)

- El tool `message` acepta `menu_commands: [{command, description}]`.
- El canal llama `setMyCommands(commands, scope={type: "chat", chat_id})` (best-effort,
  log debug si falla).
- Se registran al enviar el mensaje que los pide; se reemplazan con el siguiente
  `menu_commands` (o se limpian con lista vacía).

### 5. Ephemeral messages (10.2)

- El tool `message` acepta `ephemeral: bool` (solo tiene sentido en grupos).
- El canal envía con `is_ephemeral=True` + `receiver_user_id=<user_id del chat>`.
- Si el servidor no soporta 10.2 (BadRequest), se reintenta sin `is_ephemeral`
  (best-effort, sin latch).

### 6. Config del canal

- `rich_messages: bool = False` (ya existe) — se mantiene.
- `streaming: bool = True` (ya existe) — los drafts se usan solo si `rich_messages`
  está activo.
- `inline_keyboards: bool = False` (ya existe) — se mantiene para `buttons`.
- No se agregan configs nuevas: `reply_keyboard`, `menu_commands` y `ephemeral` son
  por-mensaje (el agente decide).

## Edge Cases

- **Servidor Bot API viejo**: `sendRichMessage`/`sendRichMessageDraft` fallan con
  "method not found" → latch-off + fallback legacy (ya implementado).
- **Draft expirado (30 s)**: si el stream tarda más, Telegram descarta el draft; el
  fix final con `sendRichMessage` igualmente envía el mensaje completo (el draft
  desaparece solo).
- **Ephemeral sin soporte (10.1 o menor)**: BadRequest → reintento sin `is_ephemeral`.
- **`setMyCommands` con scope falla**: log debug; el bot sigue funcionando.
- **Chunks rich > 32.768**: se parte en 30.000 por chunk (margen de seguridad).
- **`reply_markup` en rich**: se pasa como parámetro (ya soportado por
  `sendRichMessage`).
- **Reply keyboard en previews de streaming**: solo se adjunta en el mensaje final
  (stream_end), nunca en el preview.

## Acceptance Criteria

1. `_try_send_rich()` envía `sendRichMessage` con markdown y, si hay, `reply_markup`,
   `is_ephemeral`/`receiver_user_id`; fallback legacy intacto.
2. `send_delta()` con `rich_messages=True` usa `sendRichMessageDraft` (draft_id
   estable por stream) y fija con `sendRichMessage` en `stream_end`; sin drafts si
   `rich_messages=False` o si el servidor no lo soporta.
3. `OutboundMessage` gana `rich`, `reply_keyboard`, `menu_commands`, `ephemeral` y el
   tool `message` los expone con validación.
4. Reply keyboard: se envía `ReplyKeyboardMarkup` con one_time + placeholder en el
   mensaje final; el tap llega como mensaje normal.
5. `menu_commands` → `setMyCommands` con scope por chat (best-effort).
6. `ephemeral=True` → `is_ephemeral` + `receiver_user_id`; fallback sin ephemeral si
   el servidor no lo soporta.
7. Tests: (a) rich send con markdown/reply_markup/ephemeral, (b) chunks rich >
   30.000 chars, (c) fallback legacy con latch-off, (d) draft streaming con draft_id
   estable + fix final, (e) path legacy intacto sin rich, (f) reply keyboard en
   stream_end, (g) setMyCommands con scope por chat, (h) ephemeral fallback.
8. `pytest nanobot/channels/telegram/tests/ -v` verde; `ruff check` limpio.
