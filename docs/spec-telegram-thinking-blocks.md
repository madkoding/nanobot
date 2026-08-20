# Spec: Telegram Thinking Blocks — razonamiento interno del agente visible en el chat

## Contexto

El agente (deepseek-v4-flash y otros modelos con `reasoning_content`) ya genera
razonamiento interno en cada turno. `AgentRunner` lo emite vía
`hook.emit_reasoning()` → `ProgressEvent(reasoning_delta / reasoning_end)`, y
`ChannelManager` lo entrega al canal **solo si** el canal sobreescribe
`send_reasoning_delta` / `send_reasoning_end` y tiene `show_reasoning` habilitado
(default `True` en `BaseChannel`).

`TelegramChannel` **no** sobreescribe esos métodos → el razonamiento se descarta
silenciosamente. El usuario quiere verlo en el chat usando los **Thinking Blocks**
de Telegram (Bot API 10.1).

## Investigación: Thinking Blocks

- `InputRichBlockThinking` (Bot API 10.1): bloque con placeholder "Thinking…",
  correspondiente al tag `<tg-thinking>` en Rich Markdown.
- **Restricción**: el bloque thinking **solo puede usarse en `sendRichMessageDraft`**
  (draft efímero ~30 s en chats privados). No puede recibirse en mensajes finales.
- El draft se fija (persiste) llamando `sendRichMessage` con el mismo `draft_id` —
  Telegram reemplaza el draft por el mensaje final.
- `sendRichMessageDraft` acepta `reply_parameters` (verificado contra la API real,
  PR #22) — el reply se conserva al fijar con `sendRichMessage(draft_id=...)`.
- Límites rich: 32.768 chars UTF-8, 500 bloques, 16 niveles de anidación.
- Alternativa persistente: `<details><summary>…</summary>…</details>` (bloque
  colapsable) sí es válido en mensajes finales rich (`sendRichMessage` /
  `editMessageText(rich_message=...)`).
- Alternativa de bajo énfasis en el path legacy: blockquote expandible
  (`<blockquote expandable>` en HTML).

## Estado actual en nanobot

- `AgentRunner` ya emite reasoning en streaming: `runner.py:516-518` (one-shot),
  `1126-1127` y `1153-1159` (streaming con `IncrementalThinkExtractor`).
- `ChannelManager._send_reasoning_delta/_end` (manager.py:828-850) despachan a
  `channel.send_reasoning_delta/end` si `channel.show_reasoning` (manager.py:752-766).
- `TelegramChannel` no implementa los primitivos → no-op silencioso.
- Streaming actual de Telegram: preview legacy (`send_message` + `edit_message_text`)
  y conversión rich in-place en `stream_end` con `editMessageText(rich_message=...)`.
- Decisión durable previa (commit 26a2795f): "no usar `sendRichMessageDraft` para
  contenido — dejaba drafts huérfanos congelados". **Esta spec la refina**: el
  huérfano era un bug de fijación (no pasar `draft_id` al fijar, corregido en
  e0d00541); un draft sin fijar expira solo (~30 s) y no deja basura. Con las
  reglas anti-huérfano de la Opción D, el flujo de drafts es seguro.
- `_StreamBuf` acumula solo `text` (contenido); no hay buffer de reasoning.

## Objetivo

Mostrar el razonamiento interno del agente en el chat de Telegram durante el turno,
usando Thinking Blocks nativos (`<tg-thinking>` en draft) y con el reasoning
persistente en `<details>` en el mensaje final, sin romper el contrato de streaming
actual (un solo mensaje, sin huérfanos, sin duplicados).

**Usuario**: operador del gateway (Telegram). **Éxito**: al pedir algo que requiera
razonamiento, el bot muestra el "Thinking…" animado nativo con el razonamiento en
vivo, y el mensaje final queda con el reasoning accesible (colapsable) y el reply
conservado.

## Requisitos

| ID | Requisito |
|---|---|
| REQ-001 | `TelegramChannel` debe sobreescribir `send_reasoning_delta` / `send_reasoning_end` para renderizar el reasoning |
| REQ-002 | El reasoning debe verse **en vivo** durante el turno con el Thinking Block nativo (`<tg-thinking>` en draft) |
| REQ-003 | El mensaje final debe quedar limpio: reasoning persistente en `<details>` colapsable |
| REQ-004 | `show_reasoning` (config existente) debe controlar la visibilidad; `False` → comportamiento actual (sin reasoning) |
| REQ-005 | No debe romperse el streaming actual: un solo mensaje, sin drafts huérfanos, sin duplicados |
| REQ-006 | El reasoning no debe interferir con tool hints, reply keyboards ni ephemeral |
| REQ-007 | El reply del mensaje debe conservarse (draft con `reply_parameters` + fijado con `draft_id`) |
| REQ-008 | En grupos (drafts no soportados) → fallback a blockquote expandible legacy |
| REQ-009 | Si el draft expira (~30 s sin deltas) o falla → fallback legacy con el contenido acumulado |
| REQ-010 | Tests de regresión para los primitivos de reasoning (delta/end) en el canal Telegram |

## Decisiones de diseño

### Decisión: Opción D — Híbrida nativa (thinking draft + details final)

Aprobada por el usuario (2026-08-14). El flujo completo del stream:

1. **Fase de razonamiento** (primer `send_reasoning_delta`):
   - Si el chat es privado y rich está habilitado → `sendRichMessageDraft` con
     `draft_id` estable (por stream) y `rich_message.markdown` =
     `<tg-thinking>…</tg-thinking>` con el reasoning acumulado. El cliente muestra
     "Thinking…" animado.
   - Si no (grupo / rich deshabilitado / latch-off) → preview legacy
     (`send_message` + `edit_message_text`) con `<blockquote expandable>`.
2. **Durante el stream** (`send_delta` con reasoning abierto):
   - Draft activo → `sendRichMessageDraft` (mismo `draft_id`) con
     `<tg-thinking>…</tg-thinking>` + contenido parcial.
   - Legacy → `edit_message_text` in-place.
3. **`send_reasoning_end`**: marca el fin del reasoning; el draft/legacy se actualiza
   con el thinking completo (sin contenido aún si el modelo sigue).
4. **`stream_end`** (fijación):
   - Draft activo → `sendRichMessage` con `draft_id` (reemplaza el draft efímero) y
     markdown final = contenido + `<details><summary>🧠 Razonamiento</summary>…</details>`
     con el reasoning acumulado (persistente). `reply_parameters` se conserva.
   - Legacy → `editMessageText(rich_message=...)` in-place (path actual) con el
     mismo `<details>`.
5. **Reglas anti-huérfano**:
   - Fijar **siempre** con `draft_id` al final del stream (nunca dejar el draft vivo).
   - Si el draft expira (>30 s sin deltas): el draft se autolimpia; el contenido
     acumulado se envía por legacy (`send_message` + `edit_message_text`).
   - Si `sendRichMessage` falla al fijar: fallback legacy con el contenido acumulado.
   - Si el stream se corta (error/cancelación): fijar el draft con lo acumulado
     (best-effort) o dejarlo expirar — nunca deja basura permanente.
   - Grupos: legacy directo (drafts solo en privados).
6. **`show_reasoning=False`**: no acumular ni renderizar reasoning; comportamiento
   actual intacto.

### Costos asumidos (transparencia)

- El draft es efímero (~30 s): si el modelo piensa en silencio más de 30 s sin
  emitir deltas, el draft expira y el mensaje final va por legacy (aparece como
  mensaje nuevo, no como edición). Mitigación: los deltas de reasoning son
  frecuentes en modelos de razonamiento; no se garantiza el caso extremo.
- El reasoning se duplica en `<details>` en el mensaje final (el thinking block no
  persiste por diseño de Telegram).
- Dos paths de streaming conviven (draft rich + legacy), con latch-off si el
  servidor no soporta rich.

## Alcance

**Dentro**:
- `TelegramChannel.send_reasoning_delta` / `send_reasoning_end` (runtime.py)
- `_StreamBuf` + campos `reasoning`, `draft_id`, `draft_expires_at`, `using_draft`
- Lógica de fijación con `draft_id` + `<details>` final en `stream_end`
- Fallback legacy (expiración, grupos, latch-off)
- Tests de regresión en `nanobot/channels/telegram/tests/`

**Fuera**:
- `AgentRunner` / `ChannelManager` (ya emiten reasoning; no se tocan)
- Otros canales (Discord/WebUI ya tienen sus primitivas)
- Config: se reutiliza `show_reasoning` existente

## Notas

- El reasoning llega como `ProgressEvent` con `reasoning_delta=True` /
  `reasoning_end=True`; el contrato del bus no cambia.
- `_rich_send_disabled` (latch-off) aplica igual: si el servidor no soporta rich,
  el reasoning cae al path legacy (blockquote expandible).
- El reasoning puede ser largo (modelos de razonamiento): se trunca al límite rich
  (32.768) o se recorta a un máximo razonable (p.ej. 8.000 chars) para no inflar el
  mensaje final.
