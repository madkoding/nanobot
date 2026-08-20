# Spec: Telegram — Reply (quote) solo en el primer mensaje de la conversación

Fecha: 2026-08-15 · Rama: `fix/telegram-reply-to-first` · Upstream: `madkoding/nanobot` (main)

## Objetivo

El bot de Telegram deja de responder **siempre** con `reply_parameters` (quote) al
mensaje del usuario. El quote pasa a usarse **solo en el primer mensaje de salida
de cada conversación/chat** (ancla la conversación); los mensajes subsiguientes se
envían planos, como continuación natural.

**Usuario**: JuanPa (Telegram, @jpyunism). Hoy cada respuesta llega como quote del
mensaje anterior, y el agente interpreta cada mensaje posterior como si el usuario
repitiera la pregunta original ("Sí, ...").

**Éxito (acceptance criteria)**:
1. Con la config real del usuario (`streaming:true`, `rich_messages:true`,
   `reply_to_message` ausente → default `false`), el **primer** mensaje de salida de
   un chat lleva `reply_parameters`; los siguientes **no**.
2. `config.reply_to_message: true` conserva el comportamiento actual: **siempre** quote.
3. `/new` (y `/start`) reinicia el ancla: el siguiente mensaje vuelve a llevar quote.
4. Reiniciar el gateway reinicia el ancla de forma natural (estado en memoria).
5. `_extract_reply_context` (contexto `[Reply to bot: ...]`) se mantiene intacto.

## Comportamiento actual (raíz)

- `send()` (no-streaming, `runtime.py` ~851-857) arma `reply_params` **solo** si
  `self.config.reply_to_message` es `true` (default `false` → sin quote).
- Los caminos de streaming **ignoran** esa condición y agregan `reply_parameters`
  **incondicionalmente** si `meta.get("message_id")` existe:
  - `_send_legacy_preview()` (~1414-1418): preview legacy del stream.
  - `_finalize_stream()` (~1477-1481): fijado del draft rich vía `sendRichMessage`.
- Con `streaming:true`, el texto de la respuesta fluye por `send_delta` →
  `_send_legacy_preview` / `_finalize_stream`, por eso **siempre** quotea.
- La cadena de quotes hace que, cuando el usuario responde al mensaje del bot,
  `_extract_reply_context` (~1859-1881) inyecte `[Reply to bot: <texto previo>]`
  en el contenido del usuario; el modelo cree que el usuario repite la pregunta
  anterior y vuelve a contestarla.

## Comportamiento nuevo

| # | Caso | Antes | Después |
|---|------|-------|---------|
| 1 | Primer mensaje de salida de un chat (streaming o no) | quote (streaming) / sin quote (no-streaming) | **quote** (ancla) |
| 2 | Mensajes subsiguientes del mismo chat | quote (streaming) | **sin quote** (plano) |
| 3 | `config.reply_to_message: true` | siempre quote | siempre quote (opt-in intacto) |
| 4 | `/new` o `/start` | — | reinicia ancla → siguiente mensaje quotea |
| 5 | Reinicio del gateway | — | ancla vacía → primer mensaje quotea |
| 6 | Usuario responde explícitamente al bot | `[Reply to bot: ...]` inyectado | igual (sin cambios) |

## Tech Stack

- Python 3.11+, asyncio.
- `python-telegram-bot` 22.8 (`ReplyParameters`, `sendRichMessage` Bot API 10.1).
- pytest + pytest-asyncio (`asyncio_mode = "auto"`), fake de PTB en
  `nanobot/channels/telegram/tests/test_telegram_channel.py` (`_FakeApp`/`_FakeBot`).
- Lint: ruff (E, F, I, N, W; E501 ignorado).

## Comandos

```
# Desde el directorio del repo (/home/jyunis/.nanobot/workspace/nanobot)
Test canal:   /home/jyunis/.local/bin/uv run pytest nanobot/channels/telegram/tests/test_telegram_channel.py -q
Test puntual: /home/jyunis/.local/bin/uv run pytest nanobot/channels/telegram/tests/test_telegram_channel.py::test_<nombre> -q
Suite:        /home/jyunis/.local/bin/uv run pytest -q
Lint:         /home/jyunis/.local/bin/uv run ruff check nanobot/channels/telegram/runtime.py nanobot/channels/telegram/tests/test_telegram_channel.py
```

## Project Structure

```
nanobot/channels/telegram/runtime.py        → helper _reply_params_for + estado _reply_anchored
                                              + reset en /new y /start (único archivo funcional)
nanobot/channels/telegram/tests/test_telegram_channel.py → tests de regresión (TDD)
docs/spec-telegram-reply-to-primera-vez.md  → esta spec
```

`manifest.py` **no** se toca: `reply_to_message` no está en `SETUP_SPEC` (solo
`token`, `proxy`, `allowFrom`, `groupPolicy`, `richMessages`). El campo ya existe en
`TelegramConfig` (`runtime.py` ~386) y se puede setear manualmente en `config.json`;
exponerlo en el WebUI queda fuera de alcance (ver Open Questions).

## Code Style

- Python, docstrings y comentarios en español (regla del repo), conventional commits.
- Cambio quirúrgico: 1 helper + 1 atributo de estado + 2 puntos de reset + tests.
- Helper propuesto (contrato, no implementación final):

```python
def _reply_params_for(self, chat_id: int, reply_to_message_id) -> ReplyParameters | None:
    """Decide si un mensaje de salida lleva quote (reply_parameters).

    - reply_to_message_id ausente → None (sin quote, sin tocar el ancla).
    - config.reply_to_message=True → siempre quote (opt-in explícito).
    - default → quote solo en el primer mensaje de la conversación (ancla).
    """
    if not reply_to_message_id:
        return None
    if self.config.reply_to_message:
        return ReplyParameters(message_id=int(reply_to_message_id),
                               allow_sending_without_reply=True)
    if chat_id in self._reply_anchored:
        return None
    self._reply_anchored.add(chat_id)
    return ReplyParameters(message_id=int(reply_to_message_id),
                           allow_sending_without_reply=True)
```

Puntos de uso (sustituyen los bloques actuales):
- `send()` (~851-857): `reply_params = self._reply_params_for(chat_id, reply_to_message_id)`.
- `_send_legacy_preview()` (~1414-1418): idem con `int_chat_id`.
- `_finalize_stream()`: en el path draft rich (~1477-1481) y en el path
  "no preview legacy" que envía mensaje nuevo (~1544-1554). Los paths de
  **edición in-place** (editMessageText / edit_message_text) no llevan quote.

Estado del ancla (en `__init__`, ~490 junto a `_stream_bufs`):

```python
self._reply_anchored: set[int] = set()  # chat_id -> ya se ancló la conversación
```

Reset del ancla:
- `_process_forward_command()` (~2088): si el comando normalizado es `/new`,
  `self._reply_anchored.discard(message.chat_id)`.
- `_on_start()` (~1792): `self._reply_anchored.discard(update.message.chat_id)`.

## Testing Strategy

- TDD (RED → GREEN) en `nanobot/channels/telegram/tests/test_telegram_channel.py`,
  estilo actual: `TelegramChannel` + `_FakeApp`/`_FakeBot`, asserts sobre
  `sent_messages` / `do_api_request` payloads.
- Tests de regresión nuevos (exigidos por el PR Guardian de madkoding):
  1. **No-streaming**: primer `send()` con `metadata={"message_id": N}` y
     `reply_to_message` ausente → `reply_parameters.message_id == N`; un segundo
     `send()` al mismo chat → `reply_parameters is None`.
  2. **Streaming legacy**: primer `send_delta` (con `message_id`) → preview con
     `reply_parameters`; un segundo stream en el mismo chat → preview **sin**
     `reply_parameters`.
  3. **Streaming rich draft**: primer `_finalize_stream` (draft) con `message_id`
     → `sendRichMessage` con `reply_parameters`; segundo stream → sin
     `reply_parameters`.
  4. **Opt-in**: `reply_to_message=True` → segundo mensaje del mismo chat **sigue**
     llevando `reply_parameters`.
  5. **Reset `/new`**: tras `_process_forward_command` con `/new`, el siguiente
     `send()` vuelve a quotea.
  6. **Reset `/start`**: tras `_on_start`, el siguiente `send()` quotea.
- Tests existentes que deben seguir verdes (verificar, no romper):
  - `test_send_reply_infers_topic_from_message_id_cache` (usa `reply_to_message=True`).
  - `test_send_delta_rich_reply_parameters_propagate_to_preview_and_final`
    (primer stream → quote; su semántica pasa a "primer mensaje quotea").
  - `test_reasoning_finalizes_draft_with_details` (primer stream → quote en draft).
  - `test_send_delta_without_rich_keeps_legacy_path` y
    `test_send_delta_stream_end_splits_oversized_reply` (sin `message_id` → sin quote).
- Verificación final: suite completa + ruff limpio.

## Boundaries

- **Always**: tests antes de commit; PR al fork `jpyunism/nanobot` (nunca commit
  directo a `madkoding`); incluir test de regresión en el PR; docstrings/comentarios
  en español.
- **Ask first**: exponer `reply_to_message` en `SETUP_SPEC`/WebUI (fuera de alcance);
  acotar `TELEGRAM_REPLY_CONTEXT_MAX_LEN` (ver Open Questions); persistir el ancla.
- **Never**: eliminar `_extract_reply_context` o el contexto `[Reply to bot: ...]`;
  cambiar el contrato de `send_delta`/`send_reasoning_*`; tocar otros canales;
  cambiar el default de `reply_to_message` (sigue `false`).

## Requisitos

| ID | Requisito | Verificación |
|----|-----------|--------------|
| REQ-001 | Helper `_reply_params_for(chat_id, reply_to_message_id)` unifica la decisión de reply y devuelve `ReplyParameters \| None` | test unitario del helper |
| REQ-002 | `config.reply_to_message=True` → siempre quote (comportamiento actual de `send()`) | test opt-in (segundo mensaje quotea) |
| REQ-003 | `config.reply_to_message=False/ausente` → quote solo en el primer mensaje de la conversación | tests 1-3 |
| REQ-004 | `send()` usa el helper (reemplaza el bloque condicional ~851-857) | test no-streaming |
| REQ-005 | `_send_legacy_preview()` usa el helper (reemplaza el bloque incondicional ~1414-1418) | test streaming legacy |
| REQ-006 | `_finalize_stream()` usa el helper en el path draft (~1477-1481) y en el path "no preview legacy" (~1544-1554); los paths de edición in-place no quotean | test streaming rich draft |
| REQ-007 | Estado `self._reply_anchored: set[int]` inicializado en `__init__` | inspección + tests |
| REQ-008 | `/new` resetea el ancla del chat en `_process_forward_command` | test reset `/new` |
| REQ-009 | `/start` resetea el ancla del chat en `_on_start` | test reset `/start` |
| REQ-010 | Reinicio del gateway resetea el ancla (set vacío en `__init__`) | inspección |
| REQ-011 | `_extract_reply_context` se mantiene sin cambios | tests existentes `test_extract_reply_context_*` verdes |
| REQ-012 | Tests de regresión cubren: primer quotea / segundo no / opt-in siempre / reset | suite verde |

## Success Criteria

- [ ] Con la config real del usuario, el primer mensaje de un chat quotea y los
      siguientes no (verificado por tests de regresión).
- [ ] `reply_to_message: true` sigue quoteando siempre.
- [ ] `/new` y `/start` reinician el ancla.
- [ ] Suite completa verde (salvo fallos preexistentes de WhatsApp por deps) + ruff limpio.
- [ ] PR abierto en `madkoding/nanobot` con test de regresión.

## Open Questions

1. **Persistencia del ancla**: se decide **no persistir**. Es un hint de UX en
   memoria; reiniciar el gateway y re-anclar es el comportamiento deseado. Persistir
   añadiría un store y riesgo de estado obsoleto (un chat anclado hace semanas nunca
   volvería a quotear). Si en el futuro se quiere, se puede derivar del historial de
   sesión en vez de un flag aparte.
2. **Alcance por topic (foros)**: el ancla es por `chat_id` (según lo pedido). En
   foros con topics, cada topic es una conversación distinta; keying por
   `(chat_id, message_thread_id)` sería más correcto pero añade complejidad. Se deja
   como mejora futura si se observa el problema en foros.
3. **Acotar `TELEGRAM_REPLY_CONTEXT_MAX_LEN`**: hoy es 4000 chars. El fix principal
   (quote solo la primera vez) ya rompe la cadena de "repetición". Reducir el cap
   (p.ej. a 1000) reduciría ruido pero podría perder contexto útil. Se deja como
   follow-up opcional, no bloqueante.
4. **Exponer `reply_to_message` en `SETUP_SPEC`/WebUI**: hoy solo se setea a mano en
   `config.json`. Fuera de alcance de este fix; requiere tocar `manifest.py` y el
   WebUI del canal.
5. **`/restart` (comando)**: reinicia el loop del agente, no el canal, por lo que el
   ancla en memoria persiste. No se trata en este fix (el "restart del gateway" sí
   resetea por `__init__`). Confirmar si `/restart` debería también resetear el ancla.
