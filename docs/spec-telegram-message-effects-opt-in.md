# Spec: Telegram — Message effects opt-in (sin confeti por defecto)

Fecha: 2026-08-15 · Rama: `feature/telegram-no-default-confetti` · Upstream: `madkoding/nanobot` (main, v0.5.0)

## Objetivo

El bot de Telegram deja de aplicar confeti automáticamente a **cada** mensaje.
El efecto de mensaje (Bot API 10.2) pasa a ser **opt-in**: solo se aplica
cuando el agente lo pide explícitamente (`effect="confeti"` en el tool
`message`) o cuando el canal tiene `message_effect_id` configurado en
`config.json`.

**Usuario**: JuanPa (Telegram, @jpyunism). Hoy cada respuesta del bot llega con
animación de confeti y resulta molesto.

**Éxito**: con la config actual del usuario (sin `message_effect_id`), ningún
mensaje del bot lleva `message_effect_id`; el confeti sigue disponible como
celebración puntual vía override.

## Comportamiento actual (raíz)

- `TelegramChannel._resolve_message_effect(None)` → `_MESSAGE_EFFECT_CONFETI`
  (`5046509860389126442`). Es decir: sin override ni config → **confeti**.
- `send()` aplica `getattr(msg, "effect", None) or self.config.message_effect_id`
  y resuelve con `_resolve_message_effect` → como la config del usuario no
  define `message_effect_id`, todo mensaje cae al confeti.
- Origen: `docs/spec-telegram-ux-checklists-polls-effects.md` (D3, REQ-007:
  "config `message_effect_id` por canal, default confeti").

## Comportamiento nuevo

| # | Caso | Antes | Después |
|---|------|-------|---------|
| 1 | Sin `effect` y sin `config.message_effect_id` | confeti | **sin efecto** |
| 2 | `effect="confeti"` explícito (tool `message`) | confeti | confeti (override intacto) |
| 3 | `effect=<id crudo>` (p.ej. `5046509860389126442`) | passthrough | passthrough (intacto) |
| 4 | `config.message_effect_id="confeti"` set en canal | confeti | confeti (default opt-in por canal) |
| 5 | BadRequest "effect not supported" | reintento sin efecto | reintento sin efecto (intacto) |

## Comandos

```
Test canal:   uv run pytest nanobot/channels/telegram/tests/test_telegram_channel.py -q
Suite:        uv run pytest -q            (5759 passed / 16 failed WhatsApp por deps — preexistentes)
Lint:         uv run ruff check nanobot/channels/telegram/runtime.py nanobot/channels/telegram/tests/test_telegram_channel.py
Sync:         cp factory/runtime → site-packages pyenv 3.13.3 + uv tool (md5 verify)
Deploy:       reinicio manual del gateway por el usuario (el agente NO reinicia)
```

## Project Structure

```
nanobot/channels/telegram/runtime.py        → _resolve_message_effect (único cambio funcional)
nanobot/channels/telegram/tests/test_telegram_channel.py → tests (TDD)
docs/spec-telegram-message-effects-opt-in.md → esta spec
docs/spec-telegram-ux-checklists-polls-effects.md → actualizar D3/REQ-007 (default ya no es confeti)
tasks/plan.md, tasks/todo.md                → plan y tareas del flujo SDD
```

## Code Style

- Python, docstrings y comentarios en español (regla del repo), conventional commits.
- Cambio quirúrgico: 1 función + comentarios, sin refactors.

## Testing Strategy

- TDD (RED → GREEN) en `tests/agent/test_runner_fallback.py` no aplica; los
  tests del canal viven en `nanobot/channels/telegram/tests/test_telegram_channel.py`
  (estilo actual: `TelegramChannel` + `_FakeApp`, asserts sobre payloads).
- Nuevo test: default sin efecto. Tests existentes: actualizar el que asume
  default confeti; los de override explícito y retry quedan igual.
- Verificación final: suite completa + ruff + md5 de site-packages.

## Boundaries

- **Always**: tests antes de commit; PR al fork `jpyunism/nanobot` (nunca commit directo a madkoding); sync a ambas site-packages con md5; avisar reinicio manual.
- **Ask first**: cambiar el tool `message` / `OutboundMessage` (no hace falta: el override ya existe); aplicar efectos en el path de streaming (`send_delta` no usa efectos hoy — fuera de alcance).
- **Never**: secrets en el PR; tocar `fallbackModels`/presets (feature separada, PR #34 ya abierto); revertir el guard de reasoning/watchdog (PR #32).

## Requisitos

| ID | Requisito | Verificación |
|----|-----------|--------------|
| REQ-001 | `_resolve_message_effect(None)` devuelve `None` (sin efecto por defecto) | test unitario nuevo |
| REQ-002 | Override por mensaje (`effect` nombre o id crudo) sigue aplicando efecto en rich y legacy | tests existentes `test_send_effect_applies_message_effect_id_rich/legacy` verdes |
| REQ-003 | `config.message_effect_id` seteado se aplica como default del canal | test config default actualizado (set explícito) |
| REQ-004 | Reintento best-effort sin efecto ante BadRequest se mantiene | `test_send_effect_bad_request_retries_without_effect` verde |
| REQ-005 | Con config sin `message_effect_id`, ningún envío lleva `message_effect_id` en payload | test de regresión nuevo (rich + legacy) |
| REQ-006 | Docs del repo actualizadas (spec UX existente + esta spec) | grep "default confeti" sin hits en código/docs |

## Success Criteria

- [ ] Con la config real del usuario, el bot responde **sin** confeti.
- [ ] `message` con `effect="confeti"` sigue celebrando.
- [ ] Suite: 5759+ passed / 16 failed (WhatsApp, preexistentes); ruff limpio.
- [ ] Site-packages (pyenv + uv tool) sincronizadas (md5 idénticos); gateway listo para reinicio manual.
- [ ] PR abierto en `madkoding/nanobot` con test de regresión.

## Open Questions

- Ninguna bloqueante. Si más adelante quieres confeti solo para aprobaciones
  (specs/polls), se configura `message_effect_id` por canal o se agrega el
  override en el flujo de specs.
