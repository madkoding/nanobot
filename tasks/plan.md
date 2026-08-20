# Plan: Telegram — Message effects opt-in (sin confeti por defecto)

Spec: `docs/spec-telegram-message-effects-opt-in.md`

## Componentes y dependencias

```
TelegramChannel (runtime.py)
   └── _resolve_message_effect()   → único cambio funcional: falsy → None (sin default confeti)

Tests (nanobot/channels/telegram/tests/test_telegram_channel.py)
   ├── test_send_effect_applies_message_effect_id_rich          → intacto (override explícito)
   ├── test_send_effect_applies_message_effect_id_legacy        → intacto (override explícito)
   ├── test_send_effect_config_default_applies_when_no_override → ACTUALIZAR: config explícita
   ├── test_send_effect_bad_request_retries_without_effect      → intacto (retry best-effort)
   └── test_send_without_effect_omits_message_effect_id         → NUEVO (regresión, RED primero)

Docs
   ├── docs/spec-telegram-message-effects-opt-in.md             → spec (aprobada)
   └── docs/spec-telegram-ux-checklists-polls-effects.md        → actualizar D3/REQ-007 (default ya no es confeti)
```

## Orden de implementación

### T1: TDD (RED) — test de regresión nuevo
- `TelegramChannel.send()` con `OutboundMessage` sin `effect` y `TelegramConfig`
  sin `message_effect_id` → ningún payload (rich ni legacy) lleva
  `message_effect_id`.
- Estado RED: falla porque hoy el default es confeti.

### T2: Implementación (GREEN)
- `_resolve_message_effect`: `if not effect: return None` (en vez de confeti).
- Overrides por nombre (`confeti`) e id crudo passthrough intactos.
- BadRequest → retry sin efecto intacto.

### T3: Tests existentes + docs
- `test_send_effect_config_default_applies_when_no_override` pasa a setear
  `message_effect_id="confeti"` explícito en el config (mismo nombre, otro caso).
- Actualizar D3/REQ-007 en `spec-telegram-ux-checklists-polls-effects.md`.

### T4: Verificación y release
- `uv run pytest nanobot/channels/telegram/tests/test_telegram_channel.py -q`
- `uv run pytest -q` (suite completa; 16 failed WhatsApp preexistentes OK)
- `uv run ruff check`
- Commit conventional + push al fork + PR a `madkoding/nanobot` (test de regresión obligatorio para el PR Guardian).
- Sync a site-packages (pyenv 3.13.3 + uv tool) con md5; avisar reinicio manual del gateway.

## Riesgos y mitigaciones

- **Riesgo**: tocar el default rompe la celebración de aprobaciones de specs
  (flujo SDD). **Mitigación**: el override `effect="confeti"` del tool `message`
  queda intacto (REQ-002); si el usuario quiere confeti para aprobaciones, se
  setea `message_effect_id` en config (REQ-003).
- **Riesgo**: el PR Guardian exige test → cubierto con el test de regresión T1.
- **Riesgo**: site-packages stale → sync con md5 antes de avisar reinicio.

## Verification checkpoints

1. T1 en RED (el test nuevo falla solo por el default).
2. T2 en GREEN (test nuevo pasa; los 4 de effect existentes pasan).
3. T4: suite completa verde (sin regresiones nuevas) + ruff limpio.
4. PR mergeable en madkoding; md5 de los 3 archivos factory/runtime idénticos.
