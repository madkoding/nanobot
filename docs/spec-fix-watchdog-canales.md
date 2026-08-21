# Spec: Fix watchdog de canales — reinicio en loop y canales muertos sin reintento

Fecha: 2026-08-20 · Rama: `fix/telegram-watchdog-restart` · Upstream: `madkoding/nanobot` (main)

## Objetivo

El watchdog de canales (`ChannelManager._watchdog_loop`) tiene dos fallas que
dejaron el canal Telegram del usuario caído ~2h20m (2026-08-19 10:11 → 12:31) y
generan ~13.000 warnings "live but silent" en el log:

1. **Loop de reinicios cada 60s**: un canal sano pero sin tráfico (polling
   inactivo >10 min) se reinicia en loop infinito porque `_start_channel` nunca
   resetea `last_activity_at`.
2. **Canal muerto sin reintento**: cuando un reinicio falla (p.ej. timeout de red
   en `getMe`), el task de `start()` termina y el watchdog lo saltea para siempre
   (`if task is None or task.done(): continue`). El canal queda registrado con
   `is_running=True` (se setea al inicio de `start()`, antes de `initialize()`),
   el manager cree que está vivo, y nadie lo reintenta hasta un restart del
   gateway o un toggle manual en el WebUI.

**Usuario**: JuanPa (Telegram, @jpyunism). Incidente real: 2026-08-19 10:11:32
`telegram.error.TimedOut` en `getMe` durante un restart del watchdog → canal
caído hasta 12:31:28 (toggle manual en WebUI).

**Éxito (acceptance criteria)**:
1. Un canal sano pero inactivo NO se reinicia en loop: tras un restart del
   watchdog, `last_activity_at` se resetea y el canal tiene otros 600s de
   gracia; sin tráfico real no se vuelve a tocar.
2. Un canal cuyo `start()` falló (task done + error registrado) se reintenta
   automáticamente cada `WATCHDOG_RETRY_INTERVAL_S` (60s), sin intervención
   manual, hasta que conecta.
3. Tras un `start()` fallido, `is_running` refleja la realidad (`False`) y un
   `stop()` posterior no lanza `RuntimeError: This Updater is not running!`.
4. El canal Telegram del usuario se recupera solo tras un blip de red (caso del
   incidente) en ≤ ~2 min, sin tocar el WebUI.
5. Tests de regresión RED→GREEN para los tres puntos (watchdog, retry, stop
   seguro).

## Comportamiento actual (raíz)

- `_start_channel` (manager.py ~376-388): `errors.pop(name)` + `await
  channel.start()`; no toca `last_activity_at`. El canal arranca con el valor
  viejo (o 0.0 si es la primera vez). Con `last_activity_at` viejo >600s, el
  watchdog lo reinicia al minuto siguiente, y el loop se perpetúa porque el
  restart tampoco resetea el timer.
- `_watchdog_loop` (manager.py ~614-643): `if task is None or task.done():
  continue` — un canal cuyo start falló queda huérfano para siempre. Además
  `channel.is_running` sigue `True` porque `TelegramChannel.start()` setea
  `self._running = True` (línea 287) antes de `initialize()` y no lo limpia en
  el `except`.
- `TelegramChannel.stop()` (runtime.py ~403-425): llama `updater.stop()` sin
  verificar si el updater llegó a arrancar; PTB lanza `RuntimeError("This
  Updater is not running!")` cuando `initialize()` falló antes de
  `start_polling()` (visto en el log 2026-08-20 12:31:28).
- El watchdog es idéntico en `upstream/main` (verificado) — el bug existe
  también upstream.

## Comportamiento nuevo

| # | Caso | Antes | Después |
|---|------|-------|---------|
| 1 | Canal sano inactivo >10 min | reinicio cada 60s en loop infinito | un restart, luego 600s de gracia; sin tráfico no se toca |
| 2 | `start()` falla (timeout, red) | canal muerto para siempre hasta restart manual | reintento automático cada 60s |
| 3 | `is_running` tras start fallido | `True` (mentira) | `False` |
| 4 | `stop()` sobre canal medio-inicializado | `RuntimeError: This Updater is not running!` | no-op seguro |
| 5 | Recuperación tras blip de red | manual (WebUI/restart) | automática ≤ ~2 min |

## Tech Stack

- Python 3.11+, asyncio.
- pytest + pytest-asyncio (`asyncio_mode = "auto"`).
- Lint: ruff (E, F, I, N, W; E501 ignorado).

## Comandos

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/jyunis/.nanobot/workspace/nanobot
uv run pytest nanobot/channels/telegram/tests/test_telegram_channel.py nanobot/channels/manager_tests -q  # o ruta de tests del manager
uv run ruff check nanobot/channels/manager.py nanobot/channels/telegram/runtime.py
```

## Decisiones de diseño

<details>
<summary>D1: dónde resetear last_activity_at</summary>

En `_start_channel` (manager), no en `start()` del canal: el watchdog es el
consumidor del timer y el manager es quien orquesta los restarts. Un reset en
`_start_channel` cubre tanto el arranque inicial como cada restart del watchdog
y del WebUI. Alternativa descartada: resetear dentro de `TelegramChannel.start()`
— funcionaría, pero deja la política de liveness en el canal y el manager no
podría garantizarla para canales de terceros.
</details>

<details>
<summary>D2: reintento de canales fallidos</summary>

El watchdog pasa a tratar los canales con task done como candidatos a retry:
si el canal está en `_channel_errors` (o el task terminó sin error registrado),
se reintenta `_start_channel_task` cada `WATCHDOG_RETRY_INTERVAL_S` (60s). Se
reusa el mismo loop (sin task extra) para mantener un solo punto de orquestación.
El `_channel_errors` se limpia al reintentar (ya lo hace `_start_channel`).
</details>

<details>
<summary>D3: stop seguro en TelegramChannel</summary>

`stop()` verifica `self._app` y el estado del updater antes de llamar
`updater.stop()`; si el canal nunca llegó a `start_polling`, se salta el stop
del updater y solo se limpia `_app`. Además `start()` limpia `_running = False`
y `_app = None` en el `except` para que `is_running` sea honesto y un stop
posterior sea no-op. Esto también evita el `RuntimeError` visto en el log.
</details>

## Alcance

- `nanobot/channels/manager.py`: watchdog + `_start_channel`.
- `nanobot/channels/telegram/runtime.py`: `start()`/`stop()`.
- Tests: manager (watchdog/retry) + telegram (stop seguro).
- Sin cambios de config ni de API pública.
