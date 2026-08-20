"""Backend helpers for the bundled WebUI surface.

Serves the React WebUI over the gateway's WebSocket HTTP surface. This is the
backend, not the frontend; the React source lives in ``webui/src/``.

See ``nanobot/webui/README.md`` for a domain index of the modules in this
package and the two dispatch entry points (``ws_http``, ``settings_routes``).
"""
