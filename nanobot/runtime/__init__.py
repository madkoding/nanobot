"""Cross-cutting process infrastructure for the nanobot runtime.

Holds the pieces that are neither configuration nor domain logic but are
needed by every surface (CLI, gateway, WebUI, SDK):

- ``context`` — optional, persistent context appended to the current user prompt.
- ``process`` — cross-platform lifecycle management for background processes.
- ``features`` — optional feature discovery and enablement (channels, extras).
"""
