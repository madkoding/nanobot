"""Memory system: pure file I/O store and lightweight Consolidator.

This module is a thin facade that re-exports the implementations split into
:mod:`nanobot.agent.memory_store` and :mod:`nanobot.agent.memory_consolidator`
so existing import sites keep working unchanged.
"""

from nanobot.agent.memory_consolidator import Consolidator
from nanobot.agent.memory_store import (
    _ARCHIVE_SUMMARY_MAX_CHARS,
    _HISTORY_ENTRY_HARD_CAP,
    _RAW_ARCHIVE_MAX_CHARS,
    DreamRunProgress,
    MemoryStore,
)

__all__ = [
    "Consolidator",
    "DreamRunProgress",
    "MemoryStore",
    "_ARCHIVE_SUMMARY_MAX_CHARS",
    "_HISTORY_ENTRY_HARD_CAP",
    "_RAW_ARCHIVE_MAX_CHARS",
]
