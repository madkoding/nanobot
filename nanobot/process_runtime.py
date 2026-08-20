"""Deprecated shim for :mod:`nanobot.runtime.process`.

Moved to ``nanobot/runtime/process.py``. This module exists only so existing
imports keep working during the transition; it will be removed in a later
release.
"""

from nanobot.runtime.process import *  # noqa: F401,F403
from nanobot.runtime.process import (  # noqa: F401
    ManagedProcessRuntime,
    ProcessResult,
    ProcessRuntimePaths,
    ProcessStartOptions,
    ProcessStatus,
)
