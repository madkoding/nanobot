"""Deprecated shim for :mod:`nanobot.runtime.context`.

Moved to ``nanobot/runtime/context.py``. This module exists only so existing
imports keep working during the transition; it will be removed in a later
release.
"""

from nanobot.runtime.context import *  # noqa: F401,F403
from nanobot.runtime.context import (  # noqa: F401
    RUNTIME_CONTEXT_END,
    RUNTIME_CONTEXT_HISTORY_META,
    RUNTIME_CONTEXT_INPUT_META,
    RUNTIME_CONTEXT_MESSAGE_META,
    RUNTIME_CONTEXT_TAG,
    RuntimeContextBlock,
    RuntimeContextProvider,
    RuntimeContextResult,
    append_runtime_context,
    compile_project_context,
    detach_runtime_context,
    normalize_runtime_context_blocks,
    normalize_webui_quote,
    public_history_message,
    public_history_messages,
    reattach_runtime_context,
    resolve_runtime_context,
    runtime_context_blocks_from_metadata,
    webui_quote_runtime_context,
    wrap_runtime_context_lines,
)
