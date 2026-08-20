"""Process helpers shared across the agent/tools layer."""

from __future__ import annotations

import os

from loguru import logger


def _reap_pid(pid: int) -> None:
    """Best-effort ``waitpid`` to reap a child and prevent zombies.

    Call this after killing or after normal completion of any subprocess
    as a safety net — asyncio's child-watcher *should* have reaped it,
    but in containers / edge-cases it sometimes doesn't.

    Uses ``os`` capability checks rather than a platform flag so this is
    safe when tests patch the platform flag while still running on Windows
    (``os.waitpid`` / ``os.WNOHANG`` do not exist there).
    """
    waitpid = getattr(os, "waitpid", None)
    wnohang = getattr(os, "WNOHANG", None)
    if waitpid is None or wnohang is None:
        return
    try:
        waitpid(pid, wnohang)
    except (ProcessLookupError, ChildProcessError):
        # Already reaped, or not our child — both are fine.
        pass
    except OSError as exc:
        logger.debug("_reap_pid({}): {}", pid, exc)
