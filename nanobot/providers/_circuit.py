"""Shared per-key circuit breaker for provider implementations."""

from __future__ import annotations

import time
from typing import Any


class CircuitBreaker:
    """Simple per-key failure-count circuit breaker.

    Records consecutive failures per key. Once ``threshold`` failures are
    reached the circuit opens and ``allow_probe`` returns False until
    ``cooldown_s`` seconds have passed (half-open: one probe allowed).
    """

    def __init__(self, threshold: int, cooldown_s: float) -> None:
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._failures: dict[Any, int] = {}
        self._tripped_at: dict[Any, float] = {}

    def record_failure(self, key: Any) -> bool:
        """Increment failure count for *key* and return True if just tripped."""
        count = self._failures.get(key, 0) + 1
        self._failures[key] = count
        if count >= self._threshold:
            self._tripped_at[key] = time.monotonic()
            return True
        return False

    def record_success(self, key: Any) -> None:
        """Reset failure state for *key*."""
        self._failures.pop(key, None)
        self._tripped_at.pop(key, None)

    def allow_probe(self, key: Any) -> bool:
        """Return False when the circuit is open and cooldown has not elapsed."""
        failures = self._failures.get(key, 0)
        if failures < self._threshold:
            return True
        tripped = self._tripped_at.get(key, 0.0)
        return (time.monotonic() - tripped) >= self._cooldown_s
