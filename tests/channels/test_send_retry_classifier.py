"""Tests for the manager's send-error classifier."""

from __future__ import annotations

from nanobot.channels.manager import _is_non_retriable_send_error

# ponytail: match the real neonize.exc.SendMessageError class name
# because the manager inspects type(exc).__name__, not isinstance.
_SendMessageError = type("SendMessageError", (Exception,), {})


def test_463_send_message_error_is_non_retriable() -> None:
    exc = _SendMessageError("server returned error 463")
    assert _is_non_retriable_send_error(exc) is True


def test_429_send_message_error_is_non_retriable() -> None:
    """whatsmeow reports 429 (rate-overlimit) the same way as 463."""
    exc = _SendMessageError("server returned error 429")
    assert _is_non_retriable_send_error(exc) is True


def test_419_send_message_error_is_non_retriable() -> None:
    exc = _SendMessageError("server returned error 419")
    assert _is_non_retriable_send_error(exc) is True


def test_cooldown_active_runtime_error_is_non_retriable() -> None:
    """The channel surfaces an active 463 cooldown as a RuntimeError.
    Retrying it would only re-trigger the same gate, so the manager
    must treat it as terminal.
    """
    exc = RuntimeError("WhatsApp 463 cooldown active; 600s remaining")
    assert _is_non_retriable_send_error(exc) is True


def test_generic_send_failure_is_retriable() -> None:
    """A transient non-throttle error must NOT be marked non-retriable,
    otherwise the existing exponential-backoff loop stops working.
    """
    exc = _SendMessageError("server returned error 500")
    assert _is_non_retriable_send_error(exc) is False


def test_unrelated_runtime_error_is_retriable() -> None:
    exc = RuntimeError("connection lost")
    assert _is_non_retriable_send_error(exc) is False
