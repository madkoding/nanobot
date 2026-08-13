"""Tests for nanobot.providers._circuit.CircuitBreaker."""

import time

import pytest

from nanobot.providers._circuit import CircuitBreaker


@pytest.fixture()
def cb():
    return CircuitBreaker(threshold=3, cooldown_s=10)


def test_allow_probe_when_no_failures(cb):
    assert cb.allow_probe("key") is True


def test_allow_probe_below_threshold(cb):
    cb.record_failure("key")
    cb.record_failure("key")
    assert cb.allow_probe("key") is True


def test_allow_probe_above_threshold(cb):
    for _ in range(3):
        cb.record_failure("key")
    assert cb.allow_probe("key") is False


def test_record_failure_returns_true_when_tripped(cb):
    assert cb.record_failure("key") is False
    assert cb.record_failure("key") is False
    assert cb.record_failure("key") is True


def test_success_resets_failures(cb):
    cb.record_failure("key")
    cb.record_failure("key")
    cb.record_success("key")
    cb.record_failure("key")
    assert cb.allow_probe("key") is True


def test_keys_are_independent(cb):
    cb.record_failure("a")
    cb.record_failure("a")
    cb.record_failure("a")
    assert cb.allow_probe("a") is False
    assert cb.allow_probe("b") is True


def test_probe_after_cooldown(cb, monkeypatch):
    cb.record_failure("key")
    cb.record_failure("key")
    cb.record_failure("key")
    cb._tripped_at["key"] = time.monotonic() - 11
    assert cb.allow_probe("key") is True
