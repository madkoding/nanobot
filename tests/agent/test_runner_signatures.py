"""Unit tests for ``_content_signature``.

The runner's repetition detector relies on this fingerprint to spot a model
that is stuck emitting the same content across iterations. A simple
prefix-only fingerprint misses the "model is repeating one word many
times" pattern because each iteration's output grows in length without
changing the head; this file tests is for that regression.
"""

from __future__ import annotations

from nanobot.agent.runner_signatures import _content_signature


def test_signature_empty_inputs_return_none():
    assert _content_signature(None, None) is None
    assert _content_signature("", "") is None
    assert _content_signature("   \n  ", None) is None


def test_signature_normalizes_whitespace_and_case():
    a = _content_signature(None, "Hello   World\n\nFoo")
    b = _content_signature(None, "hello world foo")
    assert a is not None
    assert a == b


def test_signature_combines_reasoning_and_content():
    a = _content_signature("reasoning text", "answer")
    b = _content_signature(None, "reasoning text\nanswer")
    assert a is not None
    assert a == b


def test_signature_short_input_returns_just_head():
    """Inputs smaller than the head slice have no tail component."""
    sig = _content_signature(None, "short answer")
    assert sig is not None
    assert "|" not in sig


def test_signature_long_input_includes_separator():
    """Inputs larger than the head slice, where no single token dominates,
    expose a head|tail join so a repeated tail does not get hidden behind
    a moving prefix.
    """
    # Build a long blob with many distinct tokens so no single one dominates.
    base = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
        "nu xi omicron pi rho sigma tau upsilon phi chi psi omega "
    )
    long = base * 10 + "ZAP at the end"   # ~920 chars, no dominant token
    normalized = " ".join(long.split()).strip().lower()
    sig = _content_signature(None, long)
    assert sig is not None
    assert "|" in sig
    assert not sig.startswith("run:")
    head, tail = sig.split("|", 1)
    assert normalized.startswith(head)
    assert normalized.endswith(tail)


def test_signature_detects_growing_single_word_repetition():
    """Regression: two growing runs of the same word must share a fingerprint,
    otherwise the runner's repetition detector misses a model that is
    looping on one word. Both inputs collapse to the ``run:<token>`` form.
    """
    short = " ".join(["ok"] * 200)   # 599 chars
    long = " ".join(["ok"] * 400)     # 1199 chars
    sig_short = _content_signature(None, short)
    sig_long = _content_signature(None, long)
    assert sig_short is not None
    assert sig_long is not None
    assert sig_short == sig_long
    assert sig_short.startswith("run:ok")


def test_signature_distinguishes_different_tail_words():
    """When the prose has no dominant token, different tail words still
    produce different signatures.
    """
    prefix = "first sentence about tasks. second sentence about plans. "
    # Pad the prefix so it exceeds _RUN_MIN_BLOB_CHARS without becoming
    # dominant on its own.
    prefix += "alpha beta gamma delta epsilon zeta eta theta " * 6
    sig_a = _content_signature(None, prefix + "ZAP")
    sig_b = _content_signature(None, prefix + "ZOP")
    assert sig_a is not None
    assert sig_b is not None
    assert not sig_a.startswith("run:")
    assert not sig_b.startswith("run:")
    assert sig_a != sig_b


def test_signature_distinguishes_different_head_words():
    """Two long runs of different words must produce different signatures."""
    sig_a = _content_signature(None, "ZAP " * 100)
    sig_b = _content_signature(None, "ZOP " * 100)
    assert sig_a is not None
    assert sig_b is not None
    assert sig_a.startswith("run:zap")
    assert sig_b.startswith("run:zop")
    assert sig_a != sig_b


def test_signature_detects_growing_no_space_repetition():
    """Regression: ``"ok" * 200`` and ``"ok" * 500`` glued together (no
    whitespace) must share a fingerprint, otherwise the runner misses a
    model that loops on one word without separators.
    """
    short = "ok" * 200   # 400 chars
    long = "ok" * 500    # 1000 chars
    sig_short = _content_signature(None, short)
    sig_long = _content_signature(None, long)
    assert sig_short is not None
    assert sig_long is not None
    assert sig_short == sig_long
    # The dominant-token component must be present.
    assert "run:ok" in sig_short
    assert "run:ok" in sig_long


def test_signature_detects_growing_long_token_repetition():
    """A multi-character token (``"hello"``) glued together also triggers
    the dominant-token signal when its character share passes the ratio.
    """
    blob_a = "hello" * 80   # 400 chars, all "hello"
    blob_b = "hello" * 150  # 750 chars, all "hello"
    sig_a = _content_signature(None, blob_a)
    sig_b = _content_signature(None, blob_b)
    assert sig_a is not None
    assert sig_b is not None
    assert sig_a == sig_b


def test_signature_short_dominant_token_does_not_trigger():
    """Inputs where the blob is short must not trigger the dominant-token
    rule — avoids false positives on short replies like ``"answer-1"``
    where one common word trivially covers most of the content.
    """
    sig = _content_signature(None, "answer-1")
    assert sig is not None
    assert not sig.startswith("run:")


def test_signature_normal_prose_does_not_trigger_dominant():
    """Normal prose where no single token covers >60% of the blob keeps
    the head|tail fingerprint.
    """
    normal = "The quick brown fox jumps over the lazy dog " * 20  # ~860 chars
    sig = _content_signature(None, normal)
    assert sig is not None
    assert not sig.startswith("run:")


def test_signature_distinguishes_when_dominant_token_changes():
    sig_a = _content_signature(None, "ok" * 500)
    sig_b = _content_signature(None, "no" * 500)
    assert sig_a is not None
    assert sig_b is not None
    assert sig_a != sig_b
