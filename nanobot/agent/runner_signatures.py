"""Tool-call/content signature heuristics for repetition detection (extracted from runner.py)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from nanobot.providers.base import ToolCallRequest

# tool-exit-code field when the tool contract grows one.
_RETRYABLE_TOOL_ERROR_MARKERS = (
    "concurrency limit reached",
    "spawn subagent",
    "timeout",
    "timed out",
    "connection refused",
    "connection reset",
    "temporarily unavailable",
    "rate limit",
    "429",
    "503",
    "502",
    "deadline exceeded",
    "exit code 124",
    "exit code 137",
    "killed",
    "no available slot",
)


def _looks_like_retryable_tool_error(content: Any) -> bool:
    """Cheap substring check: does this tool result look like a transient
    failure worth re-prompting over? Avoids the false positive where a
    tool returned a legitimate validation rejection (e.g. goal already
    completed) and the model correctly stopped retrying.
    """
    if not isinstance(content, str):
        return False
    head = content.strip().lower()[:500]
    return any(marker in head for marker in _RETRYABLE_TOOL_ERROR_MARKERS)


def _tool_call_signature(tool_calls: list[ToolCallRequest]) -> str:
    """Canonical signature for a batch of tool calls to detect repetition.

    Uses (tool_name, target) where target is the primary path/command the tool
    operates on. Variations in secondary args (limit, offset, pattern, force)
    don't change the signature, so the model can't evade detection by tweaking
    a minor arg on the same file/command.
    """
    parts: list[str] = []
    for tc in tool_calls:
        parts.append(f"{tc.name}:{_tool_target(tc)}")
    parts.sort()
    return "|".join(parts)


def _tool_target(tc: ToolCallRequest) -> str:
    """Extract the primary target (path/command) from a tool call's args."""
    args = tc.arguments
    if not isinstance(args, dict):
        return str(args)
    # path-bearing tools: read_file, edit_file, list_dir, find_files, write_file
    path = args.get("path")
    if isinstance(path, str):
        return path
    # exec / shell tools
    command = args.get("command")
    if isinstance(command, str):
        return command
    # web tools
    url = args.get("url")
    if isinstance(url, str):
        return url
    # fallback: all args canonicalized
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(args)


def _content_signature(reasoning: str | None, content: str | None) -> str | None:
    """Canonical fingerprint for an assistant's visible output.

    Combines reasoning and content so repetition in either is detected.
    Strips whitespace, lowercases, and caps length to keep comparisons cheap.
    Returns None when there is nothing to fingerprint (blank assistant turn).

    Combines three signals so a model looping on one word is caught
    regardless of spacing:

    * ``head`` — first ``_HEAD_CONTENT_SIGNATURE_CHARS`` characters.
    * ``tail`` — last ``_TAIL_CONTENT_SIGNATURE_CHARS`` characters.
    * ``run`` — when one alphanumeric token dominates the character budget
      past ``_RUN_DOMINANCE_RATIO``, the signature collapses to just that
      token (length-capped). This catches ``"okokokok..."`` with no
      separators, where the growing blob would otherwise shift the tail
      through a different slice every iteration and evade detection.

    The three signals are combined with ``|`` separators so equality on the
    full string requires equality on every component.
    """
    blob = f"{reasoning or ''}\n{content or ''}"
    blob = blob.strip().lower()
    if not blob:
        return None
    # collapse whitespace so formatting drift does not evade detection
    blob = " ".join(blob.split())
    if len(blob) >= _RUN_MIN_BLOB_CHARS:
        dominant = _dominant_run_token(blob)
        if dominant is not None:
            # Extreme repetition: signature is the dominant token alone so
            # two runs of "ok...ok" with different lengths share a fingerprint.
            return f"run:{dominant}"
    head = blob[:_HEAD_CONTENT_SIGNATURE_CHARS]
    if len(blob) <= _HEAD_CONTENT_SIGNATURE_CHARS:
        return head
    tail = blob[-_TAIL_CONTENT_SIGNATURE_CHARS:]
    return f"{head}|{tail}"


def _dominant_run_token(blob: str) -> str | None:
    """Return the most-repeated alphanumeric token if its total character
    coverage exceeds ``_RUN_DOMINANCE_RATIO`` of *blob*, else ``None``.

    A token is a maximal alphanumeric run (``[a-z0-9_]+``). The function
    sums the character counts of *all* occurrences of the most frequent
    token (not just the longest single run), so ``"ok ok ok ..."`` with
    whitespace and ``"okokokok..."`` without both register the same way.
    Returning that token makes the signature invariant under growth: every
    iteration shares the same dominant token until the model breaks out of
    the loop.
    """
    if not blob:
        return None
    counts: dict[str, int] = {}
    current_chars = 0
    current_token: str | None = None
    for ch in blob:
        if ch.isalnum() or ch == "_":
            current_chars += 1
            if current_token is None:
                current_token = ch
            else:
                current_token += ch
        else:
            if current_token is not None:
                counts[current_token] = counts.get(current_token, 0) + current_chars
            current_token = None
            current_chars = 0
    if current_token is not None:
        counts[current_token] = counts.get(current_token, 0) + current_chars
    if not counts:
        return None
    best_token = max(counts, key=counts.get)
    best_chars = counts[best_token]
    if best_chars == 0 or best_chars / len(blob) <= _RUN_DOMINANCE_RATIO:
        return None
    return best_token[:_MAX_DOMINANT_TOKEN_CHARS]


_HEAD_CONTENT_SIGNATURE_CHARS = 300
_TAIL_CONTENT_SIGNATURE_CHARS = 200
_RUN_MIN_BLOB_CHARS = 200
_RUN_DOMINANCE_RATIO = 0.6
_MAX_DOMINANT_TOKEN_CHARS = 64


# Maximum characters of a tool result to hash for the action-observation
# detector. Large results are truncated before hashing to keep the comparison
# cheap while still catching meaningful changes in the head of the output.
_MAX_RESULT_HASH_CHARS = 50_000


def _hash_tool_results(results: list[Any]) -> str:
    """Return a stable, short fingerprint for a batch of tool results.

    Uses SHA-256 over a canonical JSON serialization. Non-serializable values
    fall back to ``str()``. Truncates very large results before hashing so a
    multi-MB tool output does not dominate CPU/memory. The hash is stable enough
    to detect "the same result again" while cheap enough to compute every turn.
    """
    try:
        blob = json.dumps(results, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        blob = str(results)
    if len(blob) > _MAX_RESULT_HASH_CHARS:
        blob = blob[:_MAX_RESULT_HASH_CHARS]
    return hashlib.sha256(blob.encode("utf-8", errors="replace")).hexdigest()[:32]


def _is_alternating_pattern(history: list[str], threshold: int) -> bool:
    """Detect an A->B->A->B alternating tool-call pattern.

    Returns True when the last ``threshold`` signatures alternate perfectly
    between two distinct signatures: [A, B, A, B, A, B] for threshold=6.
    A threshold below 4 is treated as disabled (no pattern possible).
    """
    if threshold < 4 or len(history) < threshold:
        return False
    recent = history[-threshold:]
    if len(set(recent)) != 2:
        return False
    # even indices must match each other, odd indices must match each other
    return all(recent[i] == recent[i + 2] for i in range(threshold - 2))
