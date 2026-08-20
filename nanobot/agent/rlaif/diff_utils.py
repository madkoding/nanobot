"""Utilities for extracting and normalizing unified diffs from LLM responses."""

from __future__ import annotations

import re

_DIFF_START_RE = re.compile(r"^---\s+")
_HUNK_RE = re.compile(r"@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@")


def extract_unified_diff(text: str) -> str:
    """Extract the first well-formed unified diff from a response.

    Handles:
    - Markdown ```diff ... ``` fences.
    - Bare diff blocks starting with `--- `.
    - Text before/after the diff.
    - Multiple consecutive diff blocks are merged if they are adjacent.
    """
    lines = text.splitlines()

    # If the entire response is a diff, return as-is.
    first_diff_idx = next((i for i, line in enumerate(lines) if _DIFF_START_RE.match(line)), None)
    if first_diff_idx is not None and first_diff_idx == 0:
        return _strip_markdown_fence("\n".join(lines))

    # Look inside markdown code fences for a diff block.
    fence_start: int | None = None
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped.startswith("```") and ("diff" in stripped or "patch" in stripped):
            fence_start = i + 1
            continue
        if fence_start is not None and line.strip().startswith("```"):
            candidate = "\n".join(lines[fence_start:i])
            if "--- " in candidate and "+++ " in candidate:
                return candidate
            fence_start = None

    # Fall back to first `--- ` line.
    if first_diff_idx is not None:
        block = "\n".join(lines[first_diff_idx:])
        return _strip_markdown_fence(block)

    return text


def _strip_markdown_fence(text: str) -> str:
    lines = text.splitlines()
    while lines and lines[-1].strip() == "```":
        lines.pop()
    if lines and lines[-1].endswith("```"):
        lines[-1] = lines[-1].rstrip().rstrip("`")
    return "\n".join(lines).rstrip("\n")


def summarize_unified_diff(patch: str) -> str:
    """Return a short summary of which files the patch touches."""
    files: list[str] = []
    for line in patch.splitlines():
        if line.startswith("+++ "):
            path = line[4:].split("\t")[0].strip()
            if path and path not in files:
                files.append(path)
    if not files:
        return "patch"
    if len(files) <= 3:
        return f"patch touching {', '.join(files)}"
    return f"patch touching {len(files)} files ({', '.join(files[:3])}...)"


def is_valid_unified_diff(patch: str) -> bool:
    """Cheap sanity check that the patch looks like a unified diff."""
    if "--- " not in patch or "+++ " not in patch:
        return False
    if not _HUNK_RE.search(patch):
        return False
    return True
