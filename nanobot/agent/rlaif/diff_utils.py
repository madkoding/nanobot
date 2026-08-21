"""Utilities for extracting and normalizing unified diffs from LLM responses."""

from __future__ import annotations

import re

_DIFF_START_RE = re.compile(r"^---\s+")
_HUNK_RE = re.compile(r"@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@")


def _normalize_unified_diff(patch: str, context_lines: int = 3) -> str:
    """Rebuild a unified diff with correct hunk line counts.

    Some LLMs emit hunks whose ``oldcount``/``newcount`` headers do not match
    the actual number of lines in the hunk. This function parses the hunks,
    validates them against the declared counts, and rewrites the patch with
    accurate counts so ``git apply`` accepts it.
    """
    lines = patch.splitlines()
    output: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- ") or line.startswith("+++ "):
            output.append(line)
            i += 1
            continue
        m = _HUNK_RE.match(line)
        if not m:
            output.append(line)
            i += 1
            continue
        # Parse hunk header.
        header = line
        hunk_start = i + 1
        hunk_end = hunk_start
        while hunk_end < len(lines):
            hl = lines[hunk_end]
            if hl.startswith("--- ") or hl.startswith("+++ ") or _HUNK_RE.match(hl):
                break
            hunk_end += 1
        hunk_body = lines[hunk_start:hunk_end]
        # A line like ``\ No newline at end of file`` belongs to neither side;
        # counting it inflates old_count/new_count and makes git apply reject
        # the patch. Exclude any line starting with a backslash.
        old_count = sum(
            1
            for hunk_line in hunk_body
            if not hunk_line.startswith("+") and not hunk_line.startswith("\\")
        )
        new_count = sum(
            1
            for hunk_line in hunk_body
            if not hunk_line.startswith("-") and not hunk_line.startswith("\\")
        )
        # Extract old/new start lines from header.
        header_match = re.match(
            r"@@\s+-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@",
            header,
        )
        if header_match:
            old_start = int(header_match.group(1))
            new_start = int(header_match.group(2))
        else:
            old_start = new_start = 1
        output.append(
            f"@@ -{old_start},{old_count} +{new_start},{new_count} @@"
        )
        output.extend(hunk_body)
        i = hunk_end
    return "\n".join(output)

def extract_unified_diff(text: str) -> str:
    """Extract the first well-formed unified diff from a response.

    Handles:
    - Markdown ```diff ... ``` fences.
    - Bare diff blocks starting with `--- `.
    - Text before/after the diff.
    """
    lines = text.splitlines()

    first_diff_idx = next((i for i, line in enumerate(lines) if _DIFF_START_RE.match(line)), None)
    if first_diff_idx is not None and first_diff_idx == 0:
        return _strip_markdown_fence("\n".join(lines))

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
