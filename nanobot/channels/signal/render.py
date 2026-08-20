"""Signal markdown → plain text + textStyle range rendering helpers."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class _Run:
    text: str
    styles: frozenset[str] = field(default_factory=frozenset)
    opaque: bool = False  # code / table content — skip further pattern processing


_SIG_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n?([\s\S]*?)```")
_SIG_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_SIG_HEADER_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_SIG_BLOCKQUOTE_RE = re.compile(r"^>\s*(.*)$", re.MULTILINE)
_SIG_BULLET_RE = re.compile(r"^[-*]\s+", re.MULTILINE)
_SIG_OLIST_RE = re.compile(r"^(\d+)\.\s+", re.MULTILINE)
_SIG_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_SIG_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.DOTALL)
_SIG_ITALIC_RE = re.compile(
    r"(?<!\*)\*([^*\n]+)\*(?!\*)|(?<![a-zA-Z0-9_])_([^_\n]+)_(?![a-zA-Z0-9_])"
)
_SIG_STRIKE_RE = re.compile(r"~~(.+?)~~|(?<![~\w])~([^~\n]+)~(?![~\w])", re.DOTALL)
_SIG_TOKEN_RE = re.compile(r"\x00C(\d+)\x00")

# Patterns used to strip inline markdown when rendering table cells as plain
# text. Defined separately from the styling regexes above because the cell
# stripper needs a fixed, narrow subset (no single-asterisk italic, no
# single-tilde strikethrough) and benefits from each pattern's group 1 being
# the content directly.
_SIG_CELL_STRIP_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),
    (re.compile(r"__(.+?)__"), r"\1"),
    (re.compile(r"~~(.+?)~~"), r"\1"),
    (re.compile(r"`([^`]+)`"), r"\1"),
)


def _utf16_len(s: str) -> int:
    """UTF-16 code-unit length, matching Signal BodyRange semantics."""
    return len(s.encode("utf-16-le")) // 2


def _sig_strip_cell(s: str) -> str:
    """Strip inline markdown from a table cell for plain-text rendering."""
    for pattern, repl in _SIG_CELL_STRIP_PATTERNS:
        s = pattern.sub(repl, s)
    return s.strip()


def _sig_render_table(table_lines: list[str]) -> str:
    """Render a markdown pipe-table as fixed-width plain text."""

    def dw(s: str) -> int:
        return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)

    rows: list[list[str]] = []
    has_sep = False
    for line in table_lines:
        cells = [_sig_strip_cell(c) for c in line.strip().strip("|").split("|")]
        if all(re.match(r"^:?-+:?$", c) for c in cells if c):
            has_sep = True
            continue
        rows.append(cells)
    if not rows or not has_sep:
        return "\n".join(table_lines)

    ncols = max(len(r) for r in rows)
    for r in rows:
        r.extend([""] * (ncols - len(r)))
    widths = [max(dw(r[c]) for r in rows) for c in range(ncols)]

    def dr(cells: list[str]) -> str:
        return "  ".join(f"{c}{' ' * (w - dw(c))}" for c, w in zip(cells, widths))

    out = [dr(rows[0])]
    out.append("  ".join("─" * w for w in widths))
    for row in rows[1:]:
        out.append(dr(row))
    return "\n".join(out)


def _markdown_to_signal(text: str) -> tuple[str, list[str]]:
    """Convert markdown text to Signal plain text + textStyle ranges.

    Returns ``(plain_text, text_styles)`` where ``text_styles`` are
    ``"start:length:STYLE"`` strings for the signal-cli ``textStyle`` parameter.
    """
    if not text:
        return text, []

    # Phase 1 (text-level): extract code blocks and tables with placeholder tokens
    # so they're protected from inline-style processing.
    protected: list[str] = []

    def save_code(m: re.Match) -> str:
        protected.append(m.group(1))
        return f"\x00C{len(protected) - 1}\x00"

    text = _SIG_CODE_BLOCK_RE.sub(save_code, text)

    # Detect and render pipe-tables line by line.
    lines = text.split("\n")
    rebuilt: list[str] = []
    i = 0
    while i < len(lines):
        if re.match(r"^\s*\|.+\|", lines[i]):
            tbl: list[str] = []
            while i < len(lines) and re.match(r"^\s*\|.+\|", lines[i]):
                tbl.append(lines[i])
                i += 1
            rendered = _sig_render_table(tbl)
            if rendered != "\n".join(tbl):
                protected.append(rendered)
                rebuilt.append(f"\x00C{len(protected) - 1}\x00")
            else:
                rebuilt.extend(tbl)
        else:
            rebuilt.append(lines[i])
            i += 1
    text = "\n".join(rebuilt)

    # Phase 2 (run-based): process inline patterns.
    runs: list[_Run] = [_Run(text)]

    def transform(
        pattern: re.Pattern,
        make_runs: Callable[[re.Match, frozenset[str]], list[_Run]],
    ) -> None:
        new_runs: list[_Run] = []
        for run in runs:
            if run.opaque:
                new_runs.append(run)
                continue
            pos = 0
            for m in pattern.finditer(run.text):
                if m.start() > pos:
                    new_runs.append(_Run(run.text[pos : m.start()], run.styles))
                new_runs.extend(make_runs(m, run.styles))
                pos = m.end()
            if pos < len(run.text):
                new_runs.append(_Run(run.text[pos:], run.styles))
        runs[:] = new_runs

    # Restore code/table placeholders as opaque MONOSPACE runs.
    transform(
        _SIG_TOKEN_RE,
        lambda m, s: [_Run(protected[int(m.group(1))], s | {"MONOSPACE"}, opaque=True)],
    )

    # Inline code (opaque).
    transform(_SIG_INLINE_CODE_RE, lambda m, s: [_Run(m.group(1), s | {"MONOSPACE"}, opaque=True)])

    # Headers → bold plain text.
    transform(_SIG_HEADER_RE, lambda m, s: [_Run(m.group(1), s | {"BOLD"})])

    # Blockquotes → strip marker.
    transform(_SIG_BLOCKQUOTE_RE, lambda m, s: [_Run(m.group(1), s)])

    # Bullet lists → bullet character.
    transform(_SIG_BULLET_RE, lambda m, s: [_Run("• ", s)])

    # Numbered lists → normalize spacing.
    transform(_SIG_OLIST_RE, lambda m, s: [_Run(m.group(1) + ". ", s)])

    # Links → "text (url)" or bare url when text equals url.
    def _link_runs(m: re.Match, s: frozenset) -> list[_Run]:
        link_text, url = m.group(1), m.group(2)

        def _norm(u: str) -> str:
            return re.sub(r"^https?://(www\.)?", "", u).rstrip("/").lower()

        if _norm(url) == _norm(link_text):
            return [_Run(url, s)]
        return [_Run(f"{link_text} ({url})", s)]

    transform(_SIG_LINK_RE, _link_runs)

    # Bold (before italic so ** doesn't interfere).
    transform(_SIG_BOLD_RE, lambda m, s: [_Run(m.group(1) or m.group(2), s | {"BOLD"})])

    # Italic (single * or _).
    transform(_SIG_ITALIC_RE, lambda m, s: [_Run(m.group(1) or m.group(2), s | {"ITALIC"})])

    # Strikethrough: ~~text~~ (standard) or ~text~ (single-tilde variant).
    transform(_SIG_STRIKE_RE, lambda m, s: [_Run(m.group(1) or m.group(2), s | {"STRIKETHROUGH"})])

    # Phase 3: assemble output. Offsets and lengths are emitted in UTF-16 code
    # units because Signal's BodyRange (via signal-cli's textStyle) interprets
    # them as such; Python's len() counts code points, which would shift ranges
    # left by 1 unit per non-BMP character preceding them.
    plain_text = ""
    text_styles: list[str] = []
    utf16_offset = 0
    for run in runs:
        if not run.text:
            continue
        plain_text += run.text
        start = utf16_offset
        length = _utf16_len(run.text)
        utf16_offset += length
        for style in sorted(run.styles):
            text_styles.append(f"{start}:{length}:{style}")

    return plain_text, text_styles


def _partition_styles(
    plain_text: str, chunks: list[str], text_styles: list[str]
) -> list[list[str]]:
    """Partition Signal textStyle ranges across message chunks.

    ``split_message`` slices ``plain_text`` into pieces (optionally trimming
    whitespace at the boundaries), but the style ranges produced by
    ``_markdown_to_signal`` are expressed in UTF-16 offsets relative to the
    full ``plain_text``. This redistributes them per chunk with offsets
    rebased to each chunk's start. Ranges that span a boundary are split
    across the chunks they touch; ranges that fall entirely in trimmed
    whitespace are dropped.
    """
    if not chunks:
        return []
    if not text_styles:
        return [[] for _ in chunks]

    # Locate each chunk's UTF-16 start in plain_text. split_message lstrips at
    # boundaries (but not before the first chunk), so we skip whitespace
    # between chunks to mirror that.
    chunk_ranges: list[tuple[int, int]] = []
    cursor = 0  # Python codepoint cursor in plain_text
    for i, chunk in enumerate(chunks):
        if i > 0:
            while cursor < len(plain_text) and plain_text[cursor].isspace():
                cursor += 1
        utf16_start = _utf16_len(plain_text[:cursor])
        utf16_end = utf16_start + _utf16_len(chunk)
        chunk_ranges.append((utf16_start, utf16_end))
        cursor += len(chunk)

    result: list[list[str]] = [[] for _ in chunks]
    for entry in text_styles:
        s, ln, style = entry.split(":", 2)
        r_start = int(s)
        r_end = r_start + int(ln)
        for i, (c_start, c_end) in enumerate(chunk_ranges):
            if r_end <= c_start or r_start >= c_end:
                continue
            new_start = max(r_start, c_start) - c_start
            new_end = min(r_end, c_end) - c_start
            new_length = new_end - new_start
            if new_length > 0:
                result[i].append(f"{new_start}:{new_length}:{style}")
    return result
