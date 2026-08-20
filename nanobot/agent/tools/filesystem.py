"""File system tools: read, write, edit, list."""

import difflib
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.file_state import FileStates, _hash_file, current_file_states
from nanobot.agent.tools.path_utils import resolve_workspace_path
from nanobot.agent.tools.schema import (
    BooleanSchema,
    IntegerSchema,
    StringSchema,
    tool_parameters_schema,
)
from nanobot.config_base import Base
from nanobot.security.workspace_access import current_tool_workspace, current_workspace_scope
from nanobot.utils.helpers import build_image_content_blocks, detect_image_mime

_IS_LINUX = sys.platform.startswith("linux")


class FileToolsConfig(Base):
    """Filesystem tools configuration."""

    enable: bool = True  # built-in file tools on by default


class _FsTool(Tool):
    """Shared base for filesystem tools — common init and path resolution."""

    config_key = "file"

    @classmethod
    def enabled(cls, ctx: Any) -> bool:
        return ctx.config.file.enable

    def __init__(
        self,
        workspace: Path | None = None,
        allowed_dir: Path | None = None,
        extra_allowed_dirs: list[Path] | None = None,
        extra_read_allowed_dirs: list[Path] | None = None,
        extra_write_allowed_dirs: list[Path] | None = None,
        extra_write_allowed_files: list[Path] | None = None,
        file_states: FileStates | None = None,
        restrict_to_workspace: bool | None = None,
        sandbox_restricts_workspace: bool = False,
        extra_read_allowed_files: list[Path] | None = None,
    ):
        self._workspace = workspace
        self._allowed_dir = allowed_dir
        # Legacy alias: extra_allowed_dirs is read-only. Write-capable tools
        # must opt in via extra_write_allowed_dirs.
        self._extra_read_allowed_dirs = [
            *(extra_allowed_dirs or []),
            *(extra_read_allowed_dirs or []),
        ]
        self._extra_read_allowed_files = list(extra_read_allowed_files or [])
        self._extra_write_allowed_dirs = list(extra_write_allowed_dirs or [])
        self._extra_write_allowed_files = list(extra_write_allowed_files or [])
        self._restrict_to_workspace = (
            bool(restrict_to_workspace)
            if restrict_to_workspace is not None
            else allowed_dir is not None
        )
        self._sandbox_restricts_workspace = sandbox_restricts_workspace
        # Explicit state is used by isolated runners like Dream/subagents.
        # Main AgentLoop tools leave this unset and resolve state from the
        # current async task, which keeps shared tool instances session-safe.
        self._explicit_file_states = file_states
        self._fallback_file_states = FileStates()

    def _effective_sandbox_restricts(self) -> bool:
        """Return whether the configured exec sandbox should restrict this call.

        bwrap only runs on Linux, so on other platforms the configured sandbox
        never applies. On Linux, skip the sandbox restriction when the active
        workspace scope grants full access.
        """
        if not self._sandbox_restricts_workspace:
            return False
        if not _IS_LINUX:
            return False
        scope = current_workspace_scope()
        if scope is not None and not scope.restrict_to_workspace:
            return False
        return True

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        agent_workspace = Path(ctx.workspace)
        resolved_agent_workspace = agent_workspace.expanduser().resolve(strict=False)
        restrict = (
            ctx.config.restrict_to_workspace
            or ctx.config.exec.sandbox
        )
        sandbox_restricts = bool(ctx.config.exec.sandbox)
        allowed_dir = agent_workspace if restrict else None
        # Agent-owned skills stay available from project scopes. History is a narrower
        # capability: expose only the append-only log, not the surrounding memory directory.
        return cls(
            workspace=agent_workspace,
            allowed_dir=allowed_dir,
            extra_read_allowed_dirs=[BUILTIN_SKILLS_DIR, resolved_agent_workspace / "skills"],
            extra_read_allowed_files=[resolved_agent_workspace / "memory" / "history.jsonl"],
            file_states=ctx.file_state_store,
            restrict_to_workspace=ctx.config.restrict_to_workspace,
            sandbox_restricts_workspace=sandbox_restricts,
        )

    @property
    def _file_states(self) -> FileStates:
        if self._explicit_file_states is not None:
            return self._explicit_file_states
        return current_file_states(self._fallback_file_states)

    def _effective_allowed_root(self, access_allowed_root: Path | None) -> Path | None:
        if self._allowed_dir is None or self._workspace is None:
            return access_allowed_root
        try:
            allowed_dir = Path(self._allowed_dir).expanduser().resolve(strict=False)
            workspace = Path(self._workspace).expanduser().resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            return access_allowed_root if access_allowed_root is not None else self._allowed_dir
        if allowed_dir == workspace:
            return access_allowed_root
        return allowed_dir

    def _resolve_with_extra(
        self,
        path: str,
        extra_allowed_dirs: list[Path] | None,
        extra_allowed_files: list[Path] | None,
        *,
        include_media_dir: bool,
        extra_files_require_allowed_root: bool = False,
    ) -> Path:
        access = current_tool_workspace(
            self._workspace,
            restrict_to_workspace=self._restrict_to_workspace,
            sandbox_restricts_workspace=self._effective_sandbox_restricts(),
        )
        allowed_root = self._effective_allowed_root(access.allowed_root)
        if extra_files_require_allowed_root and allowed_root is None:
            extra_allowed_files = None
        return resolve_workspace_path(
            path,
            access.project_path,
            allowed_root,
            extra_allowed_dirs,
            extra_allowed_files,
            include_media_dir=include_media_dir,
        )

    def _resolve_read(self, path: str) -> Path:
        access = current_tool_workspace(
            self._workspace,
            restrict_to_workspace=self._restrict_to_workspace,
            sandbox_restricts_workspace=self._effective_sandbox_restricts(),
        )
        extra_read_dirs = [
            *self._extra_read_allowed_dirs,
            *access.extra_read_dirs,
        ]
        return self._resolve_with_extra(
            path,
            extra_read_dirs,
            self._extra_read_allowed_files,
            include_media_dir=True,
            extra_files_require_allowed_root=True,
        )

    def _resolve_write(self, path: str) -> Path:
        return self._resolve_with_extra(
            path,
            self._extra_write_allowed_dirs,
            self._extra_write_allowed_files,
            include_media_dir=False,
        )

    def _resolve(self, path: str) -> Path:
        return self._resolve_read(path)

    def _display_workspace(self) -> Path | None:
        return current_tool_workspace(self._workspace).project_path


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


_BLOCKED_DEVICE_PATHS = frozenset({
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",
    "/dev/stdin", "/dev/stdout", "/dev/stderr",
    "/dev/tty", "/dev/console",
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",
})


def _is_blocked_device(path: str | Path) -> bool:
    """Check if path is a blocked device that could hang or produce infinite output."""
    import re
    raw = str(path)

    # Resolve symlinks to check the actual target
    try:
        resolved = str(Path(raw).resolve())
    except (OSError, ValueError):
        resolved = raw

    if raw in _BLOCKED_DEVICE_PATHS or resolved in _BLOCKED_DEVICE_PATHS:
        return True
    if re.match(r"/proc/\d+/fd/[012]$", raw) or re.match(r"/proc/self/fd/[012]$", raw):
        return True
    if re.match(r"/proc/\d+/fd/[012]$", resolved) or re.match(r"/proc/self/fd/[012]$", resolved):
        return True

    # Check if resolved path starts with /dev/ (covers symlinks to devices)
    if resolved.startswith("/dev/"):
        return True
    return False


def _builtin_skill_read_path(path: str) -> Path | None:
    """Map workspace-relative skills/<name>/... reads onto bundled skills."""
    from nanobot.agent.skills import BUILTIN_SKILLS_DIR

    requested = Path(path)
    if requested.is_absolute():
        return None
    parts = requested.parts
    if len(parts) < 2 or parts[0] != "skills":
        return None
    root = BUILTIN_SKILLS_DIR.resolve()
    candidate = (root / Path(*parts[1:])).resolve()
    if candidate != root and root not in candidate.parents:
        return None
    return candidate if candidate.is_file() else None


@tool_parameters(
    tool_parameters_schema(
        path=StringSchema("The file path to read"),
        offset=IntegerSchema(
            1,
            description="Line number to start reading from (1-indexed, default 1)",
            minimum=1,
        ),
        limit=IntegerSchema(
            2000,
            description="Maximum number of lines to read (default 2000)",
            minimum=1,
        ),
        pages=StringSchema("Page range for PDF files, e.g. '1-5' (default: all, max 20 pages)"),
        force=BooleanSchema(
            description="Bypass same-file read deduplication and return content again.",
            default=False,
        ),
        required=["path"],
    )
)
class ReadFileTool(_FsTool):
    """Read file contents with optional line-based pagination."""
    _scopes = {"core", "subagent", "plan", "validator"}

    _MAX_CHARS = 128_000
    _MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024
    _DEFAULT_LIMIT = 2000
    _MAX_PDF_PAGES = 20

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read a file (text/image/document). Output: LINE_NUM|CONTENT. "
            "Supports PDF/DOCX/XLSX/PPTX. Use offset/limit for large files, "
            "force=true to re-read unchanged files. Max ~128K chars."
        )

    @property
    def read_only(self) -> bool:
        return True

    async def execute(
        self,
        path: str | None = None,
        offset: int = 1,
        limit: int | None = None,
        pages: str | None = None,
        force: bool = False,
        **kwargs: Any,
    ) -> Any:
        try:
            if not path:
                return ToolResult.error("Error reading file: Unknown path")

            # Device path blacklist
            if _is_blocked_device(path):
                return ToolResult.error(f"Error: Reading {path} is blocked (device path that could hang or produce infinite output).")

            fp = self._resolve_read(path)
            if not fp.exists():
                fp = _builtin_skill_read_path(path) or fp
            if _is_blocked_device(fp):
                return ToolResult.error(f"Error: Reading {fp} is blocked (device path that could hang or produce infinite output).")
            if not fp.exists():
                return ToolResult.error(f"Error: File not found: {path}")
            if not fp.is_file():
                return ToolResult.error(f"Error: Not a file: {path}")

            file_size = fp.stat().st_size
            if file_size > self._MAX_FILE_SIZE_BYTES:
                size_mib = file_size / (1024 * 1024)
                max_mib = self._MAX_FILE_SIZE_BYTES // (1024 * 1024)
                return ToolResult.error(
                    f"Error: File too large to read ({size_mib:.1f} MiB). "
                    f"Maximum is {max_mib} MiB."
                )

            # PDF support
            if fp.suffix.lower() == ".pdf":
                return self._read_pdf(fp, pages)

            # Office document support
            if fp.suffix.lower() in {".docx", ".xlsx", ".pptx"}:
                return self._read_office_doc(fp)

            raw = fp.read_bytes()
            if not raw:
                return f"(Empty file: {path})"

            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if mime and mime.startswith("image/"):
                return build_image_content_blocks(raw, mime, str(fp), f"(Image file: {path})")

            # Read dedup: same path + offset + limit + unchanged mtime → stub
            # Always check for external modifications before dedup
            entry = self._file_states.get(fp)
            try:
                current_mtime = os.path.getmtime(fp)
            except OSError:
                current_mtime = 0.0
            if (
                not force
                and entry
                and entry.can_dedup
                and entry.offset == offset
                and entry.limit == limit
            ):
                if current_mtime != entry.mtime:
                    # File was modified externally - force full read and mark as not dedupable
                    entry.can_dedup = False
                    self._file_states.record_read(fp, offset=offset, limit=limit)  # Update state with new mtime
                    # Continue to read full content (don't return dedup message)
                else:
                    # File unchanged - return dedup message
                    # But only if content is actually unchanged (not just mtime)
                    current_hash = _hash_file(str(fp))
                    if current_hash == entry.content_hash:
                        return f"[File unchanged since last read: {path}]"
                    else:
                        # Content changed despite same mtime - force full read
                        entry.can_dedup = False
                        self._file_states.record_read(fp, offset=offset, limit=limit)
            else:
                # No previous state or marked as not dedupable - read full content
                self._file_states.record_read(fp, offset=offset, limit=limit)
                # Force full read by setting can_dedup to False for this read
                if entry:
                    entry.can_dedup = False

            # Read the file content after dedup check
            raw = fp.read_bytes()
            try:
                text_content = raw.decode("utf-8")
            except UnicodeDecodeError:
                # Binary file - return error message
                mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
                if mime and mime.startswith("image/"):
                    return build_image_content_blocks(raw, mime, str(fp), f"(Image file: {path})")
                return ToolResult.error(f"Error: Cannot read binary file {path} (MIME: {mime or 'unknown'}). Only UTF-8 text and images are supported.")

            # Normalize CRLF -> LF before line-splitting. Primarily a Windows
            # concern (git checkouts with autocrlf, editors saving CRLF) but
            # applied on all platforms so downstream StrReplace/Grep behavior
            # is consistent regardless of where the file was written.
            text_content = text_content.replace("\r\n", "\n")

            all_lines = text_content.splitlines()
            total = len(all_lines)

            if offset < 1:
                offset = 1
            if offset > total:
                return ToolResult.error(f"Error: offset {offset} is beyond end of file ({total} lines)")

            start = offset - 1
            end = min(start + (limit or self._DEFAULT_LIMIT), total)
            numbered = [f"{start + i + 1}| {line}" for i, line in enumerate(all_lines[start:end])]
            result = "\n".join(numbered)

            if len(result) > self._MAX_CHARS:
                trimmed, chars = [], 0
                for line in numbered:
                    chars += len(line) + 1
                    if chars > self._MAX_CHARS:
                        break
                    trimmed.append(line)
                end = start + len(trimmed)
                result = "\n".join(trimmed)

            if end < total:
                result += f"\n\n(Showing lines {offset}-{end} of {total}. Use offset={end + 1} to continue.)"
            else:
                result += f"\n\n(End of file — {total} lines total)"
            self._file_states.record_read(fp, offset=offset, limit=limit)
            return result
        except PermissionError as e:
            return ToolResult.error(f"Error: {e}")
        except Exception as e:
            return ToolResult.error(f"Error reading file: {e}")

    def _read_pdf(self, fp: Path, pages: str | None) -> str:
        from nanobot.utils.document import PdfPageRangeError, PdfSafetyError, extract_pdf_pages

        try:
            extraction = extract_pdf_pages(
                fp,
                pages=pages,
                max_pages=self._MAX_PDF_PAGES,
                max_chars=self._MAX_CHARS,
            )
        except PdfPageRangeError:
            return ToolResult.error(f"Error: Invalid page range '{pages}'. Use format like '1-5'.")
        except PdfSafetyError as e:
            return ToolResult.error(f"Error reading PDF: {e}")
        except Exception as e:
            return ToolResult.error(f"Error reading PDF: {e}")

        if not extraction.text:
            return f"(PDF has no extractable text: {fp})"

        result = extraction.text
        if extraction.end_page < extraction.total_pages - 1:
            next_start = extraction.end_page + 2
            next_end = min(extraction.end_page + 1 + self._MAX_PDF_PAGES, extraction.total_pages)
            result += (
                f"\n\n(Showing pages {extraction.start_page + 1}-{extraction.end_page + 1} "
                f"of {extraction.total_pages}. Use pages='{next_start}-{next_end}' to continue.)"
            )
        return result

    def _read_office_doc(self, fp: Path) -> str:
        from nanobot.utils.document import extract_text

        result = extract_text(fp)

        if result is None:
            return ToolResult.error(f"Error: Unsupported file format: {fp.suffix}")

        if result.startswith("[error:"):
            return ToolResult.error(f"Error reading {fp.suffix.upper()} file: {result}")

        if not result:
            return f"({fp.suffix.upper().lstrip('.')} has no extractable text: {fp})"

        if len(result) > self._MAX_CHARS:
            result = result[:self._MAX_CHARS] + "\n\n(Document text truncated at ~128K chars)"

        return result


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


@tool_parameters(
    tool_parameters_schema(
        path=StringSchema("The file path to write to"),
        content=StringSchema("The content to write"),
        required=["path", "content"],
    )
)
class WriteFileTool(_FsTool):
    """Write content to a file."""
    _scopes = {"core", "subagent"}

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Create or fully replace a file. Prefer apply_patch for code changes."

    async def execute(self, path: str | None = None, content: str | None = None, **kwargs: Any) -> str:
        try:
            if not path:
                raise ValueError("Unknown path")
            if content is None:
                raise ValueError("Unknown content")
            fp = self._resolve_write(path)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            self._file_states.record_write(fp)
            return f"Successfully wrote {len(content)} characters to {fp}"
        except PermissionError as e:
            return ToolResult.error(f"Error: {e}")
        except Exception as e:
            return ToolResult.error(f"Error writing file: {e}")


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------
# The fuzzy text-matching engine lives in filesystem_edit_match.py; re-export
# so existing import sites keep working unchanged.
from nanobot.agent.tools.filesystem_edit_match import (  # noqa: E402
    _best_window,
    _find_match,  # noqa: F401 (re-export for tests)
    _find_matches,
    _match_covers_line,
    _preserve_quote_style,
    _reindent_like_match,
)


@tool_parameters(
    tool_parameters_schema(
        path=StringSchema("The file path to edit"),
        old_text=StringSchema("The text to find and replace"),
        new_text=StringSchema("The text to replace with"),
        replace_all=BooleanSchema(description="Replace all occurrences (default false)"),
        occurrence=IntegerSchema(
            1,
            description="Optional 1-based occurrence to replace when old_text appears multiple times.",
            minimum=1,
            nullable=True,
        ),
        line_hint=IntegerSchema(
            1,
            description=(
                "Optional exact 1-based target line copied from read_file. "
                "The selected old_text match must cover this line."
            ),
            minimum=1,
            nullable=True,
        ),
        expected_replacements=IntegerSchema(
            1,
            description="Optional guard for the number of replacements that must be made.",
            minimum=1,
            nullable=True,
        ),
        required=["path", "old_text", "new_text"],
    )
)
class EditFileTool(_FsTool):
    """Edit a file by replacing text with fallback matching."""
    _scopes = {"core", "subagent"}

    _MAX_EDIT_FILE_SIZE = 1024 * 1024 * 1024  # 1 GiB
    _MARKDOWN_EXTS = frozenset({".md", ".mdx", ".markdown"})

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Exact text replacement in one file: old_text → new_text. "
            "When replacing text in an existing file, old_text and new_text "
            "must be different. "
            "For multi-file/structural edits use apply_patch. "
            "Use occurrence/line_hint/replace_all for multiple matches."
        )

    @staticmethod
    def _strip_trailing_ws(text: str) -> str:
        """Strip trailing whitespace from each line."""
        return "\n".join(line.rstrip() for line in text.split("\n"))

    async def execute(
        self, path: str | None = None, old_text: str | None = None,
        new_text: str | None = None,
        replace_all: bool = False, occurrence: int | None = None,
        line_hint: int | None = None, expected_replacements: int | None = None, **kwargs: Any,
    ) -> str:
        try:
            if not path:
                raise ValueError("Unknown path")
            if old_text is None:
                raise ValueError("Unknown old_text")
            if new_text is None:
                raise ValueError("Unknown new_text")
            if occurrence is not None and occurrence < 1:
                return ToolResult.error("Error: occurrence must be >= 1.")
            if line_hint is not None and line_hint < 1:
                return ToolResult.error("Error: line_hint must be >= 1.")
            if expected_replacements is not None and expected_replacements < 1:
                return ToolResult.error("Error: expected_replacements must be >= 1.")

            fp = self._resolve_write(path)
            file_exists = fp.exists()
            if file_exists and old_text == new_text:
                return ToolResult.error("Error: new_text must be different from old_text.")

            # Create-file semantics: old_text='' + file doesn't exist → create
            if not file_exists:
                if old_text == "":
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    fp.write_text(new_text, encoding="utf-8")
                    self._file_states.record_write(fp)
                    return f"Successfully created {fp}"
                return self._file_not_found_msg(path, fp)

            # File size protection
            try:
                fsize = fp.stat().st_size
            except OSError:
                fsize = 0
            if fsize > self._MAX_EDIT_FILE_SIZE:
                return ToolResult.error(f"Error: File too large to edit ({fsize / (1024**3):.1f} GiB). Maximum is 1 GiB.")

            # Create-file: old_text='' but file exists and not empty → reject
            if old_text == "":
                raw = fp.read_bytes()
                content = raw.decode("utf-8")
                if content.strip():
                    return ToolResult.error(f"Error: Cannot create file — {path} already exists and is not empty.")
                fp.write_text(new_text, encoding="utf-8")
                self._file_states.record_write(fp)
                return f"Successfully edited {fp}"

            # Read-before-edit check
            warning = self._file_states.check_read(fp)

            raw = fp.read_bytes()
            uses_crlf = b"\r\n" in raw
            content = raw.decode("utf-8").replace("\r\n", "\n")
            norm_old = old_text.replace("\r\n", "\n")
            matches = _find_matches(content, norm_old)

            if not matches:
                return self._not_found_msg(old_text, content, path)
            count = len(matches)
            if replace_all and occurrence is not None:
                return ToolResult.error("Error: occurrence cannot be used with replace_all=true.")
            if replace_all and line_hint is not None:
                return ToolResult.error("Error: line_hint cannot be used with replace_all=true.")
            if occurrence is not None and line_hint is not None:
                return ToolResult.error("Error: line_hint cannot be used with occurrence.")
            if occurrence is not None and occurrence > count:
                return ToolResult.error(
                    f"Error: occurrence {occurrence} is out of range; "
                    f"old_text appears {count} time(s)."
                )
            if count > 1 and not replace_all and occurrence is None and line_hint is None:
                line_numbers = [match.line for match in matches]
                preview = ", ".join(f"line {n}" for n in line_numbers[:3])
                if len(line_numbers) > 3:
                    preview += ", ..."
                location_hint = f" at {preview}" if preview else ""
                return (
                    f"Warning: old_text appears {count} times{location_hint}. "
                    "Provide more context, set occurrence to choose one match, "
                    "or set replace_all=true."
                )

            norm_new = new_text.replace("\r\n", "\n")

            # Trailing whitespace stripping (skip markdown to preserve double-space line breaks)
            if fp.suffix.lower() not in self._MARKDOWN_EXTS:
                norm_new = self._strip_trailing_ws(norm_new)

            if replace_all:
                selected = matches
            elif occurrence is not None:
                selected = [matches[occurrence - 1]]
            elif line_hint is not None:
                candidates = [match for match in matches if _match_covers_line(match, line_hint)]
                if not candidates:
                    locations = ", ".join(f"line {match.line}" for match in matches[:3])
                    if len(matches) > 3:
                        locations += ", ..."
                    return ToolResult.error(
                        f"Error: line_hint {line_hint} does not match the old_text location. "
                        f"old_text appears at {locations}. Re-read the intended region and "
                        "copy old_text that covers the target line."
                    )
                if len(candidates) > 1:
                    return ToolResult.error(
                        f"Error: line_hint {line_hint} is ambiguous; "
                        f"old_text appears {len(candidates)} times on that line."
                    )
                selected = candidates
            else:
                selected = [matches[0]]
            if expected_replacements is not None and len(selected) != expected_replacements:
                return ToolResult.error(
                    f"Error: expected {expected_replacements} replacements but "
                    f"would make {len(selected)}."
                )
            new_content = content
            for match in reversed(selected):
                replacement = _preserve_quote_style(norm_old, match.text, norm_new)
                replacement = _reindent_like_match(norm_old, match.text, replacement)

                # Delete-line cleanup: when deleting text (new_text=''), consume trailing
                # newline to avoid leaving a blank line
                end = match.end
                if replacement == "" and not match.text.endswith("\n") and content[end:end + 1] == "\n":
                    end += 1

                new_content = new_content[: match.start] + replacement + new_content[end:]
            if uses_crlf:
                new_content = new_content.replace("\n", "\r\n")

            fp.write_bytes(new_content.encode("utf-8"))
            self._file_states.record_write(fp)
            msg = f"Successfully edited {fp}"
            if warning:
                msg = f"{warning}\n{msg}"
            return msg
        except PermissionError as e:
            return ToolResult.error(f"Error: {e}")
        except Exception as e:
            return ToolResult.error(f"Error editing file: {e}")

    def _file_not_found_msg(self, path: str, fp: Path) -> str:
        """Build an error message with 'Did you mean ...?' suggestions."""
        parent = fp.parent
        suggestions: list[str] = []
        if parent.is_dir():
            siblings = [f.name for f in parent.iterdir() if f.is_file()]
            close = difflib.get_close_matches(fp.name, siblings, n=3, cutoff=0.6)
            suggestions = [str(parent / c) for c in close]
        parts = [f"Error: File not found: {path}"]
        if suggestions:
            parts.append("Did you mean: " + ", ".join(suggestions) + "?")
        return ToolResult.error("\n".join(parts))

    @staticmethod
    def _not_found_msg(old_text: str, content: str, path: str) -> str:
        best_ratio, best_start, best_window_lines, hints = _best_window(old_text, content)
        if best_ratio > 0.5:
            diff = "\n".join(difflib.unified_diff(
                old_text.splitlines(keepends=True),
                best_window_lines,
                fromfile="old_text (provided)",
                tofile=f"{path} (actual, line {best_start + 1})",
                lineterm="",
            ))
            hint_text = ""
            if hints:
                hint_text = "\nPossible cause: " + ", ".join(hints) + "."
            return ToolResult.error(
                f"Error: old_text not found in {path}."
                f"{hint_text}\nBest match ({best_ratio:.0%} similar) at line {best_start + 1}:\n{diff}"
            )

        if hints:
            return ToolResult.error(
                f"Error: old_text not found in {path}. "
                f"Possible cause: {', '.join(hints)}. "
                "Copy the exact text from read_file and try again."
            )
        return ToolResult.error(f"Error: old_text not found in {path}. No similar text found. Verify the file content.")


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------

@tool_parameters(
    tool_parameters_schema(
        path=StringSchema("The directory path to list"),
        recursive=BooleanSchema(description="Recursively list all files (default false)"),
        max_entries=IntegerSchema(
            200,
            description="Maximum entries to return (default 200)",
            minimum=1,
        ),
        required=["path"],
    )
)
class ListDirTool(_FsTool):
    """List directory contents with optional recursion."""
    _scopes = {"core", "subagent", "plan", "validator"}

    _DEFAULT_MAX = 200
    _IGNORE_DIRS = {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
        ".ruff_cache", ".coverage", "htmlcov",
        # ponytail: extra heavy dirs to skip on spinning disks
        "vendor", "target", ".cache", ".npm", ".next",
        ".pnpm-store", ".parcel-cache", ".gradle", ".idea", ".vscode",
        "__snapshots__", "coverage",
        # nanobot runtime dirs (sessions, logs, local caches) — never search these.
        ".nanobot", ".clawhub", ".imgvenv",
    }

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List directory contents. recursive=true for nested. Auto-ignores .git/node_modules/etc."

    @property
    def read_only(self) -> bool:
        return True

    async def execute(
        self, path: str | None = None, recursive: bool = False,
        max_entries: int | None = None, **kwargs: Any,
    ) -> str:
        try:
            if path is None:
                raise ValueError("Unknown path")
            dp = self._resolve(path)
            if not dp.exists():
                return ToolResult.error(f"Error: Directory not found: {path}")
            if not dp.is_dir():
                return ToolResult.error(f"Error: Not a directory: {path}")

            cap = max_entries or self._DEFAULT_MAX
            items: list[str] = []
            total = 0

            if recursive:
                for item in sorted(dp.rglob("*")):
                    if any(p in self._IGNORE_DIRS for p in item.parts):
                        continue
                    total += 1
                    if len(items) < cap:
                        rel = item.relative_to(dp)
                        items.append(f"{rel}/" if item.is_dir() else str(rel))
            else:
                for item in sorted(dp.iterdir()):
                    if item.name in self._IGNORE_DIRS:
                        continue
                    total += 1
                    if len(items) < cap:
                        pfx = "📁 " if item.is_dir() else "📄 "
                        items.append(f"{pfx}{item.name}")

            if not items and total == 0:
                return f"Directory {path} is empty"

            result = "\n".join(items)
            if total > cap:
                result += f"\n\n(truncated, showing first {cap} of {total} entries)"
            return result
        except PermissionError as e:
            return ToolResult.error(f"Error: {e}")
        except Exception as e:
            return ToolResult.error(f"Error listing directory: {e}")
