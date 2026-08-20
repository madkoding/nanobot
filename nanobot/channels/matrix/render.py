"""Matrix markdown → sanitized HTML rendering helpers."""

from __future__ import annotations

import html
import re
from urllib.parse import quote, unquote

import nh3
from mistune import HTMLRenderer, create_markdown

MATRIX_HTML_FORMAT = "org.matrix.custom.html"

MATRIX_MARKDOWN = create_markdown(
    renderer=HTMLRenderer(escape=True, allow_harmful_protocols=("mxc://",)),
    plugins=["table", "strikethrough", "url", "superscript", "subscript"],
)

MATRIX_ALLOWED_HTML_TAGS = {
    "p", "a", "strong", "em", "del", "code", "pre", "blockquote",
    "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "hr", "br", "table", "thead", "tbody", "tr", "th", "td",
    "caption", "sup", "sub", "img",
}
MATRIX_ALLOWED_HTML_ATTRIBUTES: dict[str, set[str]] = {
    "a": {"href"}, "code": {"class"}, "ol": {"start"},
    "img": {"src", "alt", "title", "width", "height"},
}
MATRIX_ALLOWED_URL_SCHEMES = {"https", "http", "matrix", "mailto", "mxc"}
_MXC_IMAGE_PLACEHOLDER_PREFIX = "https://nanobot.invalid/matrix-mxc/"
_MXC_MARKDOWN_IMAGE_RE = re.compile(
    r"(?P<prefix>!\[[^\]]*\]\()"
    r"(?P<value>mxc://[^\s)]+)"
    r"(?P<suffix>(?:\s+[^)]*)?\))"
)
_MXC_IMAGE_SRC_RE = re.compile(
    r"(?P<prefix>\bsrc=)(?P<quote>[\"'])(?P<value>mxc://[^\"']+)(?P=quote)",
    re.IGNORECASE,
)
_MXC_PLACEHOLDER_SRC_RE = re.compile(
    rf'src="{re.escape(_MXC_IMAGE_PLACEHOLDER_PREFIX)}([^"]+)"'
)


def _filter_matrix_html_attribute(tag: str, attr: str, value: str) -> str | None:
    """Filter attribute values to a safe Matrix-compatible subset."""
    if tag == "a" and attr == "href":
        return value if value.lower().startswith(("https://", "http://", "matrix:", "mailto:")) else None
    if tag == "img" and attr == "src":
        lowered = value.lower()
        if lowered.startswith("mxc://") or lowered.startswith(_MXC_IMAGE_PLACEHOLDER_PREFIX):
            return value
        return None
    if tag == "code" and attr == "class":
        classes = [c for c in value.split() if c.startswith("language-") and not c.startswith("language-_")]
        return " ".join(classes) if classes else None
    return value


MATRIX_HTML_CLEANER = nh3.Cleaner(
    tags=MATRIX_ALLOWED_HTML_TAGS,
    attributes=MATRIX_ALLOWED_HTML_ATTRIBUTES,
    attribute_filter=_filter_matrix_html_attribute,
    url_schemes=MATRIX_ALLOWED_URL_SCHEMES,
    strip_comments=True,
    link_rel="noopener noreferrer",
)


def _mask_mxc_markdown_image_sources(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        value = quote(match.group("value"), safe="")
        return (
            f"{match.group('prefix')}"
            f"{_MXC_IMAGE_PLACEHOLDER_PREFIX}{value}"
            f"{match.group('suffix')}"
        )

    return _MXC_MARKDOWN_IMAGE_RE.sub(repl, text)


def _mask_mxc_image_sources(rendered_html: str) -> str:
    def repl(match: re.Match[str]) -> str:
        value = quote(match.group("value"), safe="")
        return (
            f'{match.group("prefix")}{match.group("quote")}'
            f"{_MXC_IMAGE_PLACEHOLDER_PREFIX}{value}"
            f'{match.group("quote")}'
        )

    return _MXC_IMAGE_SRC_RE.sub(repl, rendered_html)


def _unmask_mxc_image_sources(cleaned_html: str) -> str:
    def repl(match: re.Match[str]) -> str:
        value = html.escape(unquote(match.group(1)), quote=True)
        return f'src="{value}"'

    return _MXC_PLACEHOLDER_SRC_RE.sub(repl, cleaned_html)


def _render_markdown_html(text: str) -> str | None:
    """Render markdown to sanitized HTML; returns None for plain text."""
    try:
        masked_text = _mask_mxc_markdown_image_sources(text)
        rendered = _mask_mxc_image_sources(MATRIX_MARKDOWN(masked_text))
        formatted = _unmask_mxc_image_sources(MATRIX_HTML_CLEANER.clean(rendered).strip())
    except Exception:
        return None
    if not formatted:
        return None
    # Skip formatted_body for plain <p>text</p> to keep payload minimal.
    if formatted.startswith("<p>") and formatted.endswith("</p>"):
        inner = formatted[3:-4]
        if "<" not in inner and ">" not in inner:
            return None
    return formatted


def _build_matrix_text_content(
    text: str,
    event_id: str | None = None,
    thread_relates_to: dict[str, object] | None = None,
) -> dict[str, object]:
    """
    Constructs and returns a dictionary representing the matrix text content with optional
    HTML formatting and reference to an existing event for replacement. This function is
    primarily used to create content payloads compatible with the Matrix messaging protocol.

    :param text: The plain text content to include in the message.
    :type text: str
    :param event_id: Optional ID of the event to replace. If provided, the function will
        include information indicating that the message is a replacement of the specified
        event.
    :type event_id: str | None
    :param thread_relates_to: Optional Matrix thread relation metadata. For edits this is
        stored in ``m.new_content`` so the replacement remains in the same thread.
    :type thread_relates_to: dict[str, object] | None
    :return: A dictionary containing the matrix text content, potentially enriched with
        HTML formatting and replacement metadata if applicable.
    :rtype: dict[str, object]
    """
    content: dict[str, object] = {"msgtype": "m.text", "body": text, "m.mentions": {}}
    if html := _render_markdown_html(text):
        content["format"] = MATRIX_HTML_FORMAT
        content["formatted_body"] = html
    if event_id:
        content["m.new_content"] = {
            "body": text,
            "msgtype": "m.text",
        }
        content["m.relates_to"] = {
            "rel_type": "m.replace",
            "event_id": event_id,
        }
        if thread_relates_to:
            content["m.new_content"]["m.relates_to"] = thread_relates_to
    elif thread_relates_to:
        content["m.relates_to"] = thread_relates_to

    return content
