"""Feishu/Lark message-content parsers (extracted from feishu/runtime.py)."""

from __future__ import annotations

import json

# Message type display mapping
MSG_TYPE_MAP = {
    "image": "[image]",
    "audio": "[audio]",
    "file": "[file]",
    "sticker": "[sticker]",
}


def _extract_share_card_content(content_json: dict, msg_type: str) -> str:
    """Extract text representation from share cards and interactive messages."""
    parts = []

    if msg_type == "share_chat":
        parts.append(f"[shared chat: {content_json.get('chat_id', '')}]")
    elif msg_type == "share_user":
        parts.append(f"[shared user: {content_json.get('user_id', '')}]")
    elif msg_type == "interactive":
        parts.extend(_extract_interactive_content(content_json))
    elif msg_type == "share_calendar_event":
        parts.append(f"[shared calendar event: {content_json.get('event_key', '')}]")
    elif msg_type == "system":
        parts.append("[system message]")
    elif msg_type == "merge_forward":
        parts.append("[merged forward messages]")

    return "\n".join(parts) if parts else f"[{msg_type}]"


def _extract_interactive_content(content: dict) -> list[str]:
    """Recursively extract text and links from interactive card content."""
    parts = []

    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return [content] if content.strip() else []

    if not isinstance(content, dict):
        return parts

    # user_dsl: original card definition (richest source for rendered cards)
    user_dsl = content.get("user_dsl")
    if isinstance(user_dsl, str) and user_dsl.strip():
        try:
            dsl = json.loads(user_dsl)
            if isinstance(dsl, dict):
                parts.extend(_extract_interactive_content(dsl))
                if parts:
                    return parts
        except (json.JSONDecodeError, TypeError):
            pass

    if "title" in content:
        title = content["title"]
        if isinstance(title, dict):
            title_content = title.get("content", "") or title.get("text", "")
            if title_content:
                parts.append(f"title: {title_content}")
        elif isinstance(title, str):
            parts.append(f"title: {title}")

    # Top-level elements: flat list or nested list format
    elements = content.get("elements")
    if isinstance(elements, list):
        if elements and isinstance(elements[0], list):
            # Nested list: [[{tag:"text",text:"..."}], ...]
            for row in elements:
                if isinstance(row, list):
                    for element in row:
                        parts.extend(_extract_element_content(element))
        else:
            # Flat list: [{tag:"markdown",content:"..."}, ...]
            for element in elements:
                parts.extend(_extract_element_content(element))

    # Body elements (schema 2.0)
    body = content.get("body", {})
    if isinstance(body, dict):
        body_elements = body.get("elements")
        if isinstance(body_elements, list):
            for element in body_elements:
                parts.extend(_extract_element_content(element))

    card = content.get("card", {})
    if card:
        parts.extend(_extract_interactive_content(card))

    header = content.get("header", {})
    if header:
        header_title = header.get("title", {})
        if isinstance(header_title, dict):
            header_text = header_title.get("content", "") or header_title.get("text", "")
            if header_text:
                parts.append(f"title: {header_text}")

    return parts


def _extract_element_content(element: dict) -> list[str]:
    """Extract content from a single card element."""
    parts = []

    if not isinstance(element, dict):
        return parts

    tag = element.get("tag", "")

    if tag in ("markdown", "lark_md"):
        content = element.get("content", "")
        if content:
            parts.append(content)

    elif tag == "text":
        text = element.get("text", "")
        if isinstance(text, str) and text.strip():
            parts.append(text)

    elif tag == "div":
        text = element.get("text", {})
        if isinstance(text, dict):
            text_content = text.get("content", "") or text.get("text", "")
            if text_content:
                parts.append(text_content)
        elif isinstance(text, str):
            parts.append(text)
        for field in element.get("fields") or []:
            if isinstance(field, dict):
                field_text = field.get("text", {})
                if isinstance(field_text, dict):
                    c = field_text.get("content", "")
                    if c:
                        parts.append(c)

    elif tag == "a":
        href = element.get("href", "")
        text = element.get("text", "")
        if href:
            parts.append(f"link: {href}")
        if text:
            parts.append(text)

    elif tag == "button":
        text = element.get("text", {})
        if isinstance(text, dict):
            c = text.get("content", "")
            if c:
                parts.append(c)
        multi_url = element.get("multi_url") or {}
        url = element.get("url", "") or (
            multi_url.get("url", "") if isinstance(multi_url, dict) else ""
        )
        if url:
            parts.append(f"link: {url}")

    elif tag == "img":
        alt = element.get("alt", {})
        parts.append(alt.get("content", "[image]") if isinstance(alt, dict) else "[image]")

    elif tag == "note":
        for ne in element.get("elements") or []:
            parts.extend(_extract_element_content(ne))

    elif tag == "column_set":
        for col in element.get("columns") or []:
            if not isinstance(col, dict):
                continue
            for ce in col.get("elements") or []:
                parts.extend(_extract_element_content(ce))

    elif tag == "plain_text":
        content = element.get("content", "")
        if content:
            parts.append(content)

    elif tag == "table":
        columns = [
            (column["name"], str(column.get("display_name") or column["name"]))
            for column in (element.get("columns") or [])
            if isinstance(column, dict) and column.get("name")
        ]
        rows = element.get("rows") or []
        if columns:
            parts.append(" | ".join(header for _, header in columns))
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                values = []
                for name, _ in columns:
                    value = row.get(name)
                    if isinstance(value, list):
                        value = " ".join(str(item).strip() for item in value if item is not None)
                    values.append("" if value is None else str(value).strip())
                row_text = " | ".join(values).strip()
                if row_text:
                    parts.append(row_text)

    else:
        for ne in element.get("elements") or []:
            parts.extend(_extract_element_content(ne))

    return parts


def _extract_post_content(content_json: dict) -> tuple[str, list[str]]:
    """Extract text and image keys from Feishu post (rich text) message.

    Handles three payload shapes:
    - Direct:    {"title": "...", "content": [[...]]}
    - Localized: {"zh_cn": {"title": "...", "content": [...]}}
    - Wrapped:   {"post": {"zh_cn": {"title": "...", "content": [...]}}}
    """

    def _parse_block(block: dict) -> tuple[str | None, list[str]]:
        if not isinstance(block, dict) or not isinstance(block.get("content"), list):
            return None, []
        texts, images = [], []
        title = block.get("title")
        if isinstance(title, str) and title:
            texts.append(title)
        for row in block["content"]:
            if not isinstance(row, list):
                continue
            for el in row:
                if not isinstance(el, dict):
                    continue
                tag = el.get("tag")
                if tag in ("text", "a"):
                    text = el.get("text", "")
                    if isinstance(text, str):
                        texts.append(text)
                elif tag == "at":
                    user = el.get("user_name", "user")
                    texts.append(f"@{user if isinstance(user, str) and user else 'user'}")
                elif tag == "code_block":
                    lang = el.get("language", "")
                    code_text = el.get("text", "")
                    if not isinstance(lang, str):
                        lang = ""
                    if not isinstance(code_text, str):
                        code_text = ""
                    texts.append(f"\n```{lang}\n{code_text}\n```\n")
                elif tag == "img" and (key := el.get("image_key")):
                    images.append(key)
        return (" ".join(texts).strip() or None), images

    # Unwrap optional {"post": ...} envelope
    root = content_json
    if isinstance(root, dict) and isinstance(root.get("post"), dict):
        root = root["post"]
    if not isinstance(root, dict):
        return "", []

    # Direct format
    if "content" in root:
        text, imgs = _parse_block(root)
        if text or imgs:
            return text or "", imgs

    # Localized: prefer known locales, then fall back to any dict child
    for key in ("zh_cn", "en_us", "ja_jp"):
        if key in root:
            text, imgs = _parse_block(root[key])
            if text or imgs:
                return text or "", imgs
    for val in root.values():
        if isinstance(val, dict):
            text, imgs = _parse_block(val)
            if text or imgs:
                return text or "", imgs

    return "", []
