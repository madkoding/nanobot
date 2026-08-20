"""WhatsApp message field extraction helpers (extracted from whatsapp/runtime.py)."""

from __future__ import annotations

from typing import Any, NamedTuple


class _MediaInfo(NamedTuple):
    kind: str
    message: Any
    mimetype: str
    filename: str
    is_voice: bool = False


def _has_field(message: Any, name: str) -> bool:
    if message is None:
        return False

    has_field = getattr(message, "HasField", None)
    if callable(has_field):
        try:
            return bool(has_field(name))
        except ValueError:
            pass

    list_fields = getattr(message, "ListFields", None)
    if callable(list_fields):
        try:
            return any(getattr(field, "name", "") == name for field, _ in list_fields())
        except Exception:
            pass

    value = getattr(message, name, None)
    return value is not None and value != "" and value != b""


def _message_field(message: Any, *names: str) -> Any:
    for name in names:
        if _has_field(message, name):
            return getattr(message, name)
    return None


def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    return getattr(obj, name, default)


def _context_infos(message: Any) -> list[Any]:
    infos: list[Any] = []
    for container in (
        message,
        _message_field(message, "extendedTextMessage"),
        _message_field(message, "imageMessage"),
        _message_field(message, "videoMessage"),
        _message_field(message, "audioMessage"),
        _message_field(message, "documentMessage"),
        _message_field(message, "stickerMessage"),
    ):
        context = _message_field(container, "contextInfo")
        if context is not None:
            infos.append(context)
    return infos


def _message_text(message: Any) -> str:
    conversation = str(_safe_attr(message, "conversation", "") or "").strip()
    if conversation:
        return conversation

    extended = _message_field(message, "extendedTextMessage")
    text = str(_safe_attr(extended, "text", "") or "").strip()
    if text:
        return text

    for field_name in ("imageMessage", "videoMessage", "documentMessage", "stickerMessage"):
        media_message = _message_field(message, field_name)
        caption = str(_safe_attr(media_message, "caption", "") or "").strip()
        if caption:
            return caption

    return ""


def _media_message(message: Any) -> _MediaInfo | None:
    image = _message_field(message, "imageMessage")
    if image is not None:
        return _MediaInfo(
            kind="image",
            message=image,
            mimetype=str(_safe_attr(image, "mimetype", "") or "image/jpeg"),
            filename=str(_safe_attr(image, "fileName", "") or ""),
        )

    video = _message_field(message, "videoMessage")
    if video is not None:
        return _MediaInfo(
            kind="video",
            message=video,
            mimetype=str(_safe_attr(video, "mimetype", "") or "video/mp4"),
            filename=str(_safe_attr(video, "fileName", "") or ""),
        )

    audio = _message_field(message, "audioMessage")
    if audio is not None:
        return _MediaInfo(
            kind="audio",
            message=audio,
            mimetype=str(_safe_attr(audio, "mimetype", "") or "audio/ogg"),
            filename=str(_safe_attr(audio, "fileName", "") or ""),
            is_voice=bool(_safe_attr(audio, "PTT", False) or _safe_attr(audio, "ptt", False)),
        )

    document = _message_field(message, "documentMessage")
    if document is not None:
        return _MediaInfo(
            kind="file",
            message=document,
            mimetype=str(_safe_attr(document, "mimetype", "") or "application/octet-stream"),
            filename=str(
                _safe_attr(document, "fileName", "")
                or _safe_attr(document, "title", "")
                or ""
            ),
        )

    sticker = _message_field(message, "stickerMessage")
    if sticker is not None:
        return _MediaInfo(
            kind="sticker",
            message=sticker,
            mimetype=str(_safe_attr(sticker, "mimetype", "") or "image/webp"),
            filename=str(_safe_attr(sticker, "fileName", "") or ""),
        )

    return None
