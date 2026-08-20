"""Shared OpenAI-family image generation helpers."""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from nanobot.providers.image_gen.base import (
    ImageGenerationError,
    _b64_image_data_url,
    _download_image_data_url,
)

_OPENAI_DALLE2_SUPPORTED_SIZES = {"256x256", "512x512", "1024x1024"}
_OPENAI_DALLE3_SUPPORTED_SIZES = {"1024x1024", "1792x1024", "1024x1792"}
_OPENAI_GPT_IMAGE_SUPPORTED_SIZES = {
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "auto",
}
_OPENAI_DALLE2_ASPECT_RATIO_SIZES = {
    "1:1": "1024x1024",
    "16:9": "1024x1024",
    "9:16": "1024x1024",
    "3:4": "1024x1024",
    "4:3": "1024x1024",
}
_OPENAI_DALLE3_ASPECT_RATIO_SIZES = {
    "1:1": "1024x1024",
    "16:9": "1792x1024",
    "9:16": "1024x1792",
    "3:4": "1024x1792",
    "4:3": "1792x1024",
}
_OPENAI_GPT_IMAGE_ASPECT_RATIO_SIZES = {
    "1:1": "1024x1024",
    "16:9": "1536x1024",
    "9:16": "1024x1536",
    "3:4": "1024x1536",
    "4:3": "1536x1024",
}


def _openai_size(
    model: str,
    aspect_ratio: str | None,
    image_size: str | None,
) -> str:
    """Resolve aspect ratio or image_size to an OpenAI Images API size string."""
    sizes, supported_sizes = _openai_size_options(model)
    explicit_size = _normalize_openai_image_size(image_size)
    if explicit_size and _openai_explicit_size_supported(
        explicit_size,
        supported_sizes=supported_sizes,
    ):
        return explicit_size
    if explicit_size:
        logger.warning(
            "OpenAI image size '{}' is not supported by {}; using aspect ratio/default size",
            explicit_size,
            model,
        )
    if aspect_ratio and aspect_ratio in sizes:
        return sizes[aspect_ratio]
    return "1024x1024"


def _openai_multipart_form_body(body: dict[str, Any]) -> dict[str, str]:
    form: dict[str, str] = {}
    for key, value in body.items():
        if value is None:
            continue
        if isinstance(value, bool):
            form[key] = "true" if value else "false"
        elif isinstance(value, str | int | float):
            form[key] = str(value)
        else:
            logger.warning(
                "OpenAI image edit parameter '{}' is not a scalar form field; ignoring it",
                key,
            )
    return form


def _openai_is_gpt_image_model(model: str) -> bool:
    normalized = model.lower()
    return normalized.startswith(("gpt-image", "chatgpt-image"))


def _openai_size_options(model: str) -> tuple[dict[str, str], set[str] | None]:
    normalized = model.lower()
    if normalized.startswith("dall-e-2"):
        return _OPENAI_DALLE2_ASPECT_RATIO_SIZES, _OPENAI_DALLE2_SUPPORTED_SIZES
    if normalized.startswith("dall-e-3"):
        return _OPENAI_DALLE3_ASPECT_RATIO_SIZES, _OPENAI_DALLE3_SUPPORTED_SIZES
    if normalized.startswith("gpt-image-2"):
        return _OPENAI_GPT_IMAGE_ASPECT_RATIO_SIZES, None
    return _OPENAI_GPT_IMAGE_ASPECT_RATIO_SIZES, _OPENAI_GPT_IMAGE_SUPPORTED_SIZES


def _normalize_openai_image_size(image_size: str | None) -> str | None:
    if not image_size:
        return None
    normalized = image_size.strip().lower()
    return normalized or None


def _openai_explicit_size_supported(
    size: str,
    *,
    supported_sizes: set[str] | None,
) -> bool:
    if supported_sizes is not None:
        return size in supported_sizes
    width, sep, height = size.partition("x")
    return bool(sep and width.isdecimal() and height.isdecimal())


async def _openai_images_from_payload(
    payload: dict[str, Any],
) -> list[str]:
    """Extract images from OpenAI Images API response.

    Handles both ``b64_json`` (preferred) and ``url`` (downloaded) formats.
    """
    images: list[str] = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        b64 = item.get("b64_json")
        if isinstance(b64, str) and b64:
            images.append(_b64_image_data_url(b64))
            continue
        url = item.get("url")
        if isinstance(url, str) and url:
            images.append(await _download_image_data_url(url))
    return images


async def _parse_codex_sse_images(
    response: httpx.Response,
) -> tuple[list[str], str]:
    """Parse a Codex Responses API SSE stream for image generation output.

    Returns ``(images, content_text)``.
    """
    import json as _json

    images: list[str] = []
    text_parts: list[str] = []

    buffer: list[str] = []
    async for line_bytes in response.aiter_lines():
        line = line_bytes.strip()
        if line == "":
            if buffer:
                data_lines = []
                for bl in buffer:
                    if bl.startswith("data:"):
                        data_lines.append(bl[5:].strip())
                buffer.clear()
                if data_lines:
                    raw = "".join(data_lines)
                    if raw == "[DONE]":
                        break
                    try:
                        event = _json.loads(raw)
                    except Exception:
                        continue
                    ev_type = event.get("type", "")
                    if ev_type in ("error", "response.failed"):
                        logger.error("Codex SSE failure: {}", raw[:2000])
                    _collect_images_from_sse_event(event, images)
                    _collect_text_from_sse_event(event, text_parts)
                    if ev_type == "response.completed":
                        break
            continue
        buffer.append(line)

    # flush remaining
    if buffer:
        data_lines = [bl[5:].strip() for bl in buffer if bl.startswith("data:")]
        raw = "".join(data_lines)
        if raw and raw != "[DONE]":
            try:
                event = _json.loads(raw)
            except Exception:
                pass
            else:
                _collect_images_from_sse_event(event, images)
                _collect_text_from_sse_event(event, text_parts)

    return images, "".join(text_parts).strip()


def _collect_images_from_sse_event(event: dict[str, Any], images: list[str]) -> None:
    if event.get("type") != "response.output_item.done":
        return
    item = event.get("item") or {}
    if item.get("type") != "image_generation_call":
        return
    result = item.get("result")
    if isinstance(result, str):
        if result.startswith("data:image/"):
            images.append(result)
        else:
            images.append(_b64_image_data_url(result))
    elif isinstance(result, dict):
        image_url = result.get("image_url") or result.get("image") or ""
        if isinstance(image_url, str):
            if image_url.startswith("data:image/"):
                images.append(image_url)
            else:
                images.append(_b64_image_data_url(image_url))


def _collect_text_from_sse_event(event: dict[str, Any], text_parts: list[str]) -> None:
    if event.get("type") == "response.output_text.delta":
        delta = event.get("delta")
        if isinstance(delta, str) and delta:
            text_parts.append(delta)


__all__ = [
    "ImageGenerationError",
    "_b64_image_data_url",
    "_download_image_data_url",
    "_normalize_openai_image_size",
    "_openai_explicit_size_supported",
    "_openai_images_from_payload",
    "_openai_is_gpt_image_model",
    "_openai_multipart_form_body",
    "_openai_size",
    "_openai_size_options",
    "_parse_codex_sse_images",
    "_collect_images_from_sse_event",
    "_collect_text_from_sse_event",
]
