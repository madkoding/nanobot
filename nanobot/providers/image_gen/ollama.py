"""Ollama image generation provider."""

from __future__ import annotations

import re
from typing import Any

import httpx
from loguru import logger

from nanobot.providers.image_gen.base import (
    GeneratedImageResponse,
    ImageGenerationError,
    ImageGenerationProvider,
    _b64_image_data_url,
    _http_error_detail,
    _round_to_multiple,
    register_image_gen_provider,
)

_OLLAMA_DEFAULT_SIDE = 1024
_OLLAMA_SIZE_PRESETS = {
    "1K": 1024,
    "2K": 2048,
    "4K": 4096,
}
_OLLAMA_EXPLICIT_SIZE_RE = re.compile(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$")
_OLLAMA_ASPECT_RATIO_RE = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*$")


def _ollama_dimensions(aspect_ratio: str | None, image_size: str | None) -> tuple[int, int]:
    if image_size:
        size = image_size.strip()
        explicit = _OLLAMA_EXPLICIT_SIZE_RE.fullmatch(size)
        if explicit:
            return int(explicit.group(1)), int(explicit.group(2))
        long_side = _OLLAMA_SIZE_PRESETS.get(size.upper(), _OLLAMA_DEFAULT_SIDE)
    else:
        long_side = _OLLAMA_DEFAULT_SIDE

    if not aspect_ratio:
        return long_side, long_side

    ratio = _OLLAMA_ASPECT_RATIO_RE.fullmatch(aspect_ratio.strip())
    if ratio is None:
        return long_side, long_side

    width_ratio = int(ratio.group(1))
    height_ratio = int(ratio.group(2))
    if width_ratio <= 0 or height_ratio <= 0:
        return long_side, long_side

    if width_ratio >= height_ratio:
        width = long_side
        height = _round_to_multiple(long_side * height_ratio / width_ratio)
    else:
        height = long_side
        width = _round_to_multiple(long_side * width_ratio / height_ratio)
    return max(8, width), max(8, height)


def _ollama_image_data_url(value: str) -> str:
    if value.startswith("data:image/"):
        return value
    return _b64_image_data_url(value)


def _ollama_images_from_payload(payload: dict[str, Any]) -> list[str]:
    images: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str) and value:
            images.append(_ollama_image_data_url(value))
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(payload.get("image"))
    collect(payload.get("images"))
    return images


class OllamaImageGenerationClient(ImageGenerationProvider):
    """Async client for Ollama native image generation models."""

    provider_name = "ollama"
    model_options = ("x/z-image-turbo",)
    default_timeout = 300.0

    def _default_base_url(self) -> str:
        return "http://localhost:11434/api"

    def _resolve_base_url(self, api_base: str | None) -> str:
        if api_base:
            base = api_base.rstrip("/")
            if base.endswith("/v1"):
                return f"{base[:-3]}/api"
            return base
        return self._default_base_url()

    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        reference_images: list[str] | None = None,
        aspect_ratio: str | None = None,
        image_size: str | None = None,
    ) -> GeneratedImageResponse:
        if reference_images:
            raise ImageGenerationError(
                "Ollama image generation does not support reference images"
            )

        width, height = _ollama_dimensions(aspect_ratio, image_size)
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": 0,
        }
        body.update(self.extra_body)
        body["stream"] = False

        headers = {
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.api_base}/generate"
        response = await self._http_post(url, headers=headers, body=body)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _http_error_detail(response)
            logger.error(
                "Ollama image generation failed (HTTP {}): {}",
                response.status_code,
                detail,
            )
            raise ImageGenerationError(
                f"Ollama image generation failed (HTTP {response.status_code}): {detail}"
            ) from exc

        data = response.json()
        images = _ollama_images_from_payload(data)

        self._require_images(images, data)

        response_text = data.get("response")
        content = response_text if isinstance(response_text, str) else ""

        return GeneratedImageResponse(images=images, content=content, raw=data)


register_image_gen_provider(OllamaImageGenerationClient)
