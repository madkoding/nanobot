"""Custom (user-configured OpenAI-compatible) image generation provider."""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from nanobot.providers.image_gen.base import (
    GeneratedImageResponse,
    ImageGenerationError,
    ImageGenerationProvider,
    register_image_gen_provider,
)
from nanobot.providers.image_gen.openai_shared import (
    _openai_images_from_payload,
    _openai_size,
)


class CustomImageGenerationClient(ImageGenerationProvider):
    """OpenAI-compatible Images API for user-configured custom providers."""

    provider_name = "custom"
    missing_base_message = (
        "Custom image generation API base is not configured. Set providers.custom.apiBase."
    )

    def _default_base_url(self) -> str:
        return ""

    @staticmethod
    def _custom_size(aspect_ratio: str | None, image_size: str | None) -> str:
        if image_size:
            requested = image_size.strip()
            if requested:
                if requested.lower() == "1k":
                    return "1024x1024"
                return requested
        return _openai_size("gpt-image-2", aspect_ratio, None)

    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        reference_images: list[str] | None = None,
        aspect_ratio: str | None = None,
        image_size: str | None = None,
    ) -> GeneratedImageResponse:
        if not self.api_base:
            raise ImageGenerationError(self.missing_base_message)

        if reference_images:
            logger.warning(
                "Custom image generation does not support reference images; "
                "ignoring {} reference image(s) for {}",
                len(reference_images),
                model,
            )

        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)

        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "response_format": "b64_json",
            "n": 1,
            "size": self._custom_size(aspect_ratio, image_size),
        }
        body.update(self.extra_body)

        logger.info("Custom Images API request: POST {}/images/generations body={}", self.api_base, body)

        response = await self._http_post(
            f"{self.api_base}/images/generations",
            headers=headers,
            body=body,
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text[:1000]
            logger.error("Custom Images API error ({}): {}", response.status_code, detail)
            raise ImageGenerationError(
                f"Custom image generation failed (HTTP {response.status_code}): {detail}"
            ) from exc

        payload = response.json()
        logger.info("Custom Images API response ({}): {}", response.status_code,
                       {k: v for k, v in payload.items() if k != "data"})

        images = await _openai_images_from_payload(payload)

        self._require_images(images, payload)

        return GeneratedImageResponse(images=images, content="", raw=payload)


register_image_gen_provider(CustomImageGenerationClient)
