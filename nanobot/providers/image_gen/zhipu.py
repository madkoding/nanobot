"""Zhipu (智谱) image generation provider."""

from __future__ import annotations

from typing import Any

import httpx

from nanobot.providers.image_gen.base import (
    GeneratedImageResponse,
    ImageGenerationError,
    ImageGenerationProvider,
    _download_image_data_url,
    register_image_gen_provider,
)

_ZHIPU_TIMEOUT_S = 300.0

_ZHIPU_ASPECT_RATIO_SIZES = {
    "1:1": "1280x1280",
    "16:9": "1728x960",
    "9:16": "960x1728",
    "3:4": "1088x1472",
    "4:3": "1472x1088",
}


class ZhipuImageGenerationClient(ImageGenerationProvider):
    """Async client for Zhipu (智谱) image generation API.

    Supports:
    - Text-to-image via glm-image, cogview-4, cogview-3-flash, etc.
    - Aspect ratio selection
    - Watermark control
    """

    provider_name = "zhipu"
    model_options = ("glm-image", "cogview-4", "cogview-4-250304", "cogview-3-flash")
    missing_key_message = "Zhipu API key is not configured. Set providers.zhipu.apiKey."
    default_timeout = _ZHIPU_TIMEOUT_S

    def _default_base_url(self) -> str:
        return "https://open.bigmodel.cn/api/paas/v4"

    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        reference_images: list[str] | None = None,
        aspect_ratio: str | None = None,
        image_size: str | None = None,
    ) -> GeneratedImageResponse:
        if not self.api_key:
            raise ImageGenerationError(self.missing_key_message)

        if reference_images:
            raise ImageGenerationError(
                "Zhipu image generation does not support reference images"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
        }

        size = _zhipu_size(aspect_ratio, image_size)
        if size:
            body["size"] = size

        body.update(self.extra_body)

        url = f"{self.api_base}/images/generations"

        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            return await self._generate_with_client(
                client,
                headers=headers,
                body=body,
                url=url,
            )
        finally:
            if self._client is None:
                await client.aclose()

    async def _generate_with_client(
        self,
        client: httpx.AsyncClient,
        *,
        headers: dict[str, str],
        body: dict[str, Any],
        url: str,
    ) -> GeneratedImageResponse:
        try:
            response = await self._http_post(url, headers=headers, body=body, client=client)
        except httpx.TimeoutException as exc:
            raise ImageGenerationError("Zhipu image generation timed out") from exc
        except httpx.RequestError as exc:
            raise ImageGenerationError(f"Zhipu image generation request failed: {exc}") from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text[:500]
            raise ImageGenerationError(f"Zhipu image generation failed: {detail}") from exc

        payload = response.json()
        images = await _zhipu_images_from_payload(payload)

        self._require_images(images, payload)

        return GeneratedImageResponse(images=images, content="", raw=payload)


def _zhipu_size(
    aspect_ratio: str | None,
    image_size: str | None,
) -> str:
    """Resolve aspect ratio / image_size to Zhipu size string.

    Zhipu glm-image model supports: 1280x1280 (default), 1568x1056,
    1056x1568, 1472x1088, 1088x1472, 1728x960, 960x1728.
    """
    if image_size and "x" in image_size.lower():
        return image_size
    if aspect_ratio and aspect_ratio in _ZHIPU_ASPECT_RATIO_SIZES:
        return _ZHIPU_ASPECT_RATIO_SIZES[aspect_ratio]
    return "1280x1280"


async def _zhipu_images_from_payload(
    payload: dict[str, Any],
) -> list[str]:
    """Extract image data URLs from Zhipu API response.

    Zhipu returns images as temporary URLs that expire after 30 days.
    We download and re-encode as base64 data URLs.
    """
    images: list[str] = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if isinstance(url, str) and url:
            images.append(await _download_image_data_url(url))
    return images


register_image_gen_provider(ZhipuImageGenerationClient)
