"""StepFun (阶跃星辰) image generation provider."""

from __future__ import annotations

from typing import Any

import httpx

from nanobot.providers.image_gen.base import (
    GeneratedImageResponse,
    ImageGenerationError,
    ImageGenerationProvider,
    _b64_image_data_url,
    image_path_to_data_url,
    register_image_gen_provider,
)

_STEPFUN_ASPECT_RATIO_SIZES = {
    "1:1": "1024x1024",
    "16:9": "1280x800",
    "9:16": "800x1280",
    "3:4": "768x1360",
    "4:3": "1360x768",
}


class StepFunImageGenerationClient(ImageGenerationProvider):
    """Async client for StepFun (阶跃星辰) image generation.

    Supports:
    - Text-to-image via step-image-edit-2 (default model)
    - Reference-image-guided generation via style_reference (step-1x-medium)
    """

    provider_name = "stepfun"
    model_options = ("step-image-edit-2", "step-1x-medium")
    missing_key_message = (
        "StepFun API key is not configured. Set providers.stepfun.apiKey."
    )
    default_timeout = 120.0

    def _default_base_url(self) -> str:
        return "https://api.stepfun.com/v1"

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

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "response_format": "b64_json",
            "n": 1,
        }

        # Map aspect ratio / image_size to StepFun size string
        size = _stepfun_size(aspect_ratio, image_size)
        if size:
            body["size"] = size

        # step-1x-medium supports style_reference for reference-image-guided generation
        refs = list(reference_images or [])
        if refs and "1x" in model:
            body["style_reference"] = {
                "source_url": image_path_to_data_url(refs[0]),
            }

        body.update(self.extra_body)

        response = await self._http_post(
            f"{self.api_base}/images/generations",
            headers=headers,
            body=body,
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text[:500]
            raise ImageGenerationError(
                f"StepFun image generation failed: {detail}"
            ) from exc

        payload = response.json()
        images = _stepfun_images_from_payload(payload)

        self._require_images(images, payload)

        return GeneratedImageResponse(images=images, content="", raw=payload)


def _stepfun_size(
    aspect_ratio: str | None,
    image_size: str | None,
) -> str:
    """Resolve aspect ratio / image_size to StepFun size string.

    StepFun expects ``WIDTHxHEIGHT`` (note: width x height, not the more
    common ``HxW`` order used by other providers).  The accepted sizes are
    ``1024x1024``, ``768x1360``, ``896x1184``, ``1360x768``, ``1184x896``.
    """
    if image_size and "x" in image_size.lower():
        return image_size
    if aspect_ratio and aspect_ratio in _STEPFUN_ASPECT_RATIO_SIZES:
        return _STEPFUN_ASPECT_RATIO_SIZES[aspect_ratio]
    return "1024x1024"


def _stepfun_images_from_payload(payload: dict[str, Any]) -> list[str]:
    """Extract base64 images from StepFun API response.

    StepFun returns images in ``data[].b64_json`` (base64 strings).
    """
    images: list[str] = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        b64 = item.get("b64_json")
        if isinstance(b64, str) and b64:
            images.append(_b64_image_data_url(b64))
    return images


register_image_gen_provider(StepFunImageGenerationClient)
