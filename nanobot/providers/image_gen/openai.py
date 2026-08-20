"""OpenAI image generation provider (Images API with API key)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from nanobot.providers.base import LLMProvider
from nanobot.providers.image_gen.base import (
    GeneratedImageResponse,
    ImageGenerationError,
    ImageGenerationProvider,
    register_image_gen_provider,
)
from nanobot.providers.image_gen.openai_shared import (
    _openai_images_from_payload,
    _openai_is_gpt_image_model,
    _openai_multipart_form_body,
    _openai_size,
)
from nanobot.utils.helpers import detect_image_mime


class OpenAIImageGenerationClient(ImageGenerationProvider):
    """OpenAI Images API using an API key (``providers.openai.apiKey``)."""

    provider_name = "openai"
    model_options = ("gpt-image-2", "gpt-image-1", "dall-e-3", "dall-e-2")
    missing_key_message = (
        "OpenAI API key is not configured. Set providers.openai.apiKey."
    )

    def _default_base_url(self) -> str:
        return "https://api.openai.com/v1"

    @staticmethod
    def _strip_model_prefix(model: str) -> str:
        """Remove ``openai/`` prefix if present (OpenRouter convention)."""
        return LLMProvider._strip_prefix(model, ("openai", "openai_codex"))

    async def _parse_images_response(self, payload: dict[str, Any]) -> list[str]:
        return await _openai_images_from_payload(payload)

    async def _post_image_edit(
        self,
        *,
        headers: dict[str, str],
        body: dict[str, Any],
        reference_images: list[str],
    ) -> httpx.Response:
        files: list[tuple[str, tuple[str, Any, str]]] = []
        handles: list[Any] = []
        try:
            for path in reference_images:
                p = Path(path).expanduser()
                raw = p.read_bytes()
                mime = detect_image_mime(raw)
                if mime is None:
                    raise ImageGenerationError(f"unsupported reference image: {p}")
                handle = p.open("rb")
                handles.append(handle)
                files.append(("image[]", (p.name, handle, mime)))

            client = self._client
            if client is not None:
                return await client.post(
                    f"{self.api_base}/images/edits",
                    headers=headers,
                    data=body,
                    files=files,
                )
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                return await c.post(
                    f"{self.api_base}/images/edits",
                    headers=headers,
                    data=body,
                    files=files,
                )
        finally:
            for handle in handles:
                handle.close()

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

        clean_model = self._strip_model_prefix(model)

        generation_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        edit_headers = {
            "Authorization": f"Bearer {self.api_key}",
            **self.extra_headers,
        }

        body: dict[str, Any] = {
            "model": clean_model,
            "prompt": prompt,
        }

        if not _openai_is_gpt_image_model(clean_model):
            body["response_format"] = "b64_json"
            body["n"] = 1

        size = _openai_size(clean_model, aspect_ratio, image_size)
        if size:
            body["size"] = size

        body.update(self.extra_body)
        # Drop null-valued params so extraBody can opt out of defaults like response_format.
        body = {key: value for key, value in body.items() if value is not None}

        refs = list(reference_images or [])
        if refs:
            if not _openai_is_gpt_image_model(clean_model):
                raise ImageGenerationError(
                    f"OpenAI model '{clean_model}' does not support reference images; "
                    "use a GPT Image model"
                )
            edit_body = _openai_multipart_form_body(body)
            logger.info(
                "OpenAI Images API request: POST {}/images/edits body={} reference_images={}",
                self.api_base,
                edit_body,
                len(refs),
            )
            response = await self._post_image_edit(
                headers=edit_headers,
                body=edit_body,
                reference_images=refs,
            )
        else:
            logger.info(
                "OpenAI Images API request: POST {}/images/generations body={}",
                self.api_base,
                body,
            )

            response = await self._http_post(
                f"{self.api_base}/images/generations",
                headers=generation_headers,
                body=body,
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text[:1000]
            logger.error("OpenAI Images API error ({}): {}", response.status_code, detail)
            raise ImageGenerationError(
                f"OpenAI image generation failed (HTTP {response.status_code}): {detail}"
            ) from exc

        payload = response.json()
        logger.info("OpenAI Images API response ({}): {}", response.status_code,
                       {k: v for k, v in payload.items() if k != "data"})

        images = await self._parse_images_response(payload)
        self._require_images(images, payload)

        return GeneratedImageResponse(images=images, content="", raw=payload)


register_image_gen_provider(OpenAIImageGenerationClient)
