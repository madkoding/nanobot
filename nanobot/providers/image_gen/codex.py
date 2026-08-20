"""OpenAI Codex image generation provider (Responses API via Codex OAuth)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from loguru import logger

from nanobot.providers.image_gen.base import (
    GeneratedImageResponse,
    ImageGenerationError,
    ImageGenerationProvider,
    register_image_gen_provider,
)
from nanobot.providers.image_gen.openai_shared import _parse_codex_sse_images


class CodexImageGenerationClient(ImageGenerationProvider):
    """OpenAI image generation via Codex subscription OAuth.

    Uses the Codex Responses API with the ``image_generation`` tool
    (the same mechanism ChatGPT uses internally).  No API key required —
    the Codex OAuth token from ``oauth_cli_kit`` is used instead.
    """

    provider_name = "openai_codex"
    model_options = ("gpt-5.4",)
    missing_key_message = (
        "Codex OAuth token is unavailable. "
        "Log in with Codex subscription first."
    )

    def _default_base_url(self) -> str:
        return "https://chatgpt.com/backend-api"

    def _codex_model(self, model: str) -> str:
        """Strip the ``openai-codex/`` prefix if present."""
        if model.startswith(("openai-codex/", "openai_codex/")):
            return model.split("/", 1)[1]
        return model

    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        reference_images: list[str] | None = None,
        aspect_ratio: str | None = None,
        image_size: str | None = None,
    ) -> GeneratedImageResponse:
        try:
            from oauth_cli_kit import get_token as get_codex_token
        except ImportError:
            raise ImageGenerationError(self.missing_key_message)

        try:
            token_kwargs = {"proxy": self.proxy} if self.proxy else {}
            token = await asyncio.to_thread(get_codex_token, **token_kwargs)
        except Exception as exc:
            raise ImageGenerationError(self.missing_key_message) from exc
        if not token or not token.access:
            raise ImageGenerationError(self.missing_key_message)

        logger.info(
            "Using Codex OAuth token for image generation (account: {})",
            token.account_id,
        )

        if reference_images:
            logger.warning(
                "Codex image generation does not support reference images; "
                "ignoring {} reference image(s)",
                len(reference_images),
            )

        headers = {
            "Authorization": f"Bearer {token.access}",
            "chatgpt-account-id": token.account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": "nanobot",
            "User-Agent": "nanobot (python)",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        body: dict[str, Any] = {
            "model": self._codex_model(model),
            "instructions": "Generate an image based on the user's request.",
            "input": [{"role": "user", "content": prompt}],
            "tools": [{"type": "image_generation"}],
            "tool_choice": "auto",
            "stream": True,
            "store": False,
        }
        body.update(self.extra_body)

        logger.info("Codex Responses API request: POST {}/codex/responses body={}",
                       self.api_base, {k: v for k, v in body.items() if k != "input"})

        response = await self._http_post(
            f"{self.api_base}/codex/responses",
            headers=headers,
            body=body,
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text[:1000]
            logger.error("Codex Responses API error ({}): {}", response.status_code, detail)
            raise ImageGenerationError(
                f"Codex image generation failed (HTTP {response.status_code}): {detail}"
            ) from exc

        images, content_text = await _parse_codex_sse_images(response)

        raw = {"status": "completed"}
        self._require_images(images, raw)

        return GeneratedImageResponse(images=images, content=content_text, raw=raw)


register_image_gen_provider(CodexImageGenerationClient)
