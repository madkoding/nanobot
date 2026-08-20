"""ModelScope (魔搭) image generation provider."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from nanobot.providers.image_gen.base import (
    GeneratedImageResponse,
    ImageGenerationError,
    ImageGenerationProvider,
    _download_image_data_url,
    _http_error_detail,
    image_path_to_data_url,
    register_image_gen_provider,
)

_MODELSCOPE_TIMEOUT_S = 300.0
_MODELSCOPE_POLL_INTERVAL_S = 5.0
_MODELSCOPE_POLL_MAX_ATTEMPTS = 60  # 5 min at 5s intervals
_MODELSCOPE_ASPECT_RATIOS = {
    "1:1": "1328x1328",
    "16:9": "1664x928",
    "9:16": "928x1664",
    "3:4": "1140x1472",
    "4:3": "1472x1140",
}


def _modelscope_size(
    aspect_ratio: str | None,
    image_size: str | None,
) -> str:
    """Resolve aspect ratio / image_size to a ModelScope size string."""
    if image_size and "x" in image_size.lower():
        return image_size
    if aspect_ratio and aspect_ratio in _MODELSCOPE_ASPECT_RATIOS:
        return _MODELSCOPE_ASPECT_RATIOS[aspect_ratio]
    return "1024x1024"


class ModelScopeImageGenerationClient(ImageGenerationProvider):
    """Async client for ModelScope (魔搭) AIGC image generation.

    ModelScope uses an async task pattern: POST submits the job and returns
    a task_id, then the client polls GET /tasks/{task_id} until
    task_status is SUCCEED or FAILED.
    """

    provider_name = "modelscope"
    model_options = ("Qwen/Qwen-Image-2512",)
    missing_key_message = (
        "ModelScope API key is not configured. Set providers.modelscope.apiKey."
    )
    default_timeout = _MODELSCOPE_TIMEOUT_S

    def _default_base_url(self) -> str:
        return "https://api-inference.modelscope.cn/v1"

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
            "X-ModelScope-Async-Mode": "true",
            **self.extra_headers,
        }

        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
        }

        size = _modelscope_size(aspect_ratio, image_size)
        if size:
            body["size"] = size

        refs = list(reference_images or [])
        if refs:
            image_refs = [image_path_to_data_url(path) for path in refs]
            body["image_url"] = image_refs[0] if len(image_refs) == 1 else image_refs

        body.update(self.extra_body)

        url = f"{self.api_base}/images/generations"
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            return await self._generate_with_client(
                client,
                url=url,
                headers=headers,
                body=body,
            )
        finally:
            if self._client is None:
                await client.aclose()

    async def _generate_with_client(
        self,
        client: httpx.AsyncClient,
        *,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> GeneratedImageResponse:
        try:
            response = await client.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise ImageGenerationError("ModelScope image generation request timed out") from exc
        except httpx.RequestError as exc:
            raise ImageGenerationError(f"ModelScope image generation request failed: {exc}") from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = _http_error_detail(response)
            raise ImageGenerationError(f"ModelScope image generation failed: {detail}") from exc

        task_data = response.json()
        task_id = task_data.get("task_id")
        if not task_id:
            raise ImageGenerationError(
                f"ModelScope did not return a task_id: {response.text[:500]}"
            )

        images = await self._poll_task(client, task_id, headers)

        self._require_images(images, task_data)
        return GeneratedImageResponse(images=images, content="", raw=task_data)

    async def _poll_task(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        submit_headers: dict[str, str],
    ) -> list[str]:
        poll_headers = {
            "Authorization": submit_headers["Authorization"],
            "X-ModelScope-Task-Type": "image_generation",
            **self.extra_headers,
        }
        poll_url = f"{self.api_base}/tasks/{task_id}"

        for _ in range(_MODELSCOPE_POLL_MAX_ATTEMPTS):
            try:
                response = await client.get(poll_url, headers=poll_headers)
            except httpx.RequestError:
                await asyncio.sleep(_MODELSCOPE_POLL_INTERVAL_S)
                continue

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = _http_error_detail(response)
                raise ImageGenerationError(
                    f"ModelScope task polling failed: {detail}"
                ) from exc

            data = response.json()
            status = data.get("task_status")

            if status == "SUCCEED":
                return await self._collect_images(data)
            if status == "FAILED":
                raise ImageGenerationError(
                    f"ModelScope image generation task failed: {data}"
                )

            await asyncio.sleep(_MODELSCOPE_POLL_INTERVAL_S)

        raise ImageGenerationError(
            f"ModelScope image generation timed out after "
            f"{_MODELSCOPE_POLL_MAX_ATTEMPTS} polls"
        )

    @staticmethod
    async def _collect_images(
        data: dict[str, Any],
    ) -> list[str]:
        images: list[str] = []
        for url in data.get("output_images") or []:
            if isinstance(url, str) and url:
                if url.startswith("data:image/"):
                    images.append(url)
                else:
                    images.append(await _download_image_data_url(url))
        return images


register_image_gen_provider(ModelScopeImageGenerationClient)
