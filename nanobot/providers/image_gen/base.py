"""Shared base for image generation providers: errors, base class, registry, common helpers."""

from __future__ import annotations

import base64
import binascii
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

from nanobot.providers.registry import find_by_name
from nanobot.security.network import PinnedDNSAsyncTransport, UnsafeURLRequestError
from nanobot.utils.helpers import detect_image_mime

_DEFAULT_TIMEOUT_S = 120.0
_IMAGE_DOWNLOAD_MAX_BYTES = 32 * 1024 * 1024
_IMAGE_DOWNLOAD_MAX_REDIRECTS = 5


class ImageGenerationError(RuntimeError):
    """Raised when the image generation provider cannot return images."""


@dataclass(frozen=True)
class GeneratedImageResponse:
    """Images and optional text returned by the provider."""

    images: list[str]
    content: str
    raw: dict[str, Any]


def _read_image_b64(path: str | Path) -> tuple[str, str]:
    """Return ``(mime, base64)`` for the image at ``path``."""
    p = Path(path).expanduser()
    raw = p.read_bytes()
    mime = detect_image_mime(raw)
    if mime is None:
        raise ImageGenerationError(f"unsupported reference image: {p}")
    return mime, base64.b64encode(raw).decode("ascii")


def image_path_to_data_url(path: str | Path) -> str:
    """Convert a local image path to an image data URL."""
    mime, encoded = _read_image_b64(path)
    return f"data:{mime};base64,{encoded}"


def image_path_to_inline_data(path: str | Path) -> dict[str, str]:
    """Convert a local image path to a Gemini ``inlineData`` payload dict."""
    mime, encoded = _read_image_b64(path)
    return {"mimeType": mime, "data": encoded}


def _b64_image_data_url(value: str) -> str:
    encoded = "".join(value.split())
    try:
        raw = base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ImageGenerationError("generated image payload was not valid base64") from exc
    mime = detect_image_mime(raw)
    if mime is None:
        raise ImageGenerationError("generated image payload was not a supported image")
    return f"data:{mime};base64,{encoded}"


async def _download_image_data_url(
    url: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    try:
        safe_transport = PinnedDNSAsyncTransport(inner=transport)
        # Proxies resolve the target independently and would defeat DNS pinning.
        async with httpx.AsyncClient(
            transport=safe_transport,
            follow_redirects=False,
            timeout=_DEFAULT_TIMEOUT_S,
            trust_env=False,
        ) as client:
            current_url = url
            for _ in range(_IMAGE_DOWNLOAD_MAX_REDIRECTS + 1):
                async with client.stream("GET", current_url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise ImageGenerationError(
                                "generated image URL redirected without a location"
                            )
                        current_url = urljoin(str(response.url), location)
                        continue

                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        raise ImageGenerationError(
                            f"failed to download generated image (HTTP {response.status_code})"
                        ) from exc

                    declared_size = response.headers.get("content-length")
                    if declared_size:
                        try:
                            if int(declared_size) > _IMAGE_DOWNLOAD_MAX_BYTES:
                                raise ImageGenerationError(
                                    "generated image exceeded the 32 MiB download limit"
                                )
                        except ValueError:
                            pass

                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > _IMAGE_DOWNLOAD_MAX_BYTES:
                            raise ImageGenerationError(
                                "generated image exceeded the 32 MiB download limit"
                            )
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                    break
            else:
                raise ImageGenerationError("generated image URL exceeded the redirect limit")
    except UnsafeURLRequestError as exc:
        raise ImageGenerationError(f"blocked unsafe generated image URL: {exc}") from exc
    except httpx.RequestError as exc:
        raise ImageGenerationError(f"failed to download generated image: {exc}") from exc

    mime = detect_image_mime(raw)
    if mime is None:
        raise ImageGenerationError("generated image URL did not return a supported image")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _http_error_detail(response: httpx.Response) -> str:
    """Extract a readable error message from an HTTP error response."""
    try:
        data = response.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                return err.get("message") or str(err)
            if err:
                return str(err)
    except Exception:
        pass
    return response.text[:500] or "<empty response body>"


def _round_to_multiple(value: float, multiple: int = 8) -> int:
    rounded = int(round(value / multiple) * multiple)
    return max(multiple, rounded)


_IMAGE_GEN_PROVIDERS: dict[str, type["ImageGenerationProvider"]] = {}


def register_image_gen_provider(cls: type["ImageGenerationProvider"]) -> None:
    """Register an image provider at import time only.

    The registry is populated by module side effects so provider discovery
    stays lazy and consistent across the process.
    """
    name = cls.provider_name
    if not name:
        raise ValueError(f"{cls.__name__} must set provider_name")
    _IMAGE_GEN_PROVIDERS[name] = cls


def get_image_gen_provider(name: str) -> type["ImageGenerationProvider"] | None:
    return _IMAGE_GEN_PROVIDERS.get(name)


def image_gen_provider_names() -> tuple[str, ...]:
    """Return registered image generation provider names in registry order."""
    return tuple(_IMAGE_GEN_PROVIDERS)


def image_gen_provider_configs(config: Any) -> dict[str, Any]:
    providers_cfg = config.providers
    return {
        name: pc
        for name in _IMAGE_GEN_PROVIDERS
        if (pc := getattr(providers_cfg, name, None)) is not None
    }


class ImageGenerationProvider(ABC):
    """Base class for image generation provider clients."""

    provider_name: str = ""
    model_options: tuple[str, ...] = ()
    missing_key_message: str = ""
    default_timeout: float = _DEFAULT_TIMEOUT_S

    def __init__(
        self,
        *,
        api_key: str | None,
        api_base: str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        proxy: str | None = None,
        timeout: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_base = self._resolve_base_url(api_base)
        self.extra_headers = extra_headers or {}
        self.extra_body = extra_body or {}
        self.proxy = proxy or None
        self.timeout = timeout if timeout is not None else self.default_timeout
        self._client = client

    def _resolve_base_url(self, api_base: str | None) -> str:
        if api_base:
            return api_base.rstrip("/")
        spec = find_by_name(self.provider_name)
        if spec and spec.default_api_base:
            return spec.default_api_base.rstrip("/")
        return self._default_base_url()

    def _default_base_url(self) -> str:
        return ""

    @abstractmethod
    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        reference_images: list[str] | None = None,
        aspect_ratio: str | None = None,
        image_size: str | None = None,
    ) -> GeneratedImageResponse: ...

    def _require_images(self, images: list[str], data: dict[str, Any]) -> None:
        if images:
            return
        provider_error = data.get("error") if isinstance(data, dict) else None
        label = self.provider_name
        if provider_error:
            raise ImageGenerationError(f"{label} returned no images: {provider_error}")
        raise ImageGenerationError(f"{label} returned no images for this request")

    async def _http_post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: dict[str, Any],
        client: httpx.AsyncClient | None = None,
    ) -> httpx.Response:
        if client is not None:
            return await client.post(url, headers=headers, json=body)
        if self._client is not None:
            return await self._client.post(url, headers=headers, json=body)
        client_kwargs: dict[str, Any] = {"timeout": self.timeout}
        if self.proxy:
            client_kwargs["proxy"] = self.proxy
            client_kwargs["trust_env"] = False
        async with httpx.AsyncClient(**client_kwargs) as c:
            return await c.post(url, headers=headers, json=body)
