"""Image generation providers.

This module is a thin facade over the ``nanobot.providers.image_gen`` package:
it re-exports the shared base/registry and registers every provider so existing
import sites and module-level test patches keep working unchanged. Provider
implementations live one-per-file under ``image_gen/``.
"""

from __future__ import annotations

# Provider modules register themselves on import via side effects.
from nanobot.providers.image_gen import (  # noqa: E402,F401
    aihubmix,
    codex,
    custom,
    gemini,
    minimax,
    modelscope,
    ollama,
    openai,
    openrouter,
    stepfun,
    zhipu,
)
from nanobot.providers.image_gen.aihubmix import AIHubMixImageGenerationClient  # noqa: F401
from nanobot.providers.image_gen.base import (
    _IMAGE_DOWNLOAD_MAX_BYTES,  # noqa: F401 (re-export for tests)
    GeneratedImageResponse,
    ImageGenerationError,
    ImageGenerationProvider,
    _download_image_data_url,  # noqa: F401 (re-export for tests)
    get_image_gen_provider,
    image_gen_provider_configs,
    image_gen_provider_names,
    image_path_to_data_url,
    image_path_to_inline_data,
    register_image_gen_provider,
)
from nanobot.providers.image_gen.codex import CodexImageGenerationClient  # noqa: F401
from nanobot.providers.image_gen.custom import CustomImageGenerationClient  # noqa: F401
from nanobot.providers.image_gen.gemini import GeminiImageGenerationClient  # noqa: F401
from nanobot.providers.image_gen.minimax import MiniMaxImageGenerationClient  # noqa: F401
from nanobot.providers.image_gen.modelscope import ModelScopeImageGenerationClient  # noqa: F401
from nanobot.providers.image_gen.ollama import OllamaImageGenerationClient  # noqa: F401
from nanobot.providers.image_gen.openai import OpenAIImageGenerationClient  # noqa: F401
from nanobot.providers.image_gen.openrouter import OpenRouterImageGenerationClient  # noqa: F401
from nanobot.providers.image_gen.stepfun import StepFunImageGenerationClient  # noqa: F401
from nanobot.providers.image_gen.zhipu import ZhipuImageGenerationClient  # noqa: F401

__all__ = [
    "GeneratedImageResponse",
    "ImageGenerationError",
    "ImageGenerationProvider",
    "get_image_gen_provider",
    "image_gen_provider_configs",
    "image_gen_provider_names",
    "register_image_gen_provider",
    "image_path_to_data_url",
    "image_path_to_inline_data",
]
