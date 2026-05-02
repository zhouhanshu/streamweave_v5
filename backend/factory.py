"""Backend construction for the current process."""

from __future__ import annotations

from streamweave.config import BackendConfig, RuntimeConfig

from .base import BaseBackend, MockBackend
from .gemini import GeminiBackend
from .openai import OpenAICompatibleBackend
from .pool import BackendPool


def create_backend(config: BackendConfig, runtime: RuntimeConfig) -> BaseBackend:
    name = config.backend.lower()
    if name == "mock":
        return MockBackend()
    if name in {"openai", "vllm", "openai_compatible", "local"}:
        if config.endpoints and not config.base_url:
            return BackendPool(
                [OpenAICompatibleBackend(config, runtime, endpoint=endpoint) for endpoint in config.endpoints],
                config=config,
            )
        endpoint = (config.base_url or (config.endpoints[0] if config.endpoints else "")).strip()
        return OpenAICompatibleBackend(config, runtime, endpoint=endpoint or None)
    if name == "gemini":
        return GeminiBackend(config, runtime)
    raise ValueError(f"Unknown backend: {config.backend}")
